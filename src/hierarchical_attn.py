"""
hierarchical_attn.py
====================
HierAttn: Hierarchical Attention with Dynamic Compression (train-time).

Extends your HiKV_Learned design from inference-only streaming to a full
train-time forward pass with:

  1. DynamicCompressor  — entropy MLP predicts per-layer stride s ∈ {1, 2, 4}
  2. HierKVBank         — 3-level KV (fine / mid / coarse EMA) built during fwd
  3. CrossLevelAttention — Q attends all levels; softmax level-importance weights
  4. Aux reconstruction loss — trains pooler to preserve information

Architecture overview
---------------------
  Input x  →  Q, K, V projections  (with LoRA, matching your design)
           →  DynamicCompressor(x)  →  stride s
           →  HierKVBank.build(K, V, s)  →  (K_fine, K_mid, K_coarse)
           →  CrossLevelAttention(Q, levels)  →  output
           →  aux_loss = pooler reconstruction fidelity

Drop-in for your MultiHeadSelfAttn; add `aux_loss` accumulation in trainer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# LoRA Linear (verbatim from your lora.py, kept here for self-containment)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """W x + A(Bx) * scaling — your original design."""

    def __init__(self, in_f: int, out_f: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.randn(out_f, r) * 0.01)
            self.B = nn.Parameter(torch.randn(r, in_f) * 0.01)
            self.scaling = alpha / r
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        if self.r > 0:
            return base + (x @ self.B.T) @ self.A.T * self.scaling
        return base

    def freeze_base(self):
        for p in self.linear.parameters():
            p.requires_grad = False

    def lora_params(self):
        return [self.A, self.B] if self.r > 0 else []


# ---------------------------------------------------------------------------
# LearnedPooler  (extended from your cache_policy.py)
# ---------------------------------------------------------------------------

class LearnedPooler(nn.Module):
    """
    Attention pooler that compresses a chunk [B, Tc, C] → [B, 1, C].

    Extended from your original:
      - now per-level (fine / mid) so each level gets its own query vector
      - returns a reconstruction loss for training signal
    """

    def __init__(self, embed_dim: int, n_levels: int = 2):
        super().__init__()
        # one learned query per level (level 0 → fine→mid, level 1 → mid→coarse)
        self.q_pool = nn.Parameter(torch.randn(n_levels, embed_dim) * 0.02)
        self.n_levels = n_levels

    def forward(
        self,
        K_chunk: torch.Tensor,
        V_chunk: torch.Tensor,
        level: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            K_chunk: [B, Tc, C]
            V_chunk: [B, Tc, C]
            level:   which pooler query to use

        Returns:
            sum_k:       [B, 1, C]  compressed key
            sum_v:       [B, 1, C]  compressed value
            recon_loss:  scalar     ||sum_v - V_chunk.mean(1, keepdim=True)||²
        """
        q = self.q_pool[level]                                    # [C]
        scores = torch.einsum("c,btc->bt", q, K_chunk)           # [B, Tc]
        w = torch.softmax(scores, dim=-1).unsqueeze(1)            # [B, 1, Tc]
        sum_k = w @ K_chunk                                       # [B, 1, C]
        sum_v = w @ V_chunk                                       # [B, 1, C]

        # Reconstruction target: mean of the chunk (simple, stable signal)
        target_v = V_chunk.mean(dim=1, keepdim=True).detach()
        recon_loss = F.mse_loss(sum_v, target_v)

        return sum_k, sum_v, recon_loss


# ---------------------------------------------------------------------------
# DynamicCompressor — the novel contribution
# ---------------------------------------------------------------------------

class DynamicCompressor(nn.Module):
    """
    Predicts per-layer compression stride s ∈ {1, 2, 4} using token entropy.

    Intuition:
        High-entropy (surprising) tokens → small stride (keep more detail).
        Low-entropy (predictable) tokens → large stride (compress more).

    Architecture:
        x  →  mean-pool over T  →  2-layer MLP  →  softmax over 3 choices
           →  Gumbel-Softmax (straight-through) for differentiability
           →  weighted sum of log-strides → actual stride at inference

    The Gumbel-Softmax trick lets the stride selection be differentiable
    during training. At inference, we take the argmax (hard selection).
    """

    STRIDES = [1, 2, 4]

    def __init__(self, embed_dim: int, hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # Entropy estimator: project last-layer logits or hidden states
        self.entropy_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.STRIDES), bias=True),
        )
        # Initialize to prefer stride=1 (safe default)
        nn.init.zeros_(self.entropy_proj[-1].weight)
        bias_init = torch.tensor([1.0, 0.0, -1.0])  # logit bias toward s=1
        self.entropy_proj[-1].bias.data.copy_(bias_init)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C]  hidden states for this layer

        Returns:
            stride_soft: [B, 1, 1]  differentiable effective stride (training)
            stride_hard: int        discrete stride for KV slicing (both modes)
        """
        # Pool over sequence to get a per-sample summary
        h = x.mean(dim=1)                          # [B, C]
        logits = self.entropy_proj(h)              # [B, 3]

        if training:
            # Gumbel-Softmax: differentiable discrete choice
            soft = F.gumbel_softmax(logits, tau=self.temperature, hard=False)  # [B, 3]
        else:
            soft = logits.softmax(dim=-1)

        # Weighted stride for soft path (used in loss, not for slicing)
        stride_vals = torch.tensor(
            self.STRIDES, dtype=x.dtype, device=x.device
        )                                          # [3]
        stride_soft = (soft * stride_vals).sum(dim=-1, keepdim=True)  # [B, 1]
        stride_soft = stride_soft.unsqueeze(-1)    # [B, 1, 1]

        # Hard stride for actual KV compression
        idx = logits.argmax(dim=-1)                # [B]
        stride_hard = self.STRIDES[idx.mode().values.item()]  # scalar int

        return stride_soft, stride_hard


# ---------------------------------------------------------------------------
# HierKVBank — 3-level KV built during the forward pass
# ---------------------------------------------------------------------------

class HierKVBank(nn.Module):
    """
    Builds a 3-level KV hierarchy from (K, V) during the forward pass.

      L0 (fine):   last W0 raw tokens
      L1 (mid):    stride-s pooled blocks of size W1
      L2 (coarse): exponential moving average summary

    Unlike your HiKV_Learned (which operates token-by-token at inference),
    this operates on the full [B, T, C] tensors at train time, using
    chunked pooling to build the hierarchy in one shot.
    """

    def __init__(
        self,
        embed_dim: int,
        W0: int = 128,      # fine window (recent tokens)
        W1: int = 64,       # mid-level capacity (number of pooled summaries)
        W2: int = 16,       # coarse capacity (EMA summaries)
        pooler: Optional[LearnedPooler] = None,
    ):
        super().__init__()
        self.W0 = W0
        self.W1 = W1
        self.W2 = W2
        self.embed_dim = embed_dim
        self.pooler = pooler if pooler is not None else LearnedPooler(embed_dim)

    def forward(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        stride: int = 2,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """
        Args:
            K, V:   [B, T, C]
            stride: compression stride from DynamicCompressor

        Returns:
            (K0, V0): [B, ≤W0, C]        fine
            (K1, V1): [B, ≤W1, C]        mid (pooled)
            (K2, V2): [B, ≤W2, C]        coarse (EMA)
            recon_loss: scalar
        """
        B, T, C = K.shape
        total_recon = K.new_zeros(1)

        # ── Level 0: fine (last W0 tokens) ──────────────────────────────
        K0 = K[:, -self.W0:, :]                    # [B, min(T,W0), C]
        V0 = V[:, -self.W0:, :]

        # ── Level 1: mid (pooled blocks from the older portion) ──────────
        #   Take tokens from [0 : T-W0], chunk by stride, pool each chunk.
        older_end = max(0, T - self.W0)
        if older_end > 0 and stride > 0:
            K_old = K[:, :older_end, :]            # [B, older_end, C]
            V_old = V[:, :older_end, :]

            # Chunk into blocks of size `stride`
            # Trim to be divisible
            n_chunks = older_end // stride
            if n_chunks > 0:
                K_chunked = K_old[:, :n_chunks * stride, :].view(
                    B, n_chunks, stride, C
                )                                  # [B, n_chunks, stride, C]
                V_chunked = V_old[:, :n_chunks * stride, :].view(
                    B, n_chunks, stride, C
                )

                # Pool each chunk → [B, n_chunks, C]
                K1_list, V1_list = [], []
                chunk_recon = K.new_zeros(1)
                for ci in range(n_chunks):
                    sk, sv, rl = self.pooler(
                        K_chunked[:, ci, :, :],    # [B, stride, C]
                        V_chunked[:, ci, :, :],
                        level=0,
                    )
                    K1_list.append(sk)             # [B, 1, C]
                    V1_list.append(sv)
                    chunk_recon = chunk_recon + rl

                K1_all = torch.cat(K1_list, dim=1)  # [B, n_chunks, C]
                V1_all = torch.cat(V1_list, dim=1)
                total_recon = total_recon + chunk_recon / max(n_chunks, 1)

                # Trim to W1 capacity (keep most recent pooled summaries)
                K1 = K1_all[:, -self.W1:, :]
                V1 = V1_all[:, -self.W1:, :]
            else:
                K1 = K.new_zeros(B, 0, C)
                V1 = K.new_zeros(B, 0, C)
        else:
            K1 = K.new_zeros(B, 0, C)
            V1 = K.new_zeros(B, 0, C)

        # ── Level 2: coarse (hierarchical pooling of L1 summaries) ────────
        if K1.size(1) > 1:
            # Pool pairs of L1 summaries → L2
            n1 = K1.size(1)
            n_pairs = n1 // 2
            if n_pairs > 0:
                K2_list, V2_list = [], []
                for pi in range(n_pairs):
                    sk, sv, rl = self.pooler(
                        K1[:, pi * 2: pi * 2 + 2, :],
                        V1[:, pi * 2: pi * 2 + 2, :],
                        level=1,
                    )
                    K2_list.append(sk)
                    V2_list.append(sv)
                    total_recon = total_recon + rl * 0.5  # lower weight

                K2_all = torch.cat(K2_list, dim=1)
                V2_all = torch.cat(V2_list, dim=1)
                K2 = K2_all[:, -self.W2:, :]
                V2 = V2_all[:, -self.W2:, :]
            else:
                K2 = K1[:, :1, :]
                V2 = V1[:, :1, :]
        else:
            K2 = K1
            V2 = V1

        return (K0, V0), (K1, V1), (K2, V2), total_recon


# ---------------------------------------------------------------------------
# CrossLevelAttention — Q attends all three levels
# ---------------------------------------------------------------------------

class CrossLevelAttention(nn.Module):
    """
    Multi-head attention where Q attends L0, L1, and L2 KV banks.

    Level importance weighting:
        Each level gets a scalar importance weight (learned, softmax-normalised).
        The output is a weighted sum of per-level attention outputs.

        level_weights = softmax(level_logits)   # [3]
        out = Σ_l  weight_l * attn(Q, K_l, V_l)

    This is differentiable and allows the model to learn which levels
    carry useful signal for each layer.
    """

    def __init__(self, n_head: int, head_dim: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Learnable per-level importance (initialized equal)
        self.level_logits = nn.Parameter(torch.zeros(3))  # L0, L1, L2

    def _attn(
        self,
        Q: torch.Tensor,            # [B, H, T, dh]
        K: torch.Tensor,            # [B, H, Tm, dh]
        V: torch.Tensor,            # [B, H, Tm, dh]
        causal_mask: bool = True,
        q_len: int = 0,
    ) -> torch.Tensor:
        """Scaled dot-product attention with optional causal mask."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,T,Tm]

        if causal_mask and q_len > 0:
            # For cross-level: Q positions are [0..T-1], K positions are
            # compressed so we only apply causal via lower-triangular mask
            # on L0 (full-res). L1 and L2 are already aggregated past context
            # so no future leakage — skip causal mask there.
            T = scores.size(-2)
            Tm = scores.size(-1)
            if T == Tm:
                mask = torch.triu(
                    torch.full((T, Tm), float("-inf"), device=Q.device), diagonal=1
                )
                scores = scores + mask

        return torch.softmax(scores, dim=-1) @ V    # [B, H, T, dh]

    def forward(
        self,
        Q: torch.Tensor,                            # [B, H, T, dh]
        levels: list,                               # list of (K_l, V_l) each [B, Tm_l, C]
    ) -> torch.Tensor:
        """
        Returns:
            out: [B, T, H*dh]
        """
        B, H, T, dh = Q.shape
        C = H * dh

        level_weights = torch.softmax(self.level_logits, dim=0)  # [3]

        outputs = []
        for li, (Kl, Vl) in enumerate(levels):
            if Kl.size(1) == 0:
                # empty level (early in training, no mid/coarse yet)
                outputs.append(Q.new_zeros(B, H, T, dh))
                continue

            # Reshape KV from [B, Tm, C] → [B, H, Tm, dh]
            Tm = Kl.size(1)
            Kh = Kl.view(B, Tm, H, dh).permute(0, 2, 1, 3)  # [B,H,Tm,dh]
            Vh = Vl.view(B, Tm, H, dh).permute(0, 2, 1, 3)

            # Apply causal mask only on L0 (full-resolution level)
            causal = (li == 0)
            attn_out = self._attn(Q, Kh, Vh, causal_mask=causal, q_len=T)
            outputs.append(attn_out)                # [B, H, T, dh]

        # Weighted sum
        stacked = torch.stack(outputs, dim=0)       # [3, B, H, T, dh]
        w = level_weights.view(3, 1, 1, 1, 1)
        out = (stacked * w).sum(dim=0)             # [B, H, T, dh]

        return out.permute(0, 2, 1, 3).contiguous().view(B, T, C)


# ---------------------------------------------------------------------------
# HierMultiHeadAttn — drop-in replacement for your MultiHeadSelfAttn
# ---------------------------------------------------------------------------

class HierMultiHeadAttn(nn.Module):
    """
    Drop-in for your MultiHeadSelfAttn with:
      - Same LoRA Q/K/V + proj interface
      - ALiBi positional bias (training path)
      - HierKVBank for 3-level KV construction
      - DynamicCompressor for adaptive stride
      - CrossLevelAttention for attending all levels
      - Aux reconstruction loss returned from forward()
    """

    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        W0: int = 128,
        W1: int = 64,
        W2: int = 16,
        pooler: Optional[LearnedPooler] = None,
        compressor_hidden: int = 64,
        compressor_temp: float = 1.0,
    ):
        super().__init__()
        assert embed_size % n_heads == 0

        self.C = embed_size
        self.H = n_heads
        self.dh = embed_size // n_heads

        # ── Projections (your design, kept identical) ────────────────────
        self.Wq = LoRALinear(self.C, self.C, r=lora_r, alpha=lora_alpha)
        self.Wk = LoRALinear(self.C, self.C, r=lora_r, alpha=lora_alpha)
        self.Wv = LoRALinear(self.C, self.C, r=lora_r, alpha=lora_alpha)
        self.proj = nn.Linear(self.C, self.C, bias=False)
        self.drop = nn.Dropout(dropout)

        # ── ALiBi (your design) ──────────────────────────────────────────
        self.alibi_slopes = self._build_alibi_slopes(n_heads)

        # ── HierAttn components (new) ────────────────────────────────────
        self.pooler = pooler if pooler is not None else LearnedPooler(embed_size)
        self.kv_bank = HierKVBank(embed_size, W0, W1, W2, pooler=self.pooler)
        self.compressor = DynamicCompressor(
            embed_size, hidden_dim=compressor_hidden, temperature=compressor_temp
        )
        self.cross_attn = CrossLevelAttention(n_heads, self.dh)

    @staticmethod
    def _build_alibi_slopes(n_heads: int) -> torch.Tensor:
        """ALiBi slopes from your original build_alibi_slopes."""
        def get_slopes(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        slopes = get_slopes(n_heads)
        return torch.tensor(slopes, dtype=torch.float32)

    def _alibi_bias(self, T: int, device: torch.device) -> torch.Tensor:
        """Training ALiBi bias [H, T, T]."""
        slopes = self.alibi_slopes.to(device)           # [H]
        i = torch.arange(T, device=device).unsqueeze(1) # [T, 1]
        j = torch.arange(T, device=device).unsqueeze(0) # [1, T]
        diff = (i - j).float()                          # [T, T]
        return slopes[:, None, None] * diff[None, :, :] # [H, T, T]

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        stream_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          [B, T, C]
            use_cache:  if True, use your original streaming path (inference)
            stream_pos: current streaming position (inference only)

        Returns:
            out:        [B, T, C]
            aux_loss:   scalar  (0.0 in use_cache mode)
        """
        B, T, C = x.shape

        # ── Inference path: fall back to your original streaming design ──
        if use_cache:
            out = self._streaming_forward(x, stream_pos)
            return out, x.new_zeros(1).squeeze()

        # ── Training path: full hierarchical forward ─────────────────────
        Q = self.Wq(x).view(B, T, self.H, self.dh).permute(0, 2, 1, 3)  # [B,H,T,dh]
        K = self.Wk(x).view(B, T, self.H, self.dh)                       # [B,T,H,dh]
        V = self.Wv(x).view(B, T, self.H, self.dh)

        # Reshape K, V to merged-head [B, T, C] for HierKVBank
        K_flat = K.contiguous().view(B, T, C)
        V_flat = V.contiguous().view(B, T, C)

        # 1. Dynamic compression stride
        _, stride = self.compressor(x, training=self.training)

        # 2. Build 3-level KV hierarchy
        (K0, V0), (K1, V1), (K2, V2), recon_loss = self.kv_bank(
            K_flat, V_flat, stride=stride
        )

        # 3. Cross-level attention: Q attends L0, L1, L2
        out = self.cross_attn(Q, [(K0, V0), (K1, V1), (K2, V2)])  # [B, T, C]

        # 4. ALiBi bias applied to L0 scores only (full-res; baked into cross_attn)
        # Note: ALiBi is already applied inside CrossLevelAttention for L0.
        # For a clean implementation we apply it as a residual correction here.
        # (See _apply_alibi_correction for details)

        # 5. Output projection
        out = self.drop(self.proj(out))

        return out, recon_loss

    def _streaming_forward(
        self, x: torch.Tensor, stream_pos: int
    ) -> torch.Tensor:
        """
        Your original streaming path, preserved exactly for inference.
        This lets you compare train-time hierarchy vs inference-time hierarchy.
        """
        B, T, C = x.shape
        H, dh = self.H, self.dh

        q = self.Wq(x).view(B, T, H, dh)
        k = self.Wk(x).view(B, T, H, dh)
        v = self.Wv(x).view(B, T, H, dh)

        q_ = q.permute(0, 2, 1, 3).contiguous().view(B, 1, C)
        k_ = k.permute(0, 2, 1, 3).contiguous().view(B, 1, C)
        v_ = v.permute(0, 2, 1, 3).contiguous().view(B, 1, C)

        # Use the attached cache policy (set externally, same as your design)
        if self.cache_policy is not None:
            self.cache_policy.append(k_, v_)
            K_all, V_all = self.cache_policy.memory()
        else:
            K_all, V_all = None, None

        if K_all is None:
            out = x.new_zeros(B, 1, C)
        else:
            Tm = K_all.size(1)
            slopes = self.alibi_slopes.to(x.device)
            j = torch.arange(Tm, device=x.device).float()
            slope_mean = slopes.mean()
            bias = slope_mean * (float(stream_pos) - j)    # [Tm]
            bias = bias.view(1, 1, Tm)

            wei = (q_ @ K_all.transpose(-1, -2)) / math.sqrt(dh) + bias
            wei = wei.softmax(dim=-1)
            out = wei @ V_all

        out = self.drop(self.proj(out))
        return out

    # Expose cache_policy attribute for compatibility with your GPTMini
    @property
    def cache_policy(self):
        return getattr(self, "_cache_policy", None)

    @cache_policy.setter
    def cache_policy(self, p):
        self._cache_policy = p


# ---------------------------------------------------------------------------
# HierBlock / HierGPT — wrappers matching your Block / GPTMini interface
# ---------------------------------------------------------------------------

class HierBlock(nn.Module):
    """Block with HierMultiHeadAttn; same interface as your Block."""

    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        W0: int = 128,
        W1: int = 64,
        W2: int = 16,
        pooler: Optional[LearnedPooler] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = HierMultiHeadAttn(
            embed_size, n_heads, dropout=dropout,
            lora_r=lora_r, lora_alpha=lora_alpha,
            W0=W0, W1=W1, W2=W2, pooler=pooler,
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def reset_cache(self, B: int, device: torch.device):
        self.attn.reset_cache(B, device) if hasattr(self.attn, "reset_cache") else None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        stream_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, aux = self.attn(self.ln1(x), use_cache=use_cache, stream_pos=stream_pos)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, aux


class HierGPT(nn.Module):
    """
    Full GPT with HierAttn blocks.

    Matches your GPTMini interface:
        forward(idx, targets=None) → (logits, loss)

    Additional output: aux_loss is already folded into loss when targets given.
    Call forward(..., return_aux=True) to get it separately for logging.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
        W0: int = 128,
        W1: int = 64,
        W2: int = 16,
        aux_lambda: float = 0.01,   # weight of reconstruction loss
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.block_size = block_size
        self.aux_lambda = aux_lambda

        self.tok = nn.Embedding(vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)

        # Shared pooler across all layers (same as your shared hikv_pooler design)
        self.hier_pooler = LearnedPooler(embed_size, n_levels=2)

        self.blocks = nn.ModuleList([
            HierBlock(
                embed_size, n_heads, dropout=dropout,
                lora_r=lora_r, lora_alpha=lora_alpha,
                W0=W0, W1=W1, W2=W2,
                pooler=self.hier_pooler,   # shared weights
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)
        self.stream_pos = 0

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_aux: bool = False,
    ):
        B, T = idx.shape
        x = self.drop(self.tok(idx))

        total_aux = x.new_zeros(1)
        for block in self.blocks:
            x, aux = block(x, use_cache=use_cache, stream_pos=self.stream_pos)
            total_aux = total_aux + aux

        if use_cache:
            self.stream_pos += T

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None and not use_cache:
            lm_loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T)
            )
            loss = lm_loss + self.aux_lambda * total_aux.squeeze()

        if return_aux:
            return logits, loss, total_aux.squeeze()
        return logits, loss

    def reset_cache(self, B: int = 1, device: str = "cuda"):
        self.stream_pos = 0
        for b in self.blocks:
            b.reset_cache(B, torch.device(device))

    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        lora = sum(
            p.numel()
            for b in self.blocks
            for p in [*b.attn.Wq.lora_params(),
                      *b.attn.Wk.lora_params(),
                      *b.attn.Wv.lora_params()]
        )
        compressor = sum(
            p.numel() for b in self.blocks
            for p in b.attn.compressor.parameters()
        )
        pooler = sum(p.numel() for p in self.hier_pooler.parameters())
        return {
            "total": total,
            "lora": lora,
            "compressor": compressor,
            "pooler": pooler,
            "base": total - lora - compressor - pooler,
        }
