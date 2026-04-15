#@title 🧱 Transformer (ALiBi + LoRA in Q/K/V + pluggable caches)
class MultiHeadSelfAttn(nn.Module):
    def __init__(self, embed_size, n_heads, dropout=0.1, lora_r=8, lora_alpha=16):
        super().__init__()
        assert embed_size % n_heads == 0
        self.C = embed_size
        self.H = n_heads
        self.dh = embed_size // n_heads

        # LoRA-enabled projections
        self.Wq = LoRALinear(self.C, self.C, r=lora_r, alpha=lora_alpha)
        self.Wk = LoRALinear(self.C, self.C, r=lora_r, alpha=lora_alpha)
        self.Wv = LoRALinear(self.C, self.C, r=lora_r, alpha=lora_alpha)
        self.proj = nn.Linear(self.C, self.C, bias=False)
        self.drop = nn.Dropout(dropout)

        # ALiBi slopes (per head)
        self.alibi_slopes = build_alibi_slopes(self.H)  # [H]

        # cache policy assigned later per block, with independent state
        self.cache_policy: CachePolicy = None

    def reset_cache(self, B, device):
        if self.cache_policy is not None:
            # merged-head width is C
            self.cache_policy.reset(B, self.C, device)

    def _alibi_training_bias(self, T, device):
        # bias[h,i,j] = slope[h]*(i - j)
        slopes = self.alibi_slopes.to(device)               # [H]
        i = torch.arange(T, device=device).unsqueeze(1)      # [T,1]
        j = torch.arange(T, device=device).unsqueeze(0)      # [1,T]
        diff = (i - j).to(torch.float32)                     # [T,T]
        bias = slopes[:, None, None] * diff[None, :, :]      # [H,T,T]
        return bias

    def _alibi_stream_bias(self, q_pos: int, Tm: int, device):
        # streaming: one query at pos=q_pos, keys at j=0..Tm-1 → bias[h,j] = slope[h]*(q_pos - j)
        slopes = self.alibi_slopes.to(device)                # [H]
        j = torch.arange(Tm, device=device).to(torch.float32)
        diff = (float(q_pos) - j)[None, :]                   # [1,Tm]
        # We use mean slope here since we merged heads into [B,1,C]
        slope_mean = slopes.mean()
        bias = slope_mean * diff                             # [1,Tm]
        return bias.unsqueeze(1)                             # [1,1,Tm] to add to [B,1,Tm]

    def forward(self, x, use_cache=False, cos=None, sin=None, stream_pos: int = 0):
        B, T, C = x.shape
        H, dh = self.H, self.dh

        q = self.Wq(x).view(B, T, H, dh)
        k = self.Wk(x).view(B, T, H, dh)
        v = self.Wv(x).view(B, T, H, dh)

        if not use_cache:
            # Training path: full attention with ALiBi bias
            qh = q.permute(0,2,1,3)  # [B,H,T,dh]
            kh = k.permute(0,2,1,3)
            vh = v.permute(0,2,1,3)

            scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(dh)  # [B,H,T,T]
            bias = self._alibi_training_bias(T, x.device)         # [H,T,T]
            scores = scores + bias.unsqueeze(0)                   # broadcast over batch

            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask

            wei = scores.softmax(dim=-1)
            out = (wei @ vh).transpose(1,2).contiguous().view(B, T, C)
            return self.drop(self.proj(out))
        else:
            # Streaming path: T must be 1, merged-head caching
            assert T == 1, "Streaming path expects T=1"
            q_ = q.permute(0,2,1,3).contiguous().view(B, 1, C)  # [B,1,C]
            k_ = k.permute(0,2,1,3).contiguous().view(B, 1, C)
            v_ = v.permute(0,2,1,3).contiguous().view(B, 1, C)

            self.cache_policy.append(k_, v_)
            K_all, V_all = self.cache_policy.memory()  # [B,Tm,C] or None

            if K_all is None:
                out = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
            else:
                wei = (q_ @ K_all.transpose(-1, -2)) / math.sqrt(dh)   # [B,1,Tm]
                # Add ALiBi streaming bias (approximate mean slope since heads merged)
                bias = self._alibi_stream_bias(stream_pos, K_all.size(1), x.device)  # [1,1,Tm]
                wei = wei + bias
                wei = wei.softmax(dim=-1)
                out = wei @ V_all                                       # [B,1,C]

            return self.drop(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size, embed_size),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_size, n_heads, dropout=0.1, lora_r=8, lora_alpha=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadSelfAttn(embed_size, n_heads, dropout=dropout, lora_r=lora_r, lora_alpha=lora_alpha)
        self.ff = FeedForward(embed_size, dropout=dropout)
    def reset_cache(self, B, device): self.attn.reset_cache(B, device)
    def forward(self, x, use_cache=False, stream_pos: int = 0):
        x = x + self.attn(self.ln1(x), use_cache=use_cache, stream_pos=stream_pos)
        x = x + self.ff(self.ln2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, embed_size=384, n_layers=6, n_heads=6, block_size=256, dropout=0.1, lora_r=8, lora_alpha=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.block_size = block_size

        self.tok = nn.Embedding(vocab_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(embed_size, n_heads, dropout=dropout, lora_r=lora_r, lora_alpha=lora_alpha)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

        # shared learned-pooler so its params are in the model
        self.hikv_pooler = LearnedPooler(embed_size)
        self.stream_pos = 0

    def set_cache_policy(self, policy: CachePolicy):
        """
        Assign per-layer independent caches with shared pooler (if HiKV_Learned).
        """
        for b in self.blocks:
            # create a layer-local empty copy
            b.attn.cache_policy = policy.new_empty_like()

    def set_cache_policy_hikv_learned(self, W0=64, W1=16, W2=4):
        # build per-layer HiKV_Learned referencing the model-level pooler
        for b in self.blocks:
            b.attn.cache_policy = HiKV_Learned(W0, W1, W2, embed_dim=self.embed_size, pooler=self.hikv_pooler)

    def reset_cache(self, B=1, device='cuda'):
        self.stream_pos = 0
        for b in self.blocks:
            b.reset_cache(B, device)

    def forward(self, idx, targets=None, use_cache=False):
        B, T = idx.shape
        assert (not use_cache and T <= self.block_size) or use_cache, "T exceeds block_size in training"
        x = self.tok(idx)
        x = self.drop(x)

        if not use_cache:
            for b in self.blocks:
                x = b(x, use_cache=False)
        else:
            # streaming, T should be 1
            for b in self.blocks:
                x = b(x, use_cache=True, stream_pos=self.stream_pos)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None and not use_cache:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        if use_cache:
            self.stream_pos += T
        return logits, loss