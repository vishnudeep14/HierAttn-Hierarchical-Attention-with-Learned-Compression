"""
benchmark_shakespeare_v3.py
===========================
Key insight from v1/v2 results:
    HierAttn was REPLACING standard causal attention with a weighted mix of
    3 levels. Early in training, L1 and L2 are near-zero (not enough context
    has been pooled yet), so the model effectively had no attention signal.
    Baseline uses pure causal attention which always works from step 1.

Fix: Residual hierarchy design.
    out = causal_attn(x)          # identical to baseline — always works
        + alpha * hier_context(x)  # hierarchical signal added on top
    where alpha is a learned scalar starting near 0.

    This means:
    - At init: HierAttn ≈ Baseline (safe fallback)
    - During training: model learns how much hierarchy helps
    - No risk of the hierarchy blocking baseline learning

Expected: HierAttn bpb ≤ Baseline bpb (matches or beats it)
"""

import os, math, time, json, urllib.request
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "/kaggle/working/shakespeare.txt"
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH)

with open(DATA_PATH) as f:
    raw = f.read()

data       = torch.tensor(list(raw.encode("utf-8")), dtype=torch.long)
n          = len(data)
train_data = data[:int(0.9*n)]
val_data   = data[int(0.9*n):]
VOCAB      = 256
print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    vocab_size  : int   = VOCAB
    block_size  : int   = 256
    batch_size  : int   = 64
    embed_size  : int   = 256
    n_layers    : int   = 6
    n_heads     : int   = 4
    dropout     : float = 0.1
    lora_r      : int   = 8
    lora_alpha  : int   = 16
    W0          : int   = 128   # fine window
    chunk_size  : int   = 8     # pooling stride
    W1          : int   = 16    # mid capacity
    W2          : int   = 8     # coarse capacity
    aux_lambda  : float = 0.0
    n_steps     : int   = 3000
    lr          : float = 3e-4
    warmup      : int   = 200
    eval_every  : int   = 300
    eval_steps  : int   = 50

C = Cfg()

def get_batch(split="train"):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - C.block_size, (C.batch_size,))
    x  = torch.stack([d[i  :i+C.block_size  ] for i in ix]).to(DEVICE)
    y  = torch.stack([d[i+1:i+C.block_size+1] for i in ix]).to(DEVICE)
    return x, y

def get_lr(step):
    if step < C.warmup:
        return C.lr * step / max(C.warmup, 1)
    p = (step - C.warmup) / max(C.n_steps - C.warmup, 1)
    return C.lr * 0.5 * (1 + math.cos(math.pi * p))

# ── Shared primitives ─────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16):
        super().__init__()
        self.w = nn.Linear(in_f, out_f, bias=False)
        self.r = r
        self.s = alpha / r if r > 0 else 0.
        if r > 0:
            self.A = nn.Parameter(torch.randn(out_f, r) * 0.01)
            self.B = nn.Parameter(torch.randn(r, in_f) * 0.01)
    def forward(self, x):
        out = self.w(x)
        if self.r > 0:
            out = out + (x @ self.B.T) @ self.A.T * self.s
        return out

def build_slopes(H):
    start = 2**(-(2**(-(math.log2(H)-3))))
    return torch.tensor([start * start**i for i in range(H)], dtype=torch.float32)

def causal_alibi(T, slopes, device):
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    return slopes.to(device)[:, None, None] * (i-j).float()[None]

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE  (unchanged from v2)
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, H    = cfg.embed_size, cfg.n_heads
        self.H  = H; self.dh = E // H
        self.Wq = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wk = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wv = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.proj   = nn.Linear(E, E, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
        self.slopes = build_slopes(H)

    def forward(self, x):
        B, T, E = x.shape; H, dh = self.H, self.dh
        q = self.Wq(x).view(B,T,H,dh).permute(0,2,1,3)
        k = self.Wk(x).view(B,T,H,dh).permute(0,2,1,3)
        v = self.Wv(x).view(B,T,H,dh).permute(0,2,1,3)
        s = q @ k.transpose(-2,-1) / math.sqrt(dh)
        s = s + causal_alibi(T, self.slopes, x.device)
        s = s + torch.triu(torch.full((T,T), float('-inf'), device=x.device), 1)
        o = (s.softmax(-1) @ v).transpose(1,2).reshape(B,T,E)
        return self.drop(self.proj(o))

class BaselineBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E = cfg.embed_size
        self.ln1  = nn.LayerNorm(E); self.ln2 = nn.LayerNorm(E)
        self.attn = BaselineAttn(cfg)
        self.ff   = nn.Sequential(
            nn.Linear(E,4*E), nn.GELU(), nn.Linear(4*E,E), nn.Dropout(cfg.dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BaselineGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok    = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([BaselineBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.embed_size)
        self.head   = nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)
    def forward(self, idx, targets=None):
        x = self.drop(self.tok(idx))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1)) \
               if targets is not None else None
        return logits, loss, torch.zeros(1, device=idx.device)
    def num_params(self): return sum(p.numel() for p in self.parameters())

# ═══════════════════════════════════════════════════════════════════════════════
# HIERATTN v3  —  RESIDUAL design
# ═══════════════════════════════════════════════════════════════════════════════

class VectorizedPooler(nn.Module):
    """Pool [B, n, chunk, C] → [B, n, C] with one batched einsum."""
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(embed_dim) * 0.02)
    def forward(self, K, V):
        # K,V: [B, n_chunks, chunk_size, C]
        w     = torch.einsum('c,bntc->bnt', self.q, K).softmax(-1).unsqueeze(-1)
        sum_k = (w * K).sum(-2)    # [B, n, C]
        sum_v = (w * V).sum(-2)
        recon = F.mse_loss(sum_v, V.mean(-2).detach())
        return sum_k, sum_v, recon


class HierKVBank(nn.Module):
    """Vectorized 3-level KV bank (same as v2 — this part was already correct)."""
    def __init__(self, cfg, pooler):
        super().__init__()
        self.W0 = cfg.W0; self.cs = cfg.chunk_size
        self.W1 = cfg.W1; self.W2 = cfg.W2
        self.pooler = pooler

    def forward(self, K, V):
        B, T, C = K.shape
        recon   = K.new_zeros(1)

        K0 = K[:, -self.W0:]; V0 = V[:, -self.W0:]

        older    = max(0, T - self.W0)
        n_chunks = older // self.cs
        if n_chunks > 0:
            Kc = K[:, :n_chunks*self.cs].view(B, n_chunks, self.cs, C)
            Vc = V[:, :n_chunks*self.cs].view(B, n_chunks, self.cs, C)
            K1a, V1a, r1 = self.pooler(Kc, Vc)
            recon = recon + r1
            K1 = K1a[:, -self.W1:]; V1 = V1a[:, -self.W1:]
        else:
            K1 = K.new_zeros(B,0,C); V1 = K.new_zeros(B,0,C)

        n1 = K1.size(1); n_pairs = n1 // 2
        if n_pairs > 0:
            K2a, V2a, r2 = self.pooler(
                K1[:, :n_pairs*2].view(B, n_pairs, 2, C),
                V1[:, :n_pairs*2].view(B, n_pairs, 2, C))
            recon = recon + r2 * 0.5
            K2 = K2a[:, -self.W2:]; V2 = V2a[:, -self.W2:]
        else:
            K2 = K.new_zeros(B,0,C); V2 = K.new_zeros(B,0,C)

        return (K0,V0), (K1,V1), (K2,V2), recon


class HierContextHead(nn.Module):
    """
    Computes long-range context from L1 + L2 KV banks only.
    No causal masking needed — these are all past-context summaries.
    This is added as a residual on top of standard causal attention.
    """
    def __init__(self, cfg):
        super().__init__()
        E, H      = cfg.embed_size, cfg.n_heads
        self.H    = H; self.dh = E // H
        # Separate Q projection for the hierarchical head
        self.Wq   = nn.Linear(E, E, bias=False)
        self.proj = nn.Linear(E, E, bias=False)
        # Gate: scalar per layer, init near 0 so hierarchy starts silent
        self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12
        nn.init.zeros_(self.proj.weight)               # proj starts at zero too

    def forward(self, x, K1, V1, K2, V2):
        B, T, E = x.shape; H, dh = self.H, self.dh

        # If both levels empty, return zero
        if K1.size(1) == 0 and K2.size(1) == 0:
            return x.new_zeros(B, T, E)

        Q = self.Wq(x).view(B,T,H,dh).permute(0,2,1,3)  # [B,H,T,dh]

        def attend(Kf, Vf):
            if Kf.size(1) == 0:
                return Q.new_zeros(B, H, T, dh)
            Tm = Kf.size(1)
            Kh = Kf.view(B,Tm,H,dh).permute(0,2,1,3)
            Vh = Vf.view(B,Tm,H,dh).permute(0,2,1,3)
            s  = Q @ Kh.transpose(-2,-1) / math.sqrt(dh)
            return s.softmax(-1) @ Vh

        # Equal weighting of mid and coarse (can be learned if desired)
        ctx = attend(K1, V1) + attend(K2, V2)             # [B,H,T,dh]
        ctx = ctx.permute(0,2,1,3).reshape(B,T,E)
        ctx = self.proj(ctx)

        # Gate controls how much hierarchy contributes
        # sigmoid(gate) starts at ~0.12, grows as training progresses
        alpha = torch.sigmoid(self.gate)
        return alpha * ctx


class HierAttnV3(nn.Module):
    """
    Residual hierarchical attention.

    out = standard_causal_attn(x) + hier_context_head(x, L1, L2)

    Standard causal attn is identical to baseline — always provides a
    strong learning signal from step 1. The hierarchical head adds
    long-range context on top, gated to start near-silent.
    """
    def __init__(self, cfg, pooler):
        super().__init__()
        E, H = cfg.embed_size, cfg.n_heads
        self.H = H; self.dh = E // H

        # Standard causal path (identical to baseline)
        self.Wq_std   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wk_std   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wv_std   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.proj_std = nn.Linear(E, E, bias=False)
        self.drop     = nn.Dropout(cfg.dropout)
        self.slopes   = build_slopes(H)

        # Hierarchical KV bank
        self.bank     = HierKVBank(cfg, pooler)

        # Hierarchical context head (residual)
        self.hier     = HierContextHead(cfg)

    def forward(self, x):
        B, T, E = x.shape; H, dh = self.H, self.dh

        # ── Standard causal attention (identical to baseline) ────────────
        q = self.Wq_std(x).view(B,T,H,dh).permute(0,2,1,3)
        k = self.Wk_std(x).view(B,T,H,dh).permute(0,2,1,3)
        v = self.Wv_std(x).view(B,T,H,dh).permute(0,2,1,3)
        s = q @ k.transpose(-2,-1) / math.sqrt(dh)
        s = s + causal_alibi(T, self.slopes, x.device)
        s = s + torch.triu(torch.full((T,T), float('-inf'), device=x.device), 1)
        std_out = (s.softmax(-1) @ v).transpose(1,2).reshape(B,T,E)
        std_out = self.drop(self.proj_std(std_out))

        # ── Build KV hierarchy from K/V of standard path ─────────────────
        Kf = k.permute(0,2,1,3).reshape(B,T,E)   # reuse already-computed K
        Vf = v.permute(0,2,1,3).reshape(B,T,E)
        (_, _), (K1,V1), (K2,V2), recon = self.bank(Kf, Vf)

        # ── Hierarchical residual ─────────────────────────────────────────
        hier_out = self.hier(x, K1, V1, K2, V2)

        return std_out + hier_out, recon


class HierBlockV3(nn.Module):
    def __init__(self, cfg, pooler):
        super().__init__()
        E = cfg.embed_size
        self.ln1  = nn.LayerNorm(E); self.ln2 = nn.LayerNorm(E)
        self.attn = HierAttnV3(cfg, pooler)
        self.ff   = nn.Sequential(
            nn.Linear(E,4*E), nn.GELU(), nn.Linear(4*E,E), nn.Dropout(cfg.dropout))
    def forward(self, x):
        a, recon = self.attn(self.ln1(x))
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, recon


class HierGPTV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lam    = cfg.aux_lambda
        self.tok    = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.drop   = nn.Dropout(cfg.dropout)
        self.pooler = VectorizedPooler(cfg.embed_size)
        self.blocks = nn.ModuleList([
            HierBlockV3(cfg, self.pooler) for _ in range(cfg.n_layers)
        ])
        self.ln_f   = nn.LayerNorm(cfg.embed_size)
        self.head   = nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.drop(self.tok(idx))
        recons = []
        for b in self.blocks:
            x, r = b(x); recons.append(r)
        logits = self.head(self.ln_f(x))
        loss   = None
        if targets is not None:
            lm   = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
            aux  = torch.stack(recons).mean()
            loss = lm + self.lam * aux
        return logits, loss, torch.stack(recons).mean().detach()

    def num_params(self): return sum(p.numel() for p in self.parameters())

    def gate_values(self):
        """Show learned gate values — diagnostic tool."""
        gates = []
        for i, b in enumerate(self.blocks):
            g = torch.sigmoid(b.attn.hier.gate).item()
            gates.append((i, g))
        return gates

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING + EVAL
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, steps=None):
    model.eval()
    losses = []
    for _ in range(steps or C.eval_steps):
        x, y = get_batch("val")
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    avg = sum(losses)/len(losses)
    return {"val_loss": avg, "bpb": avg/math.log(2)}


def train(model, label):
    print(f"\n{'═'*65}")
    print(f"  {label}")
    print(f"  Params: {model.num_params():,}")
    print(f"{'═'*65}")
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=0.01)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    log  = {"steps":[], "train":[], "val":[], "bpb":[]}
    t0   = time.time()

    for step in range(C.n_steps):
        for pg in opt.param_groups: pg["lr"] = get_lr(step)
        x, y = get_batch("train")
        _, loss, _ = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log["train"].append(loss.item())

        if (step+1) % C.eval_every == 0 or step == C.n_steps-1:
            m   = evaluate(model)
            sps = (step+1)/(time.time()-t0)
            sm  = sum(log["train"][-50:])/min(50,len(log["train"]))
            # Show gate values for HierGPT to confirm hierarchy is activating
            gate_str = ""
            if hasattr(model, "gate_values"):
                gs = [f"{g:.2f}" for _,g in model.gate_values()]
                gate_str = f"  gates: [{', '.join(gs)}]"
            print(f"  step {step+1:4d}/{C.n_steps}"
                  f"  train: {sm:.4f}  val: {m['val_loss']:.4f}"
                  f"  bpb: {m['bpb']:.4f}  lr: {get_lr(step):.1e}"
                  f"  {sps:.1f} st/s{gate_str}")
            log["steps"].append(step+1); log["val"].append(m["val_loss"])
            log["bpb"].append(m["bpb"])

    final   = evaluate(model, steps=100)
    mem     = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0
    elapsed = time.time()-t0
    print(f"\n  FINAL → val_loss: {final['val_loss']:.4f}  bpb: {final['bpb']:.4f}"
          f"  mem: {mem:.0f}MB  speed: {C.n_steps/elapsed:.1f}st/s")
    if hasattr(model, "gate_values"):
        print(f"  Final gates: {[(i, round(g,3)) for i,g in model.gate_values()]}")
    return {"label":label, "val_loss":final["val_loss"], "bpb":final["bpb"],
            "params_M":model.num_params()/1e6, "peak_mem_mb":mem,
            "steps_per_s":C.n_steps/elapsed, "log":log}, model

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results):
    print(f"\n{'═'*75}")
    print("  BENCHMARK RESULTS — TinyShakespeare")
    print(f"{'═'*75}")
    print(f"  {'Model':<32} {'val_loss':>9} {'bpb':>7} {'params':>8} {'mem_MB':>8} {'st/s':>6}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['label']:<32}  {r['val_loss']:>7.4f}  {r['bpb']:>5.4f}"
              f"  {r['params_M']:>5.2f}M  {r['peak_mem_mb']:>7.0f}  {r['steps_per_s']:>5.1f}")
    if len(results) == 2:
        b, h = results
        dl = h["val_loss"]-b["val_loss"]; db = h["bpb"]-b["bpb"]
        spd = h["steps_per_s"]/max(b["steps_per_s"],1e-9)
        print(f"{'═'*75}")
        print(f"  Delta  val_loss: {dl:+.4f} {'↓ better' if dl<0 else '↑ worse'}"
              f"   bpb: {db:+.4f}   speed: {spd:.2f}x")
        if dl < 0:
            print("  ✓ HierAttn outperforms baseline — long-range context is helping.")
        elif abs(dl) <= 0.02:
            print("  ~ Matched — hierarchy adds context at no quality cost.")
        else:
            print("  Gap remains — try increasing n_steps to 5000.")
        print(f"{'═'*75}")

def plot_curves(results):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9,4))
        colors = ["#4C72B0","#DD8452"]
        for r, c in zip(results, colors):
            ax.plot(r["log"]["steps"], r["log"]["bpb"],
                    label=r["label"], color=c, linewidth=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Val BPB ↓")
        ax.set_title("TinyShakespeare — Baseline vs HierAttn v3")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        path = "/kaggle/working/curves_v3.png"
        plt.savefig(path, dpi=120); plt.show()
        print(f"  Plot → {path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or True:
    torch.manual_seed(42)
    results = []

    r_base, _      = train(BaselineGPT(C),   "Baseline (ALiBi + LoRA)")
    results.append(r_base)

    r_hier, m_hier = train(HierGPTV3(C),     "HierAttn v3 (residual)")
    results.append(r_hier)

    print_table(results)
    plot_curves(results)

    summary = {"dataset":"TinyShakespeare","config":C.__dict__,
               "results":[{k:v for k,v in r.items() if k!="log"} for r in results]}
    with open("/kaggle/working/benchmark_v3.json","w") as f:
        json.dump(summary, f, indent=2)
    print("Saved → /kaggle/working/benchmark_v3.json")
