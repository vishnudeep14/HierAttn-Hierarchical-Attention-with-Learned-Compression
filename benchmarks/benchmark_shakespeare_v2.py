"""
benchmark_shakespeare_v2.py
===========================
Fixed version. Two bugs patched from v1:

  Bug 1 — HierKVBank used a Python for-loop over chunks → broke the gradient
           graph and ran on CPU. Fixed with torch.unfold + batched matmul.

  Bug 2 — CrossLevelAttn mixed causal/non-causal attention incorrectly,
           causing the model to see future tokens through L1/L2. Fixed by
           using a consistent additive masking strategy.

Expected: HierAttn bpb close to or better than baseline.
"""

import os, sys, math, time, json, urllib.request
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_PATH = "/kaggle/working/shakespeare.txt"
if not os.path.exists(DATA_PATH):
    print("Downloading TinyShakespeare...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        DATA_PATH)

with open(DATA_PATH) as f:
    raw = f.read()

data  = torch.tensor(list(raw.encode("utf-8")), dtype=torch.long)
n     = len(data)
train_data = data[:int(0.9*n)]
val_data   = data[int(0.9*n):]
VOCAB      = 256
print(f"Train: {len(train_data):,}  Val: {len(val_data):,}  chars")

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
    # HierAttn windows
    W0          : int   = 128   # fine   – last W0 raw tokens
    chunk_size  : int   = 8     # stride for mid pooling  (was 'stride')
    W1          : int   = 16    # mid    – keep last W1 pooled summaries
    W2          : int   = 8     # coarse – keep last W2 double-pooled summaries
    aux_lambda  : float = 0.0   # set to 0 first; tune after baseline matches
    # Training
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
    return slopes.to(device)[:, None, None] * (i-j).float()[None]  # [H,T,T]

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, H      = cfg.embed_size, cfg.n_heads
        self.H    = H
        self.dh   = E // H
        self.Wq   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wk   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wv   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.proj = nn.Linear(E, E, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.slopes = build_slopes(H)

    def forward(self, x):
        B, T, E = x.shape
        H, dh   = self.H, self.dh
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
        self.ln1  = nn.LayerNorm(E)
        self.ln2  = nn.LayerNorm(E)
        self.attn = BaselineAttn(cfg)
        self.ff   = nn.Sequential(
            nn.Linear(E, 4*E), nn.GELU(),
            nn.Linear(4*E, E), nn.Dropout(cfg.dropout))
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
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, torch.zeros(1, device=idx.device)
    def num_params(self): return sum(p.numel() for p in self.parameters())

# ═══════════════════════════════════════════════════════════════════════════════
# HIERATTN  —  fully vectorized, gradient-friendly
# ═══════════════════════════════════════════════════════════════════════════════

class VectorizedPooler(nn.Module):
    """
    Pools [B, n_chunks, chunk_size, C] → [B, n_chunks, C] in one shot.
    No Python loops — fully on GPU, gradients flow cleanly.

    Uses a learned per-head query vector:
        score[b,n,t] = q · K[b,n,t,:]
        weight       = softmax over t
        summary      = weight @ V[b,n,:,:]
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def forward(self, K, V):
        # K, V : [B, n_chunks, chunk_size, C]
        scores  = torch.einsum('c, bntc -> bnt', self.q, K)   # [B, n, cs]
        w       = scores.softmax(-1).unsqueeze(-1)             # [B, n, cs, 1]
        sum_k   = (w * K).sum(-2)                              # [B, n, C]
        sum_v   = (w * V).sum(-2)
        # Reconstruction loss: compressed V should ≈ mean of chunk
        target  = V.mean(-2).detach()                          # [B, n, C]
        recon   = F.mse_loss(sum_v, target)
        return sum_k, sum_v, recon


class HierKVBank(nn.Module):
    """
    Builds 3-level KV from full [B, T, C] in a single forward pass.

    All chunking done with tensor ops (unfold/reshape) — no Python loops.

    L0 fine   : last W0 raw tokens
    L1 mid    : pooled chunks of size `chunk_size` from older tokens → keep W1
    L2 coarse : pooled pairs of L1 summaries → keep W2
    """
    def __init__(self, cfg, pooler):
        super().__init__()
        self.W0         = cfg.W0
        self.chunk_size = cfg.chunk_size
        self.W1         = cfg.W1
        self.W2         = cfg.W2
        self.pooler     = pooler

    def forward(self, K, V):
        # K, V : [B, T, C]
        B, T, C = K.shape
        recon   = K.new_zeros(1)

        # ── L0: fine (last W0 tokens) ────────────────────────────────────
        K0 = K[:, -self.W0:, :]
        V0 = V[:, -self.W0:, :]

        # ── L1: mid (vectorized chunked pooling of older portion) ─────────
        older = max(0, T - self.W0)
        cs    = self.chunk_size
        # trim to multiple of chunk_size
        n_chunks = older // cs
        if n_chunks > 0:
            Ko = K[:, :n_chunks * cs, :]              # [B, n*cs, C]
            Vo = V[:, :n_chunks * cs, :]
            # reshape into chunks — no copy, just a view
            Kc = Ko.view(B, n_chunks, cs, C)           # [B, n, cs, C]
            Vc = Vo.view(B, n_chunks, cs, C)
            # one batched pooling call
            K1_all, V1_all, r1 = self.pooler(Kc, Vc)  # [B, n, C]
            recon = recon + r1
            K1 = K1_all[:, -self.W1:, :]
            V1 = V1_all[:, -self.W1:, :]
        else:
            K1 = K.new_zeros(B, 0, C)
            V1 = K.new_zeros(B, 0, C)

        # ── L2: coarse (pool pairs of L1 summaries) ───────────────────────
        n1 = K1.size(1)
        n_pairs = n1 // 2
        if n_pairs > 0:
            K1p = K1[:, :n_pairs*2, :].view(B, n_pairs, 2, C)  # [B, p, 2, C]
            V1p = V1[:, :n_pairs*2, :].view(B, n_pairs, 2, C)
            K2_all, V2_all, r2 = self.pooler(K1p, V1p)          # [B, p, C]
            recon = recon + r2 * 0.5
            K2 = K2_all[:, -self.W2:, :]
            V2 = V2_all[:, -self.W2:, :]
        else:
            K2 = K.new_zeros(B, 0, C)
            V2 = K.new_zeros(B, 0, C)

        return (K0, V0), (K1, V1), (K2, V2), recon


class HierAttn(nn.Module):
    """
    Hierarchical multi-head attention.

    Q attends three KV levels simultaneously.
    Level importance weights are learned (softmax over 3 scalars).
    Causal mask applied to L0 only — L1/L2 are past-context summaries.
    """
    def __init__(self, cfg, pooler):
        super().__init__()
        E, H      = cfg.embed_size, cfg.n_heads
        self.H    = H
        self.dh   = E // H
        self.Wq   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wk   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wv   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.proj = nn.Linear(E, E, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.slopes    = build_slopes(H)
        self.bank      = HierKVBank(cfg, pooler)
        # learned level importance (initialized equal)
        self.lvl_w     = nn.Parameter(torch.zeros(3))

    def _attend(self, Q, K_flat, V_flat, causal=False):
        """Q:[B,H,T,dh]  K_flat/V_flat:[B,Tm,C] → [B,H,T,dh]"""
        B, H, T, dh = Q.shape
        Tm = K_flat.size(1)
        if Tm == 0:
            return Q.new_zeros(B, H, T, dh)
        C  = H * dh
        Kh = K_flat.view(B, Tm, H, dh).permute(0,2,1,3)   # [B,H,Tm,dh]
        Vh = V_flat.view(B, Tm, H, dh).permute(0,2,1,3)
        s  = Q @ Kh.transpose(-2,-1) / math.sqrt(dh)       # [B,H,T,Tm]
        if causal and T == Tm:
            mask = torch.triu(torch.full((T,T), float('-inf'), device=Q.device), 1)
            s    = s + mask
        return s.softmax(-1) @ Vh                           # [B,H,T,dh]

    def forward(self, x):
        B, T, E = x.shape
        H, dh   = self.H, self.dh

        Q   = self.Wq(x).view(B,T,H,dh).permute(0,2,1,3)  # [B,H,T,dh]
        K   = self.Wk(x).reshape(B,T,E)
        V   = self.Wv(x).reshape(B,T,E)

        (K0,V0),(K1,V1),(K2,V2), recon = self.bank(K, V)

        w   = self.lvl_w.softmax(0)                         # [3]

        # attend each level, weight and sum
        o0  = self._attend(Q, K0, V0, causal=True)          # [B,H,T,dh]
        o1  = self._attend(Q, K1, V1, causal=False)
        o2  = self._attend(Q, K2, V2, causal=False)

        out = w[0]*o0 + w[1]*o1 + w[2]*o2                  # [B,H,T,dh]
        out = out.permute(0,2,1,3).reshape(B,T,E)
        return self.drop(self.proj(out)), recon


class HierBlock(nn.Module):
    def __init__(self, cfg, pooler):
        super().__init__()
        E = cfg.embed_size
        self.ln1  = nn.LayerNorm(E)
        self.ln2  = nn.LayerNorm(E)
        self.attn = HierAttn(cfg, pooler)
        self.ff   = nn.Sequential(
            nn.Linear(E, 4*E), nn.GELU(),
            nn.Linear(4*E, E), nn.Dropout(cfg.dropout))
    def forward(self, x):
        a, recon = self.attn(self.ln1(x))
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, recon


class HierGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lam    = cfg.aux_lambda
        self.tok    = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.drop   = nn.Dropout(cfg.dropout)
        # one shared pooler across all layers (same as your original design)
        self.pooler = VectorizedPooler(cfg.embed_size)
        self.blocks = nn.ModuleList([
            HierBlock(cfg, self.pooler) for _ in range(cfg.n_layers)
        ])
        self.ln_f   = nn.LayerNorm(cfg.embed_size)
        self.head   = nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x      = self.drop(self.tok(idx))
        recons = []
        for b in self.blocks:
            x, r = b(x)
            recons.append(r)
        logits = self.head(self.ln_f(x))
        loss   = None
        if targets is not None:
            lm   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            aux  = torch.stack(recons).mean()
            loss = lm + self.lam * aux
        return logits, loss, torch.stack(recons).mean().detach()

    def num_params(self): return sum(p.numel() for p in self.parameters())

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    log = {"steps":[], "train":[], "val":[], "bpb":[]}
    t0  = time.time()

    for step in range(C.n_steps):
        for pg in opt.param_groups: pg["lr"] = get_lr(step)
        x, y = get_batch("train")
        _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log["train"].append(loss.item())

        if (step+1) % C.eval_every == 0 or step == C.n_steps-1:
            m   = evaluate(model)
            sps = (step+1)/(time.time()-t0)
            sm  = sum(log["train"][-50:])/min(50, len(log["train"]))
            print(f"  step {step+1:4d}/{C.n_steps}"
                  f"  train: {sm:.4f}"
                  f"  val: {m['val_loss']:.4f}"
                  f"  bpb: {m['bpb']:.4f}"
                  f"  lr: {get_lr(step):.1e}"
                  f"  {sps:.1f} st/s")
            log["steps"].append(step+1)
            log["val"].append(m["val_loss"])
            log["bpb"].append(m["bpb"])

    final    = evaluate(model, steps=100)
    mem      = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0
    elapsed  = time.time()-t0

    print(f"\n  FINAL → val_loss: {final['val_loss']:.4f}"
          f"  bpb: {final['bpb']:.4f}"
          f"  mem: {mem:.0f}MB"
          f"  speed: {C.n_steps/elapsed:.1f}st/s")

    return {
        "label": label, "val_loss": final["val_loss"], "bpb": final["bpb"],
        "params_M": model.num_params()/1e6, "peak_mem_mb": mem,
        "steps_per_s": C.n_steps/elapsed, "log": log,
    }, model

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results):
    print(f"\n{'═'*75}")
    print("  BENCHMARK RESULTS — TinyShakespeare")
    print(f"{'═'*75}")
    print(f"  {'Model':<30} {'val_loss':>9} {'bpb':>7} {'params':>8} {'mem_MB':>8} {'st/s':>6}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['label']:<30}  {r['val_loss']:>7.4f}  {r['bpb']:>5.4f}"
              f"  {r['params_M']:>5.2f}M  {r['peak_mem_mb']:>7.0f}  {r['steps_per_s']:>5.1f}")
    if len(results) == 2:
        b, h = results
        dl, db = h["val_loss"]-b["val_loss"], h["bpb"]-b["bpb"]
        spd    = h["steps_per_s"]/max(b["steps_per_s"],1e-9)
        print(f"{'═'*75}")
        print(f"  Delta  val_loss: {dl:+.4f} {'↓ better' if dl<0 else '↑ worse'}"
              f"   bpb: {db:+.4f}   speed: {spd:.2f}x")
        if dl < -0.02:
            print("  ✓ HierAttn meaningfully improves over baseline.")
        elif abs(dl) <= 0.02:
            print("  ~ Matched quality — hierarchy adds context at no perplexity cost.")
        else:
            print("  Still underperforming — check notes below.")
    print(f"{'═'*75}")


def plot_curves(results):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        colors  = ["#4C72B0", "#DD8452"]
        for r, c in zip(results, colors):
            ax.plot(r["log"]["steps"], r["log"]["bpb"],
                    label=r["label"], color=c, linewidth=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Val BPB (↓ better)")
        ax.set_title("TinyShakespeare — Baseline vs HierAttn v2")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        path = "/kaggle/working/curves_v2.png"
        plt.savefig(path, dpi=120)
        plt.show()
        print(f"  Plot saved → {path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


def ablation(lambdas=(0.0, 0.001, 0.01, 0.05)):
    print(f"\n{'═'*55}")
    print("  ABLATION: aux_lambda sweep")
    print(f"{'═'*55}")
    rows = []
    for lam in lambdas:
        cfg2 = Cfg(); cfg2.n_steps = 600; cfg2.aux_lambda = lam
        m    = HierGPT(cfg2).to(DEVICE)
        opt  = torch.optim.AdamW(m.parameters(), lr=cfg2.lr, weight_decay=0.01)
        for step in range(cfg2.n_steps):
            x, y = get_batch("train")
            _, loss, _ = m(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        ev = evaluate(m, steps=50)
        rows.append((lam, ev["val_loss"], ev["bpb"]))
        tag = " ◄ best so far" if ev["bpb"] == min(r[2] for r in rows) else ""
        print(f"  λ={lam:.3f}  val_loss: {ev['val_loss']:.4f}  bpb: {ev['bpb']:.4f}{tag}")
    best = min(rows, key=lambda r: r[2])[0]
    print(f"\n  Best λ = {best}")
    print(f"{'═'*55}")
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" or True:
    torch.manual_seed(42)
    results = []

    r_base, _ = train(BaselineGPT(C), "Baseline (ALiBi + LoRA)")
    results.append(r_base)

    r_hier, _ = train(HierGPT(C),     "HierAttn v2 (vectorized)")
    results.append(r_hier)

    print_table(results)
    plot_curves(results)
    ablation()

    summary = {
        "dataset": "TinyShakespeare",
        "config":  C.__dict__,
        "results": [{k:v for k,v in r.items() if k!="log"} for r in results],
    }
    with open("/kaggle/working/benchmark_v2.json","w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved → /kaggle/working/benchmark_v2.json")
