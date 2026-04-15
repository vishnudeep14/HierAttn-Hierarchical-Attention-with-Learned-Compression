"""
benchmark_shakespeare.py
========================
Complete benchmark: Baseline GPT vs HierAttn on TinyShakespeare.

Run this as a single script in Kaggle — no other files needed.
Everything is self-contained: model definitions, training, eval, plots.

Expected runtime on T4: ~20 minutes
Expected results:
    Baseline  → bpb ~1.6–1.9
    HierAttn  → bpb ~1.5–1.8  (lower = better)
"""

import os, sys, math, time, json, urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Download Shakespeare ──────────────────────────────────────────────────────
DATA_PATH = "/kaggle/working/shakespeare.txt"
if not os.path.exists(DATA_PATH):
    print("Downloading TinyShakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, DATA_PATH)

with open(DATA_PATH) as f:
    raw_text = f.read()

print(f"Dataset: {len(raw_text):,} chars")
print(f"Sample: {raw_text[:120]!r}")

# ── Byte-level encoding ───────────────────────────────────────────────────────
data = torch.tensor(list(raw_text.encode("utf-8")), dtype=torch.long)
n = len(data)
split = int(0.9 * n)
train_data = data[:split]
val_data   = data[split:]
VOCAB_SIZE  = 256

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    vocab_size:    int   = VOCAB_SIZE
    block_size:    int   = 256
    batch_size:    int   = 64
    embed_size:    int   = 256
    n_layers:      int   = 6
    n_heads:       int   = 4
    dropout:       float = 0.1
    lora_r:        int   = 8
    lora_alpha:    int   = 16
    # HierAttn
    W0:            int   = 128
    W1:            int   = 64
    W2:            int   = 16
    aux_lambda:    float = 0.01
    # Training
    n_steps:       int   = 3000
    lr:            float = 3e-4
    warmup_steps:  int   = 200
    eval_interval: int   = 300
    eval_steps:    int   = 50

C = Cfg()

# ── Data helpers ──────────────────────────────────────────────────────────────
def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - C.block_size, (C.batch_size,))
    x = torch.stack([d[i:     i + C.block_size    ] for i in ix]).to(DEVICE)
    y = torch.stack([d[i + 1: i + C.block_size + 1] for i in ix]).to(DEVICE)
    return x, y

# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step):
    if step < C.warmup_steps:
        return C.lr * step / max(C.warmup_steps, 1)
    progress = (step - C.warmup_steps) / max(C.n_steps - C.warmup_steps, 1)
    return C.lr * 0.5 * (1 + math.cos(math.pi * progress))

# ── Shared building blocks ────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16):
        super().__init__()
        self.linear  = nn.Linear(in_f, out_f, bias=False)
        self.r       = r
        self.scaling = alpha / r if r > 0 else 0.0
        if r > 0:
            self.A = nn.Parameter(torch.randn(out_f, r) * 0.01)
            self.B = nn.Parameter(torch.randn(r, in_f) * 0.01)
    def forward(self, x):
        out = self.linear(x)
        if self.r > 0:
            out = out + (x @ self.B.T) @ self.A.T * self.scaling
        return out

def alibi_slopes(n_heads):
    start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
    return torch.tensor([start * start**i for i in range(n_heads)],
                        dtype=torch.float32)

def alibi_bias(T, slopes, device):
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    return slopes.to(device)[:, None, None] * (i - j).float()[None]

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A — BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineAttn(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.H, self.dh = C.n_heads, C.embed_size // C.n_heads
        self.Wq   = LoRALinear(C.embed_size, C.embed_size, C.lora_r, C.lora_alpha)
        self.Wk   = LoRALinear(C.embed_size, C.embed_size, C.lora_r, C.lora_alpha)
        self.Wv   = LoRALinear(C.embed_size, C.embed_size, C.lora_r, C.lora_alpha)
        self.proj = nn.Linear(C.embed_size, C.embed_size, bias=False)
        self.drop = nn.Dropout(C.dropout)
        self.slopes = alibi_slopes(C.n_heads)

    def forward(self, x):
        B, T, E = x.shape
        H, dh = self.H, self.dh
        q = self.Wq(x).view(B,T,H,dh).permute(0,2,1,3)
        k = self.Wk(x).view(B,T,H,dh).permute(0,2,1,3)
        v = self.Wv(x).view(B,T,H,dh).permute(0,2,1,3)
        s = (q @ k.transpose(-2,-1)) / math.sqrt(dh)
        s = s + alibi_bias(T, self.slopes, x.device)
        s = s + torch.triu(torch.full((T,T), float('-inf'), device=x.device), 1)
        out = (s.softmax(-1) @ v).transpose(1,2).reshape(B,T,E)
        return self.drop(self.proj(out))

class BaselineBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ln1  = nn.LayerNorm(C.embed_size)
        self.ln2  = nn.LayerNorm(C.embed_size)
        self.attn = BaselineAttn(C)
        self.ff   = nn.Sequential(
            nn.Linear(C.embed_size, 4*C.embed_size), nn.GELU(),
            nn.Linear(4*C.embed_size, C.embed_size), nn.Dropout(C.dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BaselineGPT(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.tok    = nn.Embedding(C.vocab_size, C.embed_size)
        self.drop   = nn.Dropout(C.dropout)
        self.blocks = nn.ModuleList([BaselineBlock(C) for _ in range(C.n_layers)])
        self.ln_f   = nn.LayerNorm(C.embed_size)
        self.head   = nn.Linear(C.embed_size, C.vocab_size, bias=False)
    def forward(self, idx, targets=None):
        x = self.drop(self.tok(idx))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, torch.zeros(1, device=idx.device)
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B — HIERATTN
# ═══════════════════════════════════════════════════════════════════════════════

class LearnedPooler(nn.Module):
    def __init__(self, embed_dim, n_levels=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_levels, embed_dim) * 0.02)
    def forward(self, K, V, level=0):
        w = torch.softmax(torch.einsum('c,btc->bt', self.q[level], K), -1).unsqueeze(1)
        sk, sv = w @ K, w @ V
        recon  = F.mse_loss(sv, V.mean(1, keepdim=True).detach())
        return sk, sv, recon

class DynamicCompressor(nn.Module):
    STRIDES = [1, 2, 4]
    def __init__(self, embed_dim, hidden=64, tau=1.0):
        super().__init__()
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data.copy_(torch.tensor([1., 0., -1.]))
    def forward(self, x):
        logits = self.net(x.mean(1))               # [B, 3]
        soft   = F.gumbel_softmax(logits, tau=self.tau, hard=False) if self.training \
                 else logits.softmax(-1)
        stride_vals = torch.tensor(self.STRIDES, dtype=x.dtype, device=x.device)
        stride_soft = (soft * stride_vals).sum(-1).mean().item()
        stride_hard = self.STRIDES[logits.argmax(-1).mode().values.item()]
        return stride_soft, stride_hard

class HierKVBank(nn.Module):
    def __init__(self, embed_dim, W0, W1, W2, pooler):
        super().__init__()
        self.W0, self.W1, self.W2 = W0, W1, W2
        self.pooler = pooler
    def forward(self, K, V, stride):
        B, T, C = K.shape
        recon = K.new_zeros(1)
        # L0: fine — last W0 tokens
        K0, V0 = K[:, -self.W0:], V[:, -self.W0:]
        # L1: mid — pool older tokens in stride-sized chunks
        older = max(0, T - self.W0)
        K1 = K.new_zeros(B, 0, C); V1 = K.new_zeros(B, 0, C)
        if older >= stride:
            n  = older // stride
            Kc = K[:, :n*stride].view(B, n, stride, C)
            Vc = V[:, :n*stride].view(B, n, stride, C)
            ks, vs = [], []
            for i in range(n):
                sk, sv, rl = self.pooler(Kc[:,i], Vc[:,i], level=0)
                ks.append(sk); vs.append(sv); recon = recon + rl
            K1 = torch.cat(ks, 1)[:, -self.W1:]
            V1 = torch.cat(vs, 1)[:, -self.W1:]
            recon = recon / n
        # L2: coarse — pool pairs of L1
        K2, V2 = K.new_zeros(B, 0, C), K.new_zeros(B, 0, C)
        if K1.size(1) >= 2:
            n2 = K1.size(1) // 2
            ks, vs = [], []
            for i in range(n2):
                sk, sv, rl = self.pooler(K1[:, i*2:i*2+2], V1[:, i*2:i*2+2], level=1)
                ks.append(sk); vs.append(sv); recon = recon + rl * 0.5
            K2 = torch.cat(ks, 1)[:, -self.W2:]
            V2 = torch.cat(vs, 1)[:, -self.W2:]
        return (K0,V0), (K1,V1), (K2,V2), recon

class CrossLevelAttn(nn.Module):
    def __init__(self, n_head, head_dim):
        super().__init__()
        self.H, self.dh = n_head, head_dim
        self.scale       = head_dim ** -0.5
        self.level_w     = nn.Parameter(torch.zeros(3))
    def _sdp(self, Q, K, V, causal=False):
        s = Q @ K.transpose(-2,-1) * self.scale
        if causal:
            T, Tm = s.size(-2), s.size(-1)
            if T == Tm:
                s = s + torch.triu(torch.full((T,Tm), float('-inf'), device=Q.device), 1)
        return s.softmax(-1) @ V
    def forward(self, Q, levels):
        B, H, T, dh = Q.shape
        C = H * dh
        w = self.level_w.softmax(0)
        out = Q.new_zeros(B, H, T, dh)
        for i, (Kl, Vl) in enumerate(levels):
            if Kl.size(1) == 0:
                continue
            Tm  = Kl.size(1)
            Kh  = Kl.view(B, Tm, H, dh).permute(0,2,1,3)
            Vh  = Vl.view(B, Tm, H, dh).permute(0,2,1,3)
            out = out + w[i] * self._sdp(Q, Kh, Vh, causal=(i==0))
        return out.permute(0,2,1,3).reshape(B, T, C)

class HierAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, H = cfg.embed_size, cfg.n_heads
        dh   = E // H
        self.H, self.dh = H, dh
        self.Wq   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wk   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.Wv   = LoRALinear(E, E, cfg.lora_r, cfg.lora_alpha)
        self.proj = nn.Linear(E, E, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.slopes     = alibi_slopes(H)
        self.pooler     = None   # set by HierGPT (shared)
        self.bank       = None
        self.compressor = DynamicCompressor(E)
        self.cross      = CrossLevelAttn(H, dh)
    def forward(self, x):
        B, T, E = x.shape
        H, dh   = self.H, self.dh
        Q = self.Wq(x).view(B,T,H,dh).permute(0,2,1,3)
        K = self.Wk(x).view(B,T,H,dh)
        V = self.Wv(x).view(B,T,H,dh)
        Kf = K.reshape(B,T,E)
        Vf = V.reshape(B,T,E)
        _, stride = self.compressor(x)
        (K0,V0),(K1,V1),(K2,V2), recon = self.bank(Kf, Vf, stride)
        out = self.cross(Q, [(K0,V0),(K1,V1),(K2,V2)])
        return self.drop(self.proj(out)), recon

class HierBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.embed_size)
        self.ln2  = nn.LayerNorm(cfg.embed_size)
        self.attn = HierAttn(cfg)
        self.ff   = nn.Sequential(
            nn.Linear(cfg.embed_size, 4*cfg.embed_size), nn.GELU(),
            nn.Linear(4*cfg.embed_size, cfg.embed_size), nn.Dropout(cfg.dropout))
    def forward(self, x):
        a, recon = self.attn(self.ln1(x))
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, recon

class HierGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.aux_lambda = cfg.aux_lambda
        self.tok    = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.drop   = nn.Dropout(cfg.dropout)
        self.pooler = LearnedPooler(cfg.embed_size, n_levels=2)
        self.blocks = nn.ModuleList([HierBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.embed_size)
        self.head   = nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)
        # Wire shared pooler + bank into every block
        for b in self.blocks:
            b.attn.pooler = self.pooler
            b.attn.bank   = HierKVBank(cfg.embed_size, cfg.W0, cfg.W1, cfg.W2,
                                       pooler=self.pooler)
    def forward(self, idx, targets=None):
        x = self.drop(self.tok(idx))
        total_recon = x.new_zeros(1)
        for b in self.blocks:
            x, recon = b(x)
            total_recon = total_recon + recon
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            lm   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = lm + self.aux_lambda * total_recon.squeeze() / len(self.blocks)
        return logits, loss, total_recon.squeeze()
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, steps=None):
    model.eval()
    steps = steps or C.eval_steps
    losses = []
    for _ in range(steps):
        x, y = get_batch("val")
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    return {"val_loss": avg, "bpb": avg / math.log(2)}


def train(model, label):
    print(f"\n{'═'*65}")
    print(f"  {label}")
    print(f"  Params: {model.num_params():,}")
    print(f"{'═'*65}")

    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=0.01)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    log = {"train_loss": [], "val_loss": [], "bpb": [], "steps": []}
    t0  = time.time()

    for step in range(C.n_steps):
        lr = get_lr(step)
        for pg in opt.param_groups: pg["lr"] = lr

        x, y      = get_batch("train")
        _, loss, recon = model(x, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        log["train_loss"].append(loss.item())

        if (step + 1) % C.eval_interval == 0 or step == C.n_steps - 1:
            m  = evaluate(model)
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            smooth = sum(log["train_loss"][-50:]) / min(50, len(log["train_loss"]))
            print(f"  step {step+1:4d}/{C.n_steps}"
                  f"  train: {smooth:.4f}"
                  f"  val: {m['val_loss']:.4f}"
                  f"  bpb: {m['bpb']:.4f}"
                  f"  lr: {lr:.1e}"
                  f"  {sps:.1f} st/s")
            log["val_loss"].append(m["val_loss"])
            log["bpb"].append(m["bpb"])
            log["steps"].append(step + 1)

    elapsed   = time.time() - t0
    final     = evaluate(model, steps=100)
    peak_mem  = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    result = {
        "label":       label,
        "val_loss":    final["val_loss"],
        "bpb":         final["bpb"],
        "params_M":    model.num_params() / 1e6,
        "peak_mem_mb": peak_mem,
        "steps_per_s": C.n_steps / elapsed,
        "log":         log,
    }
    print(f"\n  FINAL → val_loss: {final['val_loss']:.4f}  bpb: {final['bpb']:.4f}"
          f"  mem: {peak_mem:.0f}MB  speed: {result['steps_per_s']:.1f}st/s")
    return result, model


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TABLE + LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results):
    print(f"\n{'═'*75}")
    print("  BENCHMARK RESULTS — TinyShakespeare")
    print(f"{'═'*75}")
    hdr = f"  {'Model':<28} {'val_loss':>9} {'bpb':>7} {'params':>9} {'mem_MB':>9} {'st/s':>7}"
    print(hdr)
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['label']:<28}"
              f"  {r['val_loss']:>7.4f}"
              f"  {r['bpb']:>5.4f}"
              f"  {r['params_M']:>6.2f}M"
              f"  {r['peak_mem_mb']:>7.0f}"
              f"  {r['steps_per_s']:>5.1f}")
    print(f"{'═'*75}")

    if len(results) == 2:
        b, h = results[0], results[1]
        dl   = h["val_loss"] - b["val_loss"]
        db   = h["bpb"]      - b["bpb"]
        dm   = h["peak_mem_mb"] - b["peak_mem_mb"]
        spd  = h["steps_per_s"] / max(b["steps_per_s"], 1e-9)
        print(f"\n  Delta (HierAttn − Baseline):")
        print(f"    val_loss : {dl:+.4f}  {'↓ better' if dl < 0 else '↑ worse'}")
        print(f"    bpb      : {db:+.4f}  {'↓ better' if db < 0 else '↑ worse'}")
        print(f"    memory   : {dm:+.0f} MB overhead")
        print(f"    speed    : {spd:.2f}x  ({100*(1-spd):.0f}% slower)")
        print()
        if dl < -0.02:
            print("  ✓ HierAttn meaningfully improves language modelling on Shakespeare.")
        elif abs(dl) <= 0.02:
            print("  ~ Similar quality — hierarchy preserves performance with longer context.")
        else:
            print("  Try: increase n_steps, reduce aux_lambda, or widen W0.")
        print(f"{'═'*75}")


def plot_curves(results):
    """ASCII learning curves — works without matplotlib."""
    print("\n  LEARNING CURVES (val bpb over training)")
    print(f"  {'Step':<8}", end="")
    for r in results:
        print(f"  {r['label'][:18]:<20}", end="")
    print()
    print("  " + "-" * (8 + 22 * len(results)))

    # Align by step index
    min_len = min(len(r["log"]["bpb"]) for r in results)
    for i in range(min_len):
        step = results[0]["log"]["steps"][i]
        print(f"  {step:<8}", end="")
        for r in results:
            bpb = r["log"]["bpb"][i]
            bar = "█" * int((bpb / 2.0) * 12)   # normalised bar
            print(f"  {bpb:.4f} {bar:<12}", end="")
        print()

    # Try matplotlib if available
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        colors = ["#4C72B0", "#DD8452"]
        for ri, r in enumerate(results):
            axes[0].plot(r["log"]["steps"], r["log"]["bpb"],
                         label=r["label"], color=colors[ri], linewidth=2)
            axes[1].plot(r["log"]["steps"],
                         [v / math.log(2) for v in r["log"]["val_loss"]],
                         label=r["label"], color=colors[ri], linewidth=2)
        for ax, title in zip(axes, ["Val BPB (↓ better)", "Val Loss (↓ better)"]):
            ax.set_xlabel("Step"); ax.set_title(title)
            ax.legend(); ax.grid(alpha=0.3)
        plt.suptitle("TinyShakespeare — Baseline vs HierAttn", fontsize=13)
        plt.tight_layout()
        plt.savefig("/kaggle/working/learning_curves.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("\n  Plot saved → /kaggle/working/learning_curves.png")
    except Exception as e:
        print(f"\n  (matplotlib plot skipped: {e})")


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: aux_lambda on Shakespeare
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_lambda(lambdas=(0.0, 0.001, 0.01, 0.05, 0.1)):
    print(f"\n{'═'*55}")
    print("  ABLATION: aux_lambda on TinyShakespeare")
    print(f"{'═'*55}")
    rows = []
    for lam in lambdas:
        cfg_ab      = Cfg()
        cfg_ab.n_steps    = 500        # quick sweep
        cfg_ab.aux_lambda = lam
        m = HierGPT(cfg_ab).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg_ab.lr, weight_decay=0.01)
        for step in range(cfg_ab.n_steps):
            x, y = get_batch("train")
            _, loss, _ = m(x, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        ev = evaluate(m, steps=50)
        rows.append((lam, ev["val_loss"], ev["bpb"]))
        marker = " ◄ best" if ev["bpb"] == min(r[2] for r in rows) else ""
        print(f"  λ={lam:.3f}  val_loss: {ev['val_loss']:.4f}  bpb: {ev['bpb']:.4f}{marker}")
    best_lam = min(rows, key=lambda r: r[2])[0]
    print(f"\n  Best λ = {best_lam}  →  use this for your final run")
    print(f"{'═'*55}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    results = []

    # 1. Baseline
    r_base, _ = train(BaselineGPT(C), "Baseline (ALiBi + LoRA)")
    results.append(r_base)

    # 2. HierAttn
    r_hier, _ = train(HierGPT(C),     "HierAttn (dynamic compression)")
    results.append(r_hier)

    # 3. Table
    print_table(results)

    # 4. Learning curves
    plot_curves(results)

    # 5. Ablation
    abl = ablation_lambda()

    # 6. Save JSON summary
    summary = {
        "dataset": "TinyShakespeare",
        "config":  C.__dict__,
        "results": [
            {k: v for k, v in r.items() if k != "log"}
            for r in results
        ],
        "ablation_lambda": [
            {"lambda": lam, "val_loss": vl, "bpb": bpb}
            for lam, vl, bpb in abl
        ],
    }
    with open("/kaggle/working/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFull results saved → /kaggle/working/benchmark_results.json")
