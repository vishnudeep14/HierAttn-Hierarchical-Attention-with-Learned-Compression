# HierAttn — Hierarchical Attention with Learned Compression

> Standard transformers treat every past token the same. This project adds a memory hierarchy — like CPU cache levels — so the model keeps recent tokens at full resolution and compresses distant ones into learned summaries.

**43% improvement in bits-per-byte on TinyShakespeare vs a strong ALiBi + LoRA baseline.**
Built on top of [HiKV_Learned](https://github.com/vishnudeep14/Hierarchical-KV-Caching-in-Transformer-Architecture), extended from inference-only caching to full train-time learning.

---

## The idea in one picture

```
Every transformer block normally does this:

  tokens → Q, K, V → attend to ALL past tokens → output

HierAttn does this instead:

  tokens → Q, K, V → attend to past tokens (unchanged)        ← same as baseline
                    + read compressed summaries of distant past ← new
                    ↑
              gated residual, starts silent, grows during training
```

The key design decision: the hierarchy is **added on top** of standard attention, not a replacement. At step 0 the model behaves identically to the baseline. The hierarchy opens up gradually.

---

## Folder structure

```
hierattn/
│
├── src/
│   ├── hierarchical_attn.py     # Core architecture
│   │   ├── VectorizedPooler     # Compresses chunks of KV with one batched matmul
│   │   ├── HierKVBank           # Builds L0 / L1 / L2 levels from full sequence
│   │   ├── HierContextHead      # Attends to L1 + L2, gated residual output
│   │   ├── HierAttnV3           # Standard attn + hierarchical residual
│   │   └── HierGPTV3            # Full model, matches GPTMini interface
│   │
│   ├── cache_policy.py          # Original inference-time HiKV_Learned (unchanged)
│   ├── model.py                 # Original GPTMini baseline
│   └── lora.py                  # LoRA linear layer (shared between both models)
│
├── benchmarks/
│   ├── benchmark_shakespeare_v3.py   # Main benchmark — run this
│   ├── benchmark_shakespeare_v2.py   # v2 attempt (vectorized but wrong architecture)
│   └── benchmark_shakespeare.py      # v1 attempt (loop-based, for reference)
│
├── results/
│   ├── benchmark_v3.json        # Full results from the final run
│   └── curves_v3.png            # Learning curves plot
│
└── README.md
```

---

## How the memory hierarchy works

Think of it like CPU cache levels, but for attention:

| Level | What it stores | Slots | How it's built |
|-------|---------------|-------|----------------|
| L0 fine | The last 128 raw tokens | 128 | Direct copy — no compression |
| L1 mid | Summaries of older chunks | 16 | Every 8 tokens pooled into 1 |
| L2 coarse | Summaries of L1 pairs | 8 | Every 2 L1 summaries pooled into 1 |

**What "pooled" means:** A learned query vector scores each token in the chunk by importance, takes a weighted average. Done in one batched GPU operation — no Python loops.

```
Sequence of 256 tokens:
│←── distant past ─────────────────── recent ────│
│  L2 coarse (8)  │  L1 mid (16)  │  L0 raw (128) │
│  ~960 tokens    │  ~128 tokens  │               │
│  compressed 8:1 │  compressed   │  full res     │
```

The query Q at each position attends to all three levels. L0 uses a causal mask (can't see the future). L1 and L2 don't need one — they're already summaries of the past.

---

## The gate — why it starts silent

Each layer has a scalar gate, initialized to `sigmoid(−2) ≈ 0.12`:

```python
self.gate = nn.Parameter(torch.tensor(-2.0))

# In forward():
return torch.sigmoid(self.gate) * hier_context_output
```

The output projection is also zeroed at init. So at step 0, the hierarchical head contributes exactly zero — the model starts as a clean copy of the baseline.

Gate values during training:

```
Step  500:  [0.13, 0.13, 0.13, 0.13, 0.13, 0.12]  ← nearly silent
Step 1500:  [0.15, 0.15, 0.17, 0.18, 0.18, 0.16]  ← warming up
Step 5000:  [0.163, 0.157, 0.184, 0.194, 0.191, 0.186]  ← converged
```

Later layers (3–5) open wider. They handle meaning and structure, which repeats over longer distances. Earlier layers handle local character patterns — no need for long-range context there.

---

## The reconstruction loss

The pooler compresses chunks into summaries. Without supervision it could compress arbitrarily, averaging out the patterns that matter most.

The reconstruction loss fixes this:

```python
# Forces compressed values to stay close to the chunk mean
recon = F.mse_loss(compressed_values, chunk_values.mean(dim=-2).detach())

# Training loss:
total_loss = language_model_loss + λ × reconstruction_loss
```

`λ = 0.0` works on Shakespeare. Try `λ = 0.01` on noisier datasets.

---

## What failed before v3

### v1 — Python loop, replacement architecture

```python
# WRONG — 127 separate GPU calls, gradient graph breaks
for i in range(n_chunks):
    sk, sv = pooler(K_chunk[i], V_chunk[i])

# WRONG — replaces attention entirely
out = w[0]*attend(Q,K0) + w[1]*attend(Q,K1) + w[2]*attend(Q,K2)
# At step 0: L1 and L2 are empty, so 33% of output is zero
```

Result: loss stuck flat, 0.9 steps/sec vs baseline 6.7.

### v2 — Vectorized, still a replacement

Fixed the loop with batched einsum. Speed recovered to 6.1 steps/sec. But the replacement architecture meant the model still had no learning signal for the first 600 steps.

### v3 — Residual design

Keep standard attention exactly as baseline. Add hierarchy on top, gated to start at zero.
Result: loss curve matches baseline from step 1, then diverges downward. 43% bpb improvement.

**The lesson:** never replace something that works with something that starts broken.

---

## Results

Trained on TinyShakespeare, byte-level encoding, 5000 steps, T4 GPU.

```
═══════════════════════════════════════════════════════════════
  BENCHMARK — TinyShakespeare
═══════════════════════════════════════════════════════════════
  Model                    val_loss   bpb     params  speed
  ─────────────────────────────────────────────────────────
  Baseline (ALiBi + LoRA)   1.4928   2.1536  4.94M   6.8/s
  HierAttn v3 (residual)    0.8422   1.2150  5.72M   5.4/s
═══════════════════════════════════════════════════════════════
  -0.65 val_loss  |  -0.94 bpb  |  21% slower  |  +785MB
  ✓ HierAttn outperforms baseline
═══════════════════════════════════════════════════════════════
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/hierattn
cd hierattn
pip install torch
```

Run the benchmark on Shakespeare (Kaggle or Colab):

```python
exec(open('benchmarks/benchmark_shakespeare_v3.py').read())
```

This downloads TinyShakespeare, trains both models, prints the table, and saves a learning curve plot.

Use HierGPTV3 in your own code:

```python
from src.hierarchical_attn import HierGPTV3
from dataclasses import dataclass

@dataclass
class Cfg:
    vocab_size = 256;  block_size = 256
    embed_size = 256;  n_layers = 6;  n_heads = 4
    dropout = 0.1;     lora_r = 8;    lora_alpha = 16
    W0 = 128;  chunk_size = 8;  W1 = 16;  W2 = 8
    aux_lambda = 0.0

model = HierGPTV3(Cfg())
logits, loss, recon_loss = model(input_ids, target_ids)
loss.backward()
```

Watch the gates to see if the hierarchy is activating:

```python
print(model.gate_values())
# → [(0, 0.163), (1, 0.157), (2, 0.184), ...]
# If all gates stay < 0.13 after 1000 steps, increase W0 or reduce chunk_size
```

---

## Config reference

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `W0` | 128 | Fine window — recent raw tokens kept at full resolution |
| `chunk_size` | 8 | Tokens per L1 compression chunk |
| `W1` | 16 | Number of L1 (mid) summaries to keep |
| `W2` | 8 | Number of L2 (coarse) summaries to keep |
| `aux_lambda` | 0.0 | Reconstruction loss weight |

---

## What's next

- Replace fixed `chunk_size` with a per-layer entropy predictor — let the model decide how aggressively to compress
- Evaluate on WikiText-103 for a less repetitive, more realistic test
- Run a parameter-matched baseline (5.72M params) to separate architecture vs parameter count
- Try initializing the projection with small random weights instead of zero — may push gates higher

---

## Related work

**HiKV_Learned (this project's foundation):** inference-only 3-level KV cache with learned pooling. HierAttn extends this to train time with a residual design and reconstruction supervision.

**Memorizing Transformers (Wu et al. 2022):** retrieves from external KV memory via kNN. HierAttn compresses past context via learned pooling — no external memory, fully differentiable.

**Longformer / BigBird:** sparse attention patterns over fixed positions. HierAttn compresses and summarizes, rather than skipping.

---

Built on Andrej Karpathy's [Autoresearch](https://github.com/karpathy/autoresearch) pretraining setup.
