#@title 🧠 Cache policies (merged-head); includes Learned HiKV
class CachePolicy:
    def reset(self, B, D, device): raise NotImplementedError
    def append(self, k, v): raise NotImplementedError   # k,v: [B,1,D]
    def memory(self): raise NotImplementedError         # -> (K_all, V_all) [B,Tm,D]
    def new_empty_like(self): raise NotImplementedError # per-layer separate state

class FullCache(CachePolicy):
    def __init__(self): self.K=None; self.V=None
    def reset(self, B, D, device): self.K, self.V = [], []
    def append(self, k, v): self.K.append(k); self.V.append(v)
    def memory(self):
        if not self.K: return None, None
        return torch.cat(self.K, 1), torch.cat(self.V, 1)
    def new_empty_like(self): return FullCache()

class SlidingWindow(CachePolicy):
    def __init__(self, L=512):
        self.L=L; self.K=[]; self.V=[]
    def reset(self, B, D, device): self.K, self.V = [], []
    def append(self, k, v):
        self.K.append(k); self.V.append(v)
        if len(self.K) > self.L: self.K.pop(0); self.V.pop(0)
    def memory(self):
        if not self.K: return None, None
        return torch.cat(self.K, 1), torch.cat(self.V, 1)
    def new_empty_like(self): return SlidingWindow(self.L)

class LearnedPooler(nn.Module):
    """
    Tiny attention pooler: learns a query vector to pick informative tokens.
    Summary = softmax(q · K_chunk) @ {K_chunk or V_chunk}
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.q_pool = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def forward(self, K_chunk, V_chunk):
        # K_chunk, V_chunk: [B, Tchunk, C]
        scores = torch.einsum('c,btc->bt', self.q_pool, K_chunk)  # [B,Tchunk]
        w = torch.softmax(scores, dim=-1).unsqueeze(1)            # [B,1,Tchunk]
        sum_k = w @ K_chunk                                       # [B,1,C]
        sum_v = w @ V_chunk
        return sum_k, sum_v

class HiKV_Learned(CachePolicy):
    """
    Hierarchical cache with learned pooling for compression.
    Uses merged-head width D=C (embed_size).
    """
    def __init__(self, W0=64, W1=16, W2=4, embed_dim=384, pooler: LearnedPooler=None):
        self.W0, self.W1, self.W2 = W0, W1, W2
        self.embed_dim = embed_dim
        self.pooler = pooler if pooler is not None else LearnedPooler(embed_dim)
        self.L0K = deque(maxlen=W0); self.L0V = deque(maxlen=W0)
        self.L1K = deque(maxlen=W1); self.L1V = deque(maxlen=W1)
        self.L2K = deque(maxlen=W2); self.L2V = deque(maxlen=W2)
        self._D = None

    def reset(self, B, D, device):
        assert D == self.embed_dim, f"HiKV_Learned expects D={self.embed_dim}, got {D}"
        self._D = D
        self.L0K.clear(); self.L0V.clear()
        self.L1K.clear(); self.L1V.clear()
        self.L2K.clear(); self.L2V.clear()

    def _flush_L(self, Lk, Lv, Lk_next, Lv_next):
        if len(Lk)==0: return
        K = torch.cat(list(Lk), 1)  # [B,Tc,C]
        V = torch.cat(list(Lv), 1)
        sum_k, sum_v = self.pooler(K, V)  # [B,1,C] each
        Lk_next.append(sum_k); Lv_next.append(sum_v)
        Lk.clear(); Lv.clear()

    def append(self, k, v):  # [B,1,C]
        # sanity
        assert k.size(-1) == self.embed_dim, f"append D={k.size(-1)} != {self.embed_dim}"
        self.L0K.append(k); self.L0V.append(v)
        if len(self.L0K) == self.W0: self._flush_L(self.L0K, self.L0V, self.L1K, self.L1V)
        if len(self.L1K) == self.W1: self._flush_L(self.L1K, self.L1V, self.L2K, self.L2V)

    def memory(self):
        stacks = []
        if len(self.L2K): stacks.append((list(self.L2K), list(self.L2V)))
        if len(self.L1K): stacks.append((list(self.L1K), list(self.L1V)))
        if len(self.L0K): stacks.append((list(self.L0K), list(self.L0V)))
        if not stacks: return None, None
        K_all = torch.cat([torch.cat(klist, 1) for klist,_ in stacks], 1)
        V_all = torch.cat([torch.cat(vlist, 1) for _,vlist in stacks], 1)
        return K_all, V_all

    def new_empty_like(self):
        # share the pooler (parameters live in the model)
        return HiKV_Learned(self.W0, self.W1, self.W2, self.embed_dim, self.pooler)