#@title 🧩 LoRA Linear (for Q/K/V)
class LoRALinear(nn.Module):
    """
    W x + A(Bx) * scaling
    """
    def __init__(self, in_f, out_f, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.randn(out_f, r) * 0.01)
            self.B = nn.Parameter(torch.randn(r, in_f) * 0.01)
            self.scaling = alpha / r
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
            self.scaling = 0.0

    def forward(self, x):
        base = self.linear(x)
        if self.r > 0:
            lora = (x @ self.B.T) @ self.A.T * self.scaling
            return base + lora
        else:
            return base

    def freeze_base(self):
        for p in self.linear.parameters():
            p.requires_grad = False

    def lora_params(self):
        if self.r > 0:
            return [self.A, self.B]
        return []