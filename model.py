import torch
from torch import nn

class Chain(nn.Module):
    def __init__(self, n=20):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n) * 0.1)

    def forward(self, steps=10, p=0.01, train=False):
        s = torch.zeros(self.W.size(0))
        for _ in range(steps):
            s = torch.sigmoid(self.W @ s)
            if not train and p > 0:
                s += (torch.rand_like(s) < p).float()
                s = torch.clamp(s, 0, 1)
        return s
