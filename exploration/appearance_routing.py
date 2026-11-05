"""Learn correspondence descriptors, without distilling encoder features."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AppearanceMetric(nn.Module):
    def __init__(self, dim=768, hidden=128, width=64):
        super().__init__()
        self.dim, self.hidden, self.width = dim, hidden, width
        self.network = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                     nn.GELU(), nn.Linear(hidden, width))
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.)))

    def encode(self, x):
        return F.normalize(self.network(x).float(), dim=-1, eps=1e-6)

    def logits(self, reference, current):
        return self.log_scale.clamp(0., math.log(100.)).exp() * (reference @ current.transpose(-2, -1))

    def forward(self, reference, current):
        return self.logits(self.encode(reference), self.encode(current))


def route(logits, values):
    """Slots -> current locations; reverse normalization reads slots to pixels.

    These are ordinary content-attention weights, not a permutation or an
    invertible transport. Ambiguous observations are still averaged.
    """
    write = logits.softmax(-1).to(values.dtype)
    read = logits.transpose(-2, -1).softmax(-1).to(values.dtype)
    return write @ values, read
