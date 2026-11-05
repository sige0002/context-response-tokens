"""Single-observation bounded attention with an absolute-time-preserving reset.

Independent of the pinned zero-origin qualification implementation. Cache
metadata stores (next absolute offset, reset origin); allocated K/V capacity
depends on W, while valid K/V count depends on observations since the reset.
"""
import torch
from torch import nn
from torch.nn import functional as F

from temporal_siglip import CausalAttention


class ResettableWindowAttention(CausalAttention):
    def __init__(self, source, window):
        nn.Module.__init__(self)
        if type(source) is not CausalAttention or type(window) is not int or window < 1:
            raise ValueError('Expected original attention and a positive integer window')
        self.heads = source.heads
        self.qkv, self.out = source.qkv, source.out
        self.window = window
        self.train(source.training)

    def forward(self, x, cache=None, offset=0, streaming=False):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Window control requires eval and no-grad/inference mode')
        if not streaming or x.ndim != 3 or x.shape[1] != 1:
            raise ValueError('Resettable window control supports single-observation streaming only')
        if type(offset) is not int or offset < 0:
            raise ValueError('Expected a nonnegative absolute observation offset')
        b, _, d = x.shape
        if d != self.qkv.in_features or d % self.heads or d % 2:
            raise ValueError('Attention width is incompatible')
        if cache is None:
            origin = offset
        else:
            if (len(cache) != 3 or cache[2].device.type != 'cpu'
                    or cache[2].dtype != torch.int64 or cache[2].shape != (2,)):
                raise ValueError('Expected CPU int64 cursor and reset-origin metadata')
            cursor, origin = cache[2].tolist()
            if cursor != offset or not 0 <= origin < offset:
                raise ValueError('Ring cursor or reset origin does not match the stream')
        # Identical positional arithmetic to the original CausalAttention.
        pos = torch.arange(offset, offset + 1, device=x.device).float()
        freq = torch.exp(-torch.arange(0, d, 2, device=x.device).float()
                         * (torch.log(torch.tensor(10000., device=x.device)) / d))
        angles = pos[:, None] * freq
        pe = torch.stack((angles.sin(), angles.cos() - 1), -1).flatten(-2)
        q, k, v = self.qkv(x + pe.to(x.dtype)).reshape(
            b, 1, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        shape = (b, self.heads, self.window, d // self.heads)
        if cache is None:
            cache = (torch.empty(shape, dtype=k.dtype, device=k.device),
                     torch.empty(shape, dtype=v.dtype, device=v.device),
                     torch.tensor([offset, origin], dtype=torch.int64))
        elif any(tuple(z.shape) != shape or z.dtype != k.dtype or z.device != k.device
                 for z in cache[:2]):
            raise ValueError('KV shape, precision or device differs from the ring buffer')
        elapsed = offset - origin
        slot = elapsed % self.window
        cache[0][:, :, slot:slot + 1].copy_(k)
        cache[1][:, :, slot:slot + 1].copy_(v)
        valid = min(elapsed + 1, self.window)
        y = F.scaled_dot_product_attention(q, cache[0][:, :, :valid],
                                          cache[1][:, :, :valid], is_causal=False)
        cache[2][0] = offset + 1
        return self.out(y.transpose(1, 2).reshape(b, 1, d)), cache


def variants(encoder, window):
    if encoder.kind != 'attention':
        raise ValueError('Only the temporal-attention baseline is supported')
    original = {key: adapter.mixer for key, adapter in encoder.adapters.items()}
    bounded = {key: ResettableWindowAttention(mixer, window) for key, mixer in original.items()}
    return original, bounded
