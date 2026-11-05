"""Inference-only sliding-window control for the trained temporal attention.

Uses the original parameter objects and absolute positions. A ring buffer
retains W observations INCLUDING the current observation at each adapter.
This changes the model's available history after W observations. It is not
an implementation-equivalent substitute for full-history attention.
"""
import torch
from torch import nn
from torch.nn import functional as F

from temporal_siglip import CausalAttention


class WindowedAttention(CausalAttention):
    def __init__(self, source, window):
        nn.Module.__init__(self)
        if type(source) is not CausalAttention or type(window) is not int or window < 1:
            raise ValueError("Expected original attention and a positive integer window")
        self.heads = source.heads
        self.qkv, self.out = source.qkv, source.out
        self.window = window
        self.train(source.training)

    def forward(self, x, cache=None, offset=0, streaming=False):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Window control requires eval and no-grad/inference mode")
        if x.ndim != 3 or x.shape[1] < 1 or type(offset) is not int or offset < 0:
            raise ValueError("Expected a nonempty sequence and nonnegative integer offset")
        b, t, d = x.shape
        if d != self.qkv.in_features or d % self.heads or d % 2:
            raise ValueError("Attention width is incompatible")
        if not streaming and (cache is not None or offset != 0):
            raise ValueError("Clip evaluation starts with no cache and offset zero")
        if streaming:
            if cache is None:
                if offset != 0:
                    raise ValueError("A fresh ring buffer must start at observation zero")
            elif (len(cache) != 3 or cache[2].device.type != "cpu"
                  or cache[2].dtype != torch.int64 or cache[2].shape != torch.Size([])
                  or int(cache[2]) != offset):
                raise ValueError("Ring cache cursor does not match stream offset")
        # Match CausalAttention's positional arithmetic, including FP32 angles.
        pos = torch.arange(offset, offset + t, device=x.device).float()
        freq = torch.exp(-torch.arange(0, d, 2, device=x.device).float()
                         * (torch.log(torch.tensor(10000., device=x.device)) / d))
        angles = pos[:, None] * freq
        pe = torch.stack((angles.sin(), angles.cos() - 1), -1).flatten(-2)
        q, k, v = self.qkv(x + pe.to(x.dtype)).reshape(
            b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        if not streaming:
            indices = torch.arange(t, device=x.device)
            lag = indices[:, None] - indices[None, :]
            mask = (lag >= 0) & (lag < self.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
            return self.out(y.transpose(1, 2).reshape(b, t, d)), None
        shape = (b, self.heads, self.window, d // self.heads)
        if cache is None:
            cache = (torch.empty(shape, dtype=k.dtype, device=k.device),
                     torch.empty(shape, dtype=v.dtype, device=v.device),
                     torch.zeros((), dtype=torch.int64))
        elif any(tuple(z.shape) != shape or z.dtype != k.dtype or z.device != k.device
                 for z in cache[:2]):
            raise ValueError("KV shape, precision or device differs from the ring buffer")
        outputs = []
        for i in range(t):
            absolute = offset + i
            slot = absolute % self.window
            cache[0][:, :, slot:slot + 1].copy_(k[:, :, i:i + 1])
            cache[1][:, :, slot:slot + 1].copy_(v[:, :, i:i + 1])
            valid = min(absolute + 1, self.window)
            # After wrapping, slots need not be chronological. Attention is
            # invariant to a joint K/V permutation in exact arithmetic. The
            # absolute position was already encoded before storing each key.
            # Avoid both copying the whole history and reading unfilled slots.
            outputs.append(F.scaled_dot_product_attention(
                q[:, :, i:i + 1], cache[0][:, :, :valid], cache[1][:, :, :valid],
                is_causal=False))
        cache[2].fill_(offset + t)
        y = outputs[0] if t == 1 else torch.cat(outputs, dim=2)
        return self.out(y.transpose(1, 2).reshape(b, t, d)), cache


def variants(encoder, window):
    if encoder.kind != "attention":
        raise ValueError("Only the temporal-attention baseline is supported")
    original = {key: adapter.mixer for key, adapter in encoder.adapters.items()}
    windowed = {key: WindowedAttention(mixer, window) for key, mixer in original.items()}
    return original, windowed
