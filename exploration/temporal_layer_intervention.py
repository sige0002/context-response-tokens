"""Inference interventions on adapter placement, without changing weights.

Removing a trained module measures the trained model's dependence on it. It
does not measure the performance of a model trained with fewer modules.
"""
from contextlib import contextmanager
import itertools

from torch import nn


def layer_masks(keys):
    keys = tuple(keys)
    return {''.join(map(str, bits)): tuple(k for k, bit in zip(keys, bits) if bit)
            for bits in itertools.product((0, 1), repeat=len(keys))}


@contextmanager
def active_adapters(encoder, keys):
    """Call with fresh episode states inside this context; never reuse a state.

    The original modules are restored even on failure. No parameter values,
    precision, layer order, or spatial operations are changed.
    """
    keys = tuple(keys)
    original = encoder.adapters
    if encoder.training or len(set(keys)) != len(keys) or not set(keys) <= set(original):
        raise ValueError('Expected an eval encoder and a unique subset of its adapters')
    if encoder.placement != 'interleaved':
        raise ValueError('This intervention is defined for interleaved adapters')
    encoder.adapters = nn.ModuleDict({k: v for k, v in original.items() if k in keys})
    try:
        yield encoder
    finally:
        encoder.adapters = original
