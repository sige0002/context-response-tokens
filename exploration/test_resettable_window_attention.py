import numpy as np
import pytest
import torch

from exploration.buffered_attention import select
from exploration.cache_delayed_action import stream_features
from exploration.resettable_window_attention import ResettableWindowAttention, variants
from exploration.windowed_attention import WindowedAttention
from temporal_siglip import CausalAttention
from test_temporal_siglip import activate, make


def reference(source, x, window, origin):
    """Dense chronological QK/softmax reference, independent of the ring code."""
    b, t, d = x.shape
    pos = torch.arange(origin, origin + t, dtype=torch.float32)
    freq = torch.exp(-torch.arange(0, d, 2, dtype=torch.float32)
                     * (torch.log(torch.tensor(10000.)) / d))
    a = pos[:, None] * freq
    pe = torch.stack((a.sin(), a.cos() - 1), -1).flatten(-2)
    q, k, v = source.qkv(x + pe.to(x.dtype)).reshape(
        b, t, 3, source.heads, d // source.heads).permute(2, 0, 3, 1, 4)
    out = []
    for i in range(t):
        start = max(0, i - window + 1)
        logits = q[:, :, i:i + 1] @ k[:, :, start:i + 1].transpose(-2, -1)
        weights = (logits / (d // source.heads) ** 0.5).softmax(-1)
        out.append(weights @ v[:, :, start:i + 1])
    return source.out(torch.cat(out, dim=2).transpose(1, 2).reshape(b, t, d))


@pytest.mark.parametrize('window', [1, 3, 8, 32, 128])
@pytest.mark.parametrize('origin', [0, 8, 137])
def test_ring_matches_reference_after_reset_and_many_wraps(window, origin):
    torch.manual_seed(131)
    torch.set_num_threads(1)
    source = CausalAttention(32).double().eval()
    mixer = ResettableWindowAttention(source, window)
    values = torch.randn(2, 145, 32, dtype=torch.float64)
    with torch.inference_mode():
        expected = reference(source, values, window, origin)
        outputs, state, pointers = [], None, None
        for i in range(values.shape[1]):
            out, state = mixer(values[:, i:i + 1], state, origin + i, True)
            outputs.append(out)
            current = [value.data_ptr() for value in state]
            if pointers is not None:
                assert current == pointers
            pointers = current
            assert state[2].tolist() == [origin + i + 1, origin]
            assert sum(v.numel() * v.element_size() for v in state) == 2 * 2 * window * 32 * 8 + 16
            if i + 1 < window:
                for kv in state[:2]:
                    kv[:, :, i + 1:] = float('nan')
        torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-12, rtol=1e-12)


def test_zero_origin_agrees_with_pinned_implementation():
    torch.manual_seed(133)
    source = CausalAttention(32).double().eval()
    old, new = WindowedAttention(source, 8), ResettableWindowAttention(source, 8)
    values = torch.randn(2, 29, 32, dtype=torch.float64)
    with torch.inference_mode():
        a = b = None
        for i in range(29):
            left, a = old(values[:, i:i + 1], a, i, True)
            right, b = new(values[:, i:i + 1], b, i, True)
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_rejected_state_does_not_mutate_allocated_cache():
    mixer = ResettableWindowAttention(CausalAttention(32).eval(), 8)
    x = torch.randn(2, 1, 32)
    with pytest.raises(RuntimeError, match='no-grad'):
        mixer(x, offset=8, streaming=True)
    with torch.inference_mode():
        _, state = mixer(x, offset=8, streaming=True)
        before = [v.clone() for v in state]
        for value, offset, streaming in ((x, 8, True), (x[:1], 9, True),
                                          (x.repeat(1, 2, 1), 9, True), (x, 9, False)):
            with pytest.raises(ValueError):
                mixer(value, state, offset, streaming)
            assert all(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
                       for a, b in zip(state, before))


def test_original_delayed_extractor_runs_with_absolute_time_preserving_reset():
    torch.manual_seed(135)
    encoder = make('attention')
    activate(encoder)
    weight_pointers = {k: p.data_ptr() for k, p in encoder.named_parameters()}
    weights = {k: p.clone() for k, p in encoder.state_dict().items()}
    original, bounded = variants(encoder, 3)
    select(encoder, bounded)
    prefix = torch.randn(8, 3, 32, 32)
    alternate = torch.randn_like(prefix)
    suffix = torch.randn(16, 3, 32, 32)
    delays = [0, 1, 8, 16]
    full, sizes = stream_features(encoder, prefix, suffix, delays)
    changed, _ = stream_features(encoder, alternate, suffix, delays)
    reset, reset_sizes = stream_features(encoder, prefix, suffix, delays, True)
    reset_again, _ = stream_features(encoder, alternate, suffix, delays, True)
    assert np.array_equal(reset, reset_again)
    assert np.array_equal(full[2:], changed[2:])
    assert not np.allclose(full[1], changed[1])
    np.testing.assert_allclose(full[2:], reset[1:], atol=1e-6, rtol=1e-5)
    assert len(set(sizes + reset_sizes)) == 1
    # The original implementation is an independent reference before W evicts.
    select(encoder, original)
    expected, _ = stream_features(encoder, prefix, suffix, delays, True)
    np.testing.assert_allclose(reset[:1], expected[:1], atol=0, rtol=0)
    assert {k: p.data_ptr() for k, p in encoder.named_parameters()} == weight_pointers
    assert all(torch.equal(weights[k], p) for k, p in encoder.state_dict().items())
