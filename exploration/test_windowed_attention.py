import pytest
import torch

from exploration.buffered_attention import select
from exploration.windowed_attention import WindowedAttention, variants
from temporal_siglip import CausalAttention
from test_temporal_siglip import make, activate


def chronological_reference(source, values, window):
    """Independent dense score/softmax reference, keeping absolute positions."""
    b, t, d = values.shape
    pos = torch.arange(t, dtype=torch.float32)
    freq = torch.exp(-torch.arange(0, d, 2, dtype=torch.float32)
                     * (torch.log(torch.tensor(10000.)) / d))
    angles = pos[:, None] * freq
    pe = torch.stack((angles.sin(), angles.cos() - 1), -1).flatten(-2)
    qkv = source.qkv(values + pe.to(values.dtype))
    q, k, v = qkv.reshape(b, t, 3, source.heads, d // source.heads).permute(2, 0, 3, 1, 4)
    out = []
    for i in range(t):
        lo = max(0, i - window + 1)
        scores = q[:, :, i:i + 1] @ k[:, :, lo:i + 1].transpose(-2, -1)
        weights = (scores / (d // source.heads) ** 0.5).softmax(dim=-1)
        out.append(weights @ v[:, :, lo:i + 1])
    return source.out(torch.cat(out, dim=2).transpose(1, 2).reshape(b, t, d))


@pytest.mark.parametrize("window", [1, 3, 8])
def test_ring_matches_chronological_reference_across_wraps_and_chunks(window):
    torch.manual_seed(113)
    torch.set_num_threads(1)
    source = CausalAttention(32).double().eval()
    mixer = WindowedAttention(source, window)
    values = torch.randn(2, 29, 32, dtype=torch.float64)
    with torch.inference_mode():
        expected = chronological_reference(source, values, window)
        clip, _ = mixer(values)
        torch.testing.assert_close(clip, expected, atol=1e-12, rtol=1e-12)
        for lengths in ([1] * 29, [2, 7, 1, 13, 6]):
            state, offset, outputs, pointers = None, 0, [], None
            for length in lengths:
                out, state = mixer(values[:, offset:offset + length], state, offset, True)
                outputs.append(out)
                current = [v.data_ptr() for v in state]
                if pointers is not None:
                    assert pointers == current
                pointers = current
                offset += length
                assert int(state[2]) == offset
                assert sum(v.numel() * v.element_size() for v in state) == 2 * 2 * window * 32 * 8 + 8
                if offset < window:
                    for kv in state[:2]:
                        kv[:, :, offset:] = float("nan")
            torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-12, rtol=1e-12)


def test_invalid_state_and_training_are_rejected_without_mutating_cache():
    source = CausalAttention(32).eval()
    mixer = WindowedAttention(source, 3)
    x = torch.randn(2, 1, 32)
    with pytest.raises(RuntimeError, match="no-grad"):
        mixer(x, streaming=True)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="zero"):
            mixer(x, offset=1, streaming=True)
        _, state = mixer(x, streaming=True)
        before = [v.clone() for v in state]
        with pytest.raises(ValueError, match="cursor"):
            mixer(x, state, offset=0, streaming=True)
        with pytest.raises(ValueError, match="shape"):
            mixer(x[:1], state, offset=1, streaming=True)
        # Unfilled slots may have any bit pattern, including NaNs. Check byte
        # identity so the assertion itself does not read them as valid numbers.
        assert all(torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
                   for a, b in zip(before, state))
        mixer.train()
        with pytest.raises(RuntimeError, match="eval"):
            mixer(x, state, offset=1, streaming=True)


def test_full_encoder_matches_before_eviction_and_resets_with_shared_weights():
    encoder = make("attention")
    activate(encoder)
    old = {k: v.clone() for k, v in encoder.state_dict().items()}
    pointers = {k: p.data_ptr() for k, p in encoder.named_parameters()}
    original, windowed = variants(encoder, 8)
    images = torch.randn(1, 8, 3, 32, 32)
    with torch.inference_mode():
        expected = encoder(images)
        select(encoder, windowed)
        actual = encoder(images)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        first, cache1 = encoder.encode_chunk(images)
        second, cache2 = encoder.encode_chunk(images)
        assert torch.equal(first, second)
        assert cache1.offset == cache2.offset == 8
    assert {k: p.data_ptr() for k, p in encoder.named_parameters()} == pointers
    assert all(torch.equal(old[k], v) for k, v in encoder.state_dict().items())
    select(encoder, original)


def test_multilayer_history_horizon_is_wider_than_one_cache_window():
    encoder = make("attention")
    activate(encoder)
    layers, window = len(encoder.adapters), 3
    horizon = 1 + layers * (window - 1)
    original, windowed = variants(encoder, window)
    select(encoder, windowed)
    images = torch.randn(1, horizon + 4, 3, 32, 32)
    changed = images.clone()
    changed[:, 0] = torch.randn_like(changed[:, 0]) * 3
    with torch.inference_mode():
        a, _ = encoder.encode_chunk(images)
        b, _ = encoder.encode_chunk(changed)
        # Differences propagate beyond a single adapter's W observations.
        assert float((a[:, window:horizon] - b[:, window:horizon]).abs().max()) > 1e-7
        # With L causal attention adapters, the original image can no longer
        # affect outputs once its distance exceeds L*(W-1) observations.
        torch.testing.assert_close(a[:, horizon:], b[:, horizon:], atol=0, rtol=0)
        select(encoder, original)
        full_a = encoder(images)
        full_b = encoder(changed)
        assert float((full_a[:, -1] - full_b[:, -1]).abs().max()) > 1e-7
