import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel
from temporal_siglip import TemporalSiglip, TemporalAdapter


def make(kind="attention", device="cpu", placement="interleaved"):
    torch.manual_seed(17)
    config = SiglipVisionConfig(hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=16)
    return TemporalSiglip(SiglipVisionModel(config), kind=kind, every=1,
                          width=32, placement=placement).to(device).eval()


def activate(model):
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.gain.fill_(0.1)


def test_initial_identity_and_gradient():
    m = make()
    x = torch.randn(1, 3, 3, 32, 32)
    expected = m.backbone(x.flatten(0, 1)).last_hidden_state.reshape(1, 3, 4, 64)
    torch.testing.assert_close(m(x), expected, atol=0, rtol=0)
    m(x).square().mean().backward()
    assert any(a.gain.grad.abs().sum() > 0 for a in m.adapters.values())
    assert all(p.grad is None for p in m.backbone.parameters())


def test_checkpoint_preserves_algorithm_and_weights(tmp_path):
    m = make()
    activate(m)
    x = torch.randn(1, 3, 3, 32, 32)
    path = tmp_path / "encoder.pt"
    m.save_checkpoint(path)
    restored = TemporalSiglip.load_checkpoint(path)
    assert restored.architecture == m.architecture
    assert not restored.training
    assert all(not p.requires_grad for p in restored.backbone.parameters())
    with torch.no_grad():
        torch.testing.assert_close(m(x), restored(x), atol=0, rtol=0)
    with pytest.raises(ValueError, match="no temporal layer"):
        TemporalSiglip(m.backbone, every=100)


def test_attention_causality_streaming_reset_and_history():
    m = make()
    activate(m)
    x = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        full = m(x)
        changed = x.clone()
        changed[:, 2:] = torch.randn_like(changed[:, 2:])
        torch.testing.assert_close(full[:, :2], m(changed)[:, :2])
        state, outputs, sizes = None, [], []
        for frame in x.unbind(1):
            y, state = m.step(frame, state)
            outputs.append(y)
            sizes.append(state.nbytes)
        torch.testing.assert_close(full, torch.stack(outputs, 1), atol=2e-6, rtol=2e-5)
        first, chunk_state = m.encode_chunk(x[:, :2])
        second, chunk_state = m.encode_chunk(x[:, 2:], chunk_state)
        torch.testing.assert_close(full, torch.cat((first, second), 1), atol=2e-6, rtol=2e-5)
        assert chunk_state.offset == 4
        torch.testing.assert_close(m.encode_current(x), full[:, -1])
        assert sizes[-1] == 4 * sizes[0]
        reset, _ = m.step(x[:, -1])
        torch.testing.assert_close(reset, m(x[:, -1:])[:, 0])
        assert (full[:, -1] - reset).abs().max() > 1e-5
        with pytest.raises(ValueError):
            make().step(x[:, 0], state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-3 requires CUDA")
@pytest.mark.parametrize("placement", ["interleaved", "late"])
def test_mamba_gpu_causality_streaming_and_backward(placement):
    m = make("mamba", "cuda", placement=placement)
    activate(m)
    x = torch.randn(1, 4, 3, 32, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = m(x)
        changed = x.clone()
        changed[:, 2:] = torch.randn_like(changed[:, 2:])
        torch.testing.assert_close(full[:, :2], m(changed)[:, :2], atol=1e-5, rtol=1e-5)
        state, outputs, sizes = None, [], []
        for frame in x.unbind(1):
            y, state = m.step(frame, state)
            outputs.append(y)
            sizes.append(state.nbytes)
        torch.testing.assert_close(full, torch.stack(outputs, 1), atol=0.015, rtol=0.02)
        assert len(set(sizes)) == 1
        first, chunk_state = m.encode_chunk(x[:, :2])
        second, chunk_state = m.encode_chunk(x[:, 2:], chunk_state)
        torch.testing.assert_close(full, torch.cat((first, second), 1), atol=0.015, rtol=0.02)
        assert chunk_state.nbytes == sizes[0]
        reset, _ = m.step(x[:, -1])
        torch.testing.assert_close(reset, m(x[:, -1:])[:, 0], atol=0.015, rtol=0.02)
        # Backprop through later frozen spatial layers to earlier adapters.
        (full.float() * torch.randn_like(full.float())).mean().backward()
        for a in m.adapters.values():
            assert torch.isfinite(a.mixer.in_proj.weight.grad).all()
            assert a.mixer.in_proj.weight.grad.abs().sum() > 0


def test_placement_control_has_identical_initial_weights_and_capacity():
    interleaved = make()
    late = make(placement="late")
    assert all(torch.equal(v, late.state_dict()[k]) for k, v in interleaved.state_dict().items())
    assert sum(p.numel() for p in interleaved.parameters()) == sum(p.numel() for p in late.parameters())
    x = torch.randn(1, 3, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(interleaved(x), late(x), atol=0, rtol=0)
        activate(interleaved)
        activate(late)
        assert (interleaved(x) - late(x)).abs().max() > 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-3 requires CUDA")
def test_mamba_long_history_chunk_boundaries_and_patch_independence():
    # Cross the official kernel's 64-token boundary twice. Test the temporal
    # operator directly to distinguish coordinate routing from spatial mixing.
    torch.manual_seed(23)
    adapter = TemporalAdapter(64, width=32, kind="mamba").cuda().eval()
    adapter.gain.data.fill_(0.5)
    x = torch.randn(2, 129, 3, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full, _ = adapter(x)
        probe = torch.randn_like(full[:, -1])
        (full[:, -1] * probe).sum().backward()
        assert torch.isfinite(x.grad).all()
        assert x.grad[:, :-1].abs().sum() > 0  # actual path from past to output
        with torch.no_grad():
            # Change past only, and only sample 0/patch 1. Current RGB-feature
            # vector is exactly unchanged; other streams must be independent.
            altered = x.detach().clone()
            altered[0, :-1, 1] *= -1
            other, _ = adapter(altered)
            assert (full[0, -1, 1] - other[0, -1, 1]).abs().max() > 1e-4
            torch.testing.assert_close(full[1], other[1], atol=0, rtol=0)
            torch.testing.assert_close(full[0, :, 0], other[0, :, 0], atol=0, rtol=0)
            state, outputs, sizes = None, [], []
            start = 0
            for end in (1, 64, 65, 128, 129):
                y, state = adapter(x[:, start:end].detach(), cache=state,
                                   offset=start, streaming=True)
                outputs.append(y)
                sizes.append(sum(s.numel() * s.element_size() for s in state))
                start = end
            torch.testing.assert_close(full, torch.cat(outputs, 1), atol=0.02, rtol=0.02)
            assert len(set(sizes)) == 1
