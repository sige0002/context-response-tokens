import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from exploration.appearance_routing import AppearanceMetric, route
from exploration.train_correspondence import predict_xy, soft_target_loss


def test_write_and_read_follow_opposite_permutations():
    permutation = torch.tensor([2, 0, 3, 1])
    values = torch.arange(12).reshape(4, 3).float()
    logits = torch.full((4, 4), -100.)
    logits[torch.arange(4), permutation] = 100.
    written, read = route(logits, values)
    torch.testing.assert_close(written, values[permutation])
    torch.testing.assert_close(read @ written, values)


def test_ambiguous_routing_exposes_average_not_invertibility():
    values = torch.tensor([[1., -1.], [-1., 1.]])
    written, read = route(torch.zeros(2, 2), values)
    assert torch.count_nonzero(written) == 0
    assert torch.count_nonzero(read @ written) == 0


def test_correspondence_loss_uses_xy_order_and_soft_boundary_targets():
    logits = torch.tensor([[0., 100., 0., 0.]])
    target = torch.tensor([[.75, .25]])
    torch.testing.assert_close(predict_xy(logits, 2), target)
    assert soft_target_loss(logits, target, 2) < 1e-5
    assert soft_target_loss(logits, target.flip(-1), 2) > 90.
    loss = soft_target_loss(torch.zeros(1, 4), torch.tensor([[.5, .5]]), 2)
    torch.testing.assert_close(loss, torch.tensor(4.).log())


def test_metric_gradient_and_batch_independence():
    torch.manual_seed(4)
    model = AppearanceMetric(8, 16, 4)
    reference, current = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    logits = model(reference, current)
    torch.testing.assert_close(logits[0], model(reference[0], current[0]))
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Official Mamba-3 requires CUDA")
def test_routed_mamba_causality_streaming_gradients_and_checkpoint(tmp_path):
    from exploration.routed_siglip import RoutedSiglip
    torch.manual_seed(17)
    config = SiglipVisionConfig(hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=16)
    model = RoutedSiglip(SiglipVisionModel(config), every=1, width=32,
                         metric_hidden=16, metric_width=8).cuda().eval()
    video = torch.randn(1, 4, 3, 32, 32, device="cuda")
    with torch.autocast("cuda", enabled=False):
        expected = model.backbone(video.flatten(0, 1)).last_hidden_state.reshape(1, 4, 4, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(model(video), expected, atol=0, rtol=0)
        for adapter in model.adapters.values():
            adapter.gain.data.fill_(.1)
        full = model(video)
        changed = video.clone(); changed[:, 2:] = torch.randn_like(changed[:, 2:])
        torch.testing.assert_close(full[:, :2], model(changed)[:, :2], atol=1e-5, rtol=1e-5)
        state, outputs, sizes = None, [], []
        for frame in video.unbind(1):
            value, state = model.step(frame, state)
            outputs.append(value); sizes.append(state.nbytes)
            assert all(len(cache) == 5 for cache in state.layers.values())
        torch.testing.assert_close(full, torch.stack(outputs, 1), atol=.015, rtol=.02)
        assert len(set(sizes)) == 1
        first, chunk = model.encode_chunk(video[:, :2])
        second, chunk = model.encode_chunk(video[:, 2:], chunk)
        torch.testing.assert_close(full, torch.cat((first, second), 1), atol=.015, rtol=.02)
        assert chunk.offset == 4 and chunk.nbytes == state.nbytes
        reset, _ = model.step(video[:, -1])
        assert (reset-full[:, -1]).abs().max() > 1e-5
        state.owner += 1
        with pytest.raises(ValueError, match="another model"):
            model.step(video[:, 0], state)
        (full.float()*torch.randn_like(full.float())).mean().backward()
        for adapter in model.adapters.values():
            for parameter in (adapter.metric.network[1].weight, adapter.mixer.in_proj.weight):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.backbone.parameters())
        path = tmp_path/"routed.pt"
        model.save_checkpoint(path)
        restored = RoutedSiglip.load_checkpoint(path, "cuda")
        torch.testing.assert_close(model(video), restored(video), atol=0, rtol=0)
        assert restored.architecture == model.architecture
