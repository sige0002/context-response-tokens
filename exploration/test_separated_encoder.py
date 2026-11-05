import copy

import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_siglip import TemporalSiglip
from exploration.separated_encoder import SeparatedEncoder


def make_model(mode='separated'):
    config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, image_size=32, patch_size=16)
    encoder = TemporalSiglip(SiglipVisionModel(config), kind='attention', every=2, width=32)
    for adapter in encoder.adapters.values():
        adapter.gain.data.fill_(.3)
    return SeparatedEncoder(encoder, observation_dim=8, mode=mode, tap_depth=2)


def test_history_independence_causality_streaming_and_reload(tmp_path):
    torch.manual_seed(39); model = make_model().eval()
    images = torch.randn(1, 4, 3, 32, 32)
    changed_past = images.clone(); changed_past[:, :3] = torch.randn_like(changed_past[:, :3])
    full, changed = model(images), model(changed_past)
    assert full.shape == (1, 4, 4, 16)
    torch.testing.assert_close(full[:, -1, :, :8], changed[:, -1, :, :8], rtol=0, atol=0)
    assert (full[:, -1, :, 8:]-changed[:, -1, :, 8:]).abs().max() > 1e-5
    future = images.clone(); future[:, 2:] = torch.randn_like(future[:, 2:])
    torch.testing.assert_close(full[:, :2], model(future)[:, :2], rtol=0, atol=0)
    state = None; chunks = []
    for image in images.unbind(1):
        value, state = model.step(image, state); chunks.append(value)
    torch.testing.assert_close(full, torch.stack(chunks, 1), rtol=2e-5, atol=2e-5)
    first, state = model.encode_chunk(images[:, :2])
    last, _ = model.encode_chunk(images[:, 2:], state)
    torch.testing.assert_close(full, torch.cat((first, last), 1), rtol=2e-5, atol=2e-5)
    path = tmp_path/'encoder.pt'; model.save_checkpoint(path)
    loaded = SeparatedEncoder.load_checkpoint(path)
    torch.testing.assert_close(full, loaded(images), rtol=0, atol=0)


def test_gradient_partition_and_fair_final_feature_control():
    torch.manual_seed(45); model = make_model().train()
    images = torch.randn(1, 3, 3, 32, 32, requires_grad=True)
    result = model(images)
    loss = (result[:, -1, :, :8]*torch.randn_like(result[:, -1, :, :8])).sum()
    loss.backward(retain_graph=True)
    assert torch.equal(images.grad[:, :-1], torch.zeros_like(images.grad[:, :-1]))
    assert images.grad[:, -1].abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.encoder.adapters.parameters())
    model.zero_grad(); images.grad = None
    (result[:, -1, :, 8:]*torch.randn_like(result[:, -1, :, 8:])).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.adapters.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    control = copy.deepcopy(model)
    control.projection.config['mode'] = 'final'; control.config['mode'] = 'final'
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in control.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='downstream'):
        SeparatedEncoder(model.encoder, observation_dim=8, tap_depth=3)
