import copy

import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_siglip import TemporalSiglip
from exploration.temporal_layer_intervention import active_adapters, layer_masks


def test_interventions_change_only_selected_paths_and_restore_on_failure():
    torch.manual_seed(23)
    config = SiglipVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=2, image_size=32, patch_size=16)
    encoder = TemporalSiglip(SiglipVisionModel(config), kind='attention', every=2, width=32).eval()
    for adapter in encoder.adapters.values():
        adapter.gain.data.fill_(.3)
    spatial = TemporalSiglip(copy.deepcopy(encoder.backbone), kind='none').eval()
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    original = encoder.adapters
    images = torch.randn(1, 3, 3, 32, 32)
    expected = encoder(images)
    masks = layer_masks(original)
    assert len(masks) == 8 and masks['101'] == ('1', '5')
    for label, keys in masks.items():
        with active_adapters(encoder, keys):
            batched = encoder(images)
            state = None; outputs = []
            for image in images.unbind(1):
                output, state = encoder.step(image, state)
                outputs.append(output)
                assert set(state.layers) == set(keys)
            torch.testing.assert_close(batched, torch.stack(outputs, 1), atol=2e-6, rtol=2e-6)
            if label == '000':
                torch.testing.assert_close(batched, spatial(images), rtol=0, atol=0)
            if label == '111':
                torch.testing.assert_close(batched, expected, rtol=0, atol=0)
        assert encoder.adapters is original
    with pytest.raises(RuntimeError, match='deliberate'):
        with active_adapters(encoder, ()):
            raise RuntimeError('deliberate')
    assert encoder.adapters is original
    for key, value in before.items():
        torch.testing.assert_close(value, encoder.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        with active_adapters(encoder, ('missing',)):
            pass
    assert encoder.adapters is original
