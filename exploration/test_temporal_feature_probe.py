import torch
from transformers import SiglipVisionConfig,SiglipVisionModel

from temporal_siglip import TemporalSiglip
from exploration.temporal_feature_probe import extract_stream


def test_feature_taps_and_reset_preserve_first_frame_and_remove_prior_inputs():
    torch.manual_seed(19)
    config=SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=4,
        num_attention_heads=2,image_size=32,patch_size=16)
    model=TemporalSiglip(SiglipVisionModel(config),kind='attention',every=4,width=32).eval()
    for adapter in model.adapters.values(): adapter.gain.data.fill_(.3)
    pixels=torch.randn(3,3,32,32)
    full,full_states=extract_stream(model,pixels)
    reset,reset_states=extract_stream(model,pixels,reset=True)
    for depth in ('4','12'):
        torch.testing.assert_close(full[depth][0],reset[depth][0],rtol=0,atol=0)
        assert (full[depth][1:]-reset[depth][1:]).abs().max()>1e-5
    assert [s['offset'] for s in full_states]==[1,2,3]
    assert [s['offset'] for s in reset_states]==[1,1,1]
    for frame in range(len(pixels)):
        independent,_=extract_stream(model,pixels[frame:frame+1])
        for depth in ('4','12'):
            torch.testing.assert_close(independent[depth][0],reset[depth][frame],rtol=0,atol=0)
    torch.testing.assert_close(model.backbone.vision_model.post_layernorm(full['4']),full['12'])


def test_image_only_tap_is_spatial_output_before_final_normalization():
    torch.manual_seed(27)
    config=SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=4,
        num_attention_heads=2,image_size=32,patch_size=16)
    model=TemporalSiglip(SiglipVisionModel(config),kind='none').eval()
    pixels=torch.randn(2,3,32,32)
    features,states=extract_stream(model,pixels)
    assert all(s['bytes']==0 and not s['layer_states'] for s in states)
    torch.testing.assert_close(model.backbone.vision_model.post_layernorm(features['4']),features['12'])
