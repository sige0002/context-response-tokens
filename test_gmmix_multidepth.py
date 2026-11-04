"""CPU tests of the additional feature path, not a native Mamba GPU test."""
import copy

import pytest
import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

import gmmix_siglip
from gmmix_multidepth import MultiDepthGatedMambaMixSiglip, reference_enrichment_module


@pytest.fixture
def model(monkeypatch):
    # No GPU kernels in this suite; real SigLIP spatial layers and upstream
    # enrichment modules are exercised. The temporal core is an explicit stub.
    class IdentityCore(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, features, return_sequence=True):
            return features

    monkeypatch.setattr(gmmix_siglip, "reference_core", lambda: IdentityCore)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = SiglipVisionConfig(hidden_size=32, intermediate_size=64,
                               num_hidden_layers=12, num_attention_heads=4,
                               image_size=32, patch_size=16)
    return MultiDepthGatedMambaMixSiglip(SiglipVisionModel(config)).eval()


def video():
    return torch.randn(2, 3, 3, 32, 32, generator=torch.Generator().manual_seed(13))


def test_features_match_huggingface_hidden_states_at_all_four_depths(model):
    images = video()
    with torch.no_grad():
        hf = model.backbone(images.flatten(0, 1), output_hidden_states=True)
        actual = model.extract_depth_features(images)
        assert model.feature_depths == (3, 6, 9, 12)
        for depth, feature in zip(model.feature_depths, actual):
            expected = model.backbone.vision_model.post_layernorm(hf.hidden_states[depth])
            torch.testing.assert_close(feature, expected, rtol=1e-5, atol=1e-6)


def test_fusion_matches_public_dino_wrapper_recipe(model):
    images = video()
    feature_list = model.extract_depth_features(images)

    class FrozenFeatureProvider(nn.Module):
        embed_dim, n_blocks = 32, 12

        def get_intermediate_layers(self, x, n, **kwargs):
            assert n == [2, 5, 8, 11] and kwargs["norm"]
            return [(f, f[:, 0], f[:, :0]) for f in feature_list]

    public = reference_enrichment_module().DINOEncoder(
        FrozenFeatureProvider(), output_mode="patch_tokens", mix_interm_feats=True).eval()
    public.interm_mlps.load_state_dict(model.feature_mlps.state_dict())
    with torch.no_grad():
        expected = public(images.flatten(0, 1)).reshape(2, 3, 4, 32)
        actual = model.spatial_features(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_enrichment_layers_train_and_backbone_stays_frozen(model):
    before = {k:v.clone() for k,v in model.backbone.state_dict().items()}
    model.train()
    assert not model.backbone.training
    assert all(mlp.training and mlp.bn.training for mlp in model.feature_mlps)
    loss = model.spatial_features(video()).square().mean()
    loss.backward()
    for mlp in model.feature_mlps:
        assert int(mlp.bn.num_batches_tracked) == 1
        for parameter in mlp.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    optimizer = torch.optim.AdamW(model.feature_mlps.parameters(), lr=.001)
    optimizer.step()
    assert all(torch.equal(before[k], v) for k,v in model.backbone.state_dict().items())


def test_spatial_enrichment_keeps_fp32_under_outer_autocast(model):
    dtypes = []
    handles = [layer.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
               for layer in model.feature_mlps.modules() if isinstance(layer, nn.Linear)]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model.spatial_features(video())
    for handle in handles:
        handle.remove()
    assert dtypes == [torch.float32] * 8


def test_eval_features_are_independent_of_future_frames_and_chunking(model):
    images = video()
    changed = images.clone()
    changed[:, 1:] = -changed[:, 1:] * 4
    with torch.no_grad():
        full = model.spatial_features(images)
        other = model.spatial_features(changed)
        separate = torch.cat([model.spatial_features(images[:, t:t+1]) for t in range(3)], dim=1)
    torch.testing.assert_close(full[:, :1], other[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(full, separate, rtol=1e-5, atol=1e-6)


def test_training_batchnorm_mixes_clip_statistics_and_is_not_causal(model):
    original, changed_model = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    images = video()
    changed = images.clone()
    changed[:, 1:] = -changed[:, 1:] * 4
    first = original.spatial_features(images)[:, 0]
    changed_first = changed_model.spatial_features(changed)[:, 0]
    assert (first - changed_first).abs().max() > 1e-5


def test_checkpoint_preserves_enrichment_and_batchnorm_state_without_network(model, tmp_path):
    model.train()
    model.spatial_features(video()).square().mean().backward()
    optimizer = torch.optim.AdamW(model.feature_mlps.parameters(), lr=.001)
    optimizer.step()
    model.eval()
    checkpoint = tmp_path / "encoder.pt"
    model.save_checkpoint(checkpoint)
    loaded = MultiDepthGatedMambaMixSiglip.load_checkpoint(checkpoint)
    assert not loaded.training
    assert loaded.architecture == model.architecture
    assert all(torch.equal(v, loaded.state_dict()[k]) for k,v in model.state_dict().items())
    with torch.no_grad():
        torch.testing.assert_close(loaded(video()), model(video()), rtol=0, atol=0)
