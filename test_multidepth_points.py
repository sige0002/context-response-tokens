"""Point-head/freeze/checkpoint contracts; temporal core is a CPU-only stub.

Native core recurrence and streaming are checked separately on actual videos.
"""
import pytest
import torch

import multidepth_point_checkpoint as checkpoint
from point_checkpoint import load_point_checkpoint as load_original_checkpoint
from point_readout import PointReadout, PointTracker, point_tracking_loss
from test_gmmix_multidepth import model, video  # Explicit shared CPU spatial fixture.


def tracker_for(encoder, frozen):
    head = PointReadout(input_dim=32, grid_size=2, dim=32, heads=4, frequencies=2, hidden=16)
    return PointTracker(encoder, head, freeze_encoder=frozen)


def inputs():
    xy = torch.tensor([[[.2, .3], [.7, .8]]] * 2)
    frames = torch.tensor([[0, 1]] * 2)
    target = xy[:, :, None].expand(-1, -1, 3, -1).clone()
    target[:, :, 2] += .04
    return xy, frames, target, torch.zeros(2, 2, 3, dtype=torch.bool)


@pytest.mark.parametrize("frozen", [True, False], ids=["probe", "adapt"])
def test_update_preserves_backbone_and_respects_feature_freezing(model, frozen):
    tracker = tracker_for(model, frozen).train()
    before = {k:v.clone() for k,v in tracker.state_dict().items()}
    xy, frames, target, occluded = inputs()
    loss = point_tracking_loss(tracker(video(), xy, frames), target, occluded, frames)["loss"]
    loss.backward()
    torch.optim.AdamW([p for p in tracker.parameters() if p.requires_grad], lr=.001).step()
    after = tracker.state_dict()
    assert all(torch.equal(value, after[key]) for key,value in before.items()
               if key.startswith("encoder.backbone."))
    assert any(not torch.equal(value, after[key]) for key,value in before.items()
               if key.startswith("readout."))
    if frozen:
        assert not tracker.encoder.training
        assert all(torch.equal(value, after[key]) for key,value in before.items() if key.startswith("encoder."))
        assert all(p.grad is None for p in model.parameters())
    else:
        assert not tracker.encoder.backbone.training
        for index, mlp in enumerate(model.feature_mlps):
            assert mlp.training and int(mlp.bn.num_batches_tracked) == 1
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mlp.parameters())
            prefix = f"encoder.feature_mlps.{index}."
            assert any(not torch.equal(value, after[key]) for key,value in before.items()
                       if key.startswith(prefix) and "running_" not in key and "num_batches" not in key)


@pytest.mark.parametrize("frozen", [True, False], ids=["probe", "adapt"])
def test_self_contained_checkpoint_roundtrip_and_train_mode(model, frozen, monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "point_kernel_environment", lambda kind: {"test_only_stub": True})
    tracker = tracker_for(model, frozen).train()
    xy, frames, target, occluded = inputs()
    optimizer = torch.optim.AdamW([p for p in tracker.parameters() if p.requires_grad], lr=.001)
    point_tracking_loss(tracker(video(), xy, frames), target, occluded, frames)["loss"].backward()
    optimizer.step()
    path = tmp_path / "point.pt"
    state = dict(step=1, optimizer=optimizer.state_dict(), rng=torch.get_rng_state())
    checkpoint.save_point_checkpoint(tracker, path, {"scope": "unit test"}, state)
    loaded, payload = checkpoint.load_point_checkpoint(path)
    assert not loaded.training and loaded.freeze_encoder == frozen
    assert payload["checkpoint_sha256"] == checkpoint.file_hash(path)
    assert payload["training_state"]["step"] == 1
    assert torch.equal(payload["training_state"]["rng"], state["rng"])
    assert all(torch.equal(value, loaded.state_dict()[key]) for key,value in tracker.state_dict().items())
    tracker.eval()
    with torch.no_grad():
        for key, value in tracker(video(), xy, frames).items():
            torch.testing.assert_close(value, loaded(video(), xy, frames)[key], rtol=0, atol=0)
    loaded.train()
    assert not loaded.encoder.backbone.training
    assert all(mlp.training == (not frozen) for mlp in loaded.encoder.feature_mlps)
    assert all(p.requires_grad == (not frozen) for p in loaded.encoder.feature_mlps.parameters())
    with pytest.raises(ValueError, match="self-contained point-tracker"):
        load_original_checkpoint(path)


@pytest.mark.parametrize("mutation, message", [
    ("missing_source", "source inventory"), ("changed_source", "source differs"),
    ("kernel", "kernel build"), ("upstream", "reference differs")])
def test_checkpoint_rejects_incompatible_provenance(model, monkeypatch, tmp_path, mutation, message):
    monkeypatch.setattr(checkpoint, "point_kernel_environment", lambda kind: {"test_only_stub": True})
    path = tmp_path / "point.pt"
    checkpoint.save_point_checkpoint(tracker_for(model, True), path, {})
    payload = torch.load(path, weights_only=True)
    if mutation == "missing_source":
        payload["source_sha256"].pop("point_readout.py")
    elif mutation == "changed_source":
        payload["source_sha256"]["point_readout.py"] = "0" * 64
    elif mutation == "kernel":
        payload["kernel_environment"] = {"test_only_stub": False}
    else:
        payload["enrichment_sha256"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        checkpoint.load_point_checkpoint(path)
