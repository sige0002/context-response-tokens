from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from evaluate_point_tracks import evaluate_example
from point_checkpoint import file_hash, load_point_checkpoint, save_point_checkpoint
from point_readout import PointReadout, PointTracker
from temporal_siglip import TemporalSiglip


@pytest.mark.parametrize("config,step,limit,allowed", [
    ({"development_run": True, "steps": 3}, 3, 0, False),
    ({"development_run": False, "steps": 3}, 2, 0, False),
    ({}, 3, 0, False),
    ({"development_run": False, "steps": 3}, 3, 0, True),
    ({"development_run": True, "steps": 3}, 2, 1, True),
])
def test_cli_rejects_incomplete_full_benchmark_but_allows_explicit_smoke(
        tmp_path, monkeypatch, config, step, limit, allowed):
    import evaluate_point_tracks as evaluation
    config = dict(config, reproducibility=evaluation.REPRODUCIBILITY)

    class ReachedDatasetSetup(Exception):
        pass

    def processor(*args, **kwargs):
        raise ReachedDatasetSetup

    monkeypatch.setattr("sys.argv", ["evaluate_point_tracks.py", "--checkpoint", "unused.pt",
        "--output", str(tmp_path / "evaluation"), "--limit", str(limit)])
    monkeypatch.setattr(evaluation, "load_point_checkpoint", lambda *a, **kw:
        (None, {"metadata": {"config": config}, "training_state": {"step": step}}))
    monkeypatch.setattr(evaluation.SiglipImageProcessor, "from_pretrained", processor)
    # This tests the real CLI gate without loading a GPU model or RoboTAP.
    monkeypatch.setattr(evaluation.torch, "use_deterministic_algorithms", lambda *a: None)
    monkeypatch.setattr(evaluation.torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(evaluation.torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(evaluation.torch.backends.cudnn, "deterministic", True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if allowed:
        with pytest.raises(ReachedDatasetSetup):
            evaluation.main()
    else:
        with pytest.raises(ValueError, match="completed full-data point-training checkpoint"):
            evaluation.main()


class PixelClockProcessor:
    def __call__(self, images, return_tensors):
        return SimpleNamespace(pixel_values=torch.tensor([float(np.asarray(images)[0, 0, 0])]))


@dataclass
class Bytes:
    nbytes: int = 0


class ConstantVelocityOracle:
    def eval(self):
        return self

    def step(self, pixels, state=None, new_queries=None):
        t = int(pixels.item())
        if state is None:
            state = SimpleNamespace(encoder_state=Bytes(), queries=None, bases=None)
        if new_queries is not None and new_queries.shape[1]:
            base = new_queries.clone()
            base[:, :, 0] -= .01 * t
            state.bases = base if state.bases is None else torch.cat((state.bases, base), dim=1)
            state.queries = Bytes(state.bases.numel() * 4)
        if state.bases is None:
            return dict(points=torch.empty(1, 0, 2), visibility_logits=torch.empty(1, 0)), state
        points = state.bases.clone()
        points[:, :, 0] += .01 * t
        hidden = (state.bases[:, :, 0] < .25) & (t == 4)
        return dict(points=points, visibility_logits=torch.where(hidden, -10., 10.)), state


def test_online_query_arrivals_restore_original_order_and_official_metrics():
    video = np.stack([np.full((8, 8, 3), t, dtype=np.uint8) for t in range(5)])
    points = np.tile(np.array([[.2, .4], [.3, .5], [.4, .6]], dtype=np.float32)[:, None], (1, 5, 1))
    points[:, :, 0] += np.arange(5, dtype=np.float32)[None] * .01
    occluded = np.zeros((3, 5), dtype=bool)
    occluded[0, :2] = True
    occluded[2, :3] = True
    occluded[0, 4] = True
    example = dict(video=video, points=points, occluded=occluded)
    result = evaluate_example(ConstantVelocityOracle(), example, PixelClockProcessor(), torch.device("cpu"))
    assert np.array_equal(result["query_frames"], [2, 0, 3])
    assert np.array_equal(result["valid_track_indices"], [0, 1, 2])
    assert all(abs(v - 1.) < 1e-7 for v in result["metrics"].values())
    for q, first in enumerate(result["query_frames"]):
        np.testing.assert_allclose(result["predicted_xy"][q, first:], points[q, first:], atol=1e-6)
    # Changing future target positions must change the score, not predictions.
    changed_targets = points.copy()
    changed_targets[:, 4, 1] += .2
    other = evaluate_example(ConstantVelocityOracle(), dict(example, points=changed_targets),
                             PixelClockProcessor(), torch.device("cpu"))
    assert np.array_equal(result["predicted_xy"], other["predicted_xy"])
    assert other["metrics"]["average_jaccard"] < 1.
    # New episodes reset state internally, without relying on callers.
    again = evaluate_example(ConstantVelocityOracle(), example, PixelClockProcessor(), torch.device("cpu"))
    assert np.array_equal(result["predicted_xy"], again["predicted_xy"])


def test_point_checkpoint_is_self_contained_and_preserves_predictions(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=2, image_size=32, patch_size=16)
    encoder = TemporalSiglip(SiglipVisionModel(config), kind="attention", every=1, width=32)
    head = PointReadout(input_dim=32, grid_size=2, dim=16, heads=2, frequencies=3, hidden=12)
    tracker = PointTracker(encoder, head).eval()
    video = torch.randn(1, 3, 3, 32, 32)
    xy, first = torch.tensor([[[.25, .7]]]), torch.tensor([[1]])
    with torch.no_grad():
        expected = tracker(video, xy, first)
    path = tmp_path / "tracker.pt"
    save_point_checkpoint(tracker, path, dict(use="serialization test"), training_state=dict(step=3))
    restored, info = load_point_checkpoint(path)
    assert info["checkpoint_sha256"] == file_hash(path)
    assert info["training_state"]["step"] == 3
    assert not restored.training and restored.freeze_encoder
    assert all(not p.requires_grad for p in restored.encoder.parameters())
    with torch.no_grad():
        actual = restored(video, xy, first)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
