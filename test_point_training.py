from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from point_checkpoint import file_hash
from train_points import check_splits, validate


def test_split_check_rejects_relabelled_copy_of_source_scene():
    train = SimpleNamespace(groups={"a"}, rows=[dict(directory="a", source_sha256="same")])
    validation = SimpleNamespace(groups={"b"}, rows=[dict(directory="b", source_sha256="same")])
    with pytest.raises(ValueError, match="source_sha256"):
        check_splits(train, validation)
    validation.rows[0]["source_sha256"] = "different"
    check_splits(train, validation)
    validation.groups = {"a"}
    with pytest.raises(ValueError, match="IDs overlap"):
        check_splits(train, validation)


class KnownTrajectories(Dataset):
    rows = [dict(id=name) for name in ("a", "b", "c")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        first = torch.tensor([0, 1])
        points = torch.tensor([[.2, .3], [.5, .6]])[:, None].expand(-1, 4, -1).clone()
        points[:, :, 0] += torch.arange(4)[None] * .01
        points += i * .02
        return dict(video=torch.arange(4).float()[:, None], query_xy=points[torch.arange(2), first],
                    query_frames=first, target_xy=points,
                    occluded=torch.arange(4)[None] < first[:, None])


class KnownVelocity(nn.Module):
    def forward(self, video, query_xy, query_frames):
        points = query_xy[:, :, None].expand(-1, -1, video.shape[1], -1).clone()
        points[..., 0] += (video[:, None, :, 0] - query_frames[..., None]) * .01
        return dict(points=points, visibility_logits=torch.full(points.shape[:-1], 10.))


def test_validation_weights_last_batch_and_records_every_scene(tmp_path):
    dataset = KnownTrajectories()
    predictions = tmp_path / "predictions.npz"
    result = validate(KnownVelocity(), DataLoader(dataset, batch_size=2), torch.device("cpu"), predictions)
    separate = validate(KnownVelocity(), DataLoader(dataset, batch_size=1), torch.device("cpu"))
    assert result["samples"] == 3
    assert all(abs(v - 1.) < 1e-7 for v in result["metrics"].values())
    for key in ("loss", "position", "visibility"):
        assert abs(result[key] - separate[key]) < 1e-7
    assert result["predictions_sha256"] == file_hash(predictions)
    with np.load(predictions, allow_pickle=False) as saved:
        assert saved["scene_ids"].tolist() == ["a", "b", "c"]
        assert saved["predicted_xy"].shape == (3, 2, 4, 2)
        np.testing.assert_allclose(saved["target_xy"], saved["predicted_xy"], atol=1e-6)
        assert np.array_equal(saved["query_frames"], [[0, 1]] * 3)
