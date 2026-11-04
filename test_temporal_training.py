import json
from pathlib import Path

from PIL import Image
import pytest
import torch
from transformers import SiglipImageProcessor

from test_temporal_siglip import make
from train_temporal import FrameClips, VideoClassifier, check_split, run_epoch


def manifest(root, name, group, labels=(0, 1)):
    rows = []
    for i, label in enumerate(labels):
        paths = []
        for t in range(3):
            path = root / f"{name}_{i}_{t}.png"
            Image.new("RGB", (32, 32), color=(20 + 50 * t, 30 + 70 * i, 10)).save(path)
            paths.append(path.name)
        rows.append(dict(id=f"{name}_{i}", group=f"{group}_{i}", frames=paths, label=label))
    file = root / f"{name}.jsonl"
    file.write_text("\n".join(json.dumps(r) for r in rows))
    return file


def test_manifest_split_sampling_and_ablation(tmp_path):
    processor = SiglipImageProcessor(size={"height": 32, "width": 32})
    train_path = manifest(tmp_path, "train", "train")
    val_path = manifest(tmp_path, "val", "val")
    train = FrameClips(train_path, processor, 3, 2)
    val = FrameClips(val_path, processor, 3, 2)
    check_split(train, val)
    full, label = train[0]
    current, _ = FrameClips(train_path, processor, 3, 2, "current")[0]
    reversed_, _ = FrameClips(train_path, processor, 3, 2, "reverse_history")[0]
    assert label == 0
    single, _ = FrameClips(train_path, processor, 1, 2)[0]
    torch.testing.assert_close(single[0], full[-1])
    torch.testing.assert_close(current, full[-1:].expand_as(full))
    torch.testing.assert_close(reversed_[-1], full[-1])
    torch.testing.assert_close(reversed_[:-1], full[:-1].flip(0))
    with pytest.raises(ValueError, match="overlap"):
        check_split(train, train)
    val.groups = train.groups.copy()
    with pytest.raises(ValueError, match="groups overlap"):
        check_split(train, val)


def test_supervised_training_opens_gate_and_preserves_backbone():
    torch.set_num_threads(2)
    model = VideoClassifier(make(), 2)
    before = {k: v.clone() for k, v in model.encoder.backbone.state_dict().items()}
    adapter = next(iter(model.encoder.adapters.values()))
    initial_down = adapter.down.weight.detach().clone()
    data = [(torch.randn(2, 3, 3, 32, 32), torch.tensor([0, 1]))]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    for _ in range(3):
        result = run_epoch(model, data, torch.device("cpu"), optimizer)
        assert result["samples"] == 2
    assert adapter.gain.abs().sum() > 0
    assert not torch.equal(initial_down, adapter.down.weight)
    assert all(torch.equal(before[k], v) for k, v in model.encoder.backbone.state_dict().items())
