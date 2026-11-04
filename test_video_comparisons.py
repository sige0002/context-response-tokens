import copy
import json
from pathlib import Path

import pytest
import torch

from run_video_comparisons import (CONTROLS, assert_comparable, control_command,
                                   evaluation_commands, verify_completion, verify_inputs)


def test_full_controls_preserve_protocol():
    reference = dict(train="train.jsonl", val="val.jsonl", classes=174, readout="attention",
                     frames=8, batch_size=4, accumulation=8, steps=20000, warmup=1000,
                     lr=1e-4, weight_decay=1e-4, evaluate_every=1000, save_every=500,
                     seed=42, kind="mamba", placement="interleaved")
    for kind, placement in CONTROLS:
        command = control_command(reference, kind, placement, Path("fixture/run"), 1)
        arguments = dict(zip(command[5::2], command[6::2]))
        assert arguments["--steps"] == "20000"
        assert arguments["--accumulation"] == "8"
        assert arguments["--readout"] == reference["readout"]
        assert arguments["--seed"] == str(reference["seed"])
        expected = dict(reference, kind=kind, placement=placement)
        assert_comparable(expected, reference, kind, placement)
        changed = dict(expected, lr=reference["lr"] * 2)
        with pytest.raises(ValueError, match="lr"):
            assert_comparable(changed, reference, kind, placement)
        commands = evaluation_commands(Path("fixture/run"), expected, 1)
        assert "full" in commands[0] and "--modes" not in commands[0]
        assert len(commands) == (1 if kind == "none" else 2)


def test_source_and_data_drift_rejected(tmp_path):
    import hashlib
    def write(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source = write("source.py", "original")
    readout = write("vendor/frozen-video/models/rvm_modules/rvm_blocks.py", "reference")
    train, val = write("train.jsonl", "train"), write("val.jsonl", "val")
    config = dict(source_sha256={"source.py": source}, readout_sha256=readout,
                  train=str(tmp_path / "train.jsonl"), val=str(tmp_path / "val.jsonl"),
                  train_sha256=train, val_sha256=val)
    verify_inputs(config, tmp_path)
    write("source.py", "changed")
    with pytest.raises(ValueError, match="source/dependencies"):
        verify_inputs(config, tmp_path)
    write("source.py", "original")
    write("val.jsonl", "changed")
    with pytest.raises(ValueError, match="manifest"):
        verify_inputs(config, tmp_path)


def test_completion_requires_budget_all_validations_and_normal_exit(tmp_path):
    val = tmp_path / "val.jsonl"
    val.write_text('{"label": 0}\n{"label": 1}\n')
    config = dict(steps=3, evaluate_every=2, val=str(val), kind="mamba", placement="interleaved")
    metrics = dict(samples=2, predictions=[0, 0], accuracy=.5, loss=1.)
    saved = dict(format_version=2, config=config, step=3,
                 history=[dict(step=2, validation=metrics), dict(step=3, validation=metrics)])
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "progress.jsonl").write_text('{"event": "budget_complete", "step": 3}\n')
    torch.save(saved, tmp_path / "last.pt")
    assert verify_completion(tmp_path, config)["validation"] == .5
    for change in (dict(step=2), dict(history=saved["history"][:1])):
        torch.save(dict(saved, **change), tmp_path / "last.pt")
        with pytest.raises(ValueError):
            verify_completion(tmp_path, config)
    wrong = copy.deepcopy(saved)
    wrong["history"][-1]["validation"]["accuracy"] = 1.
    torch.save(wrong, tmp_path / "last.pt")
    with pytest.raises(ValueError, match="saved predictions"):
        verify_completion(tmp_path, config)
    torch.save(saved, tmp_path / "last.pt")
    (tmp_path / "progress.jsonl").write_text('{"event": "stopped", "step": 3}\n')
    with pytest.raises(ValueError, match="normal budget completion"):
        verify_completion(tmp_path, config)
