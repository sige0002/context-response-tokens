"""Check matched budgets and rejection of incomplete/miscounted point results."""
import copy
from pathlib import Path

import pytest
import torch

import run_multidepth_points as queue
from multidepth_point_checkpoint import SOURCES as CHECKPOINT_SOURCES
from multidepth_point_experiment import (PROTOCOL, SOURCES, assert_training_config,
    verify_point_run, write_json, read_json)
from point_checkpoint import file_hash
from test_point_experiments import make_completion_fixture as original_fixture


def test_queue_matches_original_point_protocol_and_rejects_different_chain(tmp_path, monkeypatch):
    # Plan construction only: native qualification is separately exercised by
    # the real integration CLI, not replaced in any submitted experiment.
    monkeypatch.setattr(queue, "verify_video_plan", lambda plan: None)
    monkeypatch.setattr(queue, "verify_original_point_plan", lambda plan: None)
    monkeypatch.setattr(queue, "verify_integration", lambda path: dict(probe_resume_checked=True))
    video, points, history, integration = [tmp_path / name for name in ("video", "points", "history", "integration")]
    for path in (video, points, history, integration):
        path.mkdir()
    write_json(integration / "verification.json", {})
    original = dict(protocol=PROTOCOL, modes=["probe", "adapt"], video_plan_sha256="same-video",
        data_root=str(tmp_path / "data"), data_plan={}, data_plan_sha256="data", robotap_root=str(tmp_path / "robotap"),
        robotap_extraction_sha256="robotap", selection="Fixed final optimizer step")
    write_json(points / "plan.json", original)
    history_plan = dict(point_queue=str(points), point_plan_sha256=file_hash(points / "plan.json"))
    write_json(history / "plan.json", history_plan)
    write_json(video / "plan.json", dict(history_queue=str(history), video_plan_sha256="same-video",
        run=str(video / "model"), config=dict(kind="gmmix_multidepth", steps=20000)))
    plan = queue.build_plan(video, points, integration, tmp_path / "output", workers=2)
    assert plan["protocol"] == original["protocol"]
    assert [row["mode"] for row in plan["runs"]] == ["probe", "adapt"]
    for row in plan["runs"]:
        command = row["command"]
        start = command.index("train_multidepth_points.py") + 1
        flags = dict(zip(command[start::2], command[start+1::2]))
        assert flags["--encoder-checkpoint"] == str(video / "model/last.pt")
        assert flags["--mode"] == row["mode"]
        assert "--development-run" not in command
        for name,value in PROTOCOL.items():
            assert flags["--"+name.replace("_", "-")] == str(value)
    write_json(history / "plan.json", dict(history_plan, point_plan_sha256="different"))
    with pytest.raises(ValueError, match="different chains"):
        queue.build_plan(video, points, integration, tmp_path / "output", workers=2)


def fixture(tmp_path):
    row, plan, dataset, encoder, saved = original_fixture(tmp_path)
    config = saved["metadata"]["config"]
    environment = dict(selective_scan_extension_sha256="fixture-kernel", causal_conv1d_available=False)
    row.update(encoder_type="gmmix_multidepth", encoder_config=dict(kind="gmmix_multidepth", steps=3, **environment))
    plan["source_sha256"] = {name:"fixture-source" for name in SOURCES}
    config.update(encoder_type=row["encoder_type"], kernel_environment=environment,
        source_sha256=plan["source_sha256"], encoder_initialization=dict(encoder, config=row["encoder_config"]))
    saved.update(format_version=2, kind="multidepth_point_tracker", kernel_environment=environment,
        source_sha256={name:plan["source_sha256"][name] for name in CHECKPOINT_SOURCES})
    write_json(Path(row["run"]) / "config.json", config)
    torch.save(saved, Path(row["run"]) / "last.pt")
    return row, plan, dataset, encoder, saved


def test_recounts_completion_and_rejects_budget_format_or_inventory_changes(tmp_path):
    row, plan, dataset, encoder, saved = fixture(tmp_path)
    result = verify_point_run(row, plan, dataset, encoder)
    assert result["scheduled_validations_recounted"] == 2
    for mutation in ("partial", "format", "inventory", "history"):
        altered = copy.deepcopy(saved)
        if mutation == "partial":
            altered["training_state"]["step"] = 2
        elif mutation == "format":
            altered["format_version"] = 1
        elif mutation == "inventory":
            altered["source_sha256"].pop("gmmix_multidepth.py")
        else:
            altered["training_state"]["history"] = altered["training_state"]["history"][:1]
        torch.save(altered, Path(row["run"]) / "last.pt")
        with pytest.raises(ValueError):
            verify_point_run(row, plan, dataset, encoder)


@pytest.mark.parametrize("field,value", [("frames", 16), ("queries", 32), ("development_run", True),
    ("kernel_environment", {}), ("encoder_type", "gmmix")])
def test_rejects_training_protocol_drift(tmp_path, field, value):
    row, plan, dataset, encoder, saved = fixture(tmp_path)
    with pytest.raises(ValueError, match="protocol differs"):
        assert_training_config(dict(saved["metadata"]["config"], **{field:value}), row, plan, dataset, encoder)


@pytest.mark.parametrize("field,value", [("average_jaccard", .123), ("loss", float("nan"))])
def test_corrupt_metrics_cannot_pass_by_matching_checkpoint_and_history(tmp_path, field, value):
    row, plan, dataset, encoder, saved = fixture(tmp_path)
    last = saved["training_state"]["history"][-1]["validation"]
    if field == "loss":
        last[field] = value
    else:
        last["metrics"][field] = value
    write_json(Path(row["run"]) / "history.json", saved["training_state"]["history"])
    torch.save(saved, Path(row["run"]) / "last.pt")
    with pytest.raises(ValueError):
        verify_point_run(row, plan, dataset, encoder)
