import copy
import json
from pathlib import Path
import select
import subprocess
import sys

import numpy as np
import pytest
import torch

from point_checkpoint import file_hash
from point_experiment import (PROTOCOL, SOURCES, assert_training_config, train_command,
                             verify_point_run, verify_robotap, write_json)
from run_point_comparisons import bind_process, build_plan
from tapvid_metrics import compute_tapvid_metrics
from train_video import REPRODUCIBILITY


@pytest.mark.parametrize("include_gmmix", [False, True])
def test_plan_separates_probe_and_adaptation_with_matched_budgets(tmp_path, monkeypatch, include_gmmix):
    import run_point_comparisons as queue
    monkeypatch.setattr(queue, "verify_inputs", lambda config: None)
    video, data, robotap, output = (tmp_path / name for name in ("video", "data", "robotap", "output"))
    for path in (video, data, robotap):
        path.mkdir()
    reference = dict(kind="mamba", placement="interleaved", seed=42, steps=20000)
    controls = [dict(kind=kind, placement=where, run=str(video / f"{kind}_{where}"))
                for kind, where in (("none", "interleaved"), ("attention", "interleaved"), ("mamba", "late"))]
    write_json(video / "plan.json", dict(reference=str(video / "reference"), reference_config=reference, controls=controls))
    write_json(data / "plan.json", dict(scenes=5869))
    write_json(robotap / "extraction.json", dict(fixture=True))
    gmmix = tmp_path / "gmmix" if include_gmmix else None
    if gmmix is not None:
        gmmix.mkdir()
        write_json(gmmix / "plan.json", dict(video_queue=str(video), video_plan_sha256=file_hash(video / "plan.json"),
            run=str(gmmix / "model"), config=dict(reference, kind="gmmix", placement="after_encoder")))
    plan = build_plan(video, data, robotap, output, ["probe", "adapt"], 2, gmmix)
    assert len(plan["runs"]) == (9 if include_gmmix else 7)
    assert [row["mode"] for row in plan["runs"]] == ["probe"] * (4 + include_gmmix) + ["adapt"] * (3 + include_gmmix)
    assert plan["reused_controls"][0]["reference"] == "none_interleaved_probe_seed42"
    for row in plan["runs"]:
        command = row["command"]
        start = command.index("train_points.py") + 1
        arguments = dict(zip(command[start::2], command[start+1::2]))
        assert command[:7] == ["uv", "run", "--project", "baselines/gmmix", "--locked", "python", "train_points.py"]
        assert arguments["--encoder-type"] == row["encoder_type"]
        assert row["encoder_type"] == ("gmmix" if row["encoder_config"]["kind"] == "gmmix" else "temporal")
        assert arguments["--mode"] == row["mode"]
        assert arguments["--encoder-checkpoint"] == str(Path(row["encoder_run"]) / "last.pt")
        for key, value in PROTOCOL.items():
            assert arguments["--" + key.replace("_", "-")] == str(value)
        assert "--development-run" not in command and "--stop-after-step" not in command
    if gmmix is not None:
        baseline = json.loads((gmmix / "plan.json").read_text())
        baseline["video_plan_sha256"] = "changed"
        write_json(gmmix / "plan.json", baseline)
        with pytest.raises(ValueError, match="different video comparison"):
            build_plan(video, data, robotap, output, ["probe", "adapt"], 2, gmmix)


def make_completion_fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"id":"train0"}\n')
    (data / "validation.jsonl").write_text('{"id":"val0"}\n{"id":"val1"}\n')
    run = tmp_path / "run"
    run.mkdir()
    protocol = dict(PROTOCOL, steps=3, warmup=1, evaluate_every=2, frames=4, queries=1)
    sources = {name:"fixture-source" for name in SOURCES}
    plan = dict(protocol=protocol, data_root=str(data), source_sha256=sources)
    row = dict(run=str(run), mode="probe", encoder_type="temporal", encoder_run=str(tmp_path / "encoder"),
               encoder_config=dict(kind="mamba", steps=3))
    dataset = dict(train_sha256=file_hash(data / "train.jsonl"), val_sha256=file_hash(data / "validation.jsonl"))
    encoder = dict(step=3, checkpoint_sha256="fixture-encoder")
    config = dict(protocol, mode="probe", development_run=False, encoder_checkpoint=str(tmp_path / "encoder/last.pt"),
        reproducibility=REPRODUCIBILITY,
        encoder_type="temporal", kernel_environment=None,
        train=str(data / "train.jsonl"), val=str(data / "validation.jsonl"), **dataset,
        source_sha256=sources, encoder_initialization=dict(encoder, config=row["encoder_config"]))
    write_json(run / "config.json", config)
    target = np.full((2, 1, 4, 2), .3, dtype=np.float32)
    xy, first = target[:, :, 0].copy(), np.zeros((2, 1), dtype=np.int64)
    occluded = np.zeros((2, 1, 4), dtype=bool)
    occluded[1, 0, 2] = True
    predicted = target.copy()
    predicted[1, 0, 3, 0] += .02
    logits = np.where(occluded, -10., 10.)
    queries = np.stack((first, xy[..., 1] * 256, xy[..., 0] * 256), -1)
    metrics = {key:float(value.mean()) for key, value in compute_tapvid_metrics(
        queries, occluded, target * 256, logits < 0, predicted * 256, query_mode="first").items()}
    history = []
    for step in (2, 3):
        path = run / f"validation_{step}.npz"
        np.savez_compressed(path, scene_ids=np.array(["val0", "val1"]), query_xy=xy, query_frames=first,
            target_xy=target, occluded=occluded, predicted_xy=predicted, visibility_logits=logits)
        history.append(dict(step=step, validation=dict(samples=2, loss=.3, metrics=metrics,
            predictions=path.name, predictions_sha256=file_hash(path))))
    saved = dict(format_version=1, kind="point_tracker", metadata=dict(config=config), freeze_encoder=True,
        training_state=dict(step=3, history=history), source_sha256={name:sources[name] for name in
            ("point_checkpoint.py", "point_readout.py", "temporal_siglip.py")})
    write_json(run / "history.json", history)
    (run / "progress.jsonl").write_text('{"event":"budget_complete","step":3}\n')
    torch.save(saved, run / "last.pt")
    return row, plan, dataset, encoder, saved


def test_completed_point_run_recounts_predictions_and_rejects_partial_run(tmp_path):
    row, plan, dataset, encoder, saved = make_completion_fixture(tmp_path)
    run = Path(row["run"])
    result = verify_point_run(row, plan, dataset, encoder)
    assert result["step"] == 3 and result["scheduled_validations_recounted"] == 2
    assert result["checkpoint_sha256"] == file_hash(run / "last.pt")
    for broken in (dict(step=2), dict(history=saved["training_state"]["history"][:1])):
        altered = copy.deepcopy(saved)
        altered["training_state"].update(broken)
        torch.save(altered, run / "last.pt")
        with pytest.raises(ValueError):
            verify_point_run(row, plan, dataset, encoder)
    torch.save(saved, run / "last.pt")
    (run / "progress.jsonl").write_text('{"event":"stopped","step":3}\n')
    with pytest.raises(ValueError, match="normal budget completion"):
        verify_point_run(row, plan, dataset, encoder)


def test_point_comparison_rejects_protocol_and_encoder_drift(tmp_path):
    row, plan, dataset, encoder, saved = make_completion_fixture(tmp_path)
    config = saved["metadata"]["config"]
    for key, value in (("seed", 43), ("development_run", True), ("mode", "adapt"), ("queries", 2),
                       ("encoder_type", "gmmix"), ("kernel_environment", {"different":"build"})):
        with pytest.raises(ValueError, match="protocol differs"):
            assert_training_config(dict(config, **{key:value}), row, plan, dataset, encoder)
    with pytest.raises(ValueError, match="encoder checkpoint differs"):
        assert_training_config(config, row, plan, dataset, dict(encoder, checkpoint_sha256="other"))


def test_saved_metric_cannot_pass_merely_by_matching_history_and_checkpoint(tmp_path):
    row, plan, dataset, encoder, saved = make_completion_fixture(tmp_path)
    run = Path(row["run"])
    saved["training_state"]["history"][-1]["validation"]["metrics"]["average_jaccard"] = .123
    write_json(run / "history.json", saved["training_state"]["history"])
    torch.save(saved, run / "last.pt")
    with pytest.raises(ValueError, match="differs from saved predictions"):
        verify_point_run(row, plan, dataset, encoder)


def test_robotap_completion_rejects_subset_wrong_checkpoint_or_changed_predictions(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, prediction=np.zeros((1, 4, 2), dtype=np.float32))
    row = dict(predictions=path.name, predictions_sha256=file_hash(path), metrics=dict(average_jaccard=.4))
    report = dict(scope="full RoboTAP first-visible online evaluation", videos=265,
        checkpoint=dict(checkpoint_sha256="checkpoint"), dataset_extraction_sha256="data",
        metrics=dict(average_jaccard=.4), rows=[dict(row, id=f"video{i}") for i in range(265)])
    write_json(tmp_path / "summary.json", report)
    completed = dict(checkpoint_sha256="checkpoint")
    assert verify_robotap(tmp_path, completed, "data")["videos"] == 265
    for change in (dict(scope="engineering subset"), dict(videos=1),
                   dict(checkpoint=dict(checkpoint_sha256="different"))):
        write_json(tmp_path / "summary.json", dict(report, **change))
        with pytest.raises(ValueError, match="partial or belongs"):
            verify_robotap(tmp_path, completed, "data")
    write_json(tmp_path / "summary.json", report)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        verify_robotap(tmp_path, completed, "data")


def test_dependency_handle_binds_running_process_and_observes_its_exit(tmp_path):
    import os
    script = tmp_path / "dependency.py"
    script.write_text("import time\ntime.sleep(30)\n")
    child = subprocess.Popen([sys.executable, str(script), "--output", str(tmp_path)])
    fd = None
    try:
        fd = bind_process(child.pid, "dependency.py", "--output", tmp_path)
        with pytest.raises(ValueError, match="different output"):
            bind_process(child.pid, "dependency.py", "--output", tmp_path / "elsewhere")
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        assert not poller.poll(0)
        child.terminate()
        child.wait(timeout=5)
        assert poller.poll(1000)
    finally:
        if fd is not None:
            os.close(fd)
        if child.poll() is None:
            child.kill()
            child.wait()
