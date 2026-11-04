import copy
import hashlib
import json

import pytest

from evaluate_temporal import classification_metrics
from summarize_video import build_report, read_run


def make_run(tmp_path, name, predictions):
    val = tmp_path / "val.jsonl"
    rows = [dict(id="a", group="a", label=0), dict(id="b", group="b", label=1)]
    val.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    config = dict(kind="none", placement="interleaved", steps=20, seed=42, frames=8, classes=2,
                  val=str(val), val_sha256=hashlib.sha256(val.read_bytes()).hexdigest())
    metrics = dict(**classification_metrics([0, 1], predictions, 2),
                   clips=[dict(row, prediction=pred) for row, pred in zip(rows, predictions)])
    report = dict(config=config, step=20, frames=8, training_frames=8, role="validation",
                  manifest_sha256=config["val_sha256"], checkpoint_sha256=name,
                  ablations=dict(full=dict(modes=dict(clip=metrics, stream=metrics),
                                          clip_vs_stream=dict(prediction_disagreements=0))))
    run = tmp_path / name
    run.mkdir()
    (run / "config.json").write_text(json.dumps(config))
    (run / "final_clip_stream.json").write_text(json.dumps(report))
    return run, report


def test_summary_uses_paired_stream_predictions_and_rejects_budget_drift(tmp_path):
    a, _ = make_run(tmp_path, "a", [0, 1])
    b, _ = make_run(tmp_path, "b", [1, 0])
    loaded = [read_run(a), read_run(b)]
    report = build_report(loaded)
    assert report["between_models"][0]["delta_percentage_points"] == 100
    assert report["between_models"][0]["conditional_95_interval_pp"] == [100, 100]
    changed = copy.deepcopy(loaded)
    changed[1]["config"]["steps"] += 1
    with pytest.raises(ValueError, match="optimizer budget"):
        build_report(changed)


def test_rejects_wrong_checkpoint_steps_labels_and_saved_metrics(tmp_path):
    run, original = make_run(tmp_path, "run", [0, 1])
    bad = copy.deepcopy(original)
    bad["step"] -= 1
    (run / "final_clip_stream.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fixed final"):
        read_run(run)
    for corruption in ("labels", "accuracy"):
        bad = copy.deepcopy(original)
        if corruption == "labels":
            bad["ablations"]["full"]["modes"]["clip"]["clips"][0]["label"] = 1
        else:
            bad["ablations"]["full"]["modes"]["clip"]["accuracy"] = 0
        (run / "final_clip_stream.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            read_run(run)
