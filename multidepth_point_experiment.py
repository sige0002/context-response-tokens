"""Additional multi-depth baseline under the existing point-transfer protocol.

Shared sampling, loss, readout and metric functions are imported unchanged.
The format-specific verifier mirrors point_experiment while requiring format 2
and the extra feature-enrichment/checkpoint sources. Existing queues stay pinned.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from multidepth_experiment import kernel_environment
from multidepth_point_checkpoint import SOURCES as CHECKPOINT_SOURCES
from point_checkpoint import file_hash
from point_experiment import PROTOCOL, read_json, write_json, verify_dataset, verify_robotap
from tapvid_metrics import compute_tapvid_metrics
from train_multidepth_points import SOURCES
from train_video import REPRODUCIBILITY

UV = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]


def train_command(protocol, encoder, data, run, mode, workers):
    command = UV + ["train_multidepth_points.py", "--encoder-checkpoint", str(Path(encoder) / "last.pt"),
        "--train", str(Path(data) / "train.jsonl"), "--val", str(Path(data) / "validation.jsonl"),
        "--output", str(run), "--mode", mode, "--workers", str(workers)]
    for key,value in protocol.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def assert_training_config(config, row, plan, dataset, encoder):
    expected = dict(plan["protocol"], mode=row["mode"], development_run=False,
        reproducibility=REPRODUCIBILITY,
        encoder_type=row["encoder_type"],
        kernel_environment=kernel_environment(row["encoder_config"]) if row["encoder_type"] == "gmmix_multidepth" else None,
        encoder_checkpoint=str(Path(row["encoder_run"]) / "last.pt"),
        train=str(Path(plan["data_root"]) / "train.jsonl"),
        val=str(Path(plan["data_root"]) / "validation.jsonl"),
        train_sha256=dataset["train_sha256"], val_sha256=dataset["val_sha256"],
        source_sha256={name:plan["source_sha256"][name] for name in SOURCES})
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Point-training protocol differs: {key}")
    initialization = config["encoder_initialization"]
    for key in ("step", "checkpoint_sha256"):
        if initialization.get(key) != encoder[key]:
            raise ValueError(f"Initial encoder checkpoint differs: {key}")
    if initialization["config"] != row["encoder_config"]:
        raise ValueError("Initial encoder training conditions differ")


def verify_point_run(row, plan, dataset, encoder):
    run = Path(row["run"])
    config = read_json(run / "config.json")
    assert_training_config(config, row, plan, dataset, encoder)
    # Open once: checkpoint provenance and payload cannot refer to different
    # versions if another process replaces last.pt.
    import hashlib
    with (run / "last.pt").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        saved = torch.load(source, map_location="cpu", weights_only=True)
    if (saved.get("format_version") != 2 or saved.get("kind") != "multidepth_point_tracker"
            or saved["metadata"]["config"] != config
            or saved.get("kernel_environment") != config["kernel_environment"]
            or saved["freeze_encoder"] != (row["mode"] == "probe")):
        raise ValueError("Point checkpoint differs from its training configuration")
    if set(saved["source_sha256"]) != set(CHECKPOINT_SOURCES):
        raise ValueError("Point checkpoint source inventory differs")
    for name, expected in saved["source_sha256"].items():
        if plan["source_sha256"].get(name) != expected:
            raise ValueError("Point checkpoint source changed")
    state = saved["training_state"]
    if state["step"] != config["steps"]:
        raise ValueError("Point run did not finish the optimizer budget")
    schedule = sorted(set(range(config["evaluate_every"], config["steps"] + 1,
                                config["evaluate_every"])) | {config["steps"]})
    history = state["history"]
    if [v["step"] for v in history] != schedule or history != read_json(run / "history.json"):
        raise ValueError("Scheduled point validation is incomplete or inconsistent")
    records = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line]
    if not records or records[-1] != dict(event="budget_complete", step=config["steps"]):
        raise ValueError("Point trainer did not record normal budget completion")
    scene_ids = np.array([json.loads(line)["id"] for line in Path(config["val"]).read_text().splitlines() if line])
    for row_history in history:
        validation = row_history["validation"]
        predictions = run / validation["predictions"]
        if (not math.isfinite(validation["loss"]) or validation["samples"] != len(scene_ids)
                or file_hash(predictions) != validation["predictions_sha256"]):
            raise ValueError("Incomplete or changed saved point predictions")
        with np.load(predictions, allow_pickle=False) as values:
            if not np.array_equal(values["scene_ids"], scene_ids):
                raise ValueError("Validation prediction order differs from scene manifest")
            if (not np.isfinite(values["predicted_xy"]).all()
                    or not np.isfinite(values["visibility_logits"]).all()):
                raise ValueError("Nonfinite saved point predictions")
            xy, first = values["query_xy"], values["query_frames"]
            queries = np.stack((first, xy[..., 1] * 256, xy[..., 0] * 256), axis=-1)
            recounted = compute_tapvid_metrics(queries, values["occluded"], values["target_xy"] * 256,
                values["visibility_logits"] < 0, values["predicted_xy"] * 256, query_mode="first")
        if recounted.keys() != validation["metrics"].keys():
            raise ValueError("Validation metric keys differ")
        for name, value in recounted.items():
            if not np.isfinite(value).all() or abs(float(value.mean()) - validation["metrics"][name]) > 1e-9:
                raise ValueError(f"Point validation metric differs from saved predictions: {name}")
    return dict(step=state["step"], checkpoint_sha256=digest,
                validation=history[-1]["validation"], scheduled_validations_recounted=len(history))

