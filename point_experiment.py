"""Protocol and completion checks for matched point-tracking experiments."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from kubric_data import KubricPointClips
from point_checkpoint import file_hash
from tapvid_metrics import compute_tapvid_metrics
from train_points import SOURCES, check_splits
from train_video import REPRODUCIBILITY
from gmmix_experiment import UV, kernel_environment


PROTOCOL = dict(frames=24, queries=64, batch_size=2, accumulation=4, steps=20000,
                warmup=1000, lr=1e-4, weight_decay=1e-4, evaluate_every=1000,
                save_every=500, seed=42)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_dataset(root, expected_plan):
    root = Path(root)
    complete = read_json(root / "complete.json")
    for key, expected in expected_plan.items():
        if complete.get(key) != expected:
            raise ValueError(f"Completed synthetic dataset differs from plan: {key}")
    for name in ("train", "validation"):
        if file_hash(root / (name + ".jsonl")) != complete[name + "_sha256"]:
            raise ValueError("Synthetic dataset manifest changed")
    train = KubricPointClips(root / "train.jsonl", None)
    validation = KubricPointClips(root / "validation.jsonl", None)
    check_splits(train, validation)
    if len(train) != 5569 or len(validation) != 300:
        raise ValueError("Expected all 5869 published scenes in the fixed split")
    ids = [f"{i:04d}" for i in range(5869)]
    random.Random(42).shuffle(ids)
    if validation.groups != set(ids[:300]) or train.groups != set(ids[300:]):
        raise ValueError("Scene assignment differs from the predetermined seed-42 split")
    # Preparation checks content hashes. Here check every file's size via the
    # dataset constructor and bind each manifest entry to its conversion record.
    for row in train.rows + validation.rows:
        metadata = read_json(row["directory"] / "metadata.json")
        if {key:value for key, value in row.items() if key != "directory"} != metadata:
            raise ValueError(f"Scene manifest differs from conversion record: {row['id']}")
    return dict(train_sha256=complete["train_sha256"], val_sha256=complete["validation_sha256"],
                complete_sha256=file_hash(root / "complete.json"), train_scenes=len(train),
                validation_scenes=len(validation),
                verification="Manifest hashes, all file sizes and conversion records; not a full prepared-content rehash")


def train_command(protocol, encoder, data, run, mode, workers, encoder_type="temporal"):
    command = UV + ["train_points.py", "--encoder-type", encoder_type,
               "--encoder-checkpoint", str(Path(encoder) / "last.pt"),
               "--train", str(Path(data) / "train.jsonl"),
               "--val", str(Path(data) / "validation.jsonl"),
               "--output", str(run), "--mode", mode, "--workers", str(workers)]
    for key, value in protocol.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def assert_training_config(config, row, plan, dataset, encoder):
    expected = dict(plan["protocol"], mode=row["mode"], development_run=False,
        reproducibility=REPRODUCIBILITY,
        encoder_type=row["encoder_type"],
        kernel_environment=kernel_environment(row["encoder_config"]) if row["encoder_type"] == "gmmix" else None,
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
    if (saved.get("format_version") != 1 or saved.get("kind") != "point_tracker"
            or saved["metadata"]["config"] != config
            or saved.get("kernel_environment") != config["kernel_environment"]
            or saved["freeze_encoder"] != (row["mode"] == "probe")):
        raise ValueError("Point checkpoint differs from its training configuration")
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
        if (validation["samples"] != len(scene_ids)
                or file_hash(predictions) != validation["predictions_sha256"]):
            raise ValueError("Incomplete or changed saved point predictions")
        with np.load(predictions, allow_pickle=False) as values:
            if not np.array_equal(values["scene_ids"], scene_ids):
                raise ValueError("Validation prediction order differs from scene manifest")
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


def verify_robotap(directory, completed, expected_extraction):
    directory = Path(directory)
    report = read_json(directory / "summary.json")
    if (report["scope"] != "full RoboTAP first-visible online evaluation" or report["videos"] != 265
            or report["checkpoint"]["checkpoint_sha256"] != completed["checkpoint_sha256"]
            or report["dataset_extraction_sha256"] != expected_extraction):
        raise ValueError("RoboTAP report is partial or belongs to different checkpoint/data")
    rows = report["rows"]
    if len(rows) != 265 or len({row["id"] for row in rows}) != 265:
        raise ValueError("RoboTAP evaluation has repeated or missing examples")
    for row in rows:
        if file_hash(directory / row["predictions"]) != row["predictions_sha256"]:
            raise ValueError("RoboTAP prediction archive changed")
    for name, value in report["metrics"].items():
        scores = np.array([row["metrics"][name] for row in rows])
        if not np.isfinite(scores).all() or abs(float(scores.mean()) - value) > 1e-9:
            raise ValueError("RoboTAP video-mean metric differs from its rows")
    return dict(report=str(directory / "summary.json"), metrics=report["metrics"],
                report_sha256=file_hash(directory / "summary.json"), videos=len(rows),
                verification="All prediction hashes and video-mean aggregates; raw RoboTAP labels not reread here")
