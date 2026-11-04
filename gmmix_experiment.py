"""Matched GatedMambaMix protocol and full-budget completion verification."""
import hashlib
import json
import math
from pathlib import Path

import torch

from run_video_comparisons import TRAIN_KEYS, digest, verify_inputs


ROOT = Path(__file__).resolve().parent
UV = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]
SHARED_KEYS = (*TRAIN_KEYS, "model", "revision", "readout_commit", "readout_sha256",
               "spatial_precision", "temporal_precision", "crop_min", "sampling",
               "train_sha256", "val_sha256", "reproducibility")


def read_json(path):
    return json.loads(Path(path).read_text())


def matched_config(reference, template):
    """Use an already GPU-validated build, replacing only the common protocol."""
    verify_inputs(reference)
    verify_inputs(template)
    if (template["kind"], template["placement"]) != ("gmmix", "after_encoder"):
        raise ValueError("Expected a verified GatedMambaMix template")
    for key in SHARED_KEYS:
        if key not in TRAIN_KEYS and not key.endswith("_sha256") and template[key] != reference[key]:
            raise ValueError(f"Reference/template preprocessing differs: {key}")
    for key in ("readout_sha256",):
        if template[key] != reference[key]:
            raise ValueError("Reference/template readout differs")
    result = dict(template)
    result.update({key:reference[key] for key in SHARED_KEYS})
    return result


def kernel_environment(config):
    return {key:config[key] for key in ("selective_scan_extension_sha256", "causal_conv1d_available")}


def train_command(config, output, workers):
    command = UV + ["train_gmmix_video.py", "--output", str(output), "--workers", str(workers)]
    for key in TRAIN_KEYS:
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    return command


def evaluation_commands(config, run, workers):
    run = Path(run)
    shared = UV + ["evaluate_gmmix_video.py", "--checkpoint", str(run / "last.pt"),
                  "--manifest", config["val"], "--role", "validation", "--workers", str(workers),
                  "--batch-size", str(config["batch_size"])]
    return [shared + ["--output", str(run / "final_clip_stream.json"), "--ablations", "full"],
            shared + ["--output", str(run / "final_history_ablations.json"),
                      "--modes", "clip", "--ablations", "current", "reverse_history"]]


def verify_completion(run, expected):
    run = Path(run)
    if read_json(run / "config.json") != expected:
        raise ValueError("GMMix training configuration differs from plan")
    with (run / "last.pt").open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    if (saved.get("format_version") != 3 or saved.get("model_type") != "gmmix_siglip_video"
            or saved["config"] != expected or saved["step"] != expected["steps"]):
        raise ValueError("GMMix did not finish the exact optimizer budget")
    schedule = sorted(set(range(expected["evaluate_every"], expected["steps"] + 1,
                                expected["evaluate_every"])) | {expected["steps"]})
    history = saved["history"]
    if [row["step"] for row in history] != schedule or history != read_json(run / "history.json"):
        raise ValueError("Scheduled full GMMix validation is incomplete or inconsistent")
    labels = [json.loads(line)["label"] for line in Path(expected["val"]).read_text().splitlines() if line]
    for row in history:
        metric = row["validation"]
        if (not labels or metric["samples"] != len(labels) or len(metric["predictions"]) != len(labels)
                or not math.isfinite(metric["loss"]) or not 0 <= metric["accuracy"] <= 1):
            raise ValueError("Invalid or partial GMMix validation result")
        if abs(sum(a == b for a, b in zip(labels, metric["predictions"])) / len(labels) - metric["accuracy"]) > 1e-9:
            raise ValueError("GMMix accuracy differs from saved predictions")
    records = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line]
    if not records or records[-1].get("event") != "budget_complete" or records[-1]["step"] != expected["steps"]:
        raise ValueError("GMMix trainer did not record normal budget completion")
    return dict(step=saved["step"], checkpoint_sha256=checkpoint_sha256,
                validation=history[-1]["validation"]["accuracy"],
                scheduled_validations_recounted=len(history))


def verify_evaluation(path, config, completed, ablations, modes):
    """Recount all labeled clips; checkpoint identity alone is insufficient."""
    from evaluate_temporal import classification_metrics
    report = read_json(path)
    if (report["step"] != config["steps"] or report["config"] != config
            or report["checkpoint_sha256"] != completed["checkpoint_sha256"]
            or report["manifest_sha256"] != config["val_sha256"] or report["role"] != "validation"
            or report["frames"] != config["frames"] or report["modes"] != modes
            or set(report["ablations"]) != set(ablations)):
        raise ValueError("GMMix evaluation protocol/checkpoint differs")
    rows = [json.loads(line) for line in Path(config["val"]).read_text().splitlines() if line]
    for name, expected in report["evaluation_source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("GMMix evaluator source differs")
    for ablation in ablations:
        result = report["ablations"][ablation]
        if set(result["modes"]) != set(modes):
            raise ValueError("GMMix evaluation modes incomplete")
        for mode in modes:
            metric = result["modes"][mode]
            clips = metric["clips"]
            if (len(clips) != len(rows) or not math.isfinite(metric["loss"])
                    or any(any(a[k] != b[k] for k in ("id", "group", "label")) for a, b in zip(rows, clips))):
                raise ValueError("GMMix evaluation is partial or misordered")
            counted = classification_metrics([r["label"] for r in clips], [r["prediction"] for r in clips], config["classes"])
            for key, value in counted.items():
                if metric[key] != value:
                    raise ValueError(f"GMMix evaluation metric differs from predictions: {key}")
        if len(modes) == 2:
            differences = sum(a["prediction"] != b["prediction"] for a, b in
                              zip(result["modes"]["clip"]["clips"], result["modes"]["stream"]["clips"]))
            if result["clip_vs_stream"]["prediction_disagreements"] != differences:
                raise ValueError("GMMix clip/stream disagreement count differs")
    return dict(report=str(path), report_sha256=digest(path), samples=len(rows), ablations=ablations, modes=modes)
