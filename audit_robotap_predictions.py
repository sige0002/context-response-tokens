"""Recompute saved tracking scores from original RoboTAP annotations.

No model is loaded and no training is performed. Predictions are checked
against query-arrival/order, raw video/label provenance and official metrics.
Subset audits require an explicit flag and remain engineering checks.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np

from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from robotap_data import iter_videos, video_info
from tapvid_metrics import compute_tapvid_metrics


ROOT = Path(__file__).resolve().parent


def official_metric_matches():
    upstream = ROOT / "vendor/tapnet/tapnet/tapvid/evaluation_datasets.py"
    def function(path):
        tree = ast.parse(path.read_text())
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_tapvid_metrics")
    if ast.dump(function(ROOT / "tapvid_metrics.py")) != ast.dump(function(upstream)):
        raise ValueError("Vendored metric differs from the pinned upstream function")
    return dict(upstream_file_sha256=file_hash(upstream), local_file_sha256=file_hash(ROOT / "tapvid_metrics.py"),
                comparison="Full function AST, excluding source locations")


def audit_example(row, example, directory):
    info = video_info(example)
    if info["video_sha256"] != row["video_sha256"] or info["frames"] != row["frames"]:
        raise ValueError("Prediction report refers to different raw video content")
    path = Path(directory) / row["predictions"]
    if file_hash(path) != row["predictions_sha256"]:
        raise ValueError("Saved prediction file changed")
    points, occluded = example["points"], example["occluded"]
    indices = np.flatnonzero((~occluded).any(axis=1))
    points, occluded = points[indices], occluded[indices]
    first = (~occluded).argmax(axis=1)
    if len(indices) != row["tracks"]:
        raise ValueError("Query count differs from raw annotations")
    with np.load(path, allow_pickle=False) as prediction:
        if (not np.array_equal(prediction["valid_track_indices"], indices)
                or not np.array_equal(prediction["query_frames"], first)):
            raise ValueError("Saved query selection/order differs from first-visible raw labels")
        predicted, logits = prediction["predicted_xy"], prediction["visibility_logits"]
    if (predicted.shape != points.shape or logits.shape != occluded.shape
            or not np.isfinite(predicted).all() or not np.isfinite(logits).all()):
        raise ValueError("Invalid saved prediction shape or values")
    xy = points[np.arange(len(points)), first]
    query_difference = float(np.max(np.abs(predicted[np.arange(len(points)), first] - xy)))
    # The public interface returns known queries exactly in float32. Preserve
    # possible float64 annotation rounding without admitting BF16 rounding.
    if query_difference > 1e-7:
        raise ValueError("Supplied query coordinates were changed by the tracking output")
    queries = np.stack((first, xy[:, 1] * 256, xy[:, 0] * 256), -1)[None]
    metrics = {name:float(value[0]) for name, value in compute_tapvid_metrics(
        queries, occluded[None], points[None] * 256, (logits < 0)[None],
        predicted[None] * 256, query_mode="first").items()}
    if metrics.keys() != row["metrics"].keys():
        raise ValueError("Saved metric keys differ")
    for name, value in metrics.items():
        if not np.isfinite(value) or abs(value - row["metrics"][name]) > 1e-9:
            raise ValueError(f"Metric differs from original annotations: {name}")
    return dict(id=row["id"], metrics=metrics, query_coordinate_max_abs_difference=query_difference,
                video_sha256=info["video_sha256"], predictions_sha256=row["predictions_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--root", default="data/robotap")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    path, root = Path(args.report), Path(args.root)
    report = read_json(path)
    subset = report["scope"] == "engineering subset"
    if subset and not args.allow_subset:
        parser.error("Subset audit requires --allow-subset and is not a benchmark result")
    if not subset and (report["scope"] != "full RoboTAP first-visible online evaluation" or report["videos"] != 265):
        raise ValueError("Not a completed full RoboTAP report")
    if file_hash(root / "extraction.json") != report["dataset_extraction_sha256"]:
        raise ValueError("Raw dataset provenance differs from evaluation")
    official = official_metric_matches()
    requested = {row["id"]:row for row in report["rows"]}
    if len(requested) != report["videos"] or len(requested) != len(report["rows"]) or not requested:
        raise ValueError("Repeated, missing, or empty evaluation rows")
    rows = []
    for shard, name, example in iter_videos(root):
        if name not in requested:
            continue
        row = requested.pop(name)
        if shard != row["shard"]:
            raise ValueError("Video shard differs from evaluation")
        rows.append(audit_example(row, example, path.parent))
        print(json.dumps(dict(audited=len(rows), total=report["videos"], id=name)), flush=True)
        if not requested:
            break
    if requested:
        raise ValueError("Report includes videos absent from original data")
    means = {name:float(np.mean([r["metrics"][name] for r in rows])) for name in rows[0]["metrics"]}
    if any(abs(value - report["metrics"][name]) > 1e-9 for name, value in means.items()):
        raise ValueError("Reported video-mean metrics differ from raw-label recount")
    write_json(args.output, dict(scope="Engineering subset metric audit" if subset else "Full RoboTAP raw-label metric audit",
        videos=len(rows), report_sha256=file_hash(path), original_report=str(path.resolve()),
        dataset_extraction_sha256=file_hash(root / "extraction.json"), metrics=means, rows=rows,
        official_metric=official, source_sha256={name:file_hash(ROOT / name) for name in
            ("audit_robotap_predictions.py", "robotap_data.py", "tapvid_metrics.py")},
        limits="Saved predictions are rescored; no model forward pass, additional training, or generalization claim"))


if __name__ == "__main__":
    main()
