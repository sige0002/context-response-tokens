"""Compare fixed-final-step video evaluations with paired source-group intervals.

Streaming full-history accuracy is primary. History ablations are compared to
clip inference, because those ablations were evaluated in clip mode. Intervals
condition on the trained weights and do not estimate training-seed variation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from evaluate_temporal import classification_metrics
from summarize_temporal import paired_interval


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path, val_manifest=None):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    val_manifest = val_manifest or config["val"]
    if sha256(val_manifest) != config["val_sha256"]:
        raise ValueError("Original validation manifest changed")
    rows = [json.loads(line) for line in Path(val_manifest).read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Validation IDs must be nonempty and unique")
    labels, groups = [row["label"] for row in rows], [row["group"] for row in rows]
    scores, predictions, sources = {}, {}, {}

    def read(name, expected_hash=None):
        file = path / name
        report = json.loads(file.read_text())
        if (report["config"] != config or report["step"] != config["steps"]
                or report["frames"] != config["frames"] or report["training_frames"] != config["frames"]
                or report["role"] != "validation" or report["manifest_sha256"] != config["val_sha256"]):
            raise ValueError("Expected matched validation at the fixed final update and training history length")
        if expected_hash is not None and report["checkpoint_sha256"] != expected_hash:
            raise ValueError("History ablations use a different checkpoint")
        sources[name] = sha256(file)
        for ablation, data in report["ablations"].items():
            for mode, metrics in data["modes"].items():
                clips = metrics["clips"]
                indexed = {row["id"]: row for row in clips}
                if len(indexed) != len(clips) or set(indexed) != set(ids):
                    raise ValueError("Predicted video IDs differ from validation")
                aligned = [indexed[id_] for id_ in ids]
                if ([row["label"] for row in aligned] != labels
                        or [row["group"] for row in aligned] != groups):
                    raise ValueError("Predicted labels/source groups differ from manifest")
                pred = [row["prediction"] for row in aligned]
                measured = classification_metrics(labels, pred, config["classes"])
                if any(metrics[key] != value for key, value in measured.items()):
                    raise ValueError("Recorded metrics disagree with predictions")
                key = (ablation, mode)
                if key in predictions:
                    raise ValueError("Duplicate evaluation condition")
                predictions[key], scores[key] = pred, measured
        return report

    primary = read("final_clip_stream.json")
    if ("full", "clip") not in predictions or ("full", "stream") not in predictions:
        raise ValueError("Both full clip and streaming evaluations are required")
    disagreements = sum(a != b for a, b in zip(predictions["full", "clip"], predictions["full", "stream"]))
    if primary["ablations"]["full"]["clip_vs_stream"]["prediction_disagreements"] != disagreements:
        raise ValueError("Clip/stream disagreement count is inconsistent")
    if config["kind"] != "none":
        read("final_history_ablations.json", primary["checkpoint_sha256"])
        if not all((ablation, "clip") in predictions for ablation in ("current", "reverse_history")):
            raise ValueError("Both history ablations are required")
    return dict(path=str(path), config=config, labels=labels, groups=groups, ids=ids,
                predictions=predictions, scores=scores, disagreements=disagreements,
                checkpoint_sha256=primary["checkpoint_sha256"], evaluation_sha256=sources)


def build_report(runs):
    if not runs:
        raise ValueError("At least one run is required")
    first = runs[0]
    controls = {k: v for k, v in first["config"].items() if k not in ("kind", "placement")}
    for run in runs[1:]:
        if {k: v for k, v in run["config"].items() if k not in ("kind", "placement")} != controls:
            raise ValueError("Runs differ in data/seed/readout/optimizer budget or source version")
        if any(run[key] != first[key] for key in ("ids", "labels", "groups")):
            raise ValueError("Runs evaluate different labeled videos")
    result = dict(scope="One-seed validation at fixed final optimizer step; not held-out test or robot success",
                  primary_metric="full-history streaming top-1 accuracy",
                  interval_scope="Paired source-video bootstrap conditional on weights; excludes seed variation and actor dependence",
                  matched_updates=first["config"]["steps"], seed=first["config"]["seed"],
                  samples=len(first["labels"]), runs=[], between_models=[])
    for run in runs:
        result["runs"].append(dict(path=run["path"], kind=run["config"]["kind"],
            placement=run["config"]["placement"], checkpoint_sha256=run["checkpoint_sha256"],
            evaluation_sha256=run["evaluation_sha256"],
            conditions={f"{ablation}/{mode}": dict(accuracy=score["accuracy"],
                macro_recall=score["macro_recall_present_classes"])
                for (ablation, mode), score in run["scores"].items()},
            stream_clip_prediction_disagreements=run["disagreements"],
            full_minus_history_ablation_clip={ablation: paired_interval(run["labels"],
                run["predictions"]["full", "clip"], pred, run["groups"])
                for (ablation, mode), pred in run["predictions"].items() if ablation != "full" and mode == "clip"}))
    for run in runs[1:]:
        result["between_models"].append(dict(reference=first["path"], other=run["path"],
            **paired_interval(first["labels"], first["predictions"]["full", "stream"],
                              run["predictions"]["full", "stream"], first["groups"])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="First run is the reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-manifest", help="Relocated original validation manifest; content hash must match")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new summary output")
    report = build_report([read_run(path, args.val_manifest) for path in args.runs])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
