"""Paired, source-group bootstrap for pilot validation predictions.

Intervals are conditional on these trained weights and validation groups. They
do not cover seed variation, actor overlap, or validation-selection optimism.
No test-set or SOTA claim is produced.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_interval(labels, predictions_a, predictions_b, groups, repeats=2000, seed=17):
    labels, a, b = map(np.asarray, (labels, predictions_a, predictions_b))
    if labels.ndim != 1 or not len(labels) or a.shape != labels.shape or b.shape != labels.shape:
        raise ValueError("Mismatched or empty prediction arrays")
    if len(groups) != len(labels) or repeats < 1:
        raise ValueError("Invalid groups or bootstrap count")
    _, inverse = np.unique(groups, return_inverse=True)
    difference = (a == labels).astype(float) - (b == labels).astype(float)
    count = np.bincount(inverse)
    sums = np.bincount(inverse, weights=difference)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats)
    for i in range(repeats):
        selected = rng.integers(len(count), size=len(count))
        bootstrap[i] = sums[selected].sum() / count[selected].sum()
    return dict(delta_percentage_points=100 * difference.mean(),
                conditional_95_interval_pp=(100 * np.quantile(bootstrap, [.025, .975])).tolist(),
                groups=len(count), samples=len(labels),
                a_only_correct=int(((a == labels) & (b != labels)).sum()),
                b_only_correct=int(((b == labels) & (a != labels)).sum()))


def read_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    history = json.loads((path / "history.json").read_text())
    if not history or [r["epoch"] for r in history] != list(range(len(history))):
        raise ValueError("Expected complete consecutive epoch records")
    manifest = Path(config["val_path"])
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["val_sha256"]:
        raise ValueError("Validation manifest changed")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    best = max(history, key=lambda r: r["validation"]["full"]["accuracy"])
    labels = [r["label"] for r in rows]
    for metrics in best["validation"].values():
        pred = metrics["predictions"]
        if len(pred) != len(labels) or metrics["samples"] != len(labels):
            raise ValueError("Prediction count differs from manifest")
        accuracy = np.mean(np.array(pred) == labels)
        if not np.isclose(accuracy, metrics["accuracy"], atol=1e-12, rtol=0):
            raise ValueError("Recorded accuracy disagrees with predictions")
    return config, history, best, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    loaded = [read_run(path) for path in args.runs]
    first = loaded[0][0]
    # These two fields are intentionally the architectural independent variables.
    controls = {k: v for k, v in first.items() if k not in ("kind", "placement")}
    for config, _, _, _ in loaded[1:]:
        if {k: v for k, v in config.items() if k not in ("kind", "placement")} != controls:
            raise ValueError("Runs differ in data/seed/training controls")
    rows = loaded[0][3]
    labels, groups = [r["label"] for r in rows], [r["group"] for r in rows]
    report = dict(scope="pilot validation; not a held-out test result",
                  interval_scope="conditional on weights and validation groups; excludes seed variation and selection bias",
                  expected_epochs=args.epochs, runs=[], between_models=[])
    for path, (config, history, best, _) in zip(args.runs, loaded):
        normal = best["validation"]["full"]["predictions"]
        report["runs"].append(dict(path=path, kind=config["kind"], placement=config.get("placement", "interleaved"),
            recorded_epochs=len(history), best_epoch_zero_based=best["epoch"],
            accuracy={a: v["accuracy"] for a, v in best["validation"].items()},
            full_minus_ablation={a: paired_interval(labels, normal, v["predictions"], groups)
                                 for a, v in best["validation"].items() if a != "full"}))
    ready = all(len(h) == args.epochs for _, h, _, _ in loaded)
    report["equal_epoch_budget_recorded"] = ready
    if ready:
        reference = loaded[0][2]["validation"]["full"]["predictions"]
        for path, (_, _, best, _) in zip(args.runs[1:], loaded[1:]):
            report["between_models"].append(dict(reference=args.runs[0], other=path,
                **paired_interval(labels, reference, best["validation"]["full"]["predictions"], groups)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
