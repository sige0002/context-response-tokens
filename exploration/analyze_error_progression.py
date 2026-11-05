"""Describe audited SSv2 prediction changes without inference or model updates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def analyze(run, audit_path, steps, reference_step):
    if steps != sorted(set(steps)) or len(steps) < 2 or reference_step not in steps[:-1]:
        raise ValueError("Use ordered distinct checkpoints and an earlier reference")
    config = json.loads((run / "config.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit["config_sha256"] != digest((run / "config.json").read_bytes()):
        raise ValueError("Audit belongs to a different configuration")
    manifest = Path(config["val"])
    manifest_bytes = manifest.read_bytes()
    if digest(manifest_bytes) != config["val_sha256"]:
        raise ValueError("Validation manifest changed")
    rows = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    labels_path = Path("data/ssv2/labels/labels.json")
    official_path = Path("data/ssv2/labels/validation.json")
    names = {int(v): k for k, v in json.loads(labels_path.read_text()).items()}
    original = json.loads(official_path.read_text())
    official = {r["id"]: r["template"].translate(str.maketrans("", "", "[]")) for r in original}
    if (len(official) != len(original) or len({r["id"] for r in rows}) != len(rows)
            or set(official) != {r["id"] for r in rows}
            or any(names[r["label"]] != official[r["id"]] for r in rows)):
        raise ValueError("Validation IDs or official labels disagree")
    target = [r["label"] for r in rows]
    history = json.loads((run / "history.json").read_text())
    records = {r["step"]: r["validation"] for r in history if r["step"] in steps}
    if len(records) != len(steps) or sum(r["step"] in steps for r in history) != len(steps):
        raise ValueError("Missing or duplicate checkpoints")
    audited = {r["step"]: r for r in audit["validation"]}
    predictions, hashes, correct = {}, {}, {}
    for step in steps:
        p = records[step]["predictions"]
        if len(p) != len(rows) or any(type(v) is not int or v not in names for v in p):
            raise ValueError("Invalid prediction array")
        hashes[str(step)] = digest(json.dumps(p).encode())
        correct[step] = sum(a == b for a, b in zip(p, target))
        if (hashes[str(step)] != audited[step]["predictions_sha256"]
                or correct[step] != audited[step]["correct"]
                or abs(correct[step] / len(rows) - records[step]["accuracy"]) > 1e-12):
            raise ValueError("Predictions disagree with the independently recounted audit")
        predictions[step] = p
    previous, latest = steps[-2:]
    old, new = predictions[previous], predictions[latest]
    gained = sum(a != y and b == y for a, b, y in zip(old, new, target))
    lost = sum(a == y and b != y for a, b, y in zip(old, new, target))
    if gained - lost != correct[latest] - correct[previous]:
        raise AssertionError("Paired recount failed")
    support = Counter(target)
    class_counts = {s: Counter(y for y, p in zip(target, predictions[s]) if y == p) for s in steps}
    confusions = Counter((y, p) for y, p in zip(target, new) if y != p)
    persistent = [all(predictions[s][i] != y for s in steps) for i, y in enumerate(target)]
    reference_steps = [s for s in steps if s <= reference_step]
    earlier_persistent = [all(predictions[s][i] != y for s in reference_steps) for i, y in enumerate(target)]
    old_examples = ("7449", "9555", "3988", "10569", "5335", "3208")
    index = {r["id"]: i for i, r in enumerate(rows)}
    result = {
        "scope": "Descriptive paired analysis of reused validation predictions; no new inference, causal diagnosis, or model selection.",
        "steps": steps, "samples": len(rows),
        "curve": [{"step": s, "correct": correct[s], "accuracy": correct[s] / len(rows),
                   "macro_accuracy": sum(class_counts[s][y] / support[y] for y in support) / len(support)} for s in steps],
        "latest_pair": {"before": previous, "after": latest, "gained": gained, "lost": lost,
                        "net_gain": gained - lost, "accuracy_gain_pp": 100 * (gained - lost) / len(rows)},
        "persistent_errors": {"all_steps_wrong": sum(persistent), "reference_steps": reference_steps,
                              "reference_all_wrong": sum(earlier_persistent),
                              "reference_errors_now_correct": sum(e and p == y for e, p, y in zip(earlier_persistent, new, target))},
        "top_confusions": [{"label": y, "prediction": p, "correct_name": names[y], "predicted_name": names[p],
                            "count": n, "class_support": support[y],
                            "same_error_at_every_step": sum(t == y and all(predictions[s][i] == p for s in steps)
                                                            for i, t in enumerate(target))}
                           for (y, p), n in sorted(confusions.items(), key=lambda item: (-item[1], item[0]))[:20]],
        "class_metrics": [{"label": y, "name": names[y], "samples": support[y],
                           "previous_correct": class_counts[previous][y], "latest_correct": class_counts[latest][y],
                           "net_gain": class_counts[latest][y] - class_counts[previous][y]} for y in sorted(support)],
        "previously_reviewed_examples": [{"id": video_id, "label": target[index[video_id]],
                                          "predictions": {str(s): predictions[s][index[video_id]] for s in steps}}
                                         for video_id in old_examples],
        "prediction_sha256": hashes,
        "source_sha256": {str(path): digest(path.read_bytes()) for path in
                          (Path(__file__), run / "config.json", audit_path, manifest, labels_path, official_path)},
        "limitations": ["Repeated validation use; not an unseen test set.",
                        "Training duration changes; this is not an architecture comparison.",
                        "Confusions and class accuracy cannot identify the cause of an error.",
                        "Source and prediction hashes are checked; original videos and model inference are not rerun."],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--reference-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.audit, args.steps, args.reference_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("latest_pair", "persistent_errors")}))
