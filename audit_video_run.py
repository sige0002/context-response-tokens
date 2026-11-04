"""Independently recount a live SSv2 run without loading or changing its model.

Checks labels against official annotations, split membership, recorded source
hashes, and every saved validation prediction. Video content hashes are read
from the extraction manifest, not recomputed by this metadata audit.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run, annotations):
    run, annotations = Path(run), Path(annotations)
    config = json.loads((run / "config.json").read_text())
    label_map = json.loads((annotations / "labels.json").read_text())
    assert sorted(map(int, label_map.values())) == list(range(config["classes"]))
    sources = {name: sha256(name) == expected for name, expected in config["source_sha256"].items()}
    assert all(sources.values()), sources
    splits, records = {}, {}
    for split, official_name in (("train", "train"), ("val", "validation")):
        path = Path(config[split])
        assert sha256(path) == config[split + "_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        official = json.loads((annotations / f"{official_name}.json").read_text())
        expected = {r["id"]: int(label_map[r["template"].translate(str.maketrans("", "", "[]"))])
                    for r in official}
        assert len(expected) == len(official)
        assert len(rows) == len(expected)
        assert {r["id"] for r in rows} == expected.keys()
        assert all(r["label"] == expected[r["id"]] and r["group"] == r["id"] for r in rows)
        assert all(Path(r["video"]).stem == r["id"] for r in rows)
        assert all((path.parent / r["video"]).stat().st_size == r["bytes"] for r in rows)
        counts = Counter(r["label"] for r in rows)
        splits[split] = dict(samples=len(rows), classes=len(counts),
                             min_per_class=min(counts.values()), max_per_class=max(counts.values()),
                             official_labels_match=True, file_sizes_match=True,
                             manifest_sha256=sha256(path), annotation_sha256=sha256(annotations / f"{official_name}.json"))
        records[split] = rows
    overlap = {key: len({r[key] for r in records["train"]} & {r[key] for r in records["val"]})
               for key in ("id", "group", "sha256")}
    assert not any(overlap.values()), overlap
    validations = []
    for entry in json.loads((run / "history.json").read_text()):
        metrics = entry["validation"]
        predictions = metrics["predictions"]
        assert len(predictions) == metrics["samples"] == len(records["val"])
        assert all(type(p) is int and 0 <= p < config["classes"] for p in predictions)
        correct = sum(p == row["label"] for p, row in zip(predictions, records["val"]))
        accuracy = correct / len(predictions)
        assert abs(accuracy - metrics["accuracy"]) < 1e-12
        assert math.isfinite(metrics["loss"]) and metrics["loss"] >= 0
        validations.append(dict(step=entry["step"], correct=correct, samples=len(predictions),
                                accuracy=accuracy, reported_loss=metrics["loss"],
                                predictions_sha256=hashlib.sha256(json.dumps(predictions).encode()).hexdigest()))
    return dict(scope="Metadata/source verification and prediction recount; not an independent forward evaluation or a proof of no implementation bugs",
                run=str(run.resolve()), sources_match=sources, splits=splits,
                cross_split_overlap=overlap, validation=validations,
                video_hash_scope="Hashes recorded during extraction; no content reread in this audit",
                config_sha256=sha256(run / "config.json"), audit_source_sha256=sha256(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--annotations", default="data/ssv2/labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.run, args.annotations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
