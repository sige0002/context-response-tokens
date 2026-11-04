"""Aggregate the nine completed point experiments with paired video intervals.

Probe and adaptation remain separate. The official mean includes all annotated
videos, while identical RGB content is one bootstrap cluster. Length strata
describe different video subsets and must not be interpreted as a controlled
memory-retention experiment. No engineering subsets are accepted.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from paired_metrics import paired_mean_interval
from point_checkpoint import file_hash
from point_experiment import assert_training_config, read_json, verify_robotap, write_json
from run_point_comparisons import verify_plan_inputs


METRICS = ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")
LENGTH_BINS = ((1, 128), (129, 256), (257, 512), (513, None))


def align_reports(reference, other):
    def indexed(report):
        rows = report["rows"]
        index = {row["id"]:row for row in rows}
        if len(index) != len(rows):
            raise ValueError("Repeated point-evaluation video IDs")
        return index
    a, b = indexed(reference), indexed(other)
    if not a or set(a) != set(b):
        raise ValueError("Point reports do not evaluate the same annotated videos")
    ids = sorted(a)
    for name in ids:
        for key in ("video_sha256", "frames", "tracks"):
            if a[name][key] != b[name][key]:
                raise ValueError(f"Point report example differs: {key}")
    return [a[name] for name in ids], [b[name] for name in ids]


def compare_reports(reference, other):
    a, b = align_reports(reference, other)
    result = {}
    for metric in METRICS:
        interval = paired_mean_interval([row["metrics"][metric] for row in a],
            [row["metrics"][metric] for row in b], [row["video_sha256"] for row in a])
        result[metric] = dict(interval, difference_percentage_points=100 * interval["difference"],
                             conditional_95_interval_pp=[100 * x for x in interval["conditional_95_interval"]])
    return result


def length_description(report):
    groups = []
    for low, high in LENGTH_BINS:
        rows = [row for row in report["rows"] if row["frames"] >= low and (high is None or row["frames"] <= high)]
        if not rows:
            groups.append(dict(min_frames=low, max_frames=high, videos=0))
            continue
        def memory(key):
            sizes = np.asarray([row[key] for row in rows])
            return dict(median_bytes=float(np.median(sizes)), max_bytes=int(sizes.max()))
        groups.append(dict(min_frames=low, max_frames=high, videos=len(rows),
            unique_rgb_videos=len({row["video_sha256"] for row in rows}),
            metrics={key:float(np.mean([row["metrics"][key] for row in rows])) for key in METRICS},
            max_per_video_encoder_state=memory("max_encoder_state_bytes"),
            max_per_video_query_state=memory("max_query_state_bytes")))
    return groups


def read_completed(queue):
    queue = Path(queue)
    plan, status = read_json(queue / "plan.json"), read_json(queue / "status.json")
    if status["status"] != "complete":
        raise ValueError("Full point experiment queue is not complete")
    if len(plan["runs"]) != 9 or set(status["results"]) != {row["name"] for row in plan["runs"]}:
        raise ValueError("Expected all five probes and four temporal adaptations")
    verify_plan_inputs(plan)
    inputs = read_json(queue / "inputs.json")
    reports, details = {}, {}
    for row in plan["runs"]:
        run = Path(row["run"])
        completed = status["results"][row["name"]]
        config = read_json(run / "config.json")
        assert_training_config(config, row, plan, inputs["dataset"], inputs["encoders"][row["encoder_run"]])
        if (completed["training"]["step"] != plan["protocol"]["steps"]
                or file_hash(run / "last.pt") != completed["training"]["checkpoint_sha256"]):
            raise ValueError("Completed point checkpoint changed or is partial")
        evaluation = Path(completed["evaluation"]["report"])
        if verify_robotap(evaluation.parent, completed["training"], plan["robotap_extraction_sha256"]) != completed["evaluation"]:
            raise ValueError("Point evaluation differs from completed queue record")
        report = read_json(evaluation)
        if report["checkpoint"]["metadata"]["config"] != config:
            raise ValueError("Point evaluation used a different training configuration")
        audit_record = completed["raw_label_audit"]
        audit_path = Path(audit_record["report"])
        if file_hash(audit_path) != audit_record["report_sha256"]:
            raise ValueError("Point raw-label audit changed")
        audit = read_json(audit_path)
        if (audit["scope"] != "Full RoboTAP raw-label metric audit" or audit["videos"] != 265
                or audit["report_sha256"] != completed["evaluation"]["report_sha256"]
                or audit["dataset_extraction_sha256"] != plan["robotap_extraction_sha256"]
                or audit["metrics"] != report["metrics"]):
            raise ValueError("Raw-label audit is partial or belongs to a different report")
        audit_rows = {value["id"]:value for value in audit["rows"]}
        if len(audit_rows) != 265 or set(audit_rows) != {value["id"] for value in report["rows"]}:
            raise ValueError("Raw-label audit does not cover every video")
        for value in report["rows"]:
            verified = audit_rows[value["id"]]
            if any(verified[key] != value[key] for key in ("video_sha256", "predictions_sha256", "metrics")):
                raise ValueError("Raw-label audit differs from per-video result")
        starts = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()
                  if line and json.loads(line).get("event") == "start"]
        counts = {start["trainable_parameters"] for start in starts}
        if len(counts) != 1:
            raise ValueError("Missing or inconsistent point-training parameter count")
        reports[row["name"]] = report
        details[row["name"]] = dict(mode=row["mode"], kind=row["encoder_config"]["kind"],
            placement=row["encoder_config"]["placement"], trainable_parameters=counts.pop(),
            report=str(evaluation), report_sha256=file_hash(evaluation),
            checkpoint_sha256=completed["training"]["checkpoint_sha256"],
            raw_label_audit=audit_record, metrics={key:report["metrics"][key] for key in METRICS},
            video_length_strata=length_description(report))
    return plan, reports, details


def summarize(queue):
    plan, reports, details = read_completed(queue)
    reused = {}
    for alias in plan["reused_controls"]:
        if alias["reference"] not in reports or alias["name"] in reports:
            raise ValueError("Invalid reused no-temporal-adapter control")
        reports[alias["name"]] = reports[alias["reference"]]
        reused[alias["name"]] = alias
    seed = plan["protocol"]["seed"]
    contrasts = {}
    for mode in ("probe", "adapt"):
        reference_name = f"mamba_interleaved_{mode}_seed{seed}"
        contrast = {}
        for kind, placement in (("attention", "interleaved"), ("none", "interleaved"),
                                ("mamba", "late"), ("gmmix", "after_encoder")):
            other = f"{kind}_{placement}_{mode}_seed{seed}"
            contrast[other] = compare_reports(reports[reference_name], reports[other])
        contrasts[mode] = dict(reference=reference_name, comparisons=contrast)
    return dict(scope="Full RoboTAP, one seed; fixed-final-step checkpoints; probe and adaptation kept separate",
        primary_metric="average_jaccard", primary_contrast="Probe: interleaved Mamba-3 minus interleaved temporal attention",
        sign="Positive difference favors interleaved Mamba-3",
        aggregation="Mean over all 265 annotated videos; identical RGB content shares a bootstrap cluster",
        interval_scope="Conditional on fitted models. Other metrics/contrasts and length strata are exploratory; no multiplicity correction or seed uncertainty.",
        video_length_scope="Different video subsets; not a controlled test of retention or time since query arrival",
        memory_scope="Recorded state tensors only; encoder and per-query state separate. Excludes weights, activations, workspace and total GPU allocation.",
        runs=details, reused_controls=reused, contrasts=contrasts,
        plan_sha256=file_hash(Path(queue) / "plan.json"), status_sha256=file_hash(Path(queue) / "status.json"),
        source_sha256={name:file_hash(Path(__file__).parent / name) for name in
            ("summarize_points.py", "paired_metrics.py", "point_experiment.py", "run_point_comparisons.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-comparisons", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new point summary output")
    report = summarize(args.point_comparisons)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(dict(output=str(output), runs=len(report["runs"]), scope=report["scope"])), flush=True)


if __name__ == "__main__":
    main()
