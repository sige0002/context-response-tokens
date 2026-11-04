"""Five-way matched SSv2 comparison, including the published GMMix baseline.

Requires completed queues, their exact planned configurations and checkpoints,
full clip/stream evaluations, and both history interventions. The primary
contrast is interleaved Mamba-3 minus interleaved time attention. Other contrasts
are diagnostic; intervals are conditional on one trained seed, not SOTA proof.
"""
import argparse
import json
from pathlib import Path

from gmmix_experiment import SHARED_KEYS, verify_completion as verify_gmmix_completion
from paired_metrics import paired_mean_interval
from point_experiment import read_json, write_json
from run_video_comparisons import digest, verify_completion, verify_inputs
from summarize_video import read_run


def verify_shared_protocol(configs):
    first = configs[0]
    for config in configs[1:]:
        differences = [key for key in SHARED_KEYS if config[key] != first[key]]
        if differences:
            raise ValueError(f"Video comparison protocol differs: {differences}")
        for name in set(first["source_sha256"]) & set(config["source_sha256"]):
            if first["source_sha256"][name] != config["source_sha256"][name]:
                raise ValueError(f"Shared implementation differs: {name}")


def paired_correctness(reference, other, condition_a=("full", "stream"), condition_b=("full", "stream")):
    for key in ("ids", "labels", "groups"):
        if reference[key] != other[key]:
            raise ValueError("Video reports evaluate different examples/order/labels/groups")
    a = [int(p == y) for p, y in zip(reference["predictions"][condition_a], reference["labels"])]
    b = [int(p == y) for p, y in zip(other["predictions"][condition_b], other["labels"])]
    interval = paired_mean_interval(a, b, reference["groups"])
    return dict(interval, difference_percentage_points=100 * interval["difference"],
                conditional_95_interval_pp=[100 * value for value in interval["conditional_95_interval"]],
                a_only_correct=sum(x and not y for x, y in zip(a, b)),
                b_only_correct=sum(y and not x for x, y in zip(a, b)))


def summarize(video_queue, gmmix_queue):
    video_queue, gmmix_queue = Path(video_queue), Path(gmmix_queue)
    original, baseline = read_json(video_queue / "plan.json"), read_json(gmmix_queue / "plan.json")
    if (read_json(video_queue / "status.json")["status"] != "complete"
            or read_json(gmmix_queue / "status.json")["status"] != "complete"):
        raise ValueError("Both full video comparison queues must finish before comparative reporting")
    if (Path(baseline["video_queue"]).resolve() != video_queue.resolve()
            or baseline["video_plan_sha256"] != digest(video_queue / "plan.json")):
        raise ValueError("GMMix belongs to a different comparison")
    config = original["reference_config"]
    if config["classes"] != 174 or sum(bool(line.strip()) for line in Path(config["train"]).read_text().splitlines()) != 168913:
        raise ValueError("Full SSv2 report requires all 168913 training videos and 174 classes")
    planned = [dict(run=original["reference"], config=config)]
    planned += [dict(run=row["run"], config=dict(config, kind=row["kind"], placement=row["placement"]))
                for row in original["controls"]]
    planned.append(dict(run=baseline["run"], config=baseline["config"]))
    expected = {("mamba", "interleaved"), ("none", "interleaved"), ("attention", "interleaved"),
                ("mamba", "late"), ("gmmix", "after_encoder")}
    if len(planned) != 5 or {(r["config"]["kind"], r["config"]["placement"]) for r in planned} != expected:
        raise ValueError("Expected exactly the five planned architectures")
    verify_shared_protocol([row["config"] for row in planned])
    runs, rows = {}, []
    for row in planned:
        run, expected_config = Path(row["run"]), row["config"]
        verify_inputs(expected_config)
        completion = (verify_gmmix_completion if expected_config["kind"] == "gmmix" else verify_completion)(run, expected_config)
        values = read_run(run)
        if len(values["ids"]) != 24777:
            raise ValueError("Full SSv2 report requires all 24777 validation videos")
        if values["config"] != expected_config or values["checkpoint_sha256"] != completion["checkpoint_sha256"]:
            raise ValueError("Video evaluation differs from the completed planned checkpoint")
        name = expected_config["kind"] + "_" + expected_config["placement"]
        runs[name] = values
        starts = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()
                  if line and json.loads(line).get("event") == "start"]
        counts = {record["trainable_parameters"] for record in starts}
        if len(counts) != 1:
            raise ValueError("Missing or inconsistent training parameter count")
        rows.append(dict(name=name, path=str(run), checkpoint_sha256=completion["checkpoint_sha256"],
            evaluation_sha256=values["evaluation_sha256"], trainable_parameters=counts.pop(),
            conditions={f"{ablation}/{mode}":dict(accuracy=metric["accuracy"],
                macro_recall=metric["macro_recall_present_classes"])
                for (ablation, mode), metric in values["scores"].items()},
            clip_stream_prediction_disagreements=values["disagreements"],
            history_diagnostics={} if expected_config["kind"] == "none" else {
                ablation:paired_correctness(values, values, ("full", "clip"), (ablation, "clip"))
                for ablation in ("current", "reverse_history")}))
    reference = runs["mamba_interleaved"]
    comparisons = {name:paired_correctness(reference, values) for name, values in runs.items()
                   if name != "mamba_interleaved"}
    return dict(scope="Full SSv2 validation, one seed, fixed final optimizer step; no held-out-test or robot-success claim",
        primary_metric="Full-history streaming top-1 accuracy",
        primary_contrast="mamba_interleaved minus attention_interleaved",
        other_contrasts="Exploratory; intervals are not adjusted for multiple comparisons",
        interval_scope="Conditional on fitted weights; excludes training-seed variation, actor dependence and model-development selection",
        samples=len(reference["labels"]), steps=config["steps"], seed=config["seed"], runs=rows,
        differences_against_mamba_interleaved=comparisons,
        sign="Positive difference favors mamba_interleaved",
        unmatched="Parameter count, wall time and optional kernel optimization; GMMix uses the documented PyTorch convolution fallback",
        video_plan_sha256=digest(video_queue / "plan.json"), gmmix_plan_sha256=digest(gmmix_queue / "plan.json"),
        source_sha256={name:digest(Path(__file__).parent / name) for name in
            ("summarize_full_video.py", "summarize_video.py", "paired_metrics.py", "run_video_comparisons.py", "gmmix_experiment.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-comparisons", required=True)
    parser.add_argument("--gmmix-comparison", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new summary output")
    result = summarize(args.video_comparisons, args.gmmix_comparison)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({key:value for key, value in result.items() if key not in ("source_sha256", "runs")}), flush=True)


if __name__ == "__main__":
    main()
