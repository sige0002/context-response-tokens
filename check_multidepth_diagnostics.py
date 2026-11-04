"""Engineering checks of multi-depth cost and conditional history CLIs.

Uses three-update fixtures, never a final performance benchmark.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess

from encoder_costs import verify_measurement
from pause_training import isolated_gpu
from point_checkpoint import file_hash
from point_experiment import read_json, write_json


UV = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-run", default="results/gmmix_multidepth_qualification/direct")
    parser.add_argument("--point-integration", default="results/multidepth_point_integration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--pause-reference")
    args = parser.parse_args()
    output, video_run, point = [Path(p).resolve() for p in (args.output, args.video_run, args.point_integration)]
    output.mkdir(parents=True, exist_ok=False)
    costs = read_json("results/encoder_costs_seed42_rgbfix_v2/plan.json")
    video = costs["videos"][0]
    assert video["selection"] == "shortest" and video["frames"] == 52
    sources = ["check_multidepth_diagnostics.py", "benchmark_multidepth_video.py", "benchmark_real_video.py",
        "evaluate_multidepth_history.py", "evaluate_point_history.py", "analyze_point_history.py", "point_history.py",
        "multidepth_point_checkpoint.py", "multidepth_checkpoint.py", "gmmix_multidepth.py", "encoder_costs.py"]
    plan = dict(scope="Engineering diagnostics; three-update fixtures, not performance evidence",
        sources={name:file_hash(name) for name in sources}, cost_video=video,
        video_checkpoint_sha256=file_hash(video_run / "last.pt"),
        point_checkpoint_sha256=file_hash(point / "direct/last.pt"),
        ordinary_report_sha256=file_hash(point / "robotap/summary.json"),
        history_periods=[1,24], cost_repeats=2, cost_warmup=1)
    write_json(output / "plan.json", plan)
    child = None

    def run(label, command):
        nonlocal child
        print(f"Running {label}", flush=True)
        with (output / (label+".log")).open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"{label} failed ({code}); inspect log")

    with isolated_gpu(args.pause_pid, args.pause_reference):
        try:
            run("cost", UV + ["benchmark_multidepth_video.py", "--checkpoint", str(video_run / "last.pt"),
                "--video-id", video["id"], "--repeats", "2", "--output", str(output / "cost")])
            run("history", UV + ["evaluate_multidepth_history.py", "--checkpoint", str(point / "direct/last.pt"),
                "--output", str(output / "history"), "--limit", "1"])
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    run("history_analysis", UV + ["analyze_point_history.py", "--interventions", str(output / "history"),
        "--baseline", str(point / "robotap/summary.json"), "--output", str(output / "history/analysis.json"), "--allow-subset"])
    measurement = verify_measurement(output / "cost", read_json(video_run / "config.json"),
        dict(checkpoint_sha256=plan["video_checkpoint_sha256"], step=3), video, 2, 1, costs["extraction_sha256"])
    if (not measurement["repeated_final_outputs_equal"] or measurement["max_state_bytes"] != 96337920
            or any(row["state_bytes_min"] != 96337920 or row["state_bytes_max"] != 96337920 for row in measurement["bins"])):
        raise ValueError("Native repeat/state accounting differs")
    analysis = read_json(output / "history/analysis.json")
    if analysis["scope"] != "Engineering subset" or analysis["videos"] != 1:
        raise ValueError("Expected the explicitly limited raw-label history audit")
    assert all(file_hash(name) == digest for name,digest in plan["sources"].items())
    write_json(output / "verification.json", dict(**plan, status="passed_scoped_checks",
        cost=measurement, history_analysis_sha256=file_hash(output / "history/analysis.json"),
        history_scope=analysis["scope"], history_videos=analysis["videos"],
        history_analysis=str(output / "history/analysis.json")))
    print("Cost replay and raw-label history checks passed; no accuracy claim", flush=True)


if __name__ == "__main__":
    main()
