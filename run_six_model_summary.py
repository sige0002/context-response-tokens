"""Wait for the final diagnostic queue and build the complete single-seed report.

CPU only. Waiting and summarizing do not start, alter or stop training jobs.
"""
import argparse
import os
from pathlib import Path
import select
import signal
import time

import torch

from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from run_point_comparisons import bind_process
from summarize_six_models import summarize_all


ROOT = Path(__file__).resolve().parent
SOURCES = ("run_six_model_summary.py", "summarize_six_models.py", "summarize_full_video.py", "summarize_video.py",
    "summarize_points.py", "paired_metrics.py", "point_experiment.py", "point_checkpoint.py", "run_point_comparisons.py",
    "run_video_comparisons.py", "gmmix_experiment.py", "multidepth_experiment.py", "run_multidepth_comparison.py",
    "run_multidepth_points.py", "multidepth_point_experiment.py", "multidepth_point_checkpoint.py",
    "run_multidepth_diagnostics.py", "encoder_costs.py", "benchmark_real_video.py", "analyze_point_history.py",
    "point_history.py", "audit_robotap_predictions.py", "robotap_data.py", "tapvid_metrics.py", "evaluate_temporal.py",
    "summarize_temporal.py", "pyproject.toml", "uv.lock", "baselines/gmmix/pyproject.toml", "baselines/gmmix/uv.lock")


def input_paths(diagnostics):
    diagnostics = Path(diagnostics).resolve()
    dp = read_json(diagnostics / "plan.json")
    multi_points = Path(dp["point_queue"])
    mp = read_json(multi_points / "plan.json")
    original_points = Path(mp["point_queue"])
    op = read_json(original_points / "plan.json")
    multi_video = Path(mp["video_queue"])
    mv = read_json(multi_video / "plan.json")
    if (dp["point_plan_sha256"] != file_hash(multi_points / "plan.json")
            or mp["point_plan_sha256"] != file_hash(original_points / "plan.json")
            or mp["video_plan_sha256"] != file_hash(multi_video / "plan.json")
            or mv["video_queue"] != op["video_queue"]):
        raise ValueError("Summary dependencies belong to different experiment chains")
    return dict(video=op["video_queue"], gmmix=op["gmmix_queue"], multidepth_video=str(multi_video),
        points=str(original_points), multidepth_points=str(multi_points), history=mv["history_queue"], diagnostics=str(diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--after-diagnostics-pid", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume-wait", action="store_true", help="Resume an untouched waiting queue only")
    args = parser.parse_args()
    os.chdir(ROOT)
    # This process uses only checkpoint/array CPU operations. Avoid even CUDA
    # enumeration by libraries while the main training chain is in progress.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    output = Path(args.output).resolve()
    paths = input_paths(args.diagnostics)
    plan = dict(paths=paths, plan_sha256={name:file_hash(Path(path) / "plan.json") for name,path in paths.items()},
        source_sha256={name:file_hash(ROOT / name) for name in SOURCES},
        primary_video="Interleaved Mamba-3 minus interleaved time attention: streaming top-1",
        primary_point="Same contrast, frozen-encoder probe AJ",
        additional="Multi-depth GMMix contrasts are exploratory; no change to the primary comparisons",
        scope="Full six-model, eleven-point-run, single-seed report; publication claim remains unproven")
    if args.resume_wait:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Summary plan/source changed")
        status = read_json(output / "status.json")
        if status["status"] not in ("planned", "waiting_for_final_diagnostics", "stopped") or (output / "report").exists():
            raise ValueError("Only a waiting queue with no started report can be resumed")
        status.pop("error", None)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        for name in SOURCES:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        status = dict(status="planned", started=time.time())
    status["dependency_pid"] = args.after_diagnostics_pid
    write_json(output / "status.json", status)
    if args.plan_only:
        print("Saved complete single-seed reporting plan; no analysis/training started", flush=True)
        return
    fd, cancelled = None, False

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True
        if status["status"] == "summarizing":
            raise InterruptedError("CPU summary interrupted; training jobs were not signalled")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.after_diagnostics_pid is not None:
            fd = bind_process(args.after_diagnostics_pid, "run_multidepth_diagnostics.py", "--output", paths["diagnostics"])
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            status["status"] = "waiting_for_final_diagnostics"
            write_json(output / "status.json", status)
            while not poller.poll(1000):
                if cancelled:
                    raise InterruptedError("Summary cancelled while waiting; dependencies unchanged")
            os.close(fd)
            fd = None
        if cancelled:
            raise InterruptedError("Summary cancelled")
        for name,digest in plan["source_sha256"].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Summary source changed: {name}")
        for name,path in paths.items():
            if file_hash(Path(path) / "plan.json") != plan["plan_sha256"][name]:
                raise ValueError(f"Summary input plan changed: {name}")
            if read_json(Path(path) / "status.json")["status"] != "complete":
                raise ValueError(f"Summary dependency is incomplete: {name}")
        status["status"] = "summarizing"
        write_json(output / "status.json", status)
        manifest = summarize_all(paths, output / "report")
        for name,digest in plan["source_sha256"].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError(f"Summary source changed during analysis: {name}")
        for name,path in paths.items():
            if file_hash(Path(path) / "plan.json") != plan["plan_sha256"][name]:
                raise ValueError(f"Summary input plan changed during analysis: {name}")
        status.update(status="complete", finished=time.time(), report=str(output / "report/manifest.json"),
                      report_sha256=file_hash(output / "report/manifest.json"), reports=manifest["reports"])
        print("Completed verified single-seed comparison; more seeds and contribution assessment remain", flush=True)
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if fd is not None:
            os.close(fd)
        write_json(output / "status.json", status)


if __name__ == "__main__":
    main()
