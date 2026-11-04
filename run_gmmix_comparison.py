"""Queue the published GMMix baseline after the four matched video experiments.

The tested baseline build is kept intact. The new run uses the reference data,
training budget, readout, and precision. Published DINOv3 scores are not reused.
"""
import argparse
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from gmmix_experiment import (ROOT, evaluation_commands, matched_config, read_json,
                              train_command, verify_completion, verify_evaluation, kernel_environment)
from point_checkpoint import point_kernel_environment
from point_experiment import write_json
from run_point_comparisons import bind_process
from run_video_comparisons import digest, verify_inputs, verify_completion as verify_temporal_completion


def build_plan(video_queue, integration, output, workers):
    original = read_json(video_queue / "plan.json")
    template = read_json(integration / "direct/config.json")
    evidence = read_json(integration / "verification.json")
    for name, expected in evidence["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"GPU-verified baseline source changed: {name}")
    template_completion = verify_completion(integration / "direct", template)
    if (template_completion["checkpoint_sha256"] != evidence["checkpoint_sha256"]
            or digest(integration / "evaluation.json") != evidence["evaluation_sha256"]
            or not evidence["evaluation_matches_saved_validation"]):
        raise ValueError("Baseline integration evidence/checkpoint changed")
    config = matched_config(original["reference_config"], template)
    run = output / f"gmmix_after_encoder_seed{config['seed']}"
    sources = tuple(dict.fromkeys((*config["source_sha256"], "gmmix_experiment.py", "run_gmmix_comparison.py",
        "evaluate_gmmix_video.py", "evaluate_video.py", "evaluate_temporal.py", "run_video_comparisons.py",
        "run_point_comparisons.py", "point_experiment.py", "point_checkpoint.py")))
    return dict(video_queue=str(video_queue), video_plan_sha256=digest(video_queue / "plan.json"),
        integration=str(integration), integration_sha256=digest(integration / "verification.json"),
        run=str(run), config=config, command=train_command(config, run, workers),
        evaluations=evaluation_commands(config, run, workers),
        source_sha256={name:digest(ROOT / name) for name in sources},
        selection="Fixed final optimizer step; full SSv2 validation, not held-out test",
        scope="Matched SigLIP2, data, updates, readout, precision. Published GMMix core unchanged. Parameters/wall time not matched; one seed.")


def verify_plan(plan):
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"GMMix experiment source changed: {name}")
    for path, expected in ((Path(plan["video_queue"]) / "plan.json", plan["video_plan_sha256"]),
                           (Path(plan["integration"]) / "verification.json", plan["integration_sha256"])):
        if digest(path) != expected:
            raise ValueError(f"GMMix input record changed: {path}")
    verify_inputs(plan["config"])
    verify_inputs(read_json(Path(plan["video_queue"]) / "plan.json")["reference_config"])
    if point_kernel_environment("gmmix") != kernel_environment(plan["config"]):
        raise ValueError("GMMix kernel/convolution path differs from verified build")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-comparisons", required=True)
    parser.add_argument("--integration", default="results/gmmix_training_integration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--after-video-pid", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("Workers must be nonnegative")
    os.chdir(ROOT)
    torch.set_num_threads(2)
    video, integration, output = [Path(p).resolve() for p in (args.video_comparisons, args.integration, args.output)]
    plan = build_plan(video, integration, output, args.workers)
    if args.resume:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Resume plan, source, or protocol changed")
        status = read_json(output / "status.json")
        status.pop("error", None)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        for name in plan["source_sha256"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        status = dict(status="planned", operations=[])
    status["dependency_pid"] = args.after_video_pid
    write_json(output / "status.json", status)
    if args.plan_only:
        print("Saved matched GMMix training/evaluation plan; no jobs started", flush=True)
        return
    child, cancelled, fd = None, False, None

    def save():
        write_json(output / "status.json", status)

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(command, name):
        nonlocal child
        if cancelled:
            raise InterruptedError("GMMix comparison cancelled")
        verify_plan(plan)
        pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(s.strip().isdigit() and Path(f"/proc/{s.strip()}").exists() for s in pids.splitlines()):
            raise RuntimeError("GPU compute process present; refusing overlapping experiment")
        row = dict(name=name, command=command, started=time.time(), status="running",
                   log=f"{len(status['operations'])+1:03d}_{name}.log")
        status["operations"].append(row)
        status["status"] = "running"
        with (output / row["log"]).open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = child.pid
            save()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"{name}: process {child.pid} is running", flush=True)
        code, child = child.returncode, None
        row.update(exit_code=code, finished=time.time(), status="stopped" if cancelled else "complete" if code == 0 else "failed")
        save()
        if cancelled:
            raise InterruptedError("GMMix comparison cancelled")
        if code:
            raise RuntimeError(f"{name} failed ({code}); see {row['log']}")

    try:
        if args.after_video_pid:
            fd = bind_process(args.after_video_pid, "run_video_comparisons.py", "--output", video)
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            status["status"] = "waiting_for_video_comparisons"
            save()
            while not poller.poll(1000):
                if cancelled:
                    raise InterruptedError("Cancelled while waiting; dependency not signalled")
            os.close(fd)
            fd = None
        verify_plan(plan)
        if read_json(video / "status.json")["status"] != "complete":
            raise ValueError("Original video comparisons did not complete normally")
        original = read_json(video / "plan.json")
        reference = original["reference_config"]
        completed = {original["reference"]:verify_temporal_completion(original["reference"], reference)}
        for row in original["controls"]:
            config = dict(reference, kind=row["kind"], placement=row["placement"])
            completed[row["run"]] = verify_temporal_completion(row["run"], config)
        status["reference_completion"] = completed
        run = Path(plan["run"])
        resume = (run / "last.pt").exists()
        progress = []
        if resume:
            if read_json(run / "config.json") != plan["config"]:
                raise ValueError("Existing GMMix run differs from plan")
            import json
            progress = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line]
        if not progress or progress[-1].get("event") != "budget_complete":
            launch(plan["command"] + (["--resume"] if resume else []), "train_gmmix")
        status["completion"] = verify_completion(run, plan["config"])
        save()
        for i, command in enumerate(plan["evaluations"]):
            if digest(run / "last.pt") != status["completion"]["checkpoint_sha256"]:
                raise ValueError("Completed baseline checkpoint changed")
            destination = Path(command[command.index("--output") + 1])
            if not destination.exists():
                launch(command, f"evaluate_{i}")
            ablations, modes = (["full"], ["clip", "stream"]) if i == 0 else (["current", "reverse_history"], ["clip"])
            checked = verify_evaluation(destination, plan["config"], status["completion"], ablations, modes)
            status.setdefault("evaluations", {})[str(i)] = checked
            save()
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if fd is not None:
            os.close(fd)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save()


if __name__ == "__main__":
    main()
