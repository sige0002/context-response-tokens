"""Queue matched point-probe and adapter-adaptation experiments on one GPU.

Wait for the exact video-comparison and dataset-preparation processes, then
require their complete artifacts. Train on synthetic scenes only, and evaluate
the fixed final checkpoints on the full external RoboTAP benchmark. No choice
of checkpoint, hyperparameters, or mode is based on RoboTAP scores.
"""
import argparse
import ctypes
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from point_checkpoint import file_hash
from point_experiment import (PROTOCOL, assert_training_config, read_json, train_command,
    verify_dataset, verify_point_run, verify_robotap, write_json)
from run_video_comparisons import verify_completion, verify_inputs
from train_points import SOURCES
from gmmix_experiment import UV, verify_completion as verify_gmmix_completion


ROOT = Path(__file__).resolve().parent
QUEUE_SOURCES = tuple(dict.fromkeys((*SOURCES, "point_experiment.py", "run_point_comparisons.py",
    "evaluate_point_tracks.py", "robotap_data.py", "run_video_comparisons.py", "prepare_kubric_points.py",
    "gmmix_experiment.py", "run_gmmix_comparison.py", "audit_robotap_predictions.py")))


def build_plan(video_queue, data, robotap, output, modes, workers, gmmix_queue=None):
    video_plan = read_json(video_queue / "plan.json")
    config = video_plan["reference_config"]
    verify_inputs(config)
    encoders = [dict(run=video_plan["reference"], config=config)]
    encoders += [dict(run=row["run"], config=dict(config, kind=row["kind"], placement=row["placement"]))
                 for row in video_plan["controls"]]
    expected = {("mamba", "interleaved"), ("none", "interleaved"),
                ("attention", "interleaved"), ("mamba", "late")}
    if len(encoders) != 4 or {(r["config"]["kind"], r["config"]["placement"]) for r in encoders} != expected:
        raise ValueError("Expected all four matched video encoders")
    if gmmix_queue is not None:
        gp = read_json(gmmix_queue / "plan.json")
        if Path(gp["video_queue"]) != video_queue or gp["video_plan_sha256"] != file_hash(video_queue / "plan.json"):
            raise ValueError("GMMix baseline uses a different video comparison")
        verify_inputs(gp["config"])
        encoders.append(dict(run=gp["run"], config=gp["config"]))
    rows, reused = [], []
    for mode in modes:
        for encoder in encoders:
            ec = encoder["config"]
            name = f"{ec['kind']}_{ec['placement']}_{mode}_seed{PROTOCOL['seed']}"
            if mode == "adapt" and ec["kind"] == "none" and "probe" in modes:
                reused.append(dict(name=name, reference=f"none_interleaved_probe_seed{PROTOCOL['seed']}",
                    reason="No temporal adapters exist; spatial backbone is frozen in both modes. Same trainable head and protocol."))
                continue
            run = output / name
            encoder_type = "gmmix" if ec["kind"] == "gmmix" else "temporal"
            rows.append(dict(name=name, run=str(run), mode=mode, encoder_run=encoder["run"],
                encoder_type=encoder_type, encoder_config=ec,
                command=train_command(PROTOCOL, encoder["run"], data, run, mode, workers, encoder_type)))
    return dict(video_queue=str(video_queue), video_plan_sha256=file_hash(video_queue / "plan.json"),
        gmmix_queue=str(gmmix_queue) if gmmix_queue is not None else None,
        gmmix_plan_sha256=file_hash(gmmix_queue / "plan.json") if gmmix_queue is not None else None,
        data_root=str(data), data_plan=read_json(data / "plan.json"), data_plan_sha256=file_hash(data / "plan.json"),
        robotap_root=str(robotap), robotap_extraction_sha256=file_hash(robotap / "extraction.json"),
        protocol=PROTOCOL, modes=modes, workers=workers, encoders=encoders, runs=rows, reused_controls=reused,
        source_sha256={name:file_hash(ROOT / name) for name in QUEUE_SOURCES},
        selection="Fixed final optimizer step, full synthetic split, external RoboTAP used only for final evaluation",
        scope="One seed; probe and adaptation reported separately. All point runs use the same isolated uv environment. Parameters and elapsed time are not matched.")


def verify_plan_inputs(plan):
    for name, expected in plan["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Experiment source/dependency changed: {name}")
    for path, expected in ((Path(plan["video_queue"]) / "plan.json", plan["video_plan_sha256"]),
                           (Path(plan["data_root"]) / "plan.json", plan["data_plan_sha256"]),
                           (Path(plan["robotap_root"]) / "extraction.json", plan["robotap_extraction_sha256"])):
        if file_hash(path) != expected:
            raise ValueError(f"Experiment input record changed: {path}")
    for encoder in plan["encoders"]:
        verify_inputs(encoder["config"])
    if plan["gmmix_queue"] is not None:
        if file_hash(Path(plan["gmmix_queue"]) / "plan.json") != plan["gmmix_plan_sha256"]:
            raise ValueError("GMMix comparison plan changed")


def bind_process(pid, script, flag, directory, default=None):
    """Bind an exact process before inspecting its command; avoid PID reuse."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes, libc.pidfd_open.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Cannot bind dependency process")
    try:
        words = [s.decode() for s in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if s]
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        if not any(Path(s).name == script for s in words):
            raise ValueError(f"Dependency is not {script}")
        value = words[words.index(flag) + 1] if flag in words else default
        if value is None or (cwd / value).resolve() != Path(directory).resolve():
            raise ValueError("Dependency uses a different output/data directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-comparisons", required=True)
    parser.add_argument("--gmmix-comparison")
    parser.add_argument("--data-root", default="data/cotracker3_kubric")
    parser.add_argument("--robotap-root", default="data/robotap")
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", nargs="+", choices=("probe", "adapt"), default=["probe", "adapt"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--after-video-pid", type=int)
    parser.add_argument("--after-data-pid", type=int)
    parser.add_argument("--after-gmmix-pid", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or len(set(args.modes)) != len(args.modes):
        parser.error("Invalid workers or repeated modes")
    if args.after_gmmix_pid is not None and args.gmmix_comparison is None:
        parser.error("--after-gmmix-pid requires --gmmix-comparison")
    os.chdir(ROOT)
    torch.set_num_threads(2)
    video, data, robotap, output = [Path(value).resolve() for value in
                                  (args.video_comparisons, args.data_root, args.robotap_root, args.output)]
    gmmix = Path(args.gmmix_comparison).resolve() if args.gmmix_comparison else None
    plan = build_plan(video, data, robotap, output, args.modes, args.workers, gmmix)
    if args.resume:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Resume plan/source/configuration changed")
        status = read_json(output / "status.json")
        status.pop("error", None)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        snapshot = output / "source"
        snapshot.mkdir()
        for name in plan["source_sha256"]:
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_bytes((ROOT / name).read_bytes())
        status = dict(status="planned", operations=[], results={})
    status["dependency_pids"] = dict(video=args.after_video_pid, data=args.after_data_pid, gmmix=args.after_gmmix_pid)
    write_json(output / "status.json", status)
    if args.plan_only:
        print(f"Saved plan for {len(plan['runs'])} point experiments; no jobs started", flush=True)
        return
    child, cancelled, descriptors = None, False, {}

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

    def check_inputs():
        if cancelled:
            raise InterruptedError("Point experiment queue cancelled")
        verify_plan_inputs(plan)
        for name in ("train", "val"):
            filename = "train.jsonl" if name == "train" else "validation.jsonl"
            if file_hash(data / filename) != dataset[name + "_sha256"]:
                raise ValueError("Synthetic manifest changed after completion")
        if file_hash(data / "complete.json") != dataset["complete_sha256"]:
            raise ValueError("Synthetic completion record changed")

    def launch(command, name):
        nonlocal child
        check_inputs()
        pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(line.strip().isdigit() and Path(f"/proc/{line.strip()}").exists() for line in pids.splitlines()):
            raise RuntimeError("GPU compute process present; refusing overlapping experiments")
        attempt = len(status["operations"]) + 1
        row = dict(name=name, command=command, started=time.time(), status="running",
                   log=f"{attempt:03d}_{name}.log")
        status["operations"].append(row)
        status["status"] = "running"
        with (output / row["log"]).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = child.pid
            save()
            print(f"Started {name}: process {child.pid}", flush=True)
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"{name}: process {child.pid} is running", flush=True)
        row.update(exit_code=child.returncode, finished=time.time(),
                   status="stopped" if cancelled else "complete" if child.returncode == 0 else "failed")
        code, child = child.returncode, None
        save()
        if cancelled:
            raise InterruptedError("Queue cancelled; active child was signalled")
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}; see {row['log']}")

    try:
        for pid, script, flag, directory, default in (
            (args.after_video_pid, "run_video_comparisons.py", "--output", video, None),
            (args.after_gmmix_pid, "run_gmmix_comparison.py", "--output", gmmix, None),
            (args.after_data_pid, "prepare_kubric_points.py", "--root", data, "data/cotracker3_kubric")):
            if pid is not None:
                descriptors[bind_process(pid, script, flag, directory, default)] = dict(pid=pid, script=script)
        poller = select.poll()
        for fd in descriptors:
            poller.register(fd, select.POLLIN)
        status["status"] = "waiting_for_dependencies"
        save()
        while descriptors:
            if cancelled:
                raise InterruptedError("Cancelled while waiting; dependency jobs were not signalled")
            for fd, _ in poller.poll(1000):
                print(f"Dependency exited: {descriptors[fd]}", flush=True)
                poller.unregister(fd)
                os.close(fd)
                del descriptors[fd]
        verify_plan_inputs(plan)
        if read_json(video / "status.json")["status"] != "complete":
            raise ValueError("Video comparison queue did not complete normally")
        if gmmix is not None and read_json(gmmix / "status.json")["status"] != "complete":
            raise ValueError("GMMix comparison queue did not complete normally")
        encoders = {row["run"]:(verify_gmmix_completion if row["config"]["kind"] == "gmmix" else verify_completion)
                    (row["run"], row["config"]) for row in plan["encoders"]}
        dataset = verify_dataset(data, plan["data_plan"])
        inputs = dict(dataset=dataset, encoders=encoders)
        if (output / "inputs.json").exists() and read_json(output / "inputs.json") != inputs:
            raise ValueError("Completed dependencies differ from the previous attempt")
        write_json(output / "inputs.json", inputs)
        canonical = None
        for row in plan["runs"]:
            check_inputs()
            run = Path(row["run"])
            encoder = encoders[row["encoder_run"]]
            if file_hash(Path(row["encoder_run"]) / "last.pt") != encoder["checkpoint_sha256"]:
                raise ValueError("Completed encoder checkpoint changed")
            resume = (run / "last.pt").exists()
            complete = False
            if resume:
                assert_training_config(read_json(run / "config.json"), row, plan, dataset, encoder)
                progress = [line for line in (run / "progress.jsonl").read_text().splitlines() if line]
                import json
                complete = bool(progress and json.loads(progress[-1]).get("event") == "budget_complete")
            if not complete:
                launch(row["command"] + (["--resume"] if resume else []), row["name"])
            result = verify_point_run(row, plan, dataset, encoder)
            config = read_json(run / "config.json")
            comparable = {key:value for key, value in config.items()
                          if key not in ("encoder_checkpoint", "encoder_initialization", "mode", "encoder_type", "kernel_environment")}
            if canonical is not None and comparable != canonical:
                raise ValueError("Point runs differ beyond encoder initialization and probe/adapt mode")
            canonical = comparable
            if row["name"] in status["results"]:
                existing = status["results"][row["name"]]
                if existing["training"] != result:
                    raise ValueError("Previously completed point checkpoint changed")
                evaluation = verify_robotap(Path(existing["evaluation"]["report"]).parent, result,
                                            plan["robotap_extraction_sha256"])
            else:
                # Retain interrupted evaluation artifacts; never mix their rows
                # into a resumed full pass or overwrite them silently.
                index = 1
                while (run / f"robotap_{index:02d}").exists():
                    index += 1
                evaluation_dir = run / f"robotap_{index:02d}"
                launch(UV + ["evaluate_point_tracks.py",
                    "--checkpoint", str(run / "last.pt"), "--root", str(robotap),
                    "--output", str(evaluation_dir)], "evaluate_" + row["name"])
                evaluation = verify_robotap(evaluation_dir, result, plan["robotap_extraction_sha256"])
            evaluation_dir = Path(evaluation["report"]).parent
            audit_path = evaluation_dir / "raw_label_metric_audit.json"
            if not audit_path.exists():
                launch(UV + ["audit_robotap_predictions.py", "--report", evaluation["report"],
                    "--root", str(robotap), "--output", str(audit_path)], "audit_" + row["name"])
            audit = read_json(audit_path)
            if (audit["scope"] != "Full RoboTAP raw-label metric audit" or audit["videos"] != 265
                    or audit["report_sha256"] != evaluation["report_sha256"]
                    or audit["dataset_extraction_sha256"] != plan["robotap_extraction_sha256"]
                    or audit["metrics"] != evaluation["metrics"]
                    or any(plan["source_sha256"].get(name) != value for name, value in audit["source_sha256"].items())):
                raise ValueError("RoboTAP raw-label audit is partial or differs from the evaluation")
            status["results"][row["name"]] = dict(training=result, evaluation=evaluation,
                raw_label_audit=dict(report=str(audit_path), report_sha256=file_hash(audit_path)))
            save()
        status["status"] = "complete"
        write_json(output / "results.json", dict(scope=plan["scope"], selection=plan["selection"],
            results=status["results"], reused_controls=plan["reused_controls"],
            remaining="Paired uncertainty estimates, more seeds, long-history ablations, and publication assessment"))
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for fd in descriptors:
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
