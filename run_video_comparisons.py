"""Run full-data temporal controls with a fixed reference training protocol.

The reference must finish its entire budget and scheduled validation. A Linux
pidfd binds a running predecessor; neither stale logs nor reused PIDs count as
proof of exit. Source/data drift and partial control runs stop the queue.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parent
TRAIN_KEYS = ("train", "val", "classes", "readout", "frames", "batch_size", "accumulation",
              "steps", "warmup", "lr", "weight_decay", "evaluate_every", "save_every", "seed")
CONTROLS = (("none", "interleaved"), ("attention", "interleaved"), ("mamba", "late"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(config, root=ROOT):
    for name, expected in config["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Training source/dependencies changed: {name}")
    for name in ("train", "val"):
        if digest(config[name]) != config[name + "_sha256"]:
            raise ValueError(f"Dataset manifest changed: {name}")
    if digest(root / "vendor/frozen-video/models/rvm_modules/rvm_blocks.py") != config["readout_sha256"]:
        raise ValueError("Reference readout source changed")


def assert_comparable(actual, reference, kind, placement):
    expected = dict(reference, kind=kind, placement=placement)
    if actual != expected:
        differences = [k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)]
        raise ValueError(f"Comparison configuration differs: {sorted(differences)}")


def control_command(config, kind, placement, output, workers):
    command = ["uv", "run", "--locked", "python", "train_video.py", "--output", str(output),
               "--kind", kind, "--placement", placement, "--workers", str(workers)]
    for key in TRAIN_KEYS:
        command.extend(["--" + key.replace("_", "-"), str(config[key])])
    return command


def verify_completion(run, expected):
    run = Path(run)
    actual = json.loads((run / "config.json").read_text())
    assert_comparable(actual, expected, expected["kind"], expected["placement"])
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    if (saved.get("format_version") != 2 or saved["config"] != expected
            or saved["step"] != expected["steps"]):
        raise ValueError("Run did not finish the exact optimizer budget")
    schedule = list(range(expected["evaluate_every"], expected["steps"] + 1, expected["evaluate_every"]))
    if not schedule or schedule[-1] != expected["steps"]:
        schedule.append(expected["steps"])
    if [row["step"] for row in saved["history"]] != schedule:
        raise ValueError("Scheduled full validation is incomplete")
    labels = [json.loads(line)["label"] for line in Path(expected["val"]).read_text().splitlines() if line.strip()]
    count = len(labels)
    for row in saved["history"]:
        metrics = row["validation"]
        if (metrics["samples"] != count or len(metrics["predictions"]) != count
                or not math.isfinite(metrics["loss"]) or not 0 <= metrics["accuracy"] <= 1):
            raise ValueError("Invalid or partial validation result")
        recounted = sum(a == b for a, b in zip(labels, metrics["predictions"])) / count
        if abs(recounted - metrics["accuracy"]) > 1e-9:
            raise ValueError("Validation accuracy differs from saved predictions")
    records = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
    if not records or records[-1]["event"] != "budget_complete" or records[-1]["step"] != expected["steps"]:
        raise ValueError("Trainer did not record normal budget completion")
    return dict(step=saved["step"], validation=saved["history"][-1]["validation"]["accuracy"],
                checkpoint_sha256=digest(run / "last.pt"))


def bind_predecessor(pid, reference):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Cannot bind predecessor process")
    try:
        words = [word.decode() for word in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if word]
        if not any(word.endswith("train_video.py") for word in words) or "--output" not in words:
            raise ValueError("Predecessor is not train_video.py")
        directory = Path(f"/proc/{pid}/cwd").resolve()
        if (directory / words[words.index("--output") + 1]).resolve() != reference:
            raise ValueError("Predecessor writes to a different reference directory")
    except BaseException:
        os.close(fd)
        raise
    return fd


def evaluation_commands(run, config, workers):
    shared = ["uv", "run", "--locked", "python", "evaluate_video.py", "--checkpoint", str(run / "last.pt"),
              "--manifest", config["val"], "--role", "validation", "--workers", str(workers),
              "--batch-size", str(config["batch_size"])]
    commands = [shared + ["--output", str(run / "final_clip_stream.json"), "--ablations", "full"]]
    if config["kind"] != "none":
        commands.append(shared + ["--output", str(run / "final_history_ablations.json"),
                                 "--modes", "clip", "--ablations", "current", "reverse_history"])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("Workers must be nonnegative")
    os.chdir(ROOT)
    reference, output = Path(args.reference).resolve(), Path(args.output).resolve()
    config = json.loads((reference / "config.json").read_text())
    if (config["kind"], config["placement"]) != ("mamba", "interleaved"):
        raise ValueError("Reference must be interleaved Mamba")
    verify_inputs(config)
    plans = []
    for kind, placement in CONTROLS:
        run = output / f"{kind}_{placement}_seed{config['seed']}"
        plans.append(dict(kind=kind, placement=placement, run=str(run),
                          command=control_command(config, kind, placement, run, args.workers)))
    output.mkdir(parents=True, exist_ok=False)
    evaluation_sources = {name: digest(ROOT / name) for name in
                          ("evaluate_video.py", "video_checkpoint.py", "evaluate_temporal.py", "run_video_comparisons.py")}
    plan = dict(reference=str(reference), reference_config=config, controls=plans,
                evaluation_sources=evaluation_sources,
                selection="Fixed final optimizer step for comparison; validation, not held-out test",
                scope="One seed, matched data/updates/readout. Parameters and wall time are not matched.")
    plan["evaluations"] = evaluation_commands(reference, config, args.workers)
    for row in plans:
        plan["evaluations"].extend(evaluation_commands(Path(row["run"]),
              dict(config, kind=row["kind"], placement=row["placement"]), args.workers))
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    source_dir = output / "source"
    source_dir.mkdir()
    for name in evaluation_sources:
        (source_dir / name).write_bytes((ROOT / name).read_bytes())
    if args.plan_only:
        print(json.dumps(plan, indent=2), flush=True)
        return
    status = dict(status="starting", predecessor_pid=args.after_pid, operations=[])
    cancelled, process = False, None

    def save():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(command, name):
        nonlocal process
        if cancelled:
            raise InterruptedError("Queue cancelled")
        verify_inputs(config)
        for source, expected in evaluation_sources.items():
            if digest(ROOT / source) != expected:
                raise ValueError(f"Evaluation/queue source changed: {source}")
        gpu_pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(line.strip().isdigit() and Path(f"/proc/{line.strip()}").exists()
               for line in gpu_pids.splitlines()):
            raise RuntimeError("GPU compute process still present; refusing concurrent experiment")
        row = dict(name=name, command=command, status="running", started=time.time())
        status["operations"].append(row)
        status["status"] = "running"
        with (output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = process.pid
            save()
            print(json.dumps(row), flush=True)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"{name}: child process {process.pid} is running", flush=True)
        row.update(exit_code=process.returncode, finished=time.time(),
                   status="stopped" if cancelled else "complete" if process.returncode == 0 else "failed")
        save()
        if cancelled:
            raise InterruptedError("Queue cancelled; current child was signalled")
        if process.returncode:
            raise RuntimeError(f"{name} exited with {process.returncode}; see log")
        process = None

    try:
        if args.after_pid is not None:
            fd = bind_predecessor(args.after_pid, reference)
            status["status"] = "waiting_for_reference"
            save()
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            try:
                while not poller.poll(1000):
                    if cancelled:
                        raise InterruptedError("Queue cancelled while waiting")
            finally:
                os.close(fd)
        if cancelled:
            raise InterruptedError("Queue cancelled")
        status["reference_completion"] = verify_completion(reference, config)
        save()
        runs = [(reference, config, status["reference_completion"])]
        for row in plans:
            launch(row["command"], Path(row["run"]).name)
            expected = dict(config, kind=row["kind"], placement=row["placement"])
            completed = verify_completion(row["run"], expected)
            status["operations"][-1]["verified_completion"] = completed
            save()
            runs.append((Path(row["run"]), expected, completed))
        for run, run_config, completed in runs:
            for i, command in enumerate(evaluation_commands(run, run_config, args.workers)):
                if digest(run / "last.pt") != completed["checkpoint_sha256"]:
                    raise ValueError(f"Final checkpoint changed since completion: {run}")
                launch(command, f"evaluate_{run.name}_{i}")
                report = json.loads(Path(command[command.index("--output") + 1]).read_text())
                if report["step"] != run_config["steps"] or report["checkpoint_sha256"] != completed["checkpoint_sha256"]:
                    raise ValueError("Evaluation loaded a different checkpoint")
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
