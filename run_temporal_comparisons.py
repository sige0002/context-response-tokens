"""Run the fixed pilot controls sequentially after a verified training process.

Linux pidfd binds the actual predecessor process, not a reused PID or a log file.
Every control uses the same seed, data, frame count, batch size, and epoch budget.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import subprocess
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after-pid", type=int, required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--output", default="results/ssv2_pilot_comparisons")
    args = p.parse_args()
    reference = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new comparison directory")
    # Some standalone aarch64 Python builds omit os.pidfd_open, although glibc
    # and the kernel expose the same interface.
    libc = ctypes.CDLL(None, use_errno=True)
    open_pidfd = libc.pidfd_open
    open_pidfd.argtypes = [ctypes.c_int, ctypes.c_uint]
    open_pidfd.restype = ctypes.c_int
    pidfd = open_pidfd(args.after_pid, 0)
    if pidfd < 0:
        raise OSError(ctypes.get_errno(), "Cannot bind predecessor process")
    command = Path(f"/proc/{args.after_pid}/cmdline").read_bytes().split(b"\0")
    words = [s.decode() for s in command if s]
    if not any(w.endswith("train_temporal.py") for w in words) or "--output" not in words:
        raise ValueError("Predecessor is not the expected training process")
    predecessor_output = Path(words[words.index("--output") + 1]).resolve()
    if predecessor_output != reference:
        raise ValueError("Predecessor output directory differs")
    config = json.loads((reference / "config.json").read_text())
    output.mkdir(parents=True)
    status = dict(reference=str(reference), predecessor_pid=args.after_pid,
                  status="waiting_for_reference", controls=[])

    def save():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "status.json")

    save()
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    print("Bound predecessor process; waiting for its exit", flush=True)
    try:
        while not poller.poll(30000):
            print("reference process still running", flush=True)
    finally:
        os.close(pidfd)
    checkpoint = torch.load(reference / "last.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] + 1 != args.epochs or checkpoint["config"] != config:
        raise ValueError("Predecessor did not finish the specified experiment")
    del checkpoint
    for kind, placement in (("none", "interleaved"), ("attention", "interleaved"), ("mamba", "late")):
        # Prevent silent source drift between comparison runs.
        for name in ("train_temporal.py", "temporal_siglip.py", "uv.lock", "pyproject.toml"):
            if (reference / "source" / name).read_bytes() != Path(name).read_bytes():
                raise ValueError(f"Source/dependencies changed since reference: {name}")
        name = f"{kind}_{placement}_seed{config['seed']}"
        run = output / name
        cmd = ["uv", "run", "--locked", "python", "train_temporal.py",
               "--train", config["train_path"], "--val", config["val_path"],
               "--classes", str(config["classes"]), "--kind", kind, "--placement", placement,
               "--epochs", str(args.epochs), "--output", str(run)]
        for key in ("frames", "batch_size", "lr", "seed", "width", "state_size", "every"):
            cmd.extend(["--" + key.replace("_", "-"), str(config[key])])
        record = dict(name=name, command=cmd, status="running", started=time.time())
        status["controls"].append(record)
        status["status"] = "running"
        with (output / (name + ".log")).open("w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            save()
            print(f"started {name} pid={process.pid}", flush=True)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"{name} still running", flush=True)
        record.update(status="complete" if code == 0 else "failed", exit_code=code,
                      finished=time.time())
        save()
        if code:
            raise RuntimeError(f"Control failed: {name}; see its log")
    status["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
