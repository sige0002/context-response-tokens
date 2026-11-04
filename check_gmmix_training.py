"""Real-video training/resume/evaluation integration for the GMMix baseline.

Uses five training and three validation clips for three updates. These are
engineering checks; their scores must not be used as recognition results.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

from benchmark_streaming import idle_compute_processes
from run_video_comparisons import bind_predecessor
from video_checkpoint import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--video-run", default="results/ssv2_full_mamba_interleaved_seed42")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    fd, paused, child = None, False, None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int

    def send(sig):
        if libc.pidfd_send_signal(fd, sig, None, 0) < 0:
            raise OSError(ctypes.get_errno(), "pidfd signal failed")

    def interrupted(signum, frame):
        raise InterruptedError(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    interpreter = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]
    common = interpreter + ["train_gmmix_video.py", "--train", "data/ssv2/video_training_fixture/train.jsonl",
        "--val", "data/ssv2/video_training_fixture/val.jsonl", "--steps", "3", "--warmup", "1",
        "--frames", "8", "--batch-size", "2", "--accumulation", "2", "--evaluate-every", "2",
        "--save-every", "1", "--workers", "0"]

    def run(command, name):
        nonlocal child
        print(json.dumps(dict(event="launch", name=name, command=command)), flush=True)
        with (output / (name + ".log")).open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}; inspect its log")

    try:
        if args.pause_pid:
            fd = bind_predecessor(args.pause_pid, Path(args.video_run).resolve())
            state = Path(f"/proc/{args.pause_pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            if state not in ("T", "t"):
                send(signal.SIGSTOP)
                paused = True
                for _ in range(100):
                    if Path(f"/proc/{args.pause_pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] in ("T", "t"):
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError("Video trainer did not pause")
            print(f"Verified paused process {args.pause_pid}", flush=True)
        idle_compute_processes()
        run(common + ["--output", str(output / "direct")], "direct")
        run(common + ["--output", str(output / "resumed"), "--stop-after-step", "2"], "resumed")
        run(common + ["--output", str(output / "resumed"), "--resume"], "resumed")
        run(common + ["--output", str(output / "repeat")], "repeat")
        saved = torch.load(output / "direct/last.pt", map_location="cpu", weights_only=True)
        report = dict(scope="Three-update real-video training/restart checks, not recognition accuracy",
            train_samples=5, val_samples=3, steps=3, frames=8, comparisons={},
            predeclared_tolerances=dict(max_weight_difference=5e-6, max_validation_loss_difference=.002))
        assert [row["step"] for row in saved["history"]] == [2, 3]
        for name in ("resumed", "repeat"):
            other = torch.load(output / name / "last.pt", map_location="cpu", weights_only=True)
            assert saved["config"] == other["config"]
            assert all(saved[key] == other[key] for key in ("step", "epoch", "cursor"))
            assert [row["step"] for row in other["history"]] == [2, 3]
            maximum = max(float((value - other["model"][key]).abs().max())
                          for key, value in saved["model"].items() if value.is_floating_point())
            loss_difference = max(abs(a["validation"]["loss"] - b["validation"]["loss"])
                                  for a, b in zip(saved["history"], other["history"]))
            result = dict(max_weight_difference=maximum, max_validation_loss_difference=loss_difference,
                          predictions_equal=all(a["validation"]["predictions"] == b["validation"]["predictions"]
                              for a, b in zip(saved["history"], other["history"])))
            report["comparisons"][name] = result
            print(json.dumps(dict(event="compare", name=name, **result)), flush=True)
            (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            assert maximum < 5e-6 and loss_difference < .002 and result["predictions_equal"], result
            assert torch.equal(saved["rng"], other["rng"])
            assert all(torch.equal(a, b) for a, b in zip(saved["cuda_rng"], other["cuda_rng"]))
            del other
        evaluation_path = output / "evaluation.json"
        run(interpreter + ["evaluate_gmmix_video.py", "--checkpoint", str(output / "direct/last.pt"),
            "--manifest", "data/ssv2/video_training_fixture/val.jsonl", "--role", "validation",
            "--batch-size", "2", "--workers", "0", "--output", str(evaluation_path)], "evaluation")
        evaluation = json.loads(evaluation_path.read_text())
        assert evaluation["checkpoint_sha256"] == file_hash(output / "direct/last.pt")
        predictions = [row["prediction"] for row in evaluation["ablations"]["full"]["modes"]["clip"]["clips"]]
        assert predictions == saved["history"][-1]["validation"]["predictions"]
        report.update(checkpoint_sha256=evaluation["checkpoint_sha256"],
            evaluation_matches_saved_validation=True, evaluation_sha256=file_hash(evaluation_path),
            clip_stream={name:row["clip_vs_stream"] for name, row in evaluation["ablations"].items()},
            scheduled_validation_preserved_across_stop=True,
            sources={name:file_hash(name) for name in
                ("check_gmmix_training.py", "train_gmmix_video.py", "gmmix_checkpoint.py", "evaluate_gmmix_video.py", "gmmix_siglip.py")})
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if paused:
            send(signal.SIGCONT)
            print(f"Resumed training process {args.pause_pid}", flush=True)
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    main()
