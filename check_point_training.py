"""Exercise real-GPU point training/restart with disjoint public scene fixtures.

No scores from this tiny fixture are recognition-performance evidence. Optionally
pause one verified video-training process while these GPU checks run, and always
restore its prior running state afterwards.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from run_video_comparisons import bind_predecessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-run", required=True)
    parser.add_argument("--encoder-type", choices=("temporal", "gmmix"), default="temporal")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--pause-reference", help="Verified running video output, if different from encoder-run")
    args = parser.parse_args()
    reference, output = Path(args.encoder_run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose an empty integration directory")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    fd, paused, child = None, False, None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int

    def send(sig):
        if libc.pidfd_send_signal(fd, sig, None, 0) < 0:
            raise OSError(ctypes.get_errno(), "pidfd signal failed")

    def interrupted(signum, frame):
        raise InterruptedError(f"Integration interrupted by signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    shared = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python", "train_points.py",
              "--encoder-type", args.encoder_type,
              "--encoder-checkpoint", str(reference / "last.pt"),
              "--train", "data/point_training_fixture/train.jsonl", "--val", "data/point_training_fixture/validation.jsonl",
              "--steps", "3", "--warmup", "1", "--batch-size", "2", "--accumulation", "2",
              "--frames", "4", "--queries", "8", "--evaluate-every", "2", "--save-every", "1",
              "--workers", "0", "--development-run"]

    def run(name, extra=()):
        nonlocal child
        command = shared + ["--output", str(output / name)] + list(extra)
        print(json.dumps(dict(event="launch", run=name, arguments=extra)), flush=True)
        with (output / f"{name}.log").open("a") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"Point training {name} exited {code}; inspect its log")

    try:
        if args.pause_pid:
            fd = bind_predecessor(args.pause_pid, Path(args.pause_reference).resolve() if args.pause_reference else reference)
            state = Path(f"/proc/{args.pause_pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            if state not in ("T", "t"):
                send(signal.SIGSTOP)
                paused = True
                for _ in range(100):
                    state = Path(f"/proc/{args.pause_pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
                    if state in ("T", "t"):
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError("Video trainer did not pause")
            print(json.dumps(dict(event="verified_paused", pid=args.pause_pid)), flush=True)
        run("direct")
        run("resumed", ("--stop-after-step", "2"))
        run("resumed", ("--resume",))
        run("repeat")
        run("adapt", ("--mode", "adapt", "--stop-after-step", "1"))
        direct = torch.load(output / "direct/last.pt", map_location="cpu", weights_only=True)
        report = dict(scope="Real synthetic scenes: learning/checkpoint/resume integration only, not tracking accuracy",
                      source_encoder=direct["metadata"]["config"]["encoder_initialization"], comparisons={})
        for name in ("resumed", "repeat"):
            other = torch.load(output / name / "last.pt", map_location="cpu", weights_only=True)
            assert direct["metadata"]["config"] == other["metadata"]["config"]
            for field in ("step", "epoch", "cursor"):
                assert direct["training_state"][field] == other["training_state"][field]
            maximum = max(float((value - other["model"][key]).abs().max())
                          for key, value in direct["model"].items() if value.is_floating_point())
            # Chosen before observing results; any breach requires diagnosis.
            assert maximum < 5e-6, (name, maximum)
            histories = (direct["training_state"]["history"], other["training_state"]["history"])
            assert [r["step"] for r in histories[0]] == [2, 3]
            assert [r["step"] for r in histories[1]] == [2, 3]
            max_loss_difference = max(abs(a["validation"]["loss"]-b["validation"]["loss"]) for a,b in zip(*histories))
            assert max_loss_difference < 1e-3
            max_prediction_difference = 0.
            for step in (2, 3):
                with np.load(output / "direct" / f"validation_{step}.npz") as a, np.load(output / name / f"validation_{step}.npz") as b:
                    for key in ("scene_ids", "query_xy", "query_frames", "target_xy", "occluded"):
                        assert np.array_equal(a[key], b[key]), (name, step, key)
                    max_prediction_difference = max(max_prediction_difference, float(np.abs(a["predicted_xy"]-b["predicted_xy"]).max()))
            assert max_prediction_difference < 1e-3
            report["comparisons"][name] = dict(max_weight_difference=maximum, max_loss_difference=max_loss_difference,
                                               max_normalized_xy_difference=max_prediction_difference)
            del other
        original = torch.load(reference / "last.pt", map_location="cpu", weights_only=True)
        frozen_changed = [key for key, value in direct["model"].items() if key.startswith("encoder.")
                          and not torch.equal(value, original["model"][key])]
        assert not frozen_changed
        adapted = torch.load(output / "adapt/last.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(value, original["model"][key]) for key, value in adapted["model"].items()
                   if key.startswith("encoder.backbone."))
        changed_by_layer = {}
        for key, value in adapted["model"].items():
            prefix = "encoder.core." if args.encoder_type == "gmmix" else "encoder.adapters."
            if key.startswith(prefix) and not torch.equal(value, original["model"][key]):
                parts = key.split(".")
                layer = ".".join(parts[2:4]) if args.encoder_type == "gmmix" else parts[2]
                changed_by_layer[layer] = changed_by_layer.get(layer, 0) + 1
        if args.encoder_type == "gmmix":
            assert all(f"{component}.{i}" in changed_by_layer
                       for component in ("mamba_blocks", "spatial_blocks", "gates") for i in range(4))
        else:
            assert set(changed_by_layer) == {"3", "7", "11"}
        report.update(probe_encoder_bitwise_unchanged=True, adapt_backbone_bitwise_unchanged=True,
                      adapt_changed_parameter_tensors_by_layer=changed_by_layer,
                      steps=3, train_scenes=3, validation_scenes=1, frames=4, queries=8,
                      scheduled_validation_preserved_across_stop=True,
                      encoder_type=args.encoder_type,
                      predeclared_tolerances=dict(weights=5e-6, loss=1e-3, normalized_xy=1e-3),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key:value for key,value in report.items() if key != "source_encoder"}), flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if fd is not None:
            try:
                if paused:
                    send(signal.SIGCONT)
                    print(json.dumps(dict(event="resumed_video_trainer", pid=args.pause_pid)), flush=True)
            finally:
                os.close(fd)


if __name__ == "__main__":
    main()
