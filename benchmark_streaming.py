"""Measure fixed-history one-frame latency and state size, not video accuracy.

Each repetition starts from a deep copy of the same prefilled state. Copying and
prefill are outside timing. Other compute processes must be absent or stopped.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor

from temporal_siglip import TemporalSiglip, StreamState, DEFAULT_MODEL, DEFAULT_REVISION
from video_checkpoint import file_hash, load_video_checkpoint


def clone_state(state):
    if state is None:
        return None
    return StreamState(owner=state.owner, batch=state.batch, image_shape=state.image_shape,
                       offset=state.offset,
                       layers={key: tuple(value.clone() for value in tensors)
                               for key, tensors in state.layers.items()})


def idle_compute_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    paused = []
    for item in output.splitlines():
        if not item.strip().isdigit():
            continue
        pid = int(item.strip())
        if pid == os.getpid():
            continue
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        except FileNotFoundError:
            continue
        if state not in ("T", "t"):
            raise RuntimeError(f"Another GPU compute process is active: pid={pid}, state={state}")
        paused.append(pid)
    return paused


def distribution(values):
    return dict(p50=float(np.percentile(values, 50)), p90=float(np.percentile(values, 90)),
                minimum=float(min(values)), maximum=float(max(values)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="One atomic train_video.py last.pt")
    source.add_argument("--kind", choices=("none", "attention", "mamba"))
    p.add_argument("--placement", choices=("interleaved", "late"))
    p.add_argument("--include-readout", action="store_true")
    p.add_argument("--histories", type=int, nargs="+", default=[0, 6, 32, 128, 256])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--image", default="data/demo_dog.jpg")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if (args.batch_size < 1 or args.warmup < 1 or args.repeats < 5
            or min(args.histories) < 0 or (args.include_readout and not args.checkpoint)
            or (args.checkpoint and args.placement)):
        p.error("Invalid benchmark configuration")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new benchmark output")
    paused = idle_compute_processes()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    readout, info = None, {}
    if args.checkpoint:
        recognizer, info = load_video_checkpoint(args.checkpoint, "cuda")
        encoder = recognizer.encoder
        if args.include_readout:
            readout = recognizer.readout
        del recognizer
    else:
        encoder = TemporalSiglip.from_pretrained(kind=args.kind or "mamba",
                                                 placement=args.placement or "interleaved").cuda().eval()
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION,
                                                     local_files_only=True)
    with Image.open(args.image) as source_image:
        pixels = processor(images=source_image.convert("RGB"), return_tensors="pt").pixel_values
    pixels = pixels.expand(args.batch_size, -1, -1, -1).contiguous().cuda()
    report = dict(scope="Repeated-image kernel timing only; not recognition or memory retention accuracy",
                  architecture=encoder.architecture, trained_checkpoint=bool(args.checkpoint),
                  checkpoint=info, batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats,
                  include_readout=args.include_readout, image_sha256=file_hash(args.image),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  initial_paused_compute_pids=paused,
                  precision=dict(spatial="float32", temporal="bfloat16"),
                  timing_scope="One current frame; excludes image decoding, preprocessing, prefill, and state copying",
                  allocation_scope="Current process only; peak increment above baseline holding prefill plus copied input state. Not total GPU usage.",
                  model_parameter_bytes=sum(v.numel()*v.element_size() for v in encoder.parameters())
                      + (sum(v.numel()*v.element_size() for v in readout.parameters()) if readout is not None else 0),
                  histories=[])

    def step(state):
        features, state = encoder.step(pixels, state)
        return (readout(features) if readout is not None else features), state

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base, seen = None, 0
        for history in sorted(set(args.histories)):
            idle_compute_processes()
            while seen < history:
                result, base = step(base)
                seen += 1
                del result
            before = base.nbytes if base is not None else 0
            for _ in range(args.warmup):
                trial = clone_state(base)
                features, trial = step(trial)
                if not torch.isfinite(features).all():
                    raise FloatingPointError(f"Nonfinite output at history {history}")
                del features, trial
            gpu_ms, wall_ms, increments = [], [], []
            for _ in range(args.repeats):
                trial = clone_state(base)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                features, trial = step(trial)
                end.record()
                torch.cuda.synchronize()
                wall_ms.append((time.perf_counter() - wall_start) * 1000)
                gpu_ms.append(start.elapsed_time(end))
                increments.append(torch.cuda.max_memory_allocated() - baseline)
                after = trial.nbytes
                assert trial.offset == history + 1
                assert base is None or base.offset == history
                del features, trial
            row = dict(history_frames=history, state_bytes_before=before, state_bytes_after=after,
                       gpu_ms=distribution(gpu_ms), synchronized_wall_ms=distribution(wall_ms),
                       peak_allocated_increment_bytes=max(increments), baseline_allocated_bytes=baseline)
            report["histories"].append(row)
            print(json.dumps(row), flush=True)
        idle_compute_processes()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
