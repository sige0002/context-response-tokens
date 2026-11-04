"""Paired history interventions on a fixed, uniformly sampled validation subset.

This diagnoses an intermediate checkpoint. It is not a final matched-model
benchmark. A verified trainer may be paused for GPU isolation and is restored
in finally; the immutable checkpoint and sampled IDs are saved before inference.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from benchmark_streaming import idle_compute_processes
from evaluate_temporal import classification_metrics
from point_experiment import write_json
from run_video_comparisons import bind_predecessor, verify_inputs
from summarize_temporal import paired_interval
from train_temporal import check_split
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips


CONDITIONS = ("full", "current", "reverse_history")


def history_view(video, condition):
    if condition == "full":
        return video
    if condition == "current":
        return video[:, -1:].expand_as(video).contiguous()
    if condition == "reverse_history":
        return torch.cat((video[:, :-1].flip(1), video[:, -1:]), dim=1)
    raise ValueError("Unknown history intervention")


def sample_manifest(source, samples, seed, destination):
    source = Path(source).resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    if not 1 <= samples <= len(rows):
        raise ValueError("Sample size outside the validation set")
    selected = sorted(random.Random(seed).sample(range(len(rows)), samples))
    selected_rows = [dict(rows[i], video=str((source.parent / rows[i]["video"]).resolve())) for i in selected]
    if len({r["id"] for r in selected_rows}) != samples:
        raise ValueError("Repeated validation IDs")
    Path(destination).write_text("".join(json.dumps(row) + "\n" for row in selected_rows))
    return selected_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pause-pid", type=int)
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 0:
        parser.error("Invalid sample/worker count")
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = json.loads((run / "config.json").read_text())
    verify_inputs(config)
    rows = sample_manifest(config["val"], args.samples, args.sample_seed, output / "subset.jsonl")
    fd, paused = None, False
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_send_signal.argtypes, libc.pidfd_send_signal.restype = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint], ctypes.c_int

    def send(sig):
        if libc.pidfd_send_signal(fd, sig, None, 0) < 0:
            raise OSError(ctypes.get_errno(), "Could not signal bound trainer")

    def stop(signum, frame):
        raise InterruptedError(f"History diagnostic interrupted: {signum}")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.pause_pid:
            fd = bind_predecessor(args.pause_pid, run)
            state_path = Path(f"/proc/{args.pause_pid}/stat")
            if state_path.read_text().rsplit(") ", 1)[1].split()[0] not in ("T", "t"):
                send(signal.SIGSTOP)
                paused = True
                for _ in range(100):
                    if state_path.read_text().rsplit(") ", 1)[1].split()[0] in ("T", "t"):
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError("Trainer did not pause")
        paused_compute = idle_compute_processes()
        # The trainer atomically replaces last.pt. An open file remains tied
        # to a single checkpoint even if a caller runs without --pause-pid.
        with (run / "last.pt").open("rb") as source, (output / "checkpoint.pt").open("wb") as target:
            shutil.copyfileobj(source, target)
        model, provenance = load_video_checkpoint(output / "checkpoint.pt", "cuda")
        if provenance["config"] != config:
            raise ValueError("Checkpoint and run configuration differ")
        processor = SiglipImageProcessor.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
        train = RawVideoClips(config["train"], processor, config["frames"], config["classes"])
        data = RawVideoClips(output / "subset.jsonl", processor, config["frames"], config["classes"])
        check_split(train, data)
        loader = DataLoader(data, batch_size=config["batch_size"], num_workers=args.workers,
                            multiprocessing_context="spawn" if args.workers else None)
        plan = dict(scope="Exploratory intermediate-checkpoint history diagnostic; not final model comparison",
            checkpoint=provenance, samples=args.samples, sampling="Uniform without replacement from validation; fixed seed before inference",
            sample_seed=args.sample_seed, subset_sha256=file_hash(output / "subset.jsonl"),
            conditions=list(CONDITIONS), primary_inference_mode="clip",
            intervention="Identical preprocessed final image and number of frames across conditions; reverse only the past",
            source_sha256={name:file_hash(name) for name in ("diagnose_video_history.py", "video_data.py", "temporal_siglip.py",
                "video_checkpoint.py", "evaluate_temporal.py", "summarize_temporal.py")},
            initially_paused_compute_pids=paused_compute)
        write_json(output / "plan.json", plan)
        all_labels, predictions = [], {name:[] for name in CONDITIONS}
        losses = dict.fromkeys(CONDITIONS, 0.)
        started = time.monotonic()
        with torch.no_grad():
            for video, labels in loader:
                video, labels = video.cuda(), labels.cuda()
                all_labels.extend(labels.cpu().tolist())
                for name in CONDITIONS:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(history_view(video, name)).float()
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Nonfinite diagnostic logits")
                    predictions[name].extend(logits.argmax(-1).cpu().tolist())
                    losses[name] += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
                if len(all_labels) % 128 == 0 or len(all_labels) == args.samples:
                    print(json.dumps(dict(processed=len(all_labels), total=args.samples,
                                          seconds=time.monotonic()-started)), flush=True)
        assert all_labels == [row["label"] for row in rows]
        groups = [row["group"] for row in rows]
        summary = dict(plan, results={name:dict(**classification_metrics(all_labels, predictions[name], config["classes"]),
            loss=losses[name]/args.samples, clips=[dict(id=row["id"], group=row["group"], label=row["label"], prediction=pred)
                for row, pred in zip(rows, predictions[name])]) for name in CONDITIONS},
            full_minus_intervention={name:paired_interval(all_labels, predictions["full"], predictions[name], groups)
                                    for name in CONDITIONS if name != "full"},
            interval_scope="Paired source-video percentile bootstrap, conditional on this checkpoint and sampled validation clips; not training-seed uncertainty",
            limits="Interventions shift input distribution. A positive gap shows sensitivity/use of real history in these examples, not causal semantic understanding or superiority over other models.")
        write_json(output / "summary.json", summary)
        print(json.dumps(dict(step=provenance["step"], accuracy={name:summary["results"][name]["accuracy"] for name in CONDITIONS},
                              full_minus_intervention=summary["full_minus_intervention"])), flush=True)
    finally:
        if fd is not None:
            try:
                if paused:
                    send(signal.SIGCONT)
                    print(json.dumps(dict(resumed_trainer=args.pause_pid)), flush=True)
            finally:
                os.close(fd)


if __name__ == "__main__":
    main()
