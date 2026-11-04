"""Train the common point head with the additional multi-depth GMMix encoder.

Full experiments require the completed synthetic scene split and a completed
video-training checkpoint. Development runs are explicitly labeled and cannot
be reported as full-data results. RoboTAP is never a training/validation input.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from kubric_data import KubricPointClips
from multidepth_point_checkpoint import file_hash, load_point_checkpoint, save_point_checkpoint, point_kernel_environment
from point_readout import PointReadout, PointTracker, point_tracking_loss
from tapvid_metrics import compute_tapvid_metrics
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from train_video import learning_rate, parameter_groups, configure_reproducibility, REPRODUCIBILITY
from multidepth_checkpoint import load_multidepth_video_checkpoint
from train_points import check_splits, to_device, forward_loss, validate


SOURCES = ("train_multidepth_points.py", "train_points.py", "multidepth_point_checkpoint.py",
           "multidepth_checkpoint.py", "gmmix_multidepth.py",
           "vendor/frozen-video/models/image_encoder/dino_wrappers.py",
           "vendor/frozen-video/models/ssm_modules/mamba_modules.py",
           "vendor/frozen-video/models/rvm_modules/rvm_blocks.py", "point_readout.py", "point_checkpoint.py", "kubric_data.py",
           "temporal_siglip.py", "video_checkpoint.py", "temporal_readout.py", "train_video.py",
           "video_data.py", "train_temporal.py", "tapvid_metrics.py", "pyproject.toml", "uv.lock",
           "gmmix_siglip.py", "gmmix_checkpoint.py", "baselines/gmmix/pyproject.toml",
           "baselines/gmmix/uv.lock", "baselines/gmmix/source_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--encoder-type", choices=("gmmix_multidepth",), default="gmmix_multidepth")
    parser.add_argument("--train", default="data/cotracker3_kubric/train.jsonl")
    parser.add_argument("--val", default="data/cotracker3_kubric/validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("probe", "adapt"), default="probe")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--evaluate-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int, default=0)
    parser.add_argument("--development-run", action="store_true")
    args = parser.parse_args()
    if (min(args.frames - 1, args.queries, args.batch_size, args.accumulation, args.steps,
            args.evaluate_every, args.save_every) < 1 or not 0 <= args.warmup < args.steps
            or args.lr < 1e-7 or args.weight_decay < 0 or min(args.workers, args.stop_after_step) < 0):
        parser.error("Invalid point-training configuration")
    configure_reproducibility(args.seed)
    if not torch.cuda.is_available():
        parser.error("This training entry requires CUDA")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty run directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    train = KubricPointClips(args.train, processor, args.frames, args.queries, training=True, seed=args.seed)
    validation = KubricPointClips(args.val, processor, args.frames, args.queries, seed=args.seed)
    check_splits(train, validation)
    config = {key:value for key, value in vars(args).items()
              if key not in ("output", "resume", "workers", "stop_after_step")}
    for name in ("train", "val", "encoder_checkpoint"):
        config[name] = str(Path(config[name]).resolve())
    config.update(train_sha256=file_hash(args.train), val_sha256=file_hash(args.val),
        source_sha256={name:file_hash(Path(__file__).parent / name) for name in SOURCES},
        scope="development only" if args.development_run else "full synthetic split; external RoboTAP evaluation separate",
        precision="FP32 spatial encoder/coordinates, BF16 temporal/readout matrix products",
        reproducibility=REPRODUCIBILITY,
        deterministic_pytorch=True, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        kernel_environment=point_kernel_environment(args.encoder_type),
        loss="Per-clip mean: 100 * visible normalized-xy Huber(delta=.05) + .1 * visibility BCE; post-query only")
    if not args.development_run:
        complete = json.loads((Path(args.train).parent / "complete.json").read_text())
        if (config["train_sha256"] != complete["train_sha256"]
                or config["val_sha256"] != complete["validation_sha256"]
                or len(train) != 5569 or len(validation) != 300):
            raise ValueError("Full experiment requires the completed, disjoint prepared split")
    info, step, epoch, cursor, history = None, 0, 0, 0, []
    if args.resume:
        previous = json.loads((output / "config.json").read_text())
        config["encoder_initialization"] = previous["encoder_initialization"]
        if config != previous:
            raise ValueError("Resume source, dataset, or configuration differs")
        model, info = load_point_checkpoint(output / "last.pt", device)
        if info["metadata"]["config"] != config:
            raise ValueError("Point checkpoint configuration differs")
    else:
        recognizer, provenance = load_multidepth_video_checkpoint(args.encoder_checkpoint, device)
        if not args.development_run and provenance["step"] != provenance["config"]["steps"]:
            raise ValueError("Full probe must start from the completed fixed-step video checkpoint")
        config["encoder_initialization"] = provenance
        encoder = recognizer.encoder
        del recognizer
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(args.seed + 1)
            head = PointReadout(input_dim=encoder.backbone.config.hidden_size,
                               grid_size=encoder.backbone.config.image_size // encoder.backbone.config.patch_size)
        model = PointTracker(encoder, head, freeze_encoder=args.mode == "probe").to(device)
        source = output / "source"
        source.mkdir()
        for name in SOURCES:
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).parent / name, source / name)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    if info is not None:
        state = info["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, cursor, history = (state[key] for key in ("step", "epoch", "cursor", "history"))
        torch.set_rng_state(state["rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        del state, info
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save():
        state = dict(optimizer=optimizer.state_dict(), step=step, epoch=epoch, cursor=cursor, history=history,
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        save_point_checkpoint(model, output / "last.pt", dict(config=config), state)

    def log(record):
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    log(dict(event="start", step=step, train_samples=len(train), val_samples=len(validation), mode=args.mode,
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    started, interval_loss, interval_samples = time.monotonic(), 0., 0
    while step < args.steps and not stop:
        if cursor == len(train):
            epoch, cursor = epoch + 1, 0
        train.set_epoch(epoch)
        permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        loader = DataLoader(train, batch_size=args.batch_size, sampler=permutation[cursor:], num_workers=args.workers,
                            pin_memory=True, multiprocessing_context="spawn" if args.workers else None,
                            generator=torch.Generator().manual_seed(args.seed + epoch + 10000))
        optimizer.zero_grad(set_to_none=True)
        microbatches = accumulated_samples = 0
        for data in loader:
            model.train()
            batch = to_device(data, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, losses = forward_loss(model, batch)
            size = len(batch["video"])
            (losses["loss"] * size).backward()
            interval_loss += float(losses["loss"].detach()) * size
            interval_samples += size
            cursor += size
            accumulated_samples += size
            microbatches += 1
            # A termination request also finishes the predefined accumulation
            # boundary, so interruption never changes the effective batch.
            if microbatches < args.accumulation and cursor < len(train):
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            lr = learning_rate(step, args.steps, args.warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            microbatches = accumulated_samples = 0
            step += 1
            if args.stop_after_step and step >= args.stop_after_step:
                stop = True
            if step % 25 == 0 or step == args.steps or stop:
                log(dict(event="train", step=step, epoch=epoch, cursor=cursor, lr=lr,
                         loss=interval_loss/interval_samples, samples=interval_samples,
                         elapsed_seconds=time.monotonic()-started))
                interval_loss = interval_samples = 0
            if step % args.evaluate_every == 0 or step == args.steps:
                val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=True, multiprocessing_context="spawn" if args.workers else None,
                    generator=torch.Generator().manual_seed(args.seed + 20000))
                metrics = validate(model, val_loader, device, output / f"validation_{step}.npz")
                history.append(dict(step=step, epoch=epoch, validation=metrics))
                (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                log(dict(event="validation", step=step, **metrics))
            if step % args.save_every == 0 or step % args.evaluate_every == 0 or step == args.steps or stop:
                save()
            if step >= args.steps or stop:
                break
    log(dict(event="stopped" if stop else "budget_complete", step=step))


if __name__ == "__main__":
    main()
