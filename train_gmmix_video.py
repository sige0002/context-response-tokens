"""Matched video training of the published GatedMambaMix/SigLIP2 baseline.

Uses current-frame readout only, deterministic clip sampling, accumulation,
warmup/cosine, source snapshots, and resumable optimizer-boundary checkpoints.
Run with uv --project baselines/gmmix. This separate entry preserves the live
Mamba-3 trainer while using its data, readout, and optimization protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from gmmix_siglip import GatedMambaMixSiglip, REFERENCE_COMMIT, REFERENCE_FILES
from temporal_readout import VideoRecognizer, READOUT_COMMIT, READOUT_SHA256
from train_temporal import atomic_save, check_split, run_epoch
from train_video import learning_rate, parameter_groups, configure_reproducibility, REPRODUCIBILITY
from video_data import RawVideoClips


ARCHITECTURE = dict(kind="gmmix", placement="after_encoder", layers=4, state_size=16,
                    spatial_heads=16, mamba_dropout=.1, chunk_size=512)
SOURCES = ("train_gmmix_video.py", "gmmix_siglip.py", "gmmix_checkpoint.py", "video_data.py",
           "temporal_readout.py", "temporal_siglip.py", "train_temporal.py", "train_video.py",
           "baselines/gmmix/pyproject.toml", "baselines/gmmix/uv.lock",
           "baselines/gmmix/source_manifest.json",
           "vendor/frozen-video/models/ssm_modules/mamba_modules.py",
           "vendor/frozen-video/models/rvm_modules/rvm_blocks.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/ssv2/videos_trainval/train.jsonl")
    p.add_argument("--val", default="data/ssv2/videos_trainval/val.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--classes", type=int, default=174)
    p.add_argument("--kind", choices=("gmmix",), default="gmmix")
    p.add_argument("--placement", choices=("after_encoder",), default="after_encoder")
    p.add_argument("--readout", choices=("mean", "attention"), default="attention")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--evaluate-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-step", type=int, default=0,
                   help="Execution limit for resumability checks; keeps the full LR schedule unchanged")
    args = p.parse_args()
    if (min(args.classes, args.frames, args.batch_size, args.accumulation, args.steps,
            args.evaluate_every, args.save_every) < 1 or not 0 <= args.warmup < args.steps
            or args.lr < 1e-7 or args.weight_decay < 0 or args.workers < 0 or args.stop_after_step < 0):
        p.error("Invalid training configuration")
    if not torch.cuda.is_available():
        p.error("This experiment requires CUDA")
    from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda, causal_conv1d_fn
    if selective_scan_cuda is None:
        p.error("Use the built baseline uv project: uv run --project baselines/gmmix --locked python train_gmmix_video.py ...")
    torch.set_num_threads(4)
    configure_reproducibility(args.seed)
    device = torch.device("cuda")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Choose an empty run directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    sources = SOURCES
    config = {k: v for k, v in vars(args).items()
              if k not in ("output", "resume", "workers", "stop_after_step")}
    config.update(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, readout_commit=READOUT_COMMIT,
                  reproducibility=REPRODUCIBILITY,
                  readout_sha256=READOUT_SHA256, spatial_precision="float32", temporal_precision="bfloat16",
                  crop_min=0.8, sampling="segmented history plus final observation; shared spatial crop; no flips",
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    config.update(encoder_architecture=ARCHITECTURE, core_reference_commit=REFERENCE_COMMIT,
                  core_reference_files=REFERENCE_FILES,
                  selective_scan_extension_sha256=hashlib.sha256(Path(selective_scan_cuda.__file__).read_bytes()).hexdigest(),
                  causal_conv1d_available=causal_conv1d_fn is not None)
    for split in ("train", "val"):
        path = Path(getattr(args, split)).resolve()
        config[split] = str(path)
        config[split + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.resume:
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Resume config, source, or data differs")
    else:
        source_dir = output / "source"
        source_dir.mkdir()
        for name in sources:
            (source_dir / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, source_dir / name)
        shutil.copyfile("vendor/frozen-video/models/rvm_modules/rvm_blocks.py", source_dir / "rvm_blocks.py")
        shutil.copyfile("vendor/frozen-video/LICENSE", source_dir / "reference_readout_LICENSE")
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    train = RawVideoClips(args.train, processor, args.frames, args.classes, training=True, seed=args.seed)
    val = RawVideoClips(args.val, processor, args.frames, args.classes)
    check_split(train, val)
    encoder = GatedMambaMixSiglip.from_pretrained(**ARCHITECTURE)
    model = VideoRecognizer(encoder, args.classes, args.readout, args.seed + 1).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    step = epoch = cursor = 0
    history, best = [], -1.
    if args.resume:
        # AdamW keeps non-capturable step counters on CPU. Let optimizer loading
        # move parameter moments to their owning device without moving counters.
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (saved.get("format_version") != 3 or saved.get("model_type") != "gmmix_siglip_video"
                or saved["config"] != config):
            raise ValueError("Checkpoint config differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, cursor = saved["step"], saved["epoch"], saved["cursor"]
        history, best = saved["history"], saved["best"]
        torch.set_rng_state(saved["rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        del saved
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save():
        atomic_save(dict(format_version=3, model_type="gmmix_siglip_video", config=config,
                         backbone_config=encoder.backbone.config.to_dict(), model=model.state_dict(),
                         optimizer=optimizer.state_dict(), step=step, epoch=epoch, cursor=cursor,
                         history=history, best=best, rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all()), output / "last.pt")

    def log(record):
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    log(dict(event="start", step=step, train_samples=len(train), val_samples=len(val),
             trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    started = time.monotonic()
    interval_loss = interval_correct = interval_samples = 0
    while step < args.steps and not stop:
        if cursor == len(train):
            epoch, cursor = epoch + 1, 0
        train.set_epoch(epoch)
        permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        loader = DataLoader(train, batch_size=args.batch_size, sampler=permutation[cursor:],
                            num_workers=args.workers, pin_memory=True,
                            multiprocessing_context="spawn" if args.workers else None,
                            generator=torch.Generator().manual_seed(args.seed + epoch + 10000))
        optimizer.zero_grad(set_to_none=True)
        microbatches = accumulated_samples = 0
        for images, labels in loader:
            model.train()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            # Divide gradients by actual accumulated sample count at the update
            # boundary, including the shorter final batch of an epoch.
            loss.backward()
            microbatches += 1
            accumulated_samples += len(labels)
            cursor += len(labels)
            interval_loss += float(loss.detach())
            interval_correct += int((logits.argmax(-1) == labels).sum())
            interval_samples += len(labels)
            if microbatches < args.accumulation and cursor < len(train):
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_samples)
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
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
                         loss=interval_loss / interval_samples, accuracy=interval_correct / interval_samples,
                         samples=interval_samples, elapsed_seconds=time.monotonic() - started))
                interval_loss = interval_correct = interval_samples = 0
            if step % args.evaluate_every == 0 or step == args.steps:
                metrics = run_epoch(model, DataLoader(val, batch_size=args.batch_size,
                                    num_workers=args.workers, pin_memory=True,
                                    multiprocessing_context="spawn" if args.workers else None,
                                    generator=torch.Generator().manual_seed(args.seed + 20000)), device)
                history.append(dict(step=step, epoch=epoch, validation=metrics))
                if metrics["accuracy"] > best:
                    best = metrics["accuracy"]
                    encoder.save_checkpoint(output / "best_encoder.pt")
                    atomic_save(dict(format_version=2, readout=model.readout.state_dict(), config=config,
                                     step=step), output / "best_readout.pt")
                (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                log(dict(event="validation", step=step,
                         **{k: v for k, v in metrics.items() if k != "predictions"}))
            if step % args.save_every == 0 or step % args.evaluate_every == 0 or step == args.steps or stop:
                save()
            if step >= args.steps or stop:
                break
    log(dict(event="stopped" if stop else "budget_complete", step=step, best_validation=best))


if __name__ == "__main__":
    main()
