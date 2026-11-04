"""Evaluate one coherent checkpoint with real videos and clip/stream inference.

No training or checkpoint selection. Streaming state is reset for every clip.
History ablations retain the same current observation and preprocessing.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from evaluate_temporal import classification_metrics
from train_temporal import check_split
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips
from train_video import configure_reproducibility, REPRODUCIBILITY


@torch.no_grad()
def streaming_logits(model, video):
    state = None
    sizes = []
    for image in video.unbind(1):
        features, state = model.encoder.step(image, state)
        sizes.append(state.nbytes)
    return model.readout(features), sizes


@torch.no_grad()
def evaluate_modes(model, loader, device, modes=("clip", "stream")):
    model.eval()
    predictions = {mode: [] for mode in modes}
    losses = dict.fromkeys(modes, 0.)
    targets, squared_error, elements, maximum = [], 0., 0, 0.
    state_bytes = None
    for video, labels in loader:
        video, labels = video.to(device), labels.to(device)
        outputs = {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if "clip" in modes:
                outputs["clip"] = model(video).float()
            if "stream" in modes:
                outputs["stream"], sizes = streaming_logits(model, video)
                outputs["stream"] = outputs["stream"].float()
                if state_bytes is None:
                    state_bytes = dict(batch_size=len(labels), by_frame=sizes)
        targets.extend(labels.cpu().tolist())
        for mode, logits in outputs.items():
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Nonfinite {mode} predictions")
            predictions[mode].extend(logits.argmax(-1).cpu().tolist())
            losses[mode] += float(F.cross_entropy(logits, labels, reduction="sum"))
        if len(outputs) == 2:
            difference = outputs["stream"] - outputs["clip"]
            squared_error += float(difference.square().sum())
            elements += difference.numel()
            maximum = max(maximum, float(difference.abs().max()))
    if not targets:
        raise ValueError("Empty evaluation set")
    paired = None
    if len(modes) == 2:
        paired = dict(logit_max_abs=maximum, logit_rmse=(squared_error / elements)**0.5,
                      prediction_disagreements=sum(a != b for a, b in
                                                   zip(predictions["clip"], predictions["stream"])))
    return targets, predictions, {m: loss/len(targets) for m, loss in losses.items()}, paired, state_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--role", choices=("validation", "test"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--modes", nargs="+", choices=("clip", "stream"), default=["clip", "stream"])
    p.add_argument("--ablations", nargs="+", choices=("full", "current", "reverse_history"),
                   default=["full", "current", "reverse_history"])
    p.add_argument("--train-manifest")
    p.add_argument("--val-manifest")
    args = p.parse_args()
    if args.batch_size < 1 or args.workers < 0 or len(set(args.modes)) != len(args.modes):
        p.error("Invalid batch, workers, or repeated modes")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation output")
    torch.set_num_threads(4)
    configure_reproducibility(42)
    device = torch.device("cuda")
    model, info = load_video_checkpoint(args.checkpoint, device=device)
    config = info["config"]
    if config.get("reproducibility") != REPRODUCIBILITY:
        raise ValueError("Evaluator requires the current explicit reproducibility protocol")
    frames = config["frames"] if args.frames is None else args.frames
    processor = SiglipImageProcessor.from_pretrained(config["model"], revision=config["revision"],
                                                     local_files_only=True)
    train_path, val_path = args.train_manifest or config["train"], args.val_manifest or config["val"]
    for name, path in (("train", train_path), ("val", val_path)):
        if file_hash(path) != config[name + "_sha256"]:
            raise ValueError(f"Original {name} manifest changed")
    train = RawVideoClips(train_path, processor, frames, config["classes"])
    validation = RawVideoClips(val_path, processor, frames, config["classes"])
    report = dict(**info, role=args.role, manifest_sha256=file_hash(args.manifest),
                  evaluation_reproducibility=REPRODUCIBILITY,
                  frames=frames, training_frames=config["frames"], modes=args.modes,
                  scope="video classification, not robot success or dense-task accuracy", ablations={})
    for ablation in args.ablations:
        data = RawVideoClips(args.manifest, processor, frames, config["classes"], ablation=ablation)
        check_split(train, data)
        if args.role == "test":
            check_split(validation, data)
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers,
                            multiprocessing_context="spawn" if args.workers else None)
        labels, predictions, losses, paired, sizes = evaluate_modes(model, loader, device, args.modes)
        report["ablations"][ablation] = dict(modes={
            mode: dict(**classification_metrics(labels, pred, config["classes"]), loss=losses[mode],
                       clips=[dict(id=row["id"], group=row["group"], label=label, prediction=value)
                              for row, label, value in zip(data.rows, labels, pred)])
            for mode, pred in predictions.items()}, clip_vs_stream=paired,
            first_batch_state_bytes=sizes)
        print(json.dumps(dict(ablation=ablation, accuracy={mode: sum(a == b for a, b in zip(labels, pred))/len(labels)
                           for mode, pred in predictions.items()}, clip_vs_stream=paired)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
