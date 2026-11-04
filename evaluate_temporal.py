"""Evaluate a saved temporal encoder/head without training or checkpoint selection."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from temporal_siglip import TemporalSiglip
from train_temporal import FrameClips, VideoClassifier, check_split, run_epoch


def classification_metrics(labels, predictions, classes):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Expected matching nonempty label/prediction sequences")
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    for target, prediction in zip(labels, predictions):
        if not (0 <= target < classes and 0 <= prediction < classes):
            raise ValueError("Class index outside range")
        matrix[target, prediction] += 1
    support = matrix.sum(1)
    present = support > 0
    recall = matrix.diag().float() / support.clamp_min(1)
    return dict(accuracy=matrix.diag().sum().item() / len(labels),
                macro_recall_present_classes=recall[present].mean().item(),
                classes_present=int(present.sum()), support=support.tolist(),
                confusion_matrix=matrix.tolist())


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--role", choices=["validation", "test"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-manifest", help="Original training manifest if moved")
    p.add_argument("--val-manifest", help="Original selection-validation manifest if moved")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation output path")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.run)
    saved = torch.load(run / "best_head.pt", map_location="cpu", weights_only=True)
    config = saved["config"]
    frames = args.frames or config["frames"]
    if frames < 1 or args.batch_size < 1:
        p.error("Positive frame count and batch size required")
    processor = SiglipImageProcessor.from_pretrained(config["model"], revision=config["revision"])
    encoder = TemporalSiglip.load_checkpoint(run / "best_encoder.pt", device=device)
    expected = dict(kind=config["kind"], width=config["width"], every=config["every"],
                    state_size=config["state_size"], freeze_backbone=True,
                    placement=config.get("placement", "interleaved"))
    if encoder.architecture != expected or encoder.checkpoint_revision != config["revision"]:
        raise ValueError("Encoder and classification head configurations differ")
    model = VideoClassifier(encoder, config["classes"]).to(device).eval()
    model.head.load_state_dict(saved["head"], strict=True)
    train_path = args.train_manifest or config["train_path"]
    val_path = args.val_manifest or config["val_path"]
    for path, key in ((train_path, "train_sha256"), (val_path, "val_sha256")):
        if sha256(path) != config[key]:
            raise ValueError(f"Original split manifest changed: {path}")
    train = FrameClips(train_path, processor, frames, config["classes"])
    val = FrameClips(val_path, processor, frames, config["classes"])
    report = dict(role=args.role, checkpoint_epoch=saved["epoch"], frames=frames,
                  training_frames=config["frames"], model=config["model"], kind=config["kind"],
                  placement=encoder.placement,
                  checkpoint_sha256=sha256(run / "best_encoder.pt"),
                  head_sha256=sha256(run / "best_head.pt"), manifest_sha256=sha256(args.manifest),
                  ablations={})
    for ablation in ("full", "current", "reverse_history"):
        data = FrameClips(args.manifest, processor, frames, config["classes"], ablation)
        check_split(train, data)
        if args.role == "test":
            check_split(val, data)  # selection-validation cannot be relabelled test
        result = run_epoch(model, DataLoader(data, batch_size=args.batch_size), device)
        predictions = result.pop("predictions")
        metrics = classification_metrics([r["label"] for r in data.rows], predictions, config["classes"])
        report["ablations"][ablation] = dict(**metrics, loss=result["loss"],
            clips=[dict(id=row["id"], group=row["group"], label=row["label"], prediction=pred)
                   for row, pred in zip(data.rows, predictions)])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({a: {k: r[k] for k in ("accuracy", "macro_recall_present_classes")}
                      for a, r in report["ablations"].items()}))


if __name__ == "__main__":
    main()
