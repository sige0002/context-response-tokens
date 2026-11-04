"""Supervised video classification for the temporal SigLIP 2 encoder.

JSONL rows: {"id": str, "group": str, "frames": [relative paths], "label": int}.
Groups are source videos/episodes, not clips. Train/validation groups must differ.
This probe uses final-time features only; there is no teacher or distillation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipImageProcessor

from temporal_siglip import TemporalSiglip, DEFAULT_MODEL, DEFAULT_REVISION


class FrameClips(Dataset):
    def __init__(self, manifest, processor, frames, classes, ablation="full"):
        self.path = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        self.processor, self.frames, self.ablation = processor, frames, ablation
        if not self.rows or frames < 1 or ablation not in ("full", "current", "reverse_history"):
            raise ValueError("Invalid dataset or sampling configuration")
        ids = set()
        self.groups, self.image_paths = set(), set()
        for row in self.rows:
            if (not isinstance(row.get("id"), str) or row["id"] in ids
                    or not isinstance(row.get("group"), str) or not row["group"]):
                raise ValueError("Each clip needs a unique string id and source group")
            ids.add(row["id"])
            self.groups.add(row["group"])
            if type(row.get("label")) is not int or not 0 <= row["label"] < classes:
                raise ValueError("Label outside the declared class range")
            if not row.get("frames"):
                raise ValueError("Empty frame list")
            row["frames"] = [(self.path.parent / f).resolve() for f in row["frames"]]
            for f in row["frames"]:
                if not f.is_file():
                    raise FileNotFoundError(f)
            self.image_paths.update(row["frames"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Fixed sampling is reproducible and includes the final observation.
        indices = ([len(row["frames"]) - 1] if self.frames == 1 else
                   torch.linspace(0, len(row["frames"]) - 1, self.frames).long().tolist())
        if self.ablation == "current":
            indices = [indices[-1]] * len(indices)
        elif self.ablation == "reverse_history":
            indices = indices[:-1][::-1] + indices[-1:]
        images = []
        for i in indices:
            with Image.open(row["frames"][i]) as image:
                images.append(image.convert("RGB"))
        pixels = self.processor(images=images, return_tensors="pt").pixel_values
        return pixels, row["label"]


def check_split(train, val):
    if train.groups & val.groups:
        raise ValueError("Source video/episode groups overlap between train and validation")
    if train.image_paths & val.image_paths:
        raise ValueError("Frame paths overlap between train and validation")


class VideoClassifier(nn.Module):
    def __init__(self, encoder, classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.backbone.config.hidden_size, classes)

    def forward(self, video):
        # Spatial pooling only: temporal learning must happen in the encoder.
        return self.head(self.encoder.encode_current(video).mean(dim=1))


def run_epoch(model, loader, device, optimizer=None, progress=None):
    model.train(optimizer is not None)
    total_loss = correct = total = 0
    predictions = []
    for batch_index, (video, labels) in enumerate(loader):
        video, labels = video.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(video)
            loss = nn.functional.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        predicted = logits.argmax(-1)
        total_loss += loss.item() * len(labels)
        correct += (predicted == labels).sum().item()
        total += len(labels)
        predictions.extend(predicted.detach().cpu().tolist())
        if progress is not None and ((batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader)):
            progress(dict(batch=batch_index + 1, batches=len(loader), samples=total,
                          loss=total_loss / total, accuracy=correct / total))
    return {"loss": total_loss / total, "accuracy": correct / total,
            "samples": total, "predictions": predictions}


def atomic_save(payload, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--classes", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["none", "attention", "mamba"], default="mamba")
    p.add_argument("--placement", choices=["interleaved", "late"], default="interleaved")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--state-size", type=int, default=64)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.classes < 2 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("Invalid classes, epochs, batch size, or learning rate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.kind == "mamba" and device.type != "cuda":
        p.error("Mamba-3 requires CUDA")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    train = FrameClips(args.train, processor, args.frames, args.classes)
    validations = {a: FrameClips(args.val, processor, args.frames, args.classes, a)
                   for a in ("full", "current", "reverse_history")}
    check_split(train, validations["full"])
    config = vars(args).copy()
    for field in ("epochs", "resume", "output", "workers", "train", "val"):
        config.pop(field)
    config.update(model=DEFAULT_MODEL, revision=DEFAULT_REVISION,
                  train_sha256=hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                  val_sha256=hashlib.sha256(Path(args.val).read_bytes()).hexdigest(),
                  train_path=str(Path(args.train).resolve()), val_path=str(Path(args.val).resolve()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory or --resume")
    source_directory = output / "source"
    if not args.resume:
        source_directory.mkdir()
        for name in ("train_temporal.py", "temporal_siglip.py", "pyproject.toml", "uv.lock"):
            shutil.copyfile(Path(__file__).parent / name, source_directory / name)
    elif source_directory.exists():
        for name in ("train_temporal.py", "temporal_siglip.py"):
            if (source_directory / name).read_bytes() != (Path(__file__).parent / name).read_bytes():
                raise ValueError(f"Training source changed since checkpoint: {name}")
    encoder = TemporalSiglip.from_pretrained(kind=args.kind, every=args.every,
                                             width=args.width, state_size=args.state_size,
                                             placement=args.placement)
    # Keep the readout initialization equal across temporal module variants.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 1)
        model = VideoClassifier(encoder, args.classes)
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    start, best, history = 0, -1., []
    if args.resume:
        # Keep AdamW's CPU step counters on CPU; load_state_dict moves moments
        # to each parameter's device.
        saved = torch.load(last, map_location="cpu", weights_only=True)
        saved["config"].setdefault("placement", "interleaved")
        if saved["config"] != config:
            raise ValueError("Resume configuration or dataset manifest changed")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best, history = saved["epoch"] + 1, saved["best"], saved["history"]
        torch.set_rng_state(saved["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for epoch in range(start, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            generator=generator, num_workers=args.workers)
        def progress(info):
            line = json.dumps(dict(epoch=epoch, phase="train", **info))
            with (output / "progress.jsonl").open("a") as f:
                f.write(line + "\n")
            print(line, flush=True)

        training = run_epoch(model, loader, device, optimizer, progress=progress)
        training.pop("predictions")
        validation = {a: run_epoch(model, DataLoader(ds, batch_size=args.batch_size,
                          num_workers=args.workers), device) for a, ds in validations.items()}
        row = dict(epoch=epoch, train=training, validation=validation)
        history.append(row)
        if validation["full"]["accuracy"] > best:
            best = validation["full"]["accuracy"]
            encoder.save_checkpoint(output / "best_encoder.pt")
            atomic_save(dict(head=model.head.state_dict(), config=config, epoch=epoch),
                        output / "best_head.pt")
        atomic_save(dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, best=best, history=history, rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []), last)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "train": training,
                          "val_accuracy": {a: v["accuracy"] for a, v in validation.items()}}), flush=True)


if __name__ == "__main__":
    main()
