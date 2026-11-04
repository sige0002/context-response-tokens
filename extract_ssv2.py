"""Deterministic SSv2 subsets from the official split archives.

Select IDs from annotations before reading videos; never select using outcomes.
Limits are per class; zero means all clips. Output is a pilot subset unless both
limits are zero. Official validation remains validation, not a held-out test.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

import av
import numpy as np

from prepare_ssv2 import PARTS
from video_data import frame_to_rgb, RGB_INTERPOLATION


def rgb_conversion_protocol():
    return dict(version="full_chroma_bitexact_v1", interpolation=int(RGB_INTERPOLATION),
                threads=1, pyav=av.__version__,
                reader_sha256=hashlib.sha256(Path(__file__).with_name("video_data.py").read_bytes()).hexdigest())


def validate_cached_frames(record, output, conversion):
    if record.get("rgb_conversion") != conversion:
        raise ValueError("Cached frames use a different or unrecorded RGB conversion; choose a new output")
    if not all((output / frame).is_file() for frame in record["frames"]):
        raise FileNotFoundError("Incomplete frame cache")


class JoinedParts:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = open(next(self.paths), "rb")

    def read(self, size):
        pieces = []
        remaining = size
        while remaining:
            block = self.file.read(remaining)
            if block:
                pieces.append(block)
                remaining -= len(block)
            else:
                path = next(self.paths, None)
                if path is None:
                    break
                self.file.close()
                self.file = open(path, "rb")
        return b"".join(pieces)

    def close(self):
        self.file.close()


def select_rows(rows, labels, limit, seed):
    grouped = {int(label): [] for label in labels.values()}
    for row in rows:
        template = row["template"].replace("[", "").replace("]", "")
        grouped[int(labels[template])].append(row)
    selected = []
    for label, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: int(r["id"]))
        random.Random(seed + label).shuffle(ordered)
        if limit:
            ordered = ordered[:limit]
        selected.extend(dict(id=row["id"], label=label) for row in ordered)
    return sorted(selected, key=lambda r: int(r["id"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/ssv2")
    p.add_argument("--output", default="data/ssv2/pilot")
    p.add_argument("--train-per-class", type=int, default=16)
    p.add_argument("--val-per-class", type=int, default=4)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.train_per_class, args.val_per_class) < 0 or args.frames < 2:
        p.error("Invalid limits/frame count")
    root, output = Path(args.root), Path(args.output)
    labels = json.loads((root / "labels/labels.json").read_text())
    splits = {s: select_rows(json.loads((root / f"labels/{name}.json").read_text()), labels, limit, args.seed)
              for s, name, limit in (("train", "train", args.train_per_class),
                                      ("val", "validation", args.val_per_class))}
    assert not ({r["id"] for r in splits["train"]} & {r["id"] for r in splits["val"]})
    plan = dict(frames=args.frames, seed=args.seed, labels=labels, selected=splits,
                rgb_conversion=rgb_conversion_protocol(),
                train_per_class=args.train_per_class, val_per_class=args.val_per_class,
                scope="pilot subset" if args.train_per_class or args.val_per_class else "full train/validation",
                annotation_sha256={s: hashlib.sha256((root / f"labels/{s}.json").read_bytes()).hexdigest()
                                   for s in ("labels", "train", "validation")})
    for name in PARTS:
        record = json.loads((root / (name + ".json")).read_text())
        if (root / name).stat().st_size != record["bytes"]:
            raise ValueError("Incomplete archive")
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / "selection.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Output already contains a different selection")
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    wanted = {r["id"] for rows in splits.values() for r in rows}
    found = {}
    joined = JoinedParts([root / name for name in PARTS])
    try:
        with tarfile.open(fileobj=joined, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.suffix != ".webm" or name.stem not in wanted:
                    continue
                clip_id = name.stem
                directory = output / "frames" / clip_id
                directory.mkdir(parents=True, exist_ok=True)
                metadata = directory / "frames.json"
                if metadata.exists():
                    found[clip_id] = json.loads(metadata.read_text())
                    validate_cached_frames(found[clip_id], output, plan["rgb_conversion"])
                    continue
                source = archive.extractfile(member)
                with av.open(io.BytesIO(source.read())) as container:
                    container.streams.video[0].thread_count = 1
                    frames = list(container.decode(video=0))
                if not frames:
                    raise ValueError(f"No decoded frames: {clip_id}")
                indices = np.linspace(0, len(frames)-1, args.frames).astype(int).tolist()
                paths, times = [], []
                for i, index in enumerate(indices):
                    destination = directory / f"{i:03}.jpg"
                    frame_to_rgb(frames[index]).save(destination, quality=95)
                    paths.append(str(destination.relative_to(output)))
                    times.append(frames[index].time)
                found[clip_id] = dict(frames=paths, timestamps=times, decoded_frames=len(frames), indices=indices,
                                     rgb_conversion=plan["rgb_conversion"])
                metadata.write_text(json.dumps(found[clip_id]) + "\n")
                if len(found) % 100 == 0:
                    print(f"decoded {len(found)}/{len(wanted)}", flush=True)
    finally:
        joined.close()
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} selected videos: {sorted(missing)[:10]}")
    for split, rows in splits.items():
        lines = [dict(id=r["id"], group=r["id"], label=r["label"], **found[r["id"]]) for r in rows]
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    print(json.dumps({s: len(rows) for s, rows in splits.items()}), flush=True)


if __name__ == "__main__":
    main()
