"""Deterministic, worker-independent sampling of original SSv2 videos."""
import json
from pathlib import Path
import random

import av
from av.video.reformatter import Interpolation
from torch.utils.data import Dataset


RGB_INTERPOLATION = (Interpolation.BILINEAR | Interpolation.ACCURATE_RND
                     | Interpolation.BITEXACT | Interpolation.FULL_CHR_H_INT)


def frame_to_rgb(frame):
    """Convert every pixel, including odd-width YUV420 right edges.

    The default unscaled conversion in the installed ARM64 FFmpeg/PyAV path
    can leave the final RGB column unwritten. Explicit full chroma conversion
    and rounding avoid that path. Do not crop or overwrite the last column:
    it can contain real visual evidence. Color metadata is left to FFmpeg.
    """
    return frame.to_image(interpolation=RGB_INTERPOLATION, threads=1).convert("RGB")


def temporal_indices(length, frames, rng=None):
    """Sample past segments, keeping the same final observation at all lengths."""
    if length < 1 or frames < 1:
        raise ValueError("Video and sample lengths must be positive")
    if frames == 1:
        return [length - 1]
    result = []
    past_length = max(1, length - 1)
    for i in range(frames - 1):
        lo, hi = i * past_length // (frames - 1), (i + 1) * past_length // (frames - 1)
        if rng is None:
            index = (lo + max(lo, hi - 1)) // 2
        else:
            index = rng.randrange(lo, max(lo + 1, hi))
        result.append(min(index, length - 1))
    return result + [length - 1]


class RawVideoClips(Dataset):
    def __init__(self, manifest, processor, frames=8, classes=174, training=False,
                 seed=42, ablation="full", crop_min=0.8):
        self.path = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.rows or frames < 1 or not 0 < crop_min <= 1:
            raise ValueError("Invalid video dataset configuration")
        if ablation not in ("full", "current", "reverse_history"):
            raise ValueError("Unknown history ablation")
        if training and ablation != "full":
            raise ValueError("Ablations are evaluation-only")
        self.processor, self.frames, self.training = processor, frames, training
        self.seed, self.epoch, self.ablation, self.crop_min = seed, 0, ablation, crop_min
        self.groups, self.image_paths, ids = set(), set(), set()
        for row in self.rows:
            if (not isinstance(row.get("id"), str) or row["id"] in ids
                    or not isinstance(row.get("group"), str) or not row["group"]):
                raise ValueError("Each row needs a unique id and source group")
            if type(row.get("label")) is not int or not 0 <= row["label"] < classes:
                raise ValueError("Invalid class label")
            path = (self.path.parent / row["video"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if "bytes" in row and path.stat().st_size != row["bytes"]:
                raise ValueError(f"Video size differs from manifest: {path}")
            row["video"] = path
            ids.add(row["id"])
            self.groups.add(row["group"])
            # Compatible with check_split; these are source-video file paths.
            self.image_paths.add(path)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def read(self, index):
        row = self.rows[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{row['id']}") if self.training else None
        try:
            with av.open(str(row["video"])) as container:
                # Bound CPU oversubscription when multiple loader workers run.
                container.streams.video[0].thread_count = 1
                decoded = list(container.decode(video=0))
        except Exception as exc:
            raise RuntimeError(f"Failed to decode video {row['id']}: {row['video']}") from exc
        if not decoded:
            raise ValueError(f"Empty decoded video: {row['id']}")
        indices = temporal_indices(len(decoded), self.frames, rng)
        if self.ablation == "current":
            indices = [indices[-1]] * self.frames
        elif self.ablation == "reverse_history":
            indices = indices[:-1][::-1] + indices[-1:]
        images = [frame_to_rgb(decoded[i]) for i in indices]
        width, height = images[0].size
        if any(image.size != (width, height) for image in images):
            raise ValueError(f"Resolution changed inside video {row['id']}")
        crop = (0, 0, width, height)
        if rng is not None:
            scale = rng.uniform(self.crop_min, 1.)
            w, h = max(1, round(width * scale)), max(1, round(height * scale))
            x, y = rng.randint(0, width - w), rng.randint(0, height - h)
            crop = (x, y, x + w, y + h)
            images = [image.crop(crop) for image in images]
        # Same crop for the entire clip. No flips: left/right labels must retain
        # their meaning. SigLIP2's own resize/rescale/normalization follows.
        values = self.processor(images=images, return_tensors="pt").pixel_values
        metadata = dict(id=row["id"], indices=indices, crop=list(crop),
                        timestamps=[decoded[i].time for i in indices],
                        decoded_frames=len(decoded), epoch=self.epoch)
        return values, row["label"], metadata

    def __getitem__(self, index):
        values, label, _ = self.read(index)
        return values, label
