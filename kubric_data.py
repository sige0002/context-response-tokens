"""Lossless point-label storage and clips from CoTracker3's public Kubric data.

Upstream data: facebook/CoTracker3_Kubric, Apache-2.0. The upstream field named
visibility is an occlusion mask: its official loader applies logical_not.
RGB is resized to 224 before lossless PNG storage. All 120 frames and all
32,768 float32 trajectories are retained; no point-label quantization is used.
"""
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def save_points(path, points, occluded):
    if (points.ndim != 3 or points.shape[-1] != 2 or points.dtype != np.float32
            or occluded.shape != points.shape[:2] or occluded.dtype != np.bool_):
        raise ValueError("Expected float32 [points,time,xy] and bool occlusion")
    if ((~np.isfinite(points).all(axis=-1)) & ~occluded).any():
        raise ValueError("Nonfinite coordinate at an upstream-visible point")
    # The pinned source contains +/-inf behind occlusion (e.g. scene 0488,
    # track 19966, frame 117). Preserve those exact bytes. They must never be
    # admitted as query coordinates or as visible position supervision.
    # Byte planes improve DEFLATE compression. This is a reversible permutation
    # of the original float32 bytes, not a numerical transform or quantization.
    little = np.ascontiguousarray(points, dtype="<f4")
    planes = little.view(np.uint8).reshape(*little.shape, 4).transpose(3, 0, 1, 2)
    np.savez_compressed(path, format_version=np.array(1), points_bytes=planes, occluded=occluded)


def load_points(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1:
            raise ValueError("Unsupported point-label format")
        raw, occluded = data["points_bytes"], data["occluded"]
    if (raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[0] != 4 or raw.shape[-1] != 2
            or occluded.dtype != np.bool_ or occluded.shape != raw.shape[1:3]):
        raise ValueError("Invalid stored point-label layout")
    points = raw.transpose(1, 2, 3, 0).copy().view("<f4").reshape(*raw.shape[1:])
    if ((~np.isfinite(points).all(axis=-1)) & ~occluded).any():
        raise ValueError("Nonfinite coordinate at an upstream-visible point")
    return points, occluded


class KubricPointClips(Dataset):
    def __init__(self, manifest, processor, frames=24, queries=64, training=False, seed=42):
        self.path = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        if not self.rows or frames < 2 or queries < 1:
            raise ValueError("Invalid point-clip dataset configuration")
        if len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Repeated scene identifiers")
        self.processor, self.frames, self.queries = processor, frames, queries
        self.training, self.seed, self.epoch = training, seed, 0
        self.groups = {r["id"] for r in self.rows}
        for row in self.rows:
            if row["frames"] < frames:
                raise ValueError("Requested clip is longer than the source video")
            row["directory"] = (self.path.parent / row["directory"]).resolve()
            for name in ("images.zip", "points.npz"):
                if (row["directory"] / name).stat().st_size != row["files"][name]["bytes"]:
                    raise ValueError("Prepared file size changed")

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Negative epoch")
        self.epoch = epoch

    def read(self, index):
        row = self.rows[index]
        epoch = self.epoch if self.training else 0
        rng = random.Random(f"{self.seed}:{epoch}:{row['id']}")
        if self.training:
            stride = rng.randint(1, min(4, (row["frames"] - 1) // (self.frames - 1)))
            start = rng.randint(0, row["frames"] - 1 - (self.frames - 1) * stride)
            indices = start + np.arange(self.frames) * stride
        else:
            indices = np.linspace(0, row["frames"] - 1, self.frames).astype(int)
        points, occluded = load_points(row["directory"] / "points.npz")
        points, occluded = points[:, indices].copy(), occluded[:, indices].copy()
        if points.shape[1] != self.frames:
            raise ValueError("Annotation frame count changed")
        height, width = row["native_height"], row["native_width"]
        # Match the reference loader's treatment of points outside the image.
        occluded |= (points[..., 0] < 0) | (points[..., 0] > width - 1)
        occluded |= (points[..., 1] < 0) | (points[..., 1] > height - 1)
        first = (~occluded).argmax(axis=1)
        valid = (~occluded).any(axis=1) & (first < self.frames - 1)
        candidates = np.flatnonzero(valid).tolist()
        if len(candidates) < self.queries:
            raise ValueError(f"Too few visible query tracks in scene {row['id']}")
        selected = np.array(rng.sample(candidates, self.queries))
        points = points[selected] / np.array([width, height], dtype=np.float32)
        occluded, first = occluded[selected], first[selected]
        xy = points[np.arange(self.queries), first]
        with zipfile.ZipFile(row["directory"] / "images.zip") as archive:
            images = [Image.open(io.BytesIO(archive.read(f"{i:03d}.png"))).convert("RGB") for i in indices]
        pixels = self.processor(images=images, return_tensors="pt").pixel_values
        sample = dict(video=pixels, query_xy=torch.from_numpy(xy), query_frames=torch.from_numpy(first),
                      target_xy=torch.from_numpy(points), occluded=torch.from_numpy(occluded))
        metadata = dict(id=row["id"], indices=indices.tolist(), track_indices=selected.tolist(), epoch=epoch)
        return sample, metadata

    def __getitem__(self, index):
        return self.read(index)[0]
