"""RoboTAP evaluation data, retaining native frames and normalized xy labels.

Dataset: Vecerik et al., RoboTAP (ICRA 2024), CC BY 4.0.
Official source: https://github.com/google-deepmind/tapnet
This module does not create a training split or fit a tracking readout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("mediapy", "_VideoArray"):
            # Released videos use mediapy's ndarray subclass. Its serialized
            # NumPy state contains the pixels, not container/FPS metadata.
            # Load that state as an ordinary ndarray without installing a
            # visualization dependency or changing the training environment.
            return np.ndarray
        allowed = {
            ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
            ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected data object: {module}.{name}")
        return super().find_class(module, name)


def video_info(example):
    video, points, occluded = (example[key] for key in ("video", "points", "occluded"))
    if (not isinstance(video, np.ndarray) or video.dtype != np.uint8
            or video.ndim != 4 or video.shape[-1] != 3 or min(video.shape) < 1):
        raise ValueError("Expected nonempty uint8 [time,height,width,3] video")
    if (not isinstance(points, np.ndarray) or points.dtype not in (np.float32, np.float64)
            or points.ndim != 3 or points.shape[1:] != (len(video), 2) or not np.isfinite(points).all()):
        raise ValueError("Expected finite xy tracks with matching frame count")
    if (not isinstance(occluded, np.ndarray) or occluded.dtype != np.bool_
            or occluded.shape != points.shape[:2]):
        raise ValueError("Occlusion labels do not match point tracks")
    visible = ~occluded
    valid = visible.any(axis=1)
    if not valid.any():
        raise ValueError("Video contains no visible query points")
    first = visible[valid].argmax(axis=1)
    later = np.arange(len(video))[None] > first[:, None]
    observed_later = later & visible[valid]
    if not observed_later.any():
        raise ValueError("No visible target positions after first-visible queries")
    out_of_bounds = ((points < 0) | (points > 1)).any(axis=-1) & visible
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(video)).cast("B")).hexdigest()
    return dict(frames=len(video), height=video.shape[1], width=video.shape[2],
                tracks=len(points), first_visible_queries=int(valid.sum()),
                visible_positions_after_query=int(observed_later.sum()),
                evaluated_positions_including_occlusions=int(later.sum()),
                query_first_frame=int(first.min()), query_last_frame=int(first.max()),
                visible_out_of_bounds_positions=int(out_of_bounds.sum()),
                visible_xy_min=float(points[visible].min()), visible_xy_max=float(points[visible].max()),
                video_sha256=digest)


def iter_videos(root, verify_hashes=True):
    root = Path(root)
    extraction = json.loads((root / "extraction.json").read_text())
    names = sorted(name for name in extraction if name.endswith(".pkl"))
    if names != [f"robotap_split{i}.pkl" for i in range(5)]:
        raise ValueError("Expected all five official RoboTAP evaluation shards")
    seen = set()
    for name in names:
        file = root / "raw" / name
        if file.stat().st_size != extraction[name]["bytes"]:
            raise ValueError(f"Shard length changed: {name}")
        with file.open("rb") as stream:
            if verify_hashes:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != extraction[name]["sha256"]:
                    raise ValueError(f"Shard content changed: {name}")
                stream.seek(0)
            videos = NumpyUnpickler(stream, encoding="latin1").load()
        if not isinstance(videos, dict):
            raise ValueError("Expected video-name dictionary")
        for video_id, example in videos.items():
            if not isinstance(video_id, str) or video_id in seen:
                raise ValueError("Video identifiers must be strings unique across shards")
            seen.add(video_id)
            yield name, video_id, example
        del videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/robotap")
    parser.add_argument("--output", default="results/robotap_data_audit.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new audit output")
    rows, hashes, duplicates = [], {}, []
    for shard, id_, example in iter_videos(args.root):
        info = video_info(example)
        digest = info["video_sha256"]
        if digest in hashes:
            duplicates.append([hashes[digest], id_])
        hashes[digest] = id_
        rows.append(dict(id=id_, shard=shard, **info))
        if len(rows) % 25 == 0:
            print(json.dumps(dict(event="audit", videos=len(rows))), flush=True)
    if len(rows) != 265:
        raise ValueError(f"Expected 265 official evaluation videos, found {len(rows)}")
    report = dict(scope="RoboTAP data integrity and coverage, not tracking accuracy",
        dataset="RoboTAP", license="CC BY 4.0", source="https://robotap.github.io/",
        videos=len(rows), frames=sum(r["frames"] for r in rows), tracks=sum(r["tracks"] for r in rows),
        frame_count_range=[min(r["frames"] for r in rows), max(r["frames"] for r in rows)],
        visible_out_of_bounds_positions=sum(r["visible_out_of_bounds_positions"] for r in rows),
        duplicate_video_content=duplicates, coordinates="normalized xy; metrics use a 256x256 reference grid",
        use="evaluation only; no labels or videos added to SSv2 training",
        archive=json.loads((Path(args.root) / "download.json").read_text()), rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows", "archive")}), flush=True)


if __name__ == "__main__":
    main()
