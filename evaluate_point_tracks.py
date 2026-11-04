"""Causal, full-frame-rate RoboTAP evaluation with first-visible queries.

The model sees RGB plus each query's coordinates only when that query arrives.
Ground-truth future tracks/occlusion are used exclusively after predictions for
the official TAP-Vid metrics. Repeated video content stays in the official mean.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor

from point_checkpoint import load_point_checkpoint, file_hash
from robotap_data import iter_videos, video_info
from tapvid_metrics import compute_tapvid_metrics
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from train_video import configure_reproducibility, REPRODUCIBILITY


@torch.no_grad()
def track_video(tracker, video, query_xy, query_frames, processor, device):
    """Return arrays in original query order. No future target argument exists."""
    tracker.eval()
    query_xy = np.asarray(query_xy, dtype=np.float32)
    query_frames = np.asarray(query_frames)
    if (query_xy.ndim != 2 or query_xy.shape[1] != 2 or query_frames.shape != query_xy.shape[:1]
            or not np.issubdtype(query_frames.dtype, np.integer)
            or ((query_frames < 0) | (query_frames >= len(video))).any()):
        raise ValueError("Invalid query coordinates/frame layout")
    count = len(query_frames)
    tracks = np.zeros((count, len(video), 2), dtype=np.float32)
    logits = np.full((count, len(video)), -30., dtype=np.float32)
    state, order, state_bytes = None, [], []
    for t, image in enumerate(video):
        pixels = processor(images=Image.fromarray(image), return_tensors="pt").pixel_values.to(device)
        arriving = np.flatnonzero(query_frames == t)
        new_queries = torch.from_numpy(query_xy[arriving])[None].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output, state = tracker.step(pixels, state, new_queries=new_queries)
        order.extend(arriving.tolist())
        if order:
            if output["points"].shape != (1, len(order), 2) or output["visibility_logits"].shape != (1, len(order)):
                raise ValueError("Tracker output does not match cumulative query arrival order")
            tracks[order, t] = output["points"][0].float().cpu().numpy()
            logits[order, t] = output["visibility_logits"][0].float().cpu().numpy()
        state_bytes.append(dict(encoder=state.encoder_state.nbytes,
                                queries=0 if state.queries is None else state.queries.nbytes))
    if len(order) != count or not np.isfinite(tracks).all() or not np.isfinite(logits).all():
        raise ValueError("Missing query predictions or nonfinite output")
    return tracks, logits, state_bytes


def evaluate_example(tracker, example, processor, device):
    video, points, occluded = (example[key] for key in ("video", "points", "occluded"))
    valid = (~occluded).any(axis=1)
    points, occluded = points[valid], occluded[valid]
    first = (~occluded).argmax(axis=1)
    xy = points[np.arange(len(points)), first]
    predicted, logits, sizes = track_video(tracker, video, xy, first, processor, device)
    # TAP-Vid queries are (t,y,x), while trajectories are (x,y). Metrics use
    # the same 256-pixel reference grid even though model images are 224.
    queries = np.stack((first, xy[:, 1] * 256, xy[:, 0] * 256), axis=-1)[None]
    metrics = compute_tapvid_metrics(queries, occluded[None], points[None] * 256,
                                     (logits < 0)[None], predicted[None] * 256,
                                     query_mode="first")
    metrics = {name:float(value[0]) for name, value in metrics.items()}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite official tracking metric")
    return dict(metrics=metrics, valid_track_indices=np.flatnonzero(valid), query_frames=first,
                predicted_xy=predicted, visibility_logits=logits, state_bytes=sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="data/robotap")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Engineering subset only; never reported as a full benchmark")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("Negative video limit")
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError("Choose an empty evaluation directory")
    output.mkdir(parents=True)
    configure_reproducibility(42)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    tracker, provenance = load_point_checkpoint(args.checkpoint, device)
    training_config = provenance.get("metadata", {}).get("config", {})
    if not args.limit and training_config.get("reproducibility") != REPRODUCIBILITY:
        raise ValueError("Full evaluation requires the current explicit reproducibility protocol")
    training_state = provenance.get("training_state") or {}
    if not args.limit and (training_config.get("development_run", True)
                          or "steps" not in training_config
                          or training_state.get("step") != training_config["steps"]):
        raise ValueError("Full evaluation requires a completed full-data point-training checkpoint; use --limit for engineering checks")
    # Optimizer/RNG tensors are unnecessary for evaluation.
    provenance.pop("training_state", None)
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    rows, started = [], time.monotonic()
    for shard, name, example in iter_videos(root):
        info = video_info(example)
        result = evaluate_example(tracker, example, processor, device)
        filename = hashlib.sha256(name.encode()).hexdigest() + ".npz"
        np.savez_compressed(output / filename, valid_track_indices=result["valid_track_indices"],
                            query_frames=result["query_frames"], predicted_xy=result["predicted_xy"],
                            visibility_logits=result["visibility_logits"])
        row = dict(id=name, shard=shard, video_sha256=info["video_sha256"], frames=info["frames"],
                   tracks=len(result["valid_track_indices"]), metrics=result["metrics"],
                   predictions=filename, predictions_sha256=file_hash(output / filename),
                   max_encoder_state_bytes=max(s["encoder"] for s in result["state_bytes"]),
                   max_query_state_bytes=max(s["queries"] for s in result["state_bytes"]))
        rows.append(row)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(dict(videos=len(rows), id=name, metrics=row["metrics"], seconds=time.monotonic()-started)), flush=True)
        if args.limit and len(rows) >= args.limit:
            break
    if not args.limit and len(rows) != 265:
        raise ValueError("Incomplete official RoboTAP evaluation")
    summary = dict(scope="engineering subset" if args.limit else "full RoboTAP first-visible online evaluation",
        videos=len(rows), metrics={k:float(np.mean([r["metrics"][k] for r in rows])) for k in rows[0]["metrics"]},
        average="unweighted mean over official videos, including duplicate content with distinct annotations",
        precision="FP32 spatial encoder and coordinate outputs; BF16 temporal/readout matrix products",
        evaluation_reproducibility=REPRODUCIBILITY,
        deterministic_pytorch=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        checkpoint=provenance, dataset_extraction_sha256=file_hash(root / "extraction.json"),
        rows=rows, source_sha256={name:file_hash(name) for name in
            ("evaluate_point_tracks.py", "point_readout.py", "point_checkpoint.py", "robotap_data.py", "tapvid_metrics.py", "train_video.py")})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
