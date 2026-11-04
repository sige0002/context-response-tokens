"""Full RoboTAP encoder-state interventions, conditional on common query anchors.

The normal branch constructs descriptors at first-visible query arrival. Reset
branches share those descriptors, clear only their encoder memory periodically,
and preserve absolute frame counters. This is a mechanism diagnostic, not a
standalone runtime comparison or a completely history-free baseline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import SiglipImageProcessor

from point_checkpoint import file_hash, load_point_checkpoint
from point_experiment import write_json
from point_history import track_history_interventions
from robotap_data import iter_videos, video_info
from summarize_points import compare_reports
from tapvid_metrics import compute_tapvid_metrics
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from train_video import configure_reproducibility, REPRODUCIBILITY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="data/robotap")
    parser.add_argument("--output", required=True)
    parser.add_argument("--periods", nargs="+", type=int, default=[1,24])
    parser.add_argument("--limit", type=int, default=0, help="Explicit engineering subset only")
    args = parser.parse_args()
    if args.limit < 0 or min(args.periods) < 1 or len(set(args.periods)) != len(args.periods):
        parser.error("Invalid limit or repeated/nonpositive reset periods")
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    configure_reproducibility(42)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    tracker, provenance = load_point_checkpoint(args.checkpoint, device)
    config = provenance.get("metadata", {}).get("config", {})
    if not args.limit and config.get("reproducibility") != REPRODUCIBILITY:
        raise ValueError("Full interventions require the current explicit reproducibility protocol")
    state = provenance.get("training_state") or {}
    if not args.limit and (config.get("development_run", True) or "steps" not in config or state.get("step") != config["steps"]):
        raise ValueError("Full interventions require a completed full-data point checkpoint; use --limit for engineering checks")
    provenance.pop("training_state", None)
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    conditions = ["full"] + [f"reset_{period}" for period in args.periods]
    rows = {name:[] for name in conditions}
    sources = {name:file_hash(name) for name in ("evaluate_point_history.py", "point_history.py", "point_checkpoint.py",
        "point_readout.py", "temporal_siglip.py", "gmmix_siglip.py", "robotap_data.py", "tapvid_metrics.py", "summarize_points.py", "paired_metrics.py", "train_video.py")}
    intervention = dict(query_anchors="Common full-history descriptors created only at query arrival",
        frame_numbers="Absolute counters preserved in all branches",
        reset_schedule="Clear encoder cache before each frame t>0 divisible by period; query descriptors persist",
        scope="Conditional ongoing-feature history intervention; not wholly history-free and not a speed benchmark")
    write_json(output / "plan.json", dict(checkpoint_sha256=provenance["checkpoint_sha256"], periods=args.periods,
        limit=args.limit, intervention=intervention, dataset_extraction_sha256=file_hash(root/"extraction.json"),
        source_sha256=sources))
    for name in conditions:
        (output/name).mkdir()
    started = time.monotonic()
    for shard, name, example in iter_videos(root):
        info = video_info(example)
        indices = np.flatnonzero((~example["occluded"]).any(axis=1))
        target, occluded = example["points"][indices], example["occluded"][indices]
        first = (~occluded).argmax(axis=1)
        xy = target[np.arange(len(indices)),first]
        results = track_history_interventions(tracker,example["video"],xy,first,processor,device,args.periods)
        query_points = np.stack((first,xy[:,1]*256,xy[:,0]*256),-1)[None]
        filename = hashlib.sha256(name.encode()).hexdigest()+".npz"
        for condition, values in results.items():
            predicted, logits = values["predicted_xy"], values["visibility_logits"]
            metrics = {key:float(value[0]) for key,value in compute_tapvid_metrics(query_points,
                occluded[None], target[None]*256, (logits<0)[None], predicted[None]*256,query_mode="first").items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite intervention metrics")
            destination = output/condition/filename
            np.savez_compressed(destination,valid_track_indices=indices,query_frames=first,predicted_xy=predicted,visibility_logits=logits)
            row = dict(id=name,shard=shard,video_sha256=info["video_sha256"],frames=info["frames"],tracks=len(indices),
                metrics=metrics,predictions=filename,predictions_sha256=file_hash(destination),
                max_encoder_state_bytes=max(v["encoder"] for v in values["state_bytes"]),
                max_query_state_bytes=max(v["queries"] for v in values["state_bytes"]),reset_frames=values["resets"])
            rows[condition].append(row)
            with (output/condition/"progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
        print(json.dumps(dict(videos=len(rows["full"]),id=name,seconds=time.monotonic()-started,
            average_jaccard={key:rows[key][-1]["metrics"]["average_jaccard"] for key in conditions})),flush=True)
        if args.limit and len(rows["full"])>=args.limit:
            break
    count = len(rows["full"])
    if not args.limit and count!=265:
        raise ValueError("Incomplete full RoboTAP intervention")
    reports = {}
    for condition in conditions:
        report = dict(scope="engineering subset" if args.limit else "full RoboTAP first-visible online evaluation",
            videos=count,metrics={key:float(np.mean([r["metrics"][key] for r in rows[condition]])) for key in rows[condition][0]["metrics"]},
            checkpoint=provenance,dataset_extraction_sha256=file_hash(root/"extraction.json"),rows=rows[condition],
            source_sha256=sources,history_intervention=dict(intervention,condition=condition),
            evaluation_reproducibility=REPRODUCIBILITY,
            precision="FP32 spatial encoder and coordinates; BF16 temporal/readout matrix products")
        write_json(output/condition/"summary.json",report)
        reports[condition] = report
    differences = None if args.limit else {name:compare_reports(reports["full"],reports[name]) for name in conditions if name!="full"}
    write_json(output/"summary.json",dict(scope="Engineering only" if args.limit else "Full paired encoder-state intervention",
        videos=count,intervention=intervention,checkpoint_sha256=provenance["checkpoint_sha256"],
        conditions={name:dict(report=str(output/name/"summary.json"),report_sha256=file_hash(output/name/"summary.json"),
                    metrics=reports[name]["metrics"]) for name in conditions},full_minus_reset=differences,
        limits="Shared query anchors retain history at arrival. Resets shift feature distributions. Positive gaps do not establish semantic causality or long-term retention guarantees."))


if __name__ == "__main__":
    main()
