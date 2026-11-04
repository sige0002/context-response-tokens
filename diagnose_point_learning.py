"""Memorize real synthetic point labels to check the readout learning path.

One fixed clip, cached frozen encoder features, no validation/model selection.
The resulting training metrics are never benchmark or generalization evidence.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from benchmark_streaming import idle_compute_processes
from kubric_data import KubricPointClips
from point_checkpoint import file_hash, load_point_checkpoint
from point_readout import PointReadout, point_tracking_loss
from tapvid_metrics import compute_tapvid_metrics
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from train_video import parameter_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", required=True, help="Self-contained point checkpoint; only encoder weights are reused")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or args.steps < 1:
        raise ValueError("Choose a new output and positive update budget")
    output.mkdir(parents=True)
    idle_compute_processes()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    torch.manual_seed(67)
    model, provenance = load_point_checkpoint(args.encoder_checkpoint, "cuda")
    provenance.pop("training_state", None)
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    dataset = KubricPointClips("data/point_training_fixture/train.jsonl", processor, frames=8, queries=8, seed=67)
    sample, selection = dataset.read(0)
    batch = {key:value[None].cuda() for key,value in sample.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = model.encoder(batch["video"]).detach()
    torch.manual_seed(67)
    head = PointReadout(**model.readout.config).cuda()
    optimizer = torch.optim.AdamW(parameter_groups(head, 1e-4), lr=3e-4)
    xy, first = sample["query_xy"].numpy(), sample["query_frames"].numpy()
    query = np.stack((first, xy[:, 1] * 256, xy[:, 0] * 256), axis=-1)[None]
    report = dict(scope="Memorization of one fixed synthetic training clip; not recognition accuracy or generalization",
        encoder=provenance, selection=selection, frames=8, points=8, updates=args.steps,
        cached_frozen_features=True, fresh_readout=True, seed=67, trace=[],
        predetermined_gate=dict(relative_loss_below=.05, training_average_jaccard_at_least=.9),
        source_sha256={name:file_hash(name) for name in ("diagnose_point_learning.py", "point_readout.py", "point_checkpoint.py", "kubric_data.py")})

    def record(step):
        head.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = head(features, batch["query_xy"], batch["query_frames"])
            losses = point_tracking_loss(predictions, batch["target_xy"], batch["occluded"], batch["query_frames"])
        positions = predictions["points"].float().cpu().numpy()
        logits = predictions["visibility_logits"].float().cpu().numpy()
        metrics = compute_tapvid_metrics(query, sample["occluded"].numpy()[None],
            sample["target_xy"].numpy()[None] * 256, logits < 0, positions * 256, query_mode="first")
        row = dict(step=step, **{key:float(value) for key,value in losses.items()},
                   metrics={key:float(value[0]) for key,value in metrics.items()})
        report["trace"].append(row)
        (output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        return positions, logits

    record(0)
    for step in range(1, args.steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = 1e-5 + (3e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = head(features, batch["query_xy"], batch["query_frames"])
            losses = point_tracking_loss(predictions, batch["target_xy"], batch["occluded"], batch["query_frames"])
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Nonfinite diagnostic loss")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            positions, logits = record(step)
    initial, final = report["trace"][0], report["trace"][-1]
    report["passed"] = bool(final["loss"] < initial["loss"] * .05 and final["metrics"]["average_jaccard"] >= .9)
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output / "training_predictions.npz", predicted_xy=positions, visibility_logits=logits,
                        target_xy=sample["target_xy"].numpy()[None], occluded=sample["occluded"].numpy()[None],
                        query_xy=xy, query_frames=first)
    torch.save(dict(readout={name:value.detach().cpu() for name,value in head.state_dict().items()},
                    config=head.config, encoder_checkpoint_sha256=provenance["checkpoint_sha256"]), output / "diagnostic_readout.pt")
    if not report["passed"]:
        raise RuntimeError("Fixed-budget memorization gate did not pass; inspect saved evidence")


if __name__ == "__main__":
    main()
