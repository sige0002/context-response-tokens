"""Audit the real SSv2 pilot; counterfactual memorization is NOT a benchmark.

Does not modify the experiment, checkpoints, optimizer, or dataset. The last
test trains a separate binary classifier on paired real clips with identical
current images and reversed histories. Success tests the temporal learning
path, not semantic recognition or generalization.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor, SiglipVisionModel

from temporal_siglip import TemporalSiglip, DEFAULT_MODEL, DEFAULT_REVISION
from train_temporal import FrameClips, VideoClassifier, run_epoch


def data_audit(root):
    labels = json.loads((root / "labels/labels.json").read_text())
    result, sets = {}, {}
    for split, official in (("train", "train"), ("val", "validation")):
        annotations = {r["id"]: r for r in json.loads((root / f"labels/{official}.json").read_text())}
        rows = [json.loads(s) for s in (root / f"pilot/{split}.jsonl").read_text().splitlines()]
        errors, identical, short = [], [], []
        counts, paths, hashes = Counter(), set(), set()
        for row in rows:
            expected = int(labels[annotations[row["id"]]["template"].replace("[", "").replace("]", "")])
            counts[row["label"]] += 1
            if row["label"] != expected:
                errors.append([row["id"], "label"])
            files = [root / "pilot" / f for f in row["frames"]]
            if any(f.parent.name != row["id"] for f in files):
                errors.append([row["id"], "video_path"])
            for key in ("indices", "timestamps"):
                if not all(a <= b for a, b in zip(row[key], row[key][1:])):
                    errors.append([row["id"], key])
            contents = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
            if len(set(contents)) == 1:
                identical.append(row["id"])
            if len(set(row["indices"])) < len(row["frames"]):
                short.append(row["id"])
            paths.update(map(str, files))
            hashes.update(contents)
        result[split] = dict(videos=len(rows), classes=len(counts),
                             per_class_counts=sorted(set(counts.values())),
                             official_label_or_order_errors=errors,
                             identical_frames=identical, short_clips=short,
                             unique_frame_files=len(paths), unique_frame_contents=len(hashes))
        sets[split] = (set(r["id"] for r in rows), paths, hashes)
        assert not errors
    result["cross_split_overlap"] = {
        key: len(sets["train"][i] & sets["val"][i])
        for i, key in enumerate(("id", "path", "frame_sha256"))}
    assert not any(result["cross_split_overlap"].values())
    return result


def norm(tensor):
    return float(tensor.detach().float().norm()) if tensor is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/ssv2_pilot_mamba_interleaved_seed42")
    parser.add_argument("--output", default="results/ssv2_implementation_audit")
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "implementation and optimization diagnostics, not a generalization benchmark"}

    def record(key, value):
        report[key] = value
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(key, json.dumps(value), flush=True)

    record("data", data_audit(Path("data/ssv2")))
    config = json.loads((run / "config.json").read_text())
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    train = FrameClips(config["train_path"], processor, 6, 174)
    val = FrameClips(config["val_path"], processor, 6, 174)
    # Independent PIL/NumPy preprocessing, instead of comparing two calls to
    # the same Transformers processor.
    with Image.open(train.rows[0]["frames"][0]) as im:
        rgb = im.convert("RGB")
        resized = rgb.resize((224, 224), resample=Image.Resampling.BILINEAR)
        independent = (np.asarray(resized).astype(np.float32) / 255.0 - 0.5) / 0.5
        expected = torch.from_numpy(independent).permute(2, 0, 1)
        actual = processor(images=rgb, return_tensors="pt").pixel_values[0]
    preprocessing_error = float((expected - actual).abs().max())
    assert preprocessing_error < 1e-6
    record("preprocessing", dict(max_abs_difference=preprocessing_error,
                                  minimum=float(actual.min()), maximum=float(actual.max()),
                                  size=list(actual.shape), processor=processor.to_dict()))

    encoder = TemporalSiglip.load_checkpoint(run / "best_encoder.pt", device="cuda")
    head = torch.load(run / "best_head.pt", map_location="cpu", weights_only=True)
    model = VideoClassifier(encoder, 174).cuda().eval()
    model.head.load_state_dict(head["head"])
    original = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    trained_backbone = encoder.backbone.state_dict()
    changed = [k for k, v in original.state_dict().items()
               if not torch.equal(v, trained_backbone[k].cpu())]
    assert not changed
    record("pretrained_backbone", dict(revision=DEFAULT_REVISION,
           tensors=len(trained_backbone), changed_tensors=changed,
           all_frozen=all(not p.requires_grad for p in encoder.backbone.parameters())))
    del trained_backbone
    chosen = []
    seen = set()
    for i, row in enumerate(train.rows):
        if row["label"] not in seen:
            chosen.append(i)
            seen.add(row["label"])
        if len(chosen) == 4:
            break
    clips = torch.stack([train[i][0] for i in chosen]).cuda()
    labels = torch.tensor([train.rows[i]["label"] for i in chosen], device="cuda")
    saved_gains = {k: a.gain.detach().clone() for k, a in encoder.adapters.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a in encoder.adapters.values():
            a.gain.zero_()
        ours = encoder(clips[:2])
        with torch.autocast("cuda", enabled=False):
            reference = encoder.backbone(clips[:2].flatten(0, 1)).last_hidden_state.reshape_as(ours)
        identity_error = float((ours - reference).abs().max())
        for k, a in encoder.adapters.items():
            a.gain.copy_(saved_gains[k])
        batched = encoder(clips[:2])
        state, outputs = None, []
        for frame in clips[:2].unbind(1):
            y, state = encoder.step(frame, state)
            outputs.append(y)
        streamed = torch.stack(outputs, 1)
        stream_difference = (streamed - batched).float()
        stream_logits = model.head(streamed[:, -1].mean(1)).float()
        batch_logits = model.head(batched[:, -1].mean(1)).float()
        altered = clips[:2].clone()
        altered[:, -1] *= -1
        future_error = float((encoder(altered)[:, :-1] - batched[:, :-1]).abs().max())
    assert identity_error < 1e-5 and future_error < 1e-5
    record("real_clip_forward", dict(ids=[train.rows[i]["id"] for i in chosen[:2]],
           identity_max_abs=identity_error, future_leak_max_abs=future_error,
           batch_stream_rmse=float(stream_difference.square().mean().sqrt()),
           batch_stream_max_abs=float(stream_difference.abs().max()),
           batch_stream_logit_max_abs=float((stream_logits - batch_logits).abs().max()),
           batch_stream_prediction_equal=bool(torch.equal(stream_logits.argmax(-1), batch_logits.argmax(-1)))))
    del ours, reference, batched, streamed, stream_difference, state, outputs, altered

    model.train()
    differentiable = clips[:2].detach().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(differentiable).float(), labels[:2])
    loss.backward()
    grad_report = {}
    for k, a in encoder.adapters.items():
        grad_report[k] = dict(gain_mean_abs=float(a.gain.detach().abs().mean()),
            gain_max_abs=float(a.gain.detach().abs().max()),
            gain_gradient=norm(a.gain.grad), down_gradient=norm(a.down.weight.grad),
            up_gradient=norm(a.up.weight.grad), mixer_gradient=norm(a.mixer.in_proj.weight.grad),
            all_gradients_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in a.parameters()))
        assert grad_report[k]["mixer_gradient"] > 0 and grad_report[k]["all_gradients_finite"]
    pixel_grads = differentiable.grad.detach().float().square().sum((0, 2, 3, 4)).sqrt().tolist()
    assert sum(pixel_grads[:-1]) > 0
    record("trained_gradients", dict(adapters=grad_report, input_gradient_norm_per_frame=pixel_grads,
           frozen_backbone_has_grad=any(p.grad is not None for p in encoder.backbone.parameters())))
    model.zero_grad(set_to_none=True)
    del differentiable, loss

    # Independent counting and comparison to saved per-video predictions.
    model.eval()
    predictions, targets = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for pixels, target in DataLoader(val, batch_size=4):
            features = encoder(pixels.cuda())[:, -1].mean(1)
            logits = torch.nn.functional.linear(features, model.head.weight, model.head.bias)
            predictions.extend(logits.argmax(-1).cpu().tolist())
            targets.extend(target.tolist())
    saved = json.loads((run / "history.json").read_text())[head["epoch"]]["validation"]["full"]
    correct = sum(a == b for a, b in zip(predictions, targets))
    record("validation_recount", dict(epoch=head["epoch"] + 1, correct=correct, total=len(targets),
           accuracy=correct / len(targets), saved_accuracy=saved["accuracy"],
           changed_predictions=sum(a != b for a, b in zip(predictions, saved["predictions"]))))
    assert predictions == saved["predictions"]
    del model, encoder, saved_gains
    gc.collect()
    torch.cuda.empty_cache()

    # Same current image within every pair; labels describe artificial ordering,
    # not the SSv2 semantic classes. A deterministic current-only model is capped
    # at 50% on this balanced set. No held-out accuracy is claimed.
    paired = torch.stack([v for clip in clips for v in
                          (clip, torch.cat((clip[:-1].flip(0), clip[-1:])))]).detach()
    order_labels = torch.tensor([0, 1] * len(clips), device="cuda")
    assert torch.equal(paired[0::2, -1], paired[1::2, -1])
    torch.manual_seed(123)
    fresh_encoder = TemporalSiglip(original, kind="mamba", every=4, width=256, state_size=64)
    fresh = VideoClassifier(fresh_encoder, 2).cuda()
    optimizer = torch.optim.AdamW((p for p in fresh.parameters() if p.requires_grad), lr=1e-3)
    trace, updates = [], 0
    while updates < args.steps:
        order = torch.randperm(len(paired), device="cuda")
        batches = [(paired[idx], order_labels[idx]) for idx in order.split(4)]
        run_epoch(fresh, batches, torch.device("cuda"), optimizer)
        updates += len(batches)
        if updates % 10 == 0 or updates >= args.steps:
            metrics = run_epoch(fresh, [(paired[:4], order_labels[:4]),
                                       (paired[4:], order_labels[4:])], torch.device("cuda"))
            trace.append(dict(updates=updates, **metrics))
            print("counterfactual", json.dumps(trace[-1]), flush=True)
            if metrics["accuracy"] == 1 and metrics["loss"] < 0.05:
                break
    current_only = paired[:, -1:].expand_as(paired).contiguous()
    current = run_epoch(fresh, [(current_only[:4], order_labels[:4]),
                               (current_only[4:], order_labels[4:])], torch.device("cuda"))
    record("counterfactual_overfit", dict(source_ids=[train.rows[i]["id"] for i in chosen],
           clips=len(paired), classes=2, frames=6, optimizer="AdamW", learning_rate=1e-3,
           fresh_adapters=True, pretrained_frozen_backbone=True, artificial_order_labels=True,
           updates=updates, trace=trace, current_only=current,
           interpretation="Memorization of paired clips; not recognition accuracy or generalization."))
    print("diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
