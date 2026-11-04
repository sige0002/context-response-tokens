"""Isolate optimization of the pilot's frozen SigLIP2/current-frame readout.

Same data, pretrained features and mean-pool linear head. A separate diagnostic
uses full-batch AdamW for 300 updates instead of the pilot's 3 short epochs.
These conditions differ; this is not a temporal-model comparison or SOTA score.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor, SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from train_temporal import FrameClips


def main():
    torch.set_num_threads(4)
    output = Path("results/ssv2_implementation_audit")
    cache = output / "frozen_current_features.pt"
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
        backbone = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION).cuda().eval()
        data = {}
        for split in ("train", "val"):
            ds = FrameClips(f"data/ssv2/pilot/{split}.jsonl", processor, 1, 174)
            features, labels = [], []
            with torch.no_grad():
                for i, (images, target) in enumerate(DataLoader(ds, batch_size=16)):
                    values = backbone(images[:, 0].cuda()).last_hidden_state.mean(1)
                    features.append(values.cpu())
                    labels.append(target)
                    if i % 50 == 0:
                        print("extract", split, i, flush=True)
            data[split] = (torch.cat(features), torch.cat(labels))
        data["revision"] = DEFAULT_REVISION
        torch.save(data, cache)
        del backbone
        torch.cuda.empty_cache()
    x, y = data["train"]
    vx, vy = data["val"]
    torch.manual_seed(43)
    head = nn.Linear(x.shape[1], 174)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    trace = []
    for step in range(301):
        logits = head(x)
        loss = nn.functional.cross_entropy(logits, y)
        if step % 50 == 0:
            with torch.no_grad():
                val_logits = head(vx)
                row = dict(updates=step, train_loss=float(loss.detach()),
                           train_accuracy=float((logits.argmax(-1) == y).float().mean()),
                           val_loss=float(nn.functional.cross_entropy(val_logits, vy)),
                           val_accuracy=float((val_logits.argmax(-1) == vy).float().mean()))
                trace.append(row)
                print(json.dumps(row), flush=True)
        if step == 300:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
    record = dict(scope="Optimization diagnostic, not a matched temporal-model benchmark",
                  model=DEFAULT_MODEL, revision=DEFAULT_REVISION,
                  feature="current-frame last_hidden_state.mean(patches), frozen, float32",
                  train_samples=len(y), val_samples=len(vy), classes=174,
                  optimizer="AdamW", lr=0.01, weight_decay=0.01, full_batch=True,
                  predefined_updates=300, selected_by_validation=False, trace=trace)
    (output / "readout_optimization.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
