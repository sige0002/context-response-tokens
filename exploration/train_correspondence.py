"""Train a descriptor metric from Kubric correspondence labels on frozen SigLIP.

Small mechanism experiment, not full benchmark training or feature distillation.
Validation uses distinct scenes and the final fixed-step checkpoint only.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from exploration.appearance_routing import AppearanceMetric
from exploration.oracle_transport import basis, digest, sample


def soft_target_loss(logits, target_xy, grid=14):
    indices, weights = basis(target_xy, grid)
    return -(logits.log_softmax(-1).gather(-1, indices)*weights).sum(-1).mean()


def predict_xy(logits, grid=14):
    index = logits.argmax(-1)
    return torch.stack(((index % grid+.5)/grid, (index//grid+.5)/grid), -1)


@torch.no_grad()
def evaluate(models, cache, scenes, seed, queries):
    rows = []
    for scene in scenes:
        data = cache[scene]
        candidates = torch.where(data["visible"][:, 0])[0]
        if len(candidates) < queries:
            raise ValueError("Too few initially visible validation queries")
        generator = torch.Generator().manual_seed(seed+int(scene))
        selected = candidates[torch.randperm(len(candidates), generator=generator)[:queries]]
        for depth, model in models.items():
            features = data["features"][depth]
            reference = sample(features[0], data["points"][selected, 0], 14)
            for mode in ("frozen_cosine", "learned_metric"):
                totals = {name:dict(points=0, error_sum=0., below_4=0, below_8=0, below_16=0)
                          for name in ("all_visible", "continuous", "after_occlusion")}
                continuous = torch.ones(len(selected), dtype=torch.bool)
                for frame in range(1, len(features)):
                    visible = data["visible"][selected, frame]
                    continuous &= visible
                    if mode == "frozen_cosine":
                        logits = F.normalize(reference, dim=-1) @ F.normalize(features[frame], dim=-1).T
                    else:
                        logits = model(reference, features[frame])
                    predicted = predict_xy(logits)
                    distance = torch.linalg.vector_norm((predicted-data["points"][selected, frame])*256, dim=-1)
                    masks = dict(all_visible=visible, continuous=continuous,
                                 after_occlusion=visible & ~continuous)
                    for name, mask in masks.items():
                        values = distance[mask]
                        totals[name]["points"] += len(values)
                        totals[name]["error_sum"] += float(values.sum())
                        for threshold in (4, 8, 16):
                            totals[name][f"below_{threshold}"] += int((values < threshold).sum())
                for stratum, values in totals.items():
                    count = values["points"]
                    rows.append(dict(scene=scene, depth=int(depth), method=mode, stratum=stratum, **values,
                        mean_error=values["error_sum"]/count if count else None,
                        **{f"pck_{q}":values[f"below_{q}"]/count if count else None for q in (4, 8, 16)}))
    summary = []
    for depth in models:
        for mode in ("frozen_cosine", "learned_metric"):
            for stratum in ("all_visible", "continuous", "after_occlusion"):
                values = [r for r in rows if (r["depth"], r["method"], r["stratum"]) == (int(depth), mode, stratum) and r["points"]]
                summary.append(dict(depth=int(depth), method=mode, stratum=stratum, scenes=len(values),
                    points=sum(r["points"] for r in values),
                    **{key:float(np.mean([r[key] for r in values])) if values else None
                       for key in ("mean_error", "pck_4", "pck_8", "pck_16")}))
    return dict(rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Run an unchanged prewritten plan; no mid-training resume")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    directory, output = Path(args.cache).resolve(), Path(args.output).resolve()
    cache_plan = json.loads((directory/"plan.json").read_text())
    config = dict(seed=17, steps=600, batch_scenes=4, queries=64, evaluation_queries=256,
        lr=1e-3, weight_decay=1e-4, hidden=128, width=64, grid=14,
        train=cache_plan["config"]["train"], val=cache_plan["config"]["val"], depths=cache_plan["config"]["depths"],
        objective="Bilinear distribution over true target cells; known visible point pairs only. No teacher-feature target.",
        evaluation="Frame-zero query location given; later GT only scores; no GT correspondence as model input. Argmax cell center in normalized 256 coordinates; visibility given for scoring, not predicted.",
        limits="24 training and 8 validation scenes, one seed, 200 updates per depth. Pairwise metric only; no Mamba or memory updates trained. Reappearance covers sampled frames only. Not tracking AJ, SOTA, or full-encoder gain.")
    plan = dict(cache=str(directory), cache_plan_sha256=digest(directory/"plan.json"), config=config,
        sources={str(p.resolve()):digest(p) for p in (Path(__file__), Path("exploration/appearance_routing.py"), Path("exploration/oracle_transport.py"))})
    if args.resume:
        if json.loads((output/"plan.json").read_text()) != plan or (output/"training.jsonl").exists():
            raise ValueError("Changed plan/source or already-started training")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.plan_only:
        print("Fixed 600-update CPU correspondence protocol saved", flush=True)
        return
    cache_summary = json.loads((directory/"summary.json").read_text())
    if cache_summary["plan_sha256"] != plan["cache_plan_sha256"]:
        raise ValueError("Cache protocol changed")
    for name, value in cache_plan["sources"].items():
        if digest(name) != value:
            raise ValueError("Cache extraction source changed")
    cache = {}
    for scene in config["train"]+config["val"]:
        path = directory/(scene+".pt")
        if digest(path) != cache_summary["scenes"][scene]["sha256"]:
            raise ValueError("Cache tensor changed")
        cache[scene] = torch.load(path, map_location="cpu", weights_only=True)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    models = nn.ModuleDict({str(depth):AppearanceMetric(768, config["hidden"], config["width"]) for depth in config["depths"]})
    optimizer = torch.optim.AdamW(models.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    initial = evaluate(models, cache, config["val"], config["seed"], config["evaluation_queries"])
    (output/"initial.json").write_text(json.dumps(initial, indent=2)+"\n")
    started = time.monotonic()
    for step in range(config["steps"]):
        depth = str(config["depths"][step % len(config["depths"])])
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for scene in rng.sample(config["train"], config["batch_scenes"]):
            data = cache[scene]
            for _ in range(100):
                a, b = sorted(rng.sample(range(len(data["frames"])), 2))
                candidates = torch.where(data["visible"][:, a] & data["visible"][:, b])[0]
                if len(candidates) >= config["queries"]:
                    break
            else:
                raise ValueError("No usable training pair")
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:config["queries"]]]
            reference = sample(data["features"][depth][a], data["points"][chosen, a], config["grid"])
            logits = models[depth](reference, data["features"][depth][b])
            losses.append(soft_target_loss(logits, data["points"][chosen, b], config["grid"]))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite correspondence objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 1.)
        optimizer.step()
        if (step+1) % 30 == 0:
            row = dict(step=step+1, depth=int(depth), loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started)
            with (output/"training.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
    torch.save(dict(models=models.state_dict(), config=config), output/"metric.pt")
    final = evaluate(models, cache, config["val"], config["seed"], config["evaluation_queries"])
    final.update(plan_sha256=digest(output/"plan.json"), cache_summary_sha256=digest(directory/"summary.json"),
        checkpoint_sha256=digest(output/"metric.pt"), elapsed_seconds=time.monotonic()-started)
    (output/"final.json").write_text(json.dumps(final, indent=2)+"\n")
    print("Completed fixed-budget correspondence experiment", flush=True)


if __name__ == "__main__":
    main()
