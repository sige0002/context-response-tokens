"""Matched, small-budget end-to-end point-head/adapters pilot on real RGB.

SigLIP 2 is frozen. Both arms use identical learned metric initialization,
trainable parameters, clips, optimizer and head; only routing changes. This is
exploration on Kubric rendered validation scenes, not the main benchmark.
"""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zipfile

from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from point_readout import PointReadout, PointTracker, point_tracking_loss
from exploration.routed_siglip import RoutedSiglip
from exploration.oracle_transport import digest


def trainable_digest(model):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value.update(name.encode()); value.update(parameter.detach().cpu().float().numpy().tobytes())
    return value.hexdigest()


def build_model(config, metric):
    torch.manual_seed(config["seed"])
    backbone = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    encoder = RoutedSiglip(backbone, routing=config["routing"])
    for key, adapter in encoder.adapters.items():
        prefix = str(int(key)+1)+"."
        adapter.metric.load_state_dict({k.removeprefix(prefix):v for k,v in metric.items() if k.startswith(prefix)}, strict=True)
    return PointTracker(encoder, PointReadout(), freeze_encoder=False).cuda()


@torch.no_grad()
def evaluate(model, cache, scenes, config):
    model.eval()
    rows = []
    for scene in scenes:
        data = cache[scene]
        candidates = torch.where(data["visible"][:, 0])[0]
        generator = torch.Generator().manual_seed(config["seed"]+int(scene))
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:config["evaluation_queries"]]]
        if len(chosen) != config["evaluation_queries"]:
            raise ValueError("Too few evaluation query points")
        query = data["points"][chosen, 0].cuda()[None]
        outputs, logits, state, sizes = [], [], None, []
        for frame in range(len(data["pixels"])):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result, state = model.step(data["pixels"][frame:frame+1].cuda(), state,
                                           new_queries=query if frame == 0 else None)
            outputs.append(result["points"][0].float().cpu())
            logits.append(result["visibility_logits"][0].float().cpu())
            sizes.append(state.nbytes)
        if len(set(sizes)) != 1:
            raise ValueError("Fixed query set has growing inference state")
        prediction = torch.stack(outputs, 1)
        visibility_prediction = torch.stack(logits, 1) > 0
        truth, visible = data["points"][chosen], data["visible"][chosen]
        continuous = visible.long().cumprod(1).bool()
        moving = torch.linalg.vector_norm((truth-truth[:, :1])*256, dim=-1) > 2.
        post = torch.arange(len(data["pixels"]))[None] > 0
        for method, value in (("model", prediction), ("static_query", truth[:, :1].expand_as(truth))):
            distance = torch.linalg.vector_norm((value-truth)*256, dim=-1)
            for stratum, mask in dict(all_visible=visible & post, moving=visible & post & moving,
                                      after_occlusion=visible & post & ~continuous).items():
                errors = distance[mask]
                n = len(errors)
                rows.append(dict(scene=scene, method=method, stratum=stratum, points=n,
                    mean_error=float(errors.mean()) if n else None,
                    **{f"pck_{q}":float((errors < q).float().mean()) if n else None for q in (4, 8, 16)},
                    visibility_accuracy=float((visibility_prediction[:, 1:] == visible[:, 1:]).float().mean()) if method == "model" else None,
                    state_bytes=sizes[-1] if method == "model" else None))
    summary = []
    for method in ("model", "static_query"):
        for stratum in ("all_visible", "moving", "after_occlusion"):
            selected = [r for r in rows if r["method"] == method and r["stratum"] == stratum and r["points"]]
            summary.append(dict(method=method, stratum=stratum, scenes=len(selected), points=sum(r["points"] for r in selected),
                **{k:sum(r[k] for r in selected)/len(selected) if selected else None for k in ("mean_error", "pck_4", "pck_8", "pck_16")}))
    return dict(rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Execute an unchanged plan; not a partial-training resume")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    run, output = Path(args.metric_run).resolve(), Path(args.output).resolve()
    metric_plan = json.loads((run/"plan.json").read_text())
    metric_final = json.loads((run/"final.json").read_text())
    if digest(run/"metric.pt") != metric_final["checkpoint_sha256"]:
        raise ValueError("Metric checkpoint changed")
    directory = Path(metric_plan["cache"])
    cache_plan = json.loads((directory/"plan.json").read_text())
    config = dict(seed=17, arms=["coordinate", "appearance"], steps_per_arm=256, batch=1,
        clip_frames=4, queries=32, evaluation_queries=128, lr=1e-4, weight_decay=1e-4,
        train=metric_plan["config"]["train"], val=metric_plan["config"]["val"],
        backbone=DEFAULT_MODEL, revision=DEFAULT_REVISION, precision="FP32 spatial / BF16 temporal and readout",
        scope="Matched short-budget pilot only; interleaved trainable adapters and common full point head. No backbone fine-tuning or distillation. No score-based stopping or winner selection.",
        limits="24 train / 8 validation rendered scenes, one seed, 256 updates each. Four-frame clips versus 24-frame streaming evaluation. Metric validation scenes have already been inspected in exploration. Neither convergence nor SOTA evidence; no AJ or unseen real-video claim.")
    source_files = [Path(__file__), Path("exploration/routed_siglip.py"), Path("exploration/appearance_routing.py"),
        Path("temporal_siglip.py"), Path("point_readout.py"), Path("exploration/oracle_transport.py"), Path("pyproject.toml"), Path("uv.lock")]
    plan = dict(config=config, metric_run=str(run), metric_final_sha256=digest(run/"final.json"),
        cache_summary_sha256=digest(directory/"summary.json"),
        sources={str(p.resolve()):digest(p) for p in source_files})
    if args.resume:
        if json.loads((output/"plan.json").read_text()) != plan or (output/"training.jsonl").exists():
            raise ValueError("Changed plan/source or already-started pilot")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.plan_only:
        print("Two-arm 256-update pilot plan fixed", flush=True)
        return
    # Acquire the CUDA context before staging CPU image arrays on unified RAM.
    torch.empty(1, device="cuda")
    cache_summary = json.loads((directory/"summary.json").read_text())
    cache = {}
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    for scene in config["train"]+config["val"]:
        path = directory/(scene+".pt")
        if digest(path) != cache_summary["scenes"][scene]["sha256"]:
            raise ValueError("Point cache changed")
        data = torch.load(path, weights_only=True, map_location="cpu")
        data.pop("features")
        image_path = Path(cache_plan["root"])/scene/"images.zip"
        metadata = json.loads(image_path.with_name("metadata.json").read_text())
        if digest(image_path) != metadata["files"]["images.zip"]["sha256"]:
            raise ValueError("RGB archive changed")
        with zipfile.ZipFile(image_path) as archive:
            images = [Image.open(io.BytesIO(archive.read(f"{i:03d}.png"))).convert("RGB") for i in data["frames"]]
        data["pixels"] = processor(images=images, return_tensors="pt").pixel_values
        cache[scene] = data
    metric = torch.load(run/"metric.pt", map_location="cpu", weights_only=True)["models"]
    started, results, initial_hash = time.monotonic(), {}, None
    for arm in config["arms"]:
        model = build_model(dict(config, routing=arm), metric)
        current_hash = trainable_digest(model)
        if initial_hash is not None and current_hash != initial_hash:
            raise ValueError("Arm initial weights do not match")
        initial_hash = current_hash
        rng = random.Random(config["seed"])
        generator = torch.Generator().manual_seed(config["seed"])
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=config["lr"], weight_decay=config["weight_decay"])
        model.train()
        for step in range(config["steps_per_arm"]):
            scene = rng.choice(config["train"])
            data = cache[scene]
            start = rng.randrange(len(data["frames"])-config["clip_frames"]+1)
            frames = slice(start, start+config["clip_frames"])
            candidates = torch.where(data["visible"][:, start])[0]
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:config["queries"]]]
            if len(chosen) != config["queries"]:
                raise ValueError("Too few training queries")
            pixels = data["pixels"][frames][None].cuda()
            points = data["points"][chosen, frames][None].cuda()
            occluded = (~data["visible"][chosen, frames])[None].cuda()
            query_frames = torch.zeros(1, len(chosen), device="cuda", dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model(pixels, points[:, :, 0], query_frames)
                losses = point_tracking_loss(predicted, points, occluded, query_frames)
            if not torch.isfinite(losses["loss"]):
                raise ValueError("Nonfinite pilot loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if (step+1) % 32 == 0:
                row = dict(arm=arm, step=step+1, scene=scene, start=start,
                    loss=float(losses["loss"].detach()), elapsed_seconds=time.monotonic()-started)
                with (output/"training.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
                print(json.dumps(row), flush=True)
        destination = output/(arm+".pt")
        torch.save(dict(architecture=model.encoder.architecture, backbone=DEFAULT_MODEL, revision=DEFAULT_REVISION,
            adapters=model.encoder.adapters.state_dict(), readout_config=model.readout.config,
            readout=model.readout.state_dict()), destination)
        results[arm] = dict(initial_trainable_sha256=initial_hash, checkpoint_sha256=digest(destination),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            gains={k:float(a.gain.detach().abs().mean()) for k,a in model.encoder.adapters.items()},
            **evaluate(model, cache, config["val"], config))
        (output/(arm+".json")).write_text(json.dumps(results[arm], indent=2)+"\n")
        print(json.dumps(dict(arm=arm, status="complete", summary=results[arm]["summary"])), flush=True)
        del model, optimizer, losses, predicted
        gc.collect(); torch.cuda.empty_cache()
    result = dict(plan_sha256=digest(output/"plan.json"), results=results, elapsed_seconds=time.monotonic()-started)
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
