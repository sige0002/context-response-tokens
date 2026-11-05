"""Freeze SigLIP 2 intermediate features for a small correspondence experiment."""
import argparse
import io
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

from kubric_data import load_points
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from exploration.oracle_transport import digest


def select_scenes(root):
    ids = [f"{i:04d}" for i in range(5869)]
    random.Random(42).shuffle(ids)
    validation = set(ids[:300])
    previous = {"0016", "0020", "0030", "0061", "0065", "0081", "0123", "0128"}
    complete = sorted(p.name for p in root.iterdir() if (p/"metadata.json").is_file())
    train = [s for s in complete if s not in validation][:24]
    val = [s for s in complete if s in validation and s not in previous][:8]
    if len(train) != 24 or len(val) != 8 or set(train)&set(val):
        raise ValueError("Need 24 train and 8 separate validation scenes")
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/cotracker3_kubric/scenes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    train, val = select_scenes(root)
    config = dict(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, train=train, val=val,
        frames=list(range(0, 96, 4)), depths=[4, 8, 12], tracks=4096, seed=17,
        input_size=224, feature_precision="float32", batch_size=8,
        selection="First prepared IDs in fixed train split; next eight prepared validation IDs excluding the eight oracle scenes. Chosen before scores.",
        scope="Frozen spatial features; GT positions supervise a correspondence metric, not a teacher-feature loss. Not full-dataset or recognition evaluation.")
    plan = dict(config=config, root=str(root),
        sources={str(p.resolve()):digest(p) for p in (Path(__file__), Path("kubric_data.py"), Path("temporal_siglip.py"), Path("exploration/oracle_transport.py"))},
        metadata={s:digest(root/s/"metadata.json") for s in train+val})
    if args.resume:
        if json.loads((output/"plan.json").read_text()) != plan:
            raise ValueError("Cache plan or source changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.plan_only:
        print(json.dumps(dict(train=train, val=val, frames=len(config["frames"]), status="plan_saved")), flush=True)
        return
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    model = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    started, records = time.monotonic(), {}
    for scene in train+val:
        directory = root/scene
        metadata = json.loads((directory/"metadata.json").read_text())
        for name in ("images.zip", "points.npz"):
            if digest(directory/name) != metadata["files"][name]["sha256"]:
                raise ValueError("Prepared input changed")
        destination = output/(scene+".pt")
        receipt = output/(scene+".json")
        if destination.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            if record["plan_sha256"] != digest(output/"plan.json") or record["sha256"] != digest(destination):
                raise ValueError("Existing scene cache changed")
        else:
            with zipfile.ZipFile(directory/"images.zip") as archive:
                images = [Image.open(io.BytesIO(archive.read(f"{i:03d}.png"))).convert("RGB") for i in config["frames"]]
            features = {depth:[] for depth in config["depths"]}
            for offset in range(0, len(images), config["batch_size"]):
                pixels = processor(images=images[offset:offset+config["batch_size"]], return_tensors="pt").pixel_values.cuda()
                with torch.inference_mode():
                    x = model.vision_model.embeddings(pixels.float())
                    for i, layer in enumerate(model.vision_model.encoder.layers, 1):
                        x = layer(x, attention_mask=None)
                        if i in features:
                            features[i].append(x.float().cpu())
                del pixels, x
            points, occluded = load_points(directory/"points.npz")
            indices = np.random.default_rng(config["seed"]+int(scene)).permutation(len(points))[:config["tracks"]]
            points, occluded = points[indices][:, config["frames"]], occluded[indices][:, config["frames"]]
            visible = (~occluded) & np.isfinite(points).all(-1)
            visible &= (points[..., 0] >= 0) & (points[..., 0] <= metadata["native_width"]-1)
            visible &= (points[..., 1] >= 0) & (points[..., 1] <= metadata["native_height"]-1)
            points = points / np.array([metadata["native_width"], metadata["native_height"]], dtype=np.float32)
            # Invalid values are represented only by visible=False.
            points = np.where(visible[..., None], points, 0)
            payload = dict(features={str(d):torch.cat(features[d]) for d in config["depths"]},
                points=torch.from_numpy(points.copy()), visible=torch.from_numpy(visible.copy()),
                track_indices=torch.from_numpy(indices.copy()), frames=config["frames"])
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, temporary); temporary.replace(destination)
            record = dict(scene=scene, sha256=digest(destination), bytes=destination.stat().st_size,
                plan_sha256=digest(output/"plan.json"), metadata_sha256=plan["metadata"][scene])
            receipt.write_text(json.dumps(record, indent=2)+"\n")
        records[scene] = record
        print(json.dumps(dict(scene=scene, completed=len(records), total=len(train+val), elapsed_seconds=time.monotonic()-started)), flush=True)
    summary = dict(plan_sha256=digest(output/"plan.json"), scenes=records, elapsed_seconds=time.monotonic()-started,
                   total_bytes=sum(r["bytes"] for r in records.values()))
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
