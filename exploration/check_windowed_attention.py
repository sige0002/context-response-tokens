"""Prepare real RGB on CPU; qualify sliding-window inference after main training.

No timing or accuracy claims. The chronological reference delegates each
single-observation update to the unchanged original attention implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel

from exploration.buffered_attention import select
from exploration.evaluate_video_history import model_hash, tensor_hash
from exploration.windowed_attention import variants
from temporal_siglip import CausalAttention, TemporalSiglip, DEFAULT_MODEL, DEFAULT_REVISION
from train_video import configure_reproducibility
from video_data import RawVideoClips


SOURCES = ["exploration/check_windowed_attention.py", "exploration/windowed_attention.py",
           "exploration/buffered_attention.py", "exploration/evaluate_video_history.py",
           "temporal_siglip.py", "video_data.py", "train_video.py", "pyproject.toml", "uv.lock"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class ChronologicalWindowAttention(CausalAttention):
    """Slow reference: trim chronological K/V, then call original attention."""
    def __init__(self, source, window):
        nn.Module.__init__(self)
        self.heads, self.qkv, self.out = source.heads, source.qkv, source.out
        self.window = window
        self.eval()

    def forward(self, x, cache=None, offset=0, streaming=False):
        if not streaming or x.shape[1] != 1 or self.training or torch.is_grad_enabled():
            raise ValueError("Reference supports only no-grad single-observation inference")
        if cache is None:
            if offset != 0:
                raise ValueError("Reference starts at observation zero")
            history = None
        else:
            if len(cache) != 3 or int(cache[2]) != offset:
                raise ValueError("Reference cursor mismatch")
            keep = self.window - 1
            history = tuple(v[:, :, -keep:].contiguous() for v in cache[:2]) if keep else None
        # For a single query, every retained key is in the past/current set.
        # The original absolute query offset therefore admits all these keys.
        y, (k, v) = super().forward(x, history, offset, True)
        return y, (k, v, torch.tensor(offset + 1, dtype=torch.int64))


def require_main_complete(run):
    run = Path(run).resolve()
    progress = run / "progress.jsonl"
    events = [json.loads(line) for line in progress.read_text().splitlines() if line]
    if not events or events[-1].get("event") != "budget_complete" or events[-1].get("step") != 20000:
        raise RuntimeError("Main training has not completed its 20000-update budget")
    # A completed log alone is not permission to overlap a still-live trainer.
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            if "train_video.py" not in [Path(a).name for a in args] or "--output" not in args:
                continue
            output = (process / "cwd").resolve() / args[args.index("--output") + 1]
            if output.resolve() == run:
                raise RuntimeError("The main training process is still live")
        except (OSError, UnicodeDecodeError, IndexError):
            continue


@torch.inference_mode()
def prepare(output, data_path, main_run):
    if output.exists():
        raise FileExistsError("Use a fresh qualification directory")
    torch.set_num_threads(2)
    configure_reproducibility(71)
    data = json.loads(data_path.read_text())
    record = data["manifests"]["validation"]
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("Input manifest changed")
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    dataset = RawVideoClips(record["path"], processor, frames=8, classes=174)
    index = {r["id"]: i for i, r in enumerate(dataset.rows)}
    block = data["splits"]["validation"][0]
    chosen = block["evidence"][:1] + block["suffix"]
    pixels, inputs = [], []
    for row in chosen:
        i = index[row["id"]]
        if sha(dataset.rows[i]["video"]) != row["sha256"]:
            raise ValueError("Source video changed")
        values, _, metadata = dataset.read(i)
        pixels.append(values)
        inputs.append(dict(**metadata, source_sha256=row["sha256"], pixels_sha256=tensor_hash(values)))
    pixels = torch.cat(pixels)
    if len(pixels) != 136:
        raise ValueError("Expected the fixed 17-video, 136-observation qualification input")
    backbone = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    encoder = TemporalSiglip(backbone, kind="attention").eval().requires_grad_(False)
    for adapter in encoder.adapters.values():
        adapter.gain.fill_(0.1)
    output.mkdir(parents=True)
    encoder.save_checkpoint(output / "encoder.pt")
    torch.save(pixels, output / "pixels.pt")
    plan = dict(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, frames=len(pixels), windows=[8, 32, 128],
                main_run=str(main_run.resolve()), model_state_sha256=model_hash(encoder),
                encoder_sha256=sha(output / "encoder.pt"), pixels_file_sha256=sha(output / "pixels.pt"),
                pixels_tensor_sha256=tensor_hash(pixels), data_plan_sha256=sha(data_path), inputs=inputs,
                sources={p: sha(p) for p in SOURCES},
                tolerances={"fp32": {"max_abs": 0.001, "relative_l2": 1e-5},
                            "bf16": {"max_abs": 0.125, "relative_l2": 0.005}},
                initialization="Pretrained SigLIP2; seed71 random attention; residual gates 0.1",
                scope="Real-input numerical qualification with untrained temporal layers; not recognition or timing",
                tolerance_scope="Predeclared engineering alarms on full features; not mathematical error bounds or accuracy guarantees")
    write(output / "plan.json", plan)
    print(json.dumps({"status": "prepared_cpu_only", "frames": len(pixels), "plan_sha256": sha(output / "plan.json")}))


@torch.inference_mode()
def run(output):
    plan = json.loads((output / "plan.json").read_text())
    require_main_complete(plan["main_run"])
    if (output / "summary.json").exists() or (output / "rows.jsonl").exists():
        raise FileExistsError("Preserve previous qualification or inspect partial execution")
    for path, expected in plan["sources"].items():
        if sha(path) != expected:
            raise ValueError("Qualification source changed: " + path)
    for name, key in (("encoder.pt", "encoder_sha256"), ("pixels.pt", "pixels_file_sha256")):
        if sha(output / name) != plan[key]:
            raise ValueError("Qualification fixture changed")
    torch.set_num_threads(2)
    configure_reproducibility(71)
    encoder = TemporalSiglip.load_checkpoint(output / "encoder.pt").requires_grad_(False)
    pixels = torch.load(output / "pixels.pt", map_location="cpu", weights_only=True)
    if model_hash(encoder) != plan["model_state_sha256"] or tensor_hash(pixels) != plan["pixels_tensor_sha256"]:
        raise ValueError("Decoded fixture changed")
    encoder.cuda()
    original = {key: adapter.mixer for key, adapter in encoder.adapters.items()}
    rows = []
    try:
        with (output / "rows.jsonl").open("x") as stream:
            for precision, limits in plan["tolerances"].items():
                for window in plan["windows"]:
                    select(encoder, original)
                    _, ring = variants(encoder, window)
                    reference = {key: ChronologicalWindowAttention(m, window) for key, m in original.items()}
                    states = {"ring": None, "reference": None}
                    pointers = None
                    for frame, image in enumerate(pixels):
                        features = {}
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                            for name, mixers in (("ring", ring), ("reference", reference)):
                                select(encoder, mixers)
                                features[name], states[name] = encoder.step(image[None].cuda(), states[name])
                        a, b = features["ring"].float(), features["reference"].float()
                        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                            raise FloatingPointError("Nonfinite feature")
                        difference = a - b
                        maximum = float(difference.abs().max())
                        relative = float(difference.norm() / b.norm().clamp_min(1e-12))
                        current = [[v.data_ptr() for v in cache] for cache in states["ring"].layers.values()]
                        if pointers is not None and pointers != current:
                            raise ValueError("Ring storage was reallocated")
                        pointers = current
                        expected_bytes = 602112 * window * (2 if precision == "fp32" else 1) + 24
                        if states["ring"].nbytes != expected_bytes:
                            raise ValueError("Unexpected full-model ring capacity")
                        row = dict(precision=precision, window=window, frame=frame,
                                   max_abs=maximum, relative_l2=relative, state_bytes=states["ring"].nbytes,
                                   passed=maximum <= limits["max_abs"] and relative <= limits["relative_l2"])
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(json.dumps({"precision": precision, "window": window, "frames": len(pixels)}), flush=True)
    finally:
        select(encoder, original)
    if model_hash(encoder) != plan["model_state_sha256"]:
        raise ValueError("Parameters changed during qualification")
    for path, expected in plan["sources"].items():
        if sha(path) != expected:
            raise ValueError("Source changed during qualification: " + path)
    summary = dict(passed=all(row["passed"] for row in rows), plan_sha256=sha(output / "plan.json"),
                   rows_sha256=sha(output / "rows.jsonl"), rows=len(rows), weights_unchanged=True,
                   max_abs=max(row["max_abs"] for row in rows),
                   max_relative_l2=max(row["relative_l2"] for row in rows),
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   scope=plan["scope"], trained_recognition_evaluated=False, speed_measured=False)
    write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise ValueError("Predeclared numerical alarm exceeded; retain results for diagnosis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("results/delayed_action_memory_v1/data_plan.json"))
    parser.add_argument("--main-run", type=Path, default=Path("results/ssv2_full_mamba_interleaved_seed42_rgbfix"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output, args.data, args.main_run)
    else:
        run(args.output)
