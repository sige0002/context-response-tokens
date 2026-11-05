"""Qualify the routed encoder on actual 224 Kubric RGB; not an accuracy test."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile

from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from exploration.oracle_transport import digest
from exploration.routed_siglip import RoutedSiglip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    run = Path(args.metric_run).resolve()
    metric_report = json.loads((run/"final.json").read_text())
    if digest(run/"metric.pt") != metric_report["checkpoint_sha256"]:
        raise ValueError("Metric checkpoint changed")
    metric_plan = json.loads((run/"plan.json").read_text())
    cache_plan = json.loads((Path(metric_plan["cache"])/"plan.json").read_text())
    scene = cache_plan["config"]["val"][0]
    directory = Path(cache_plan["root"])/scene
    metadata = json.loads((directory/"metadata.json").read_text())
    if digest(directory/"images.zip") != metadata["files"]["images.zip"]["sha256"]:
        raise ValueError("Qualification images changed")
    with zipfile.ZipFile(directory/"images.zip") as archive:
        images = [Image.open(io.BytesIO(archive.read(f"{i:03d}.png"))).convert("RGB") for i in (0, 4, 8, 12)]
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    backbone = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    model = RoutedSiglip(backbone).cuda().eval()
    payload = torch.load(run/"metric.pt", map_location="cpu", weights_only=True)
    for key, adapter in model.adapters.items():
        prefix = str(int(key)+1)+"."
        adapter.metric.load_state_dict({k.removeprefix(prefix):v for k,v in payload["models"].items() if k.startswith(prefix)}, strict=True)
    pixels = processor(images=images, return_tensors="pt").pixel_values.cuda()[None]
    started = time.monotonic()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        baseline = backbone(pixels.flatten(0, 1)).last_hidden_state[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            initial = model(pixels)
            torch.testing.assert_close(initial, baseline, atol=0, rtol=0)
        for adapter in model.adapters.values():
            adapter.gain.data.fill_(.05)
        full = model(pixels)
        with torch.no_grad():
            changed = pixels.clone(); changed[:, 2:] = torch.randn_like(changed[:, 2:])
            changed_output = model(changed)
            torch.testing.assert_close(full[:, :2], changed_output[:, :2], atol=1e-5, rtol=1e-5)
            outputs, sizes, state = [], [], None
            references = None
            for frame in pixels.unbind(1):
                value, state = model.step(frame, state)
                outputs.append(value); sizes.append(state.nbytes)
                if references is None:
                    references = {key:values[0].clone() for key,values in state.layers.items()}
                else:
                    for key, values in state.layers.items():
                        torch.testing.assert_close(values[0], references[key], atol=0, rtol=0)
                assert all(len(values) == 5 for values in state.layers.values())
            streamed = torch.stack(outputs, 1)
            torch.testing.assert_close(full, streamed, atol=.015, rtol=.02)
            assert len(set(sizes)) == 1
            first, chunks = model.encode_chunk(pixels[:, :2])
            last, chunks = model.encode_chunk(pixels[:, 2:], chunks)
            chunked = torch.cat((first, last), 1)
            torch.testing.assert_close(full, chunked, atol=.015, rtol=.02)
            reset, _ = model.step(pixels[:, -1])
            history_effect = float((reset-full[:, -1]).abs().max())
            assert history_effect > 1e-5
        (full.float()*torch.randn_like(full.float())).mean().backward()
    gradients = {}
    for key, adapter in model.adapters.items():
        gradients[key] = {}
        for name, parameter in (("metric", adapter.metric.network[1].weight),
                                ("mamba", adapter.mixer.in_proj.weight)):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            norm = float(parameter.grad.float().norm())
            assert norm > 0
            gradients[key][name] = norm
    assert all(p.grad is None for p in model.backbone.parameters())
    torch.cuda.synchronize()
    report = dict(model=DEFAULT_MODEL, revision=DEFAULT_REVISION, scene=scene, frames=[0, 4, 8, 12],
        shape=list(full.shape), architecture=model.architecture, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        metric_checkpoint_sha256=digest(run/"metric.pt"),
        sources={str(p):digest(p) for p in (Path(__file__), Path("exploration/routed_siglip.py"), Path("exploration/appearance_routing.py"), Path("temporal_siglip.py"))},
        initial_image_path_max_error=float((initial-baseline).abs().max()),
        causal_prefix_max_error=float((full[:, :2]-changed_output[:, :2]).detach().abs().max()),
        stream_max_error=float((full-streamed).detach().abs().max()),
        chunk_max_error=float((full-chunked).detach().abs().max()), state_bytes=sizes,
        cache_contract="one immutable FP32 descriptor plus all four official Mamba-3 buffers per adapter",
        history_effect_max=history_effect, gradient_norms=gradients,
        elapsed_seconds=time.monotonic()-started, status="passed",
        limits="Pretrained SigLIP plus briefly trained correspondence metric; random temporal weights and manually activated residual gains. Checks implementation only, not recognition performance or deployment latency. Main learner running concurrently.")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
