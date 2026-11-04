"""Pretrained 224px GPU smoke test; not a video recognition benchmark."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import SiglipImageProcessor
from temporal_siglip import TemporalSiglip, DEFAULT_MODEL, DEFAULT_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--images", nargs="+", default=["data/demo_dog.jpg"])
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--output", default="results/temporal_siglip2/smoke.json")
    args = p.parse_args()
    if args.frames < 2:
        p.error("--frames must be >= 2")
    torch.manual_seed(42)
    revision = DEFAULT_REVISION if args.model == DEFAULT_MODEL else "main"
    processor = SiglipImageProcessor.from_pretrained(args.model, revision=revision)
    images = [Image.open(path).convert("RGB") for path in args.images]
    frames = [images[i % len(images)] for i in range(args.frames)]
    video = processor(images=frames, return_tensors="pt").pixel_values[None].cuda()
    report = {"purpose": "implementation verification, not accuracy evaluation",
              "images": args.images, "repeated_images": len(images) < args.frames,
              "model": args.model, "input_shape": list(video.shape),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "variants": {}}
    for kind in ("none", "attention", "mamba"):
        torch.manual_seed(42)
        model = TemporalSiglip.from_pretrained(args.model, kind=kind).cuda().eval()
        revision = model.checkpoint_revision
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.autocast("cuda", enabled=False):
                baseline = model.backbone(video.flatten(0, 1)).last_hidden_state
                baseline = baseline.reshape(1, args.frames, *baseline.shape[1:])
            initial = model(video)
            identity_error = (initial - baseline).abs().max().item()
            assert identity_error < 1e-5, identity_error
            # Deliberately enable untrained adapters to test nontrivial paths.
            for a in model.adapters.values():
                a.gain.fill_(0.1)
            full = model(video)
            altered = video.clone()
            altered[:, -1] = -altered[:, -1]
            causal_error = (model(altered)[:, :-1] - full[:, :-1]).abs().max().item()
            assert causal_error < 1e-5, causal_error
            # Warm up both full and official recurrent kernels before timing.
            warm_state = None
            for frame in video.unbind(1):
                _, warm_state = model.step(frame, warm_state)
            del warm_state
            torch.cuda.synchronize()
            state, outputs, sizes, elapsed = None, [], [], []
            for frame in video.unbind(1):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                y, state = model.step(frame, state)
                end.record()
                torch.cuda.synchronize()
                outputs.append(y)
                sizes.append(state.nbytes)
                elapsed.append(start.elapsed_time(end))
            stream = torch.stack(outputs, 1)
            difference = (stream - full).float()
            torch.testing.assert_close(stream, full, atol=0.04, rtol=0.03)
            if kind == "mamba":
                assert len(set(sizes)) == 1
        grad_ok = None
        if kind != "none":
            model.train()
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model(video[:, :2])
                # Arbitrary probe objective only to verify the training path.
                loss = (features.float() * torch.randn_like(features.float())).mean()
            loss.backward()
            grads = [p.grad for p in model.adapters.parameters() if p.grad is not None]
            grad_ok = bool(grads and all(torch.isfinite(g).all() for g in grads))
            assert grad_ok
            assert all(a.down.weight.grad.abs().sum() > 0 for a in model.adapters.values())
            optimizer.step()
            del optimizer, features, loss, grads
        report["variants"][kind] = {
            "checkpoint_revision": revision,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "output_shape": list(full.shape), "initial_identity_max_abs": identity_error,
            "future_leak_max_abs": causal_error,
            "stream_batch_max_abs": difference.abs().max().item(),
            "stream_batch_rmse": difference.square().mean().sqrt().item(),
            "state_bytes_by_frame": sizes, "step_gpu_ms": elapsed,
            "finite_backward_and_optimizer_step": grad_ok,
        }
        print(kind, json.dumps(report["variants"][kind]), flush=True)
        del model, full, stream, initial, baseline, state, outputs
        torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
