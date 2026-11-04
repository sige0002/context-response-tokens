"""Replay native selective-scan backward on identical real-video operands.

This diagnostic observes the baseline without changing its numerical path.
Repeated native backward calls return the first result to the model. A captured
case is independently differentiated through the upstream PyTorch reference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import SiglipImageProcessor

from gmmix_multidepth import MultiDepthGatedMambaMixSiglip
from temporal_readout import VideoRecognizer
from train_video import configure_reproducibility
from video_data import RawVideoClips


def difference(a, b):
    if a is None or b is None:
        assert a is None and b is None
        return None
    d = a.detach().float() - b.detach().float()
    return dict(dtype=str(a.dtype), shape=list(a.shape), different=int((d != 0).sum()),
                max_abs=float(d.abs().max()),
                relative_l2=float(d.norm() / a.detach().float().norm().clamp_min(1e-12)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    assert args.repeats >= 2
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda, selective_scan_ref
    original = selective_scan_cuda.bwd
    report = dict(scope="Same-input native backward repeats on two actual SSv2 microbatches",
                  repeats=args.repeats, calls=[])
    captured = {}

    def observed(*operands):
        # u, delta, A, B, C, D, z, delta_bias, dout, x, out,
        # optional preallocated dz, delta_softplus, recompute_out_z.
        assert operands[11] is None and operands[13] is False
        first = original(*operands)
        largest = {}
        for repeat in range(1, args.repeats):
            other = original(*operands)
            for name, a, b in zip(("du", "ddelta", "dA", "dB", "dC", "dD", "ddelta_bias", "dz"), first, other):
                stats = difference(a, b)
                if stats is not None and stats["different"]:
                    if name not in largest or stats["max_abs"] > largest[name]["max_abs"]:
                        largest[name] = dict(**stats, repeat=repeat)
        index = len(report["calls"])
        report["calls"].append(dict(index=index, differences=largest))
        if not captured and ("dB" in largest or "dC" in largest):
            captured.update(index=index,
                inputs=[v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                        for v in operands[:9]],
                gradients=[v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in first],
                delta_softplus=operands[12])
        return first

    torch.set_num_threads(4)
    configure_reproducibility(42)
    encoder = MultiDepthGatedMambaMixSiglip.from_pretrained()
    model = VideoRecognizer(encoder, 174, "attention", 43).cuda().train()
    processor = SiglipImageProcessor.from_pretrained("google/siglip2-base-patch16-224",
        revision=encoder.checkpoint_revision, local_files_only=True)
    data = RawVideoClips("data/ssv2/video_training_fixture/train.jsonl", processor, 8, 174,
                        training=True, seed=42)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(42)).tolist()
    report["order"] = order[:4]
    selective_scan_cuda.bwd = observed
    try:
        for start in (0, 2):
            pairs = [data[i] for i in order[start:start+2]]
            images = torch.stack([p[0] for p in pairs]).cuda()
            labels = torch.tensor([p[1] for p in pairs], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
            loss.backward()
    finally:
        selective_scan_cuda.bwd = original
    del model, encoder, images, logits, loss
    torch.cuda.empty_cache()
    if captured:
        torch.save(captured, output / "captured.pt")
        inputs = [v.cuda().requires_grad_(True) if isinstance(v, torch.Tensor) else v
                  for v in captured["inputs"][:8]]
        dout = captured["inputs"][8].cuda()
        # Upstream reference already promotes the recurrence to FP32. Preserve
        # the original input/output dtypes and disable ambient autocast.
        with torch.autocast("cuda", enabled=False):
            y = selective_scan_ref(*inputs, delta_softplus=captured["delta_softplus"])
            grads = torch.autograd.grad(y, inputs, dout)
        # The reference argument order has z before delta_bias; native bwd
        # returns delta_bias before z.
        names = ("du", "ddelta", "dA", "dB", "dC", "dD", "dz", "ddelta_bias")
        native_indices = (0, 1, 2, 3, 4, 5, 7, 6)
        report["reference_comparison"] = dict(call=captured["index"], gradients={
            name:difference(captured["gradients"][i].cuda(), value)
            for name, i, value in zip(names, native_indices, grads)})
    report["source_sha256"] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ("diagnose_multidepth_scan.py", "gmmix_multidepth.py", "gmmix_siglip.py")}
    (output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
