"""Check native FP32 scan against an independent FP64 recurrence.

Uses operands captured from real SSv2 training, with two sequences retained to
keep this accuracy check small. Includes all eight input derivatives.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from train_video import configure_reproducibility


def recurrence(u, delta, a, b, c, d, z, bias):
    dt = torch.nn.functional.softplus(delta + bias[None, :, None])
    state = torch.zeros(u.shape[0], u.shape[1], a.shape[1], device=u.device, dtype=u.dtype)
    outputs = []
    assert b.shape[1] == c.shape[1] == 1
    for t in range(u.shape[-1]):
        state = torch.exp(dt[:, :, t, None] * a) * state + (
            dt[:, :, t, None] * b[:, 0, :, t][:, None, :] * u[:, :, t, None])
        y = (state * c[:, 0, :, t][:, None, :]).sum(-1) + d * u[:, :, t]
        outputs.append(y * torch.nn.functional.silu(z[:, :, t]))
    return torch.stack(outputs, -1)


def compare(a, b):
    x, y = a.detach().double(), b.detach().double()
    diff = x - y
    return dict(max_abs=float(diff.abs().max()), relative_l2=float(diff.norm()/y.norm().clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_cuda
    configure_reproducibility(42)
    torch.set_num_threads(4)
    saved = torch.load(args.capture, map_location="cpu", weights_only=True)
    assert saved["delta_softplus"]
    arrays = [x[:2] if i in (0, 1, 3, 4, 6, 8) else x for i, x in enumerate(saved["inputs"])]
    native = [x.cuda().float().requires_grad_() for x in arrays[:8]]
    reference = [x.cuda().double().requires_grad_() for x in arrays[:8]]
    dout = arrays[8].cuda().float()
    result = selective_scan_fn(*native, delta_softplus=True)
    expected = recurrence(*reference)
    gradients = torch.autograd.grad(result, native, dout)
    expected_gradients = torch.autograd.grad(expected, reference, dout.double())
    report = dict(scope="Native FP32 forward/backward versus independent FP64 recurrence on two real-input sequences",
        sequences=2, frames=arrays[0].shape[-1], predeclared_relative_l2_tolerance=1e-4,
        forward=compare(result, expected), gradients={name:compare(a, b) for name, a, b in
            zip(("u", "delta", "A", "B", "C", "D", "z", "delta_bias"), gradients, expected_gradients)},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256(Path(args.capture).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(selective_scan_cuda.__file__).read_bytes()).hexdigest())
    report["passed"] = all(row["relative_l2"] < report["predeclared_relative_l2_tolerance"]
                           for row in [report["forward"], *report["gradients"].values()])
    Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"], report


if __name__ == "__main__":
    main()
