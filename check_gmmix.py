"""GPU correctness checks for the closest-prior GatedMambaMix baseline.

Run in the isolated baseline uv project while other GPU jobs are paused.
Native kernels are compared to their upstream differentiable reference, then
the untrained published core is exercised on a real SigLIP2 video clip.
These checks are not recognition-performance evidence.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import SiglipImageProcessor

from benchmark_streaming import idle_compute_processes
from gmmix_siglip import GatedMambaMixSiglip, REFERENCE_COMMIT, REFERENCE_FILES
from point_checkpoint import file_hash
from temporal_readout import VideoRecognizer
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from train_video import parameter_groups
from video_data import RawVideoClips


def difference(actual, expected):
    actual, expected = actual.float(), expected.float()
    return dict(max_abs=float((actual - expected).abs().max()),
                relative_l2=float((actual - expected).norm() / expected.norm().clamp_min(1e-12)))


def check_scan(dtype, length):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    b, d, n = 2, 64, 16
    def random(*shape, kind=dtype):
        return (torch.randn(*shape, device="cuda", dtype=kind) * .2).requires_grad_()
    values = [random(b, d, length), random(b, d, length),
        (-torch.rand(d, n, device="cuda") - .1).requires_grad_(), random(b, n, length),
        random(b, n, length), random(d, kind=torch.float32), random(b, d, length),
        random(d, kind=torch.float32)]
    reference = [value.detach().clone().requires_grad_() for value in values]
    def calculate(fn, arguments):
        return fn(*arguments[:7], delta_bias=arguments[7], delta_softplus=True)
    output = calculate(selective_scan_fn, values)
    expected = calculate(selective_scan_ref, reference)
    weights = torch.randn_like(output)
    actual_grads = torch.autograd.grad((output * weights).sum(), values)
    reference_grads = torch.autograd.grad((expected * weights).sum(), reference)
    result = dict(dtype=str(dtype), length=length, forward=difference(output, expected),
                  backward=[difference(a, b) for a, b in zip(actual_grads, reference_grads)])
    tolerance = 2e-5 if dtype == torch.float32 else .01
    result["predeclared_relative_l2_tolerance"] = tolerance
    print(json.dumps(dict(event="native_scan", **result)), flush=True)
    assert result["forward"]["relative_l2"] < tolerance, result
    assert all(row["relative_l2"] < tolerance for row in result["backward"]), result
    assert all(torch.isfinite(value).all() for value in actual_grads)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paused = idle_compute_processes()
    torch.set_num_threads(4)
    torch.manual_seed(53)
    report = dict(scope="Native-kernel and real-video baseline integration only; no accuracy claim",
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), paused_compute_pids=paused,
        reference_commit=REFERENCE_COMMIT, reference_files=REFERENCE_FILES)
    report["native_scan"] = [check_scan(dtype, length)
        for dtype in (torch.float32, torch.bfloat16) for length in (7, 257)]
    (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    encoder = GatedMambaMixSiglip.from_pretrained().cuda().eval()
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    dataset = RawVideoClips("data/ssv2/videos_trainval/val.jsonl", processor, frames=4, classes=174)
    images, label, sampling = dataset.read(0)
    video = images[None].cuda()
    report.update(core_config=encoder.core_config, architecture=encoder.architecture, video=sampling,
                  core_parameters=sum(p.numel() for p in encoder.core.parameters()),
                  spatial_parameters=sum(p.numel() for p in encoder.backbone.parameters()))
    with torch.no_grad():
        fp32_batch = encoder(video)
        fp32_stream, _ = encoder.encode_chunk(video)
        report["fp32_clip_stream"] = difference(fp32_stream, fp32_batch)
        assert report["fp32_clip_stream"]["relative_l2"] < 2e-5, report["fp32_clip_stream"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batched = encoder(video)
        online, state = encoder.encode_chunk(video)
        report["bf16_clip_stream"] = difference(online, batched)
        assert report["bf16_clip_stream"]["relative_l2"] < .01, report["bf16_clip_stream"]
        state_bytes = state.nbytes
        _, state = encoder.step(video[:, -1], state)
        assert state.nbytes == state_bytes and state.offset == 5
        changed = video.clone()
        changed[:, 2:] = -changed[:, 2:]
        altered = encoder(changed)
        future_difference = float((altered[:, :2] - batched[:, :2]).abs().max())
        assert future_difference == 0, future_difference
        reset, reset_state = encoder.step(video[:, 0])
        torch.testing.assert_close(reset, online[:, 0], atol=0, rtol=0)
        assert reset_state.offset == 1
        try:
            encoder.step(video[:, 0].expand(2, -1, -1, -1), reset_state)
        except ValueError:
            pass
        else:
            raise AssertionError("State layout mismatch was accepted")
    report.update(state_bytes=state_bytes, state_size_constant_after_another_frame=True,
                  future_change_max_difference=future_difference, reset_exact=True)
    (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    del fp32_batch, fp32_stream, online, batched, altered, state, reset, reset_state
    before = {name:value.detach().cpu().clone() for name, value in encoder.backbone.state_dict().items()}
    recognizer = VideoRecognizer(encoder, classes=174, readout="attention", seed=43).cuda().train()
    assert not encoder.backbone.training and encoder.core.training
    optimizer = torch.optim.AdamW(parameter_groups(recognizer, 1e-4), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = recognizer(video)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([int(label)], device="cuda"))
    loss.backward()
    gradients = {name:float(p.grad.float().norm()) for name, p in encoder.core.named_parameters() if p.grad is not None}
    assert len(gradients) == len(list(encoder.core.parameters()))
    assert all(torch.isfinite(p.grad).all() for p in encoder.core.parameters())
    for layer in range(4):
        assert gradients[f"mamba_blocks.{layer}.mamba.in_proj.weight"] > 0
        assert gradients[f"gates.{layer}.weight"] > 0
        assert gradients[f"spatial_blocks.{layer}.sa_attn.in_proj_weight"] > 0
    optimizer.step()
    assert all(p.grad is None and not p.requires_grad for p in encoder.backbone.parameters())
    assert all(torch.equal(before[name], value.cpu()) for name, value in encoder.backbone.state_dict().items())
    assert all(torch.isfinite(p).all() for p in encoder.core.parameters())
    encoder.eval().save_checkpoint(output / "encoder.pt")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = encoder(video)
    restored = GatedMambaMixSiglip.load_checkpoint(output / "encoder.pt", "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = restored(video)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    import selective_scan_cuda
    report.update(backward_core_gradients=gradients, initial_one_clip_loss=float(loss.detach()),
        backbone_bitwise_unchanged=True, self_contained_checkpoint_exact=True,
        trainable_parameters=sum(p.numel() for p in recognizer.parameters() if p.requires_grad),
        extension_sha256=file_hash(selective_scan_cuda.__file__),
        optional_causal_conv1d=False,
        source_sha256={name:file_hash(name) for name in
            ("check_gmmix.py", "gmmix_siglip.py", "temporal_siglip.py", "temporal_readout.py", "video_data.py",
             "baselines/gmmix/pyproject.toml", "baselines/gmmix/uv.lock")})
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key:value for key, value in report.items() if key not in
        ("native_scan", "backward_core_gradients", "source_sha256", "reference_files")}), flush=True)


if __name__ == "__main__":
    main()
