"""Real-GPU integration of an untrained point probe, not tracking accuracy."""
import argparse
import json
from pathlib import Path

import torch
from transformers import SiglipImageProcessor

from benchmark_streaming import idle_compute_processes
from point_readout import PointReadout, PointTracker, point_tracking_loss
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError("Choose a new integration result")
    paused = idle_compute_processes()
    torch.set_num_threads(4)
    torch.manual_seed(37)
    recognizer, provenance = load_video_checkpoint(args.checkpoint, "cuda")
    encoder = recognizer.encoder
    del recognizer
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, local_files_only=True)
    dataset = RawVideoClips(provenance["config"]["val"], processor, frames=8, classes=174)
    images, _, sampling = dataset.read(0)
    video = images[None].cuda()
    head = PointReadout().cuda()
    tracker = PointTracker(encoder, head).eval()
    xy = torch.tensor([[[.35, .45], [.65, .55], [.5, .7]]], device="cuda")
    query_frames = torch.tensor([[0, 2, 5]], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batched = tracker(video, xy, query_frames)
        state, max_point_difference, max_visibility_difference = None, 0., 0.
        state_bytes = []
        for frame in range(video.shape[1]):
            new = xy[:, query_frames[0] == frame]
            online, state = tracker.step(video[:, frame], state, new_queries=new)
            active = query_frames[0] <= frame
            dp = (online["points"].float() - batched["points"][:, active, frame].float()).abs().max()
            dv = (online["visibility_logits"].float() - batched["visibility_logits"][:, active, frame].float()).abs().max()
            max_point_difference = max(max_point_difference, float(dp))
            max_visibility_difference = max(max_visibility_difference, float(dv))
            state_bytes.append(dict(encoder=state.encoder_state.nbytes, queries=state.queries.nbytes))
        # Numerical tolerances are engineering checks, not error to ground truth.
        assert max_point_difference < .005, max_point_difference
        assert max_visibility_difference < .025, max_visibility_difference
        changed = video.clone()
        changed[:, 6:] = -changed[:, 6:]
        changed_output = tracker(changed, xy, query_frames)
        future_difference = float((batched["points"][:, :, :6] - changed_output["points"][:, :, :6]).abs().max())
        assert future_difference == 0
        fresh, fresh_state = tracker.step(video[:, 0], new_queries=xy[:, :1])
        torch.testing.assert_close(fresh["points"], xy[:, :1], atol=0, rtol=0)
        assert fresh_state.encoder_state.offset == 1 and fresh_state.queries.xy.shape[1] == 1
    tracker.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = tracker(video, xy, query_frames)
        # Arbitrary targets exercise the loss/backward route; no accuracy claim.
        target = xy[:, :, None].expand(-1, -1, 8, -1).clone()
        target[:, :, 1:, 0] += .01
        occluded = torch.zeros(1, 3, 8, device="cuda", dtype=torch.bool)
        occluded[:, :, 4] = True
        losses = point_tracking_loss(output, target, occluded, query_frames)
    losses["loss"].backward()
    grads = {name: float(p.grad.float().norm()) for name, p in head.named_parameters() if p.grad is not None}
    assert len(grads) == len(list(head.parameters()))
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    assert grads["query_encoder.0.weight"] > 0 and grads["attention.in_proj_weight"] > 0
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in head.parameters())
    result = dict(scope="Real SigLIP2/Mamba-3 integration and one probe optimizer update with artificial targets; not tracking accuracy",
        encoder_checkpoint=provenance, video=sampling, queries=xy.tolist(), query_frames=query_frames.tolist(),
        readout=head.config, readout_parameters=sum(p.numel() for p in head.parameters()),
        max_clip_stream_normalized_xy_difference=max_point_difference,
        max_clip_stream_xy_difference_on_256_grid=max_point_difference * 256,
        max_clip_stream_visibility_logit_difference=max_visibility_difference,
        future_change_max_difference=future_difference, frozen_encoder_has_grad=False,
        gradients=grads, initial_artificial_loss=float(losses["loss"].detach()), state_bytes=state_bytes,
        reset_verified=True, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        initial_paused_compute_pids=paused,
        source_sha256={name:file_hash(name) for name in ("check_point_readout.py", "point_readout.py", "temporal_siglip.py", "video_checkpoint.py")})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("encoder_checkpoint", "gradients", "source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
