"""Native GPU integration of the additional multi-depth GMMix baseline.

Five training and three validation videos, three updates: an implementation
check, not recognition evidence. The primary trainer is restored in finally.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

import torch
from transformers import SiglipImageProcessor

from gmmix_multidepth import MultiDepthGatedMambaMixSiglip
from multidepth_checkpoint import load_multidepth_video_checkpoint
from pause_training import isolated_gpu
from temporal_readout import VideoRecognizer
from train_video import configure_reproducibility
from video_checkpoint import file_hash
from video_data import RawVideoClips


UV = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]
SOURCES = ("check_multidepth_training.py", "train_multidepth_video.py", "multidepth_checkpoint.py",
           "evaluate_multidepth_video.py", "gmmix_multidepth.py", "gmmix_siglip.py")


def verify_native_checkpoint(checkpoint):
    model, info = load_multidepth_video_checkpoint(checkpoint, device="cuda")
    config = info["config"]
    processor = SiglipImageProcessor.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
    dataset = RawVideoClips(config["val"], processor, config["frames"], config["classes"])
    video = dataset[0][0][None].cuda()
    buffers = {k:v.clone() for k,v in model.encoder.feature_mlps.named_buffers()}
    state = None
    outputs, sizes = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model.encoder(video)
        for frame in video.unbind(1):
            features, state = model.encoder.step(frame, state)
            outputs.append(features)
            sizes.append(state.nbytes)
        streamed = torch.stack(outputs, 1)
        changed = video.clone()
        changed[:, 4:] = -changed[:, 4:]
        altered = model.encoder(changed)
    relative = float((full.float() - streamed.float()).norm() / full.float().norm().clamp_min(1e-12))
    causal_difference = float((full[:, :4] - altered[:, :4]).abs().max())
    assert relative < .01, relative
    assert causal_difference == 0., causal_difference
    assert len(set(sizes)) == 1 and sizes[0] > 0 and state.offset == video.shape[1]
    assert all(torch.equal(value, buffers[key]) for key,value in model.encoder.feature_mlps.named_buffers())
    result = dict(frames=video.shape[1], feature_relative_l2=relative,
                  future_change_max_abs=causal_difference, state_bytes=sizes[0],
                  state_offset=state.offset, batchnorm_unchanged_in_eval=True)
    del model, video, state, full, streamed, changed, altered, outputs, buffers
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--video-run", default="results/ssv2_full_mamba_interleaved_seed42_rgbfix")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    common = UV + ["train_multidepth_video.py", "--train", "data/ssv2/video_training_fixture/train.jsonl",
        "--val", "data/ssv2/video_training_fixture/val.jsonl", "--steps", "3", "--warmup", "1",
        "--frames", "8", "--batch-size", "2", "--accumulation", "2", "--evaluate-every", "2",
        "--save-every", "1", "--workers", "0"]
    report = dict(scope="Three-update native multi-depth GMMix pipeline; not recognition accuracy",
        training_videos=5, validation_videos=3, steps=3, comparisons={},
        predeclared_tolerances=dict(max_weight_difference=5e-6, max_validation_loss_difference=.002,
                                  clip_stream_feature_relative_l2=.01, future_change_max_abs=0.))
    (output / "plan.json").write_text(json.dumps(dict(**report,
        source_sha256={name:file_hash(name) for name in SOURCES}), indent=2) + "\n")

    def run(command, label):
        print(json.dumps(dict(event="launch", label=label, command=command)), flush=True)
        with (output / (label + ".log")).open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
        if code:
            raise RuntimeError(f"{label} failed ({code}); inspect its log")

    with isolated_gpu(args.pause_pid, args.video_run):
        run(common + ["--output", str(output / "direct")], "direct")
        run(common + ["--output", str(output / "resumed"), "--stop-after-step", "2"], "resumed")
        run(common + ["--output", str(output / "resumed"), "--resume"], "resumed")
        run(common + ["--output", str(output / "repeat")], "repeat")
        saved = torch.load(output / "direct/last.pt", map_location="cpu", weights_only=True)
        assert [row["step"] for row in saved["history"]] == [2, 3]
        for name in ("resumed", "repeat"):
            other = torch.load(output / name / "last.pt", map_location="cpu", weights_only=True)
            assert saved["config"] == other["config"]
            assert all(saved[k] == other[k] for k in ("step", "epoch", "cursor"))
            maximum = max(float((value-other["model"][key]).abs().max())
                          for key,value in saved["model"].items() if value.is_floating_point())
            assert all(torch.equal(value, other["model"][key])
                       for key,value in saved["model"].items() if not value.is_floating_point())
            assert [row["step"] for row in other["history"]] == [2,3]
            loss_difference = max(abs(a["validation"]["loss"]-b["validation"]["loss"])
                                  for a,b in zip(saved["history"],other["history"]))
            predictions_equal = all(a["validation"]["predictions"] == b["validation"]["predictions"]
                                    for a,b in zip(saved["history"],other["history"]))
            result = dict(max_weight_difference=maximum, max_validation_loss_difference=loss_difference,
                          predictions_equal=predictions_equal)
            report["comparisons"][name] = result
            (output / "progress.json").write_text(json.dumps(report, indent=2)+"\n")
            assert maximum < 5e-6 and loss_difference < .002 and predictions_equal, result
            assert torch.equal(saved["rng"],other["rng"])
            assert all(torch.equal(a,b) for a,b in zip(saved["cuda_rng"],other["cuda_rng"]))
            del other
        configure_reproducibility(saved["config"]["seed"])
        initial_encoder = MultiDepthGatedMambaMixSiglip.from_pretrained()
        initial = VideoRecognizer(initial_encoder, saved["config"]["classes"], saved["config"]["readout"],
                                  saved["config"]["seed"]+1).state_dict()
        spatial_keys = [key for key in initial if key.startswith("encoder.backbone.")]
        assert all(torch.equal(initial[key],saved["model"][key]) for key in spatial_keys)
        updated = []
        bn_counts = []
        for index in range(4):
            prefix = f"encoder.feature_mlps.{index}."
            weights = [key for key in initial if key.startswith(prefix) and key.endswith("weight")]
            assert all(not torch.equal(initial[key],saved["model"][key]) for key in weights)
            assert all(torch.isfinite(saved["model"][key]).all() for key in weights)
            bn_counts.append(int(saved["model"][prefix + "bn.num_batches_tracked"]))
            updated.append(index)
        assert bn_counts == [5]*4, bn_counts
        report.update(spatial_tensors_unchanged=len(spatial_keys), enrichment_layers_updated=updated,
                      batchnorm_training_batches=bn_counts)
        del initial, initial_encoder
        evaluation = output / "evaluation.json"
        run(UV + ["evaluate_multidepth_video.py", "--checkpoint", str(output / "direct/last.pt"),
            "--manifest", saved["config"]["val"], "--role", "validation", "--batch-size", "2",
            "--workers", "0", "--output", str(evaluation)], "evaluation")
        evaluated = json.loads(evaluation.read_text())
        clip = evaluated["ablations"]["full"]["modes"]["clip"]
        assert [row["prediction"] for row in clip["clips"]] == saved["history"][-1]["validation"]["predictions"]
        assert abs(clip["loss"] - saved["history"][-1]["validation"]["loss"]) < 1e-9
        report.update(checkpoint_sha256=file_hash(output / "direct/last.pt"),
                      evaluation_sha256=file_hash(evaluation), evaluation_matches_saved_validation=True,
                      native_checks=verify_native_checkpoint(output / "direct/last.pt"),
                      clip_stream={name:result["clip_vs_stream"] for name,result in evaluated["ablations"].items()},
                      sources={name:file_hash(name) for name in SOURCES})
        assert evaluated["checkpoint_sha256"] == report["checkpoint_sha256"]
        (output / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
