"""Qualify the multi-depth baseline without claiming bitwise native training.

Retains the failed native parameter-repeatability gate. Separately verifies
the recurrence, resume with reference scan, and native evaluation/causality.
Only the original native implementation is eligible for benchmark execution.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from check_multidepth_training import UV, verify_native_checkpoint
from gmmix_multidepth import MultiDepthGatedMambaMixSiglip
from pause_training import isolated_gpu
from temporal_readout import VideoRecognizer
from train_video import configure_reproducibility
from video_checkpoint import file_hash


SOURCES = ("qualify_multidepth.py", "check_multidepth_training.py", "train_multidepth_video.py",
    "multidepth_checkpoint.py", "evaluate_multidepth_video.py", "gmmix_multidepth.py",
    "gmmix_siglip.py", "reference_multidepth_training.py", "check_scan_reference.py",
    "diagnose_multidepth_scan.py")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native", default="results/gmmix_multidepth_training_integration_fp32")
    parser.add_argument("--reference", default="results/multidepth_reference_resume")
    parser.add_argument("--scan", default="results/multidepth_scan_diagnostic")
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--video-run", default="results/ssv2_full_mamba_interleaved_seed42_rgbfix")
    args = parser.parse_args()
    output, native, reference, scan = [Path(x).resolve() for x in
                                      (args.output, args.native, args.reference, args.scan)]
    output.mkdir(parents=True, exist_ok=False)
    evidence_paths = [native / "progress.json", native / "repeatability_diagnosis.json",
                      reference / "verification.json", reference / "kernel_reference.json", scan / "report.json"]
    evidence = {str(p):file_hash(p) for p in evidence_paths}
    resume, recurrence, replay = [read(p) for p in evidence_paths[2:]]
    assert resume["passed"] and recurrence["passed"]
    for name, expected in resume["source_plan"]["sources"].items():
        assert file_hash(name) == expected, name
    for name, expected in read(native / "plan.json")["source_sha256"].items():
        assert file_hash(name) == expected, name
    for name, expected in replay["source_sha256"].items():
        assert file_hash(name) == expected, name
    assert recurrence["capture_sha256"] == file_hash(scan / "captured.pt")
    assert recurrence["source_sha256"] == file_hash("check_scan_reference.py")
    assert all(c["max_weight_difference"] == 0 and c["optimizer_equal"] and c["history_equal"]
               and c["cpu_rng_equal"] and c["cuda_rng_equal"] and c["integer_buffers_equal"]
               for c in resume["comparisons"].values())
    assert {"resumed", "repeat"} == set(resume["comparisons"])
    assert any("dB" in c["differences"] or "dC" in c["differences"] for c in replay["calls"])
    report = dict(qualification_version=1,
        decision="eligible_for_benchmark_with_nondeterministic_native_backward",
        scope="Three-update implementation qualification; no recognition or timing claim",
        native_weight_gate="failed_original_5e-6_criterion; not waived or relabeled as passed",
        native_bitwise_training=False, reference_resume_exact=True, evidence_sha256=evidence,
        sources={name:file_hash(name) for name in SOURCES}, comparisons={})
    (output / "plan.json").write_text(json.dumps(report, indent=2)+"\n")
    # This read-only link reuses the completed native fixture. Never rerun a
    # trainer against this directory or alter the failed integration records.
    (output / "direct").symlink_to(os.path.relpath(native / "direct", output), target_is_directory=True)
    torch.set_num_threads(2)
    saved = torch.load(native / "direct/last.pt", map_location="cpu", weights_only=True)
    assert saved["config"]["selective_scan_extension_sha256"] == recurrence["kernel_sha256"]
    assert [h["step"] for h in saved["history"]] == [2, 3]
    for name in ("resumed", "repeat"):
        other = torch.load(native / name / "last.pt", map_location="cpu", weights_only=True)
        assert saved["config"] == other["config"]
        assert all(saved[k] == other[k] for k in ("step", "epoch", "cursor"))
        assert torch.equal(saved["rng"], other["rng"])
        assert all(torch.equal(a, b) for a, b in zip(saved["cuda_rng"], other["cuda_rng"]))
        assert all(torch.isfinite(v).all() for v in other["model"].values() if v.is_floating_point())
        assert all(torch.equal(v, other["model"][k]) for k, v in saved["model"].items()
                   if not v.is_floating_point())
        losses = [abs(a["validation"]["loss"]-b["validation"]["loss"])
                  for a, b in zip(saved["history"], other["history"])]
        assert [h["step"] for h in other["history"]] == [2, 3]
        assert max(losses) < .002
        assert all(a["validation"]["predictions"] == b["validation"]["predictions"]
                   for a, b in zip(saved["history"], other["history"]))
        report["comparisons"][name] = dict(max_validation_loss_difference=max(losses),
            max_weight_difference=max(float((v-other["model"][k]).abs().max())
                                      for k, v in saved["model"].items() if v.is_floating_point()),
            predictions_equal=True)
        del other
    with isolated_gpu(args.pause_pid, args.video_run):
        configure_reproducibility(saved["config"]["seed"])
        initial_encoder = MultiDepthGatedMambaMixSiglip.from_pretrained()
        initial = VideoRecognizer(initial_encoder, saved["config"]["classes"],
            saved["config"]["readout"], saved["config"]["seed"]+1).state_dict()
        spatial = [k for k in initial if k.startswith("encoder.backbone.")]
        assert all(torch.equal(initial[k], saved["model"][k]) for k in spatial)
        assert all(torch.isfinite(v).all() for v in saved["model"].values() if v.is_floating_point())
        counts = []
        for i in range(4):
            prefix = f"encoder.feature_mlps.{i}."
            weights = [k for k in initial if k.startswith(prefix) and k.endswith("weight")]
            assert weights and all(not torch.equal(initial[k], saved["model"][k]) for k in weights)
            counts.append(int(saved["model"][prefix + "bn.num_batches_tracked"]))
        assert counts == [5]*4
        del initial, initial_encoder
        evaluation = output / "evaluation.json"
        with (output / "evaluation.log").open("w") as log:
            subprocess.run(UV + ["evaluate_multidepth_video.py", "--checkpoint", str(native / "direct/last.pt"),
                "--manifest", saved["config"]["val"], "--role", "validation", "--batch-size", "2",
                "--workers", "0", "--output", str(evaluation)], stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=120)
        evaluated = read(evaluation)
        clip = evaluated["ablations"]["full"]["modes"]["clip"]
        assert [r["prediction"] for r in clip["clips"]] == saved["history"][-1]["validation"]["predictions"]
        assert abs(clip["loss"]-saved["history"][-1]["validation"]["loss"]) < 1e-9
        report.update(spatial_tensors_unchanged=len(spatial), enrichment_layers_updated=list(range(4)),
            batchnorm_training_batches=counts, checkpoint_sha256=file_hash(native / "direct/last.pt"),
            evaluation_sha256=file_hash(evaluation), evaluation_matches_saved_validation=True,
            native_checks=verify_native_checkpoint(native / "direct/last.pt"),
            clip_stream={n:r["clip_vs_stream"] for n, r in evaluated["ablations"].items()})
        assert evaluated["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert all(file_hash(p) == h for p, h in evidence.items())
    (output / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
