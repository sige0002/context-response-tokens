"""Native multi-depth point training, frozen probe restart, and real-video audit.

Small public fixtures only; these checks do not establish tracking accuracy.
Native GMMix backward is known to use nondeterministic atomic reductions.
The restart/repeat test is for the frozen probe, with no encoder backward;
adaptation is separately checked for finite updates and backbone preservation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

import numpy as np
import torch

from pause_training import isolated_gpu
from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from train_multidepth_points import SOURCES


UV = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]
TOLERANCES = dict(weights=5e-6, optimizer=5e-6, loss=1e-3, normalized_xy=1e-3)


def tree_difference(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.is_floating_point():
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            return float((a-b).abs().max()) if a.numel() else 0.
        assert torch.equal(a, b)
        return 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((tree_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (tuple, list)):
        assert type(a) == type(b) and len(a) == len(b)
        return max((tree_difference(x, y) for x, y in zip(a, b)), default=0.)
    assert a == b
    return 0.


def verify_training(output, encoder_checkpoint):
    load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    direct = load(output / "direct/last.pt")
    comparisons = {}
    for name in ("resumed", "repeat"):
        other = load(output / name / "last.pt")
        assert direct["metadata"]["config"] == other["metadata"]["config"]
        a, b = direct["training_state"], other["training_state"]
        for field in ("step", "epoch", "cursor", "rng", "cuda_rng"):
            assert tree_difference(a[field], b[field]) == 0
        maximum = tree_difference(direct["model"], other["model"])
        optimizer = tree_difference(a["optimizer"], b["optimizer"])
        assert maximum < TOLERANCES["weights"], (name, maximum)
        assert optimizer < TOLERANCES["optimizer"], (name, optimizer)
        assert [r["step"] for r in a["history"]] == [2, 3]
        assert [r["step"] for r in b["history"]] == [2, 3]
        loss_difference = max(abs(x["validation"]["loss"]-y["validation"]["loss"])
                              for x,y in zip(a["history"], b["history"]))
        assert loss_difference < TOLERANCES["loss"]
        prediction_difference = 0.
        for step in (2, 3):
            with np.load(output / "direct" / f"validation_{step}.npz") as x, np.load(output / name / f"validation_{step}.npz") as y:
                for key in ("scene_ids", "query_xy", "query_frames", "target_xy", "occluded"):
                    assert np.array_equal(x[key], y[key]), (name, step, key)
                prediction_difference = max(prediction_difference, float(np.abs(x["predicted_xy"]-y["predicted_xy"]).max()))
        assert prediction_difference < TOLERANCES["normalized_xy"]
        comparisons[name] = dict(max_weight_difference=maximum, max_optimizer_difference=optimizer,
            max_loss_difference=loss_difference, max_normalized_xy_difference=prediction_difference,
            rng_and_integer_buffers_exact=True)
        del other
    original = load(encoder_checkpoint)["model"]
    assert all(torch.equal(value, original[key]) for key,value in direct["model"].items() if key.startswith("encoder."))
    assert direct["freeze_encoder"] is True
    adapted = load(output / "adapt/last.pt")
    assert adapted["freeze_encoder"] is False
    assert adapted["training_state"]["step"] == 1
    assert all(torch.isfinite(v).all() for v in adapted["model"].values() if v.is_floating_point())
    assert all(torch.equal(value, original[key]) for key,value in adapted["model"].items() if key.startswith("encoder.backbone."))
    changed = {}
    for component in ("core.mamba_blocks", "core.spatial_blocks", "core.gates", "feature_mlps"):
        for index in range(4):
            prefix = f"encoder.{component}.{index}."
            keys = [key for key,value in adapted["model"].items()
                    if key.startswith(prefix) and "running_" not in key and "num_batches" not in key
                    and not torch.equal(value, original[key])]
            assert keys, prefix
            changed[prefix] = len(keys)
    bn_updates = []
    for index in range(4):
        key = f"encoder.feature_mlps.{index}.bn.num_batches_tracked"
        count = int(adapted["model"][key]-original[key])
        assert count == 2, (key, count)  # Three scenes: one full + one partial microbatch.
        bn_updates.append(count)
    return dict(comparisons=comparisons, probe_encoder_bitwise_unchanged=True,
        adapt_backbone_bitwise_unchanged=True, adapt_changed_parameter_tensors=changed,
        adapt_batchnorm_updates=bn_updates, probe_resume_checked=True,
        native_adapt_bitwise_resume_claim=False, native_adapt_check="One finite update, all four enrichment/core layers update, frozen backbone unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--pause-reference")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    encoder = Path(args.encoder_checkpoint).resolve()
    torch.set_num_threads(2)
    sources = tuple(dict.fromkeys((*SOURCES, "check_multidepth_points.py", "evaluate_multidepth_points.py",
        "evaluate_point_tracks.py", "robotap_data.py", "audit_robotap_predictions.py", "pause_training.py")))
    plan = dict(scope="Engineering fixtures only; not accuracy evidence", encoder_checkpoint=str(encoder),
        encoder_sha256=file_hash(encoder), tolerances=TOLERANCES,
        source_sha256={name:file_hash(name) for name in sources},
        train_manifest_sha256=file_hash("data/point_training_fixture/train.jsonl"),
        val_manifest_sha256=file_hash("data/point_training_fixture/validation.jsonl"))
    write_json(output / "plan.json", plan)
    shared = UV + ["train_multidepth_points.py", "--encoder-checkpoint", str(encoder),
        "--train", "data/point_training_fixture/train.jsonl", "--val", "data/point_training_fixture/validation.jsonl",
        "--steps", "3", "--warmup", "1", "--batch-size", "2", "--accumulation", "2",
        "--frames", "4", "--queries", "8", "--evaluate-every", "2", "--save-every", "1", "--workers", "0", "--development-run"]
    child = None

    def run(name, command):
        nonlocal child
        print(json.dumps(dict(event="launch", name=name, command=command)), flush=True)
        with (output / f"{name}.log").open("a") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect its log")

    try:
        with isolated_gpu(args.pause_pid, args.pause_reference):
            try:
                for name, extra in (("direct", []), ("resumed", ["--stop-after-step", "2"]),
                        ("resumed", ["--resume"]), ("repeat", []), ("adapt", ["--mode", "adapt", "--stop-after-step", "1"])):
                    run(name, shared + ["--output", str(output / name)] + extra)
                checks = verify_training(output, encoder)
                write_json(output / "training_verification.json", checks)
                run("robotap", UV + ["evaluate_multidepth_points.py", "--checkpoint", str(output / "direct/last.pt"),
                    "--output", str(output / "robotap"), "--limit", "1"])
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        # Raw-label metric audit is CPU-only, so resume the primary trainer first.
        run("metric_audit", UV + ["audit_robotap_predictions.py", "--report", str(output / "robotap/summary.json"),
            "--output", str(output / "robotap/raw_label_metric_audit.json"), "--allow-subset"])
        evaluation = read_json(output / "robotap/summary.json")
        audit = read_json(output / "robotap/raw_label_metric_audit.json")
        assert evaluation["videos"] == audit["videos"] == 1
        assert audit["report_sha256"] == file_hash(output / "robotap/summary.json")
        assert evaluation["metrics"] == audit["metrics"]
        assert evaluation["checkpoint"]["checkpoint_sha256"] == file_hash(output / "direct/last.pt")
        assert all(file_hash(name) == expected for name,expected in plan["source_sha256"].items())
        report = dict(**plan, **checks, status="passed_scoped_checks",
            steps=3, train_scenes=3, validation_scenes=1, frames=4, queries=8,
            robotap_videos=1, robotap_frames=evaluation["rows"][0]["frames"],
            robotap_tracks=evaluation["rows"][0]["tracks"],
            evaluation_sha256=file_hash(output / "robotap/summary.json"),
            raw_label_audit_sha256=file_hash(output / "robotap/raw_label_metric_audit.json"),
            probe_checkpoint_sha256=file_hash(output / "direct/last.pt"),
            adapt_checkpoint_sha256=file_hash(output / "adapt/last.pt"))
        write_json(output / "verification.json", report)
        print(json.dumps({k:v for k,v in report.items() if k != "source_sha256"}), flush=True)
    except BaseException as exc:
        write_json(output / "failure.json", dict(error=f"{type(exc).__name__}: {exc}", plan_sha256=file_hash(output / "plan.json")))
        raise


if __name__ == "__main__":
    main()
