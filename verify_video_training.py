"""Real-video integration test, including an incomplete accumulation batch.

Five training videos and three validation videos; metrics are NOT performance
results. Compare uninterrupted training with optimizer-boundary stop/resume.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/video_training_integration_v2")
    args = parser.parse_args()
    root = Path("data/ssv2/video_training_fixture")
    root.mkdir(exist_ok=True)
    for split, count in (("train", 5), ("val", 3)):
        source = Path("data/ssv2/videos_trainval") / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()[:count]]
        for row in rows:
            row["video"] = "../videos_trainval/" + row["video"]
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = Path(args.output)
    command = [sys.executable, "train_video.py", "--train", str(root / "train.jsonl"),
               "--val", str(root / "val.jsonl"), "--steps", "3", "--warmup", "1",
               "--batch-size", "2", "--accumulation", "2", "--save-every", "1",
               "--evaluate-every", "2", "--workers", "1"]
    subprocess.run(command + ["--output", str(output / "direct")], check=True)
    subprocess.run(command + ["--output", str(output / "direct_repeat")], check=True)
    subprocess.run(command + ["--output", str(output / "resumed"), "--stop-after-step", "1"], check=True)
    stopped = torch.load(output / "resumed/last.pt", map_location="cpu", weights_only=True)
    assert stopped["step"] == 1 and stopped["cursor"] == 4
    del stopped
    subprocess.run(command + ["--output", str(output / "resumed"), "--resume"], check=True)
    locations = {}
    def load(name):
        counts = {}
        def locate(storage, location):
            counts[location] = counts.get(location, 0) + 1
            return storage
        state = torch.load(output / name / "last.pt", map_location=locate, weights_only=True)
        locations[name] = counts
        return state
    direct, repeat, resumed = load("direct"), load("direct_repeat"), load("resumed")
    assert locations["direct"] == locations["resumed"]
    assert direct["step"] == resumed["step"] == 3
    assert direct["epoch"] == resumed["epoch"] == 1
    assert direct["cursor"] == resumed["cursor"] == 4
    errors = {name: float((value - resumed["model"][name]).abs().max())
              for name, value in direct["model"].items()}
    worst = max(errors.values())
    assert worst < 1e-6, sorted(errors.items(), key=lambda x: -x[1])[:5]
    repeat_error = max(float((value - repeat["model"][name]).abs().max())
                       for name, value in direct["model"].items())
    assert repeat_error < 1e-6
    # The released Triton backward uses atomic additions. Quantify same-seed
    # uninterrupted rerun variation separately; do not promise bitwise equality.
    metric_errors = {}
    for name, other in (("repeat", repeat), ("resume", resumed)):
        errors_loss = []
        for a, b in zip(direct["history"], other["history"], strict=True):
            assert a["step"] == b["step"]
            assert a["validation"]["predictions"] == b["validation"]["predictions"]
            errors_loss.append(abs(a["validation"]["loss"] - b["validation"]["loss"]))
        metric_errors[name] = max(errors_loss)
        assert metric_errors[name] < 1e-3, metric_errors
    gains = {name: float(value.abs().sum()) for name, value in direct["model"].items() if name.endswith("gain")}
    assert all(v > 0 for v in gains.values())
    result = dict(scope="Three-update real-video pipeline/resume check; not recognition accuracy",
                  training_videos=5, validation_videos=3, frames=8, steps=3,
                  effective_batch_target=4, included_partial_batch_samples=1,
                  resumed_parameter_max_abs=worst, repeated_parameter_max_abs=repeat_error,
                  parameter_tolerance=1e-6, validation_loss_tolerance=1e-3,
                  validation_loss_max_abs=metric_errors, validation_predictions_equal=True,
                  saved_storage_locations=locations,
                  gain_absolute_sums=gains, readout=direct["config"]["readout"])
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
