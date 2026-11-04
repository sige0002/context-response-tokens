"""Read-only audit of saved video weights against the pinned spatial backbone.

Checks checkpoint tensors, optimizer counters, and nonzero temporal gates.
These checks do not establish recognition accuracy or useful history use.
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    # A trainer may replace last.pt atomically; hash and read the same inode.
    with Path(args.checkpoint).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 2:
        raise ValueError("Expected full video-training checkpoint format 2")
    config, state = saved["config"], saved["model"]
    official_path = hf_hub_download(config["model"], "model.safetensors",
        revision=config["revision"], local_files_only=True)
    prefix = "encoder.backbone."
    matched, unequal = [], []
    with safe_open(official_path, framework="pt", device="cpu") as official:
        for key, value in state.items():
            if key.startswith(prefix):
                expected = official.get_tensor(key[len(prefix):])
                (matched if torch.equal(value, expected) else unequal).append(key)
    if not matched or unequal:
        raise ValueError(f"Frozen spatial weights differ: {unequal}")
    nonfinite = [key for key, value in state.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(f"Nonfinite saved weights: {nonfinite}")
    gates = {}
    for name, value in state.items():
        if name.startswith("encoder.adapters.") and name.endswith(".gain"):
            gates[name] = dict(elements=value.numel(), nonzero=int(torch.count_nonzero(value)),
                mean_abs=float(value.abs().mean()), max_abs=float(value.abs().max()))
    counters = []
    for slot in saved["optimizer"]["state"].values():
        for name, value in slot.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite optimizer tensor: {name}")
        counters.append(int(slot["step"]))
    if not counters or min(counters) != saved["step"] or max(counters) != saved["step"]:
        raise ValueError("Optimizer update counters differ from saved step")
    report = dict(scope="Saved-weight audit, not independent inference or proof of useful temporal memory",
        step=saved["step"], checkpoint_sha256=digest, model=config["model"], revision=config["revision"],
        spatial_tensors_equal_to_official=len(matched), all_model_tensors_finite=True,
        model_tensors=len(state), temporal_gates=gates, all_optimizer_tensors_finite=True,
        optimizer_states=len(counters), optimizer_steps_min=min(counters), optimizer_steps_max=max(counters),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
