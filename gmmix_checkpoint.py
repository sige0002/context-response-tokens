"""Load the complete GatedMambaMix video baseline without replacing weights."""
import hashlib
from pathlib import Path

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from gmmix_siglip import GatedMambaMixSiglip, REFERENCE_COMMIT, REFERENCE_FILES
from temporal_readout import VideoRecognizer, READOUT_SHA256


def load_gmmix_video_checkpoint(path, device="cpu"):
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 3 or saved.get("model_type") != "gmmix_siglip_video":
        raise ValueError("Expected a complete GatedMambaMix/SigLIP2 video checkpoint")
    config = saved["config"]
    from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda, causal_conv1d_fn
    if selective_scan_cuda is None:
        raise RuntimeError("Use the baseline uv project with the compiled Mamba-1 selective-scan extension")
    if (hashlib.sha256(Path(selective_scan_cuda.__file__).read_bytes()).hexdigest()
            != config["selective_scan_extension_sha256"]
            or (causal_conv1d_fn is not None) != config["causal_conv1d_available"]):
        raise ValueError("Baseline kernel build/convolution path differs from training")
    root = Path(__file__).resolve().parent
    for name in ("gmmix_siglip.py", "gmmix_checkpoint.py", "temporal_siglip.py", "temporal_readout.py", "video_data.py"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != config["source_sha256"][name]:
            raise ValueError(f"Baseline inference source differs from training: {name}")
    if (config["readout_sha256"] != READOUT_SHA256 or config["core_reference_commit"] != REFERENCE_COMMIT
            or config["core_reference_files"] != REFERENCE_FILES):
        raise ValueError("Published baseline core/readout reference differs")
    encoder = GatedMambaMixSiglip(SiglipVisionModel(SiglipVisionConfig.from_dict(saved["backbone_config"])),
                                 **config["encoder_architecture"])
    encoder.checkpoint_revision = config["revision"]
    model = VideoRecognizer(encoder, config["classes"], config["readout"], config["seed"] + 1)
    model.load_state_dict(saved["model"], strict=True)
    info = dict(step=saved["step"], epoch=saved["epoch"], cursor=saved["cursor"], checkpoint_sha256=digest,
                checkpoint_selection="fixed optimizer step, not best selection", config=config)
    return model.to(device).eval(), info
