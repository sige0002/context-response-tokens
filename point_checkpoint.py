"""Atomic, self-contained point tracker checkpoints; no external weight reload."""
import hashlib
from pathlib import Path

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from point_readout import PointReadout, PointTracker
from temporal_siglip import TemporalSiglip


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def point_kernel_environment(encoder_kind):
    if encoder_kind != "gmmix":
        return None
    from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda, causal_conv1d_fn
    if selective_scan_cuda is None:
        raise RuntimeError("GMMix requires the compiled baseline uv project: --project baselines/gmmix")
    return dict(selective_scan_extension_sha256=file_hash(selective_scan_cuda.__file__),
                causal_conv1d_available=causal_conv1d_fn is not None)


def save_point_checkpoint(tracker, path, metadata, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    sources = ["point_readout.py", "temporal_siglip.py", "point_checkpoint.py"]
    if tracker.encoder.kind == "gmmix":
        sources.append("gmmix_siglip.py")
    payload = dict(format_version=1, kind="point_tracker", freeze_encoder=tracker.freeze_encoder,
                   encoder_architecture=tracker.encoder.architecture,
                   backbone_config=tracker.encoder.backbone.config.to_dict(),
                   checkpoint_revision=getattr(tracker.encoder, "checkpoint_revision", None),
                   readout_config=tracker.readout.config,
                   model={name:value.detach().cpu() for name, value in tracker.state_dict().items()},
                   metadata=metadata, training_state=training_state,
                   kernel_environment=point_kernel_environment(tracker.encoder.kind),
                   source_sha256={name:file_hash(source / name) for name in sources})
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_point_checkpoint(path, device="cpu"):
    # Bind to one open inode if a trainer atomically replaces last.pt.
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("kind") != "point_tracker":
        raise ValueError("Expected a self-contained point-tracker checkpoint")
    for name, expected in payload["source_sha256"].items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError(f"Point-tracker source differs: {name}")
    backbone = SiglipVisionModel(SiglipVisionConfig.from_dict(payload["backbone_config"]))
    architecture = payload["encoder_architecture"]
    if architecture["kind"] == "gmmix":
        from gmmix_siglip import GatedMambaMixSiglip
        if payload.get("kernel_environment") != point_kernel_environment("gmmix"):
            raise ValueError("Point-tracker kernel build/convolution path differs from training")
        encoder = GatedMambaMixSiglip(backbone, **architecture)
    else:
        encoder = TemporalSiglip(backbone, **architecture)
    encoder.checkpoint_revision = payload["checkpoint_revision"]
    tracker = PointTracker(encoder, PointReadout(**payload["readout_config"]), payload["freeze_encoder"])
    tracker.load_state_dict(payload.pop("model"), strict=True)
    payload["checkpoint_sha256"] = digest
    return tracker.to(device).eval(), payload
