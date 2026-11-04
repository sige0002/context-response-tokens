"""Self-contained point checkpoints for the multi-depth SigLIP2/GMMix baseline.

Separate format and source inventory preserve the already running comparisons.
The common point readout and loss are unchanged.
"""
import hashlib
from pathlib import Path

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from gmmix_multidepth import MultiDepthGatedMambaMixSiglip, ENRICHMENT_FILE, ENRICHMENT_SHA256
from gmmix_siglip import REFERENCE_COMMIT, REFERENCE_FILES
from point_checkpoint import file_hash, point_kernel_environment as original_kernel_environment
from point_readout import PointReadout, PointTracker


SOURCES = ("multidepth_point_checkpoint.py", "point_checkpoint.py", "point_readout.py",
           "gmmix_multidepth.py", "gmmix_siglip.py", "temporal_siglip.py",
           "vendor/frozen-video/" + ENRICHMENT_FILE,
           *("vendor/frozen-video/" + name for name in REFERENCE_FILES))


def point_kernel_environment(kind):
    if kind != "gmmix_multidepth":
        raise ValueError("Expected the multi-depth GMMix baseline")
    return original_kernel_environment("gmmix")


def save_point_checkpoint(tracker, path, metadata, training_state=None):
    if tracker.encoder.kind != "gmmix_multidepth":
        raise ValueError("Expected the multi-depth GMMix point tracker")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    payload = dict(format_version=2, kind="multidepth_point_tracker",
        freeze_encoder=tracker.freeze_encoder, encoder_architecture=tracker.encoder.architecture,
        backbone_config=tracker.encoder.backbone.config.to_dict(),
        checkpoint_revision=getattr(tracker.encoder, "checkpoint_revision", None),
        readout_config=tracker.readout.config,
        model={name:value.detach().cpu() for name, value in tracker.state_dict().items()},
        metadata=metadata, training_state=training_state,
        kernel_environment=point_kernel_environment(tracker.encoder.kind),
        reference_commit=REFERENCE_COMMIT, reference_files=REFERENCE_FILES,
        enrichment_file=ENRICHMENT_FILE, enrichment_sha256=ENRICHMENT_SHA256,
        source_sha256={name:file_hash(root / name) for name in SOURCES})
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_point_checkpoint(path, device="cpu"):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2 or payload.get("kind") != "multidepth_point_tracker":
        raise ValueError("Expected a self-contained multi-depth point checkpoint")
    root = Path(__file__).resolve().parent
    if set(payload["source_sha256"]) != set(SOURCES):
        raise ValueError("Point checkpoint source inventory differs")
    for name, expected in payload["source_sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Point checkpoint source differs: {name}")
    if (payload["reference_commit"] != REFERENCE_COMMIT or payload["reference_files"] != REFERENCE_FILES
            or payload["enrichment_file"] != ENRICHMENT_FILE or payload["enrichment_sha256"] != ENRICHMENT_SHA256):
        raise ValueError("Published multi-depth baseline reference differs")
    if payload["kernel_environment"] != point_kernel_environment("gmmix_multidepth"):
        raise ValueError("Point checkpoint kernel build/convolution path differs")
    backbone = SiglipVisionModel(SiglipVisionConfig.from_dict(payload["backbone_config"]))
    encoder = MultiDepthGatedMambaMixSiglip(backbone, **payload["encoder_architecture"])
    encoder.checkpoint_revision = payload["checkpoint_revision"]
    tracker = PointTracker(encoder, PointReadout(**payload["readout_config"]), payload["freeze_encoder"])
    tracker.load_state_dict(payload.pop("model"), strict=True)
    payload["checkpoint_sha256"] = digest
    return tracker.to(device).eval(), payload
