"""Load one atomic full-training checkpoint for coherent encoder/readout weights."""
import hashlib
from pathlib import Path

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_readout import VideoRecognizer, READOUT_SHA256
from temporal_siglip import TemporalSiglip


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_video_checkpoint(path, device="cpu"):
    # Opening the file once binds its inode even if the running trainer replaces
    # last.pt. Hash and deserialize the same open file, not two different saves.
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 2:
        raise ValueError("Expected a format-2 full-training checkpoint")
    config = saved["config"]
    for name in ("temporal_siglip.py", "temporal_readout.py", "video_data.py"):
        if file_hash(Path(__file__).parent / name) != config["source_sha256"][name]:
            raise ValueError(f"Inference source differs from training: {name}")
    if config["readout_sha256"] != READOUT_SHA256:
        raise ValueError("Reference readout version differs")
    # Configs are already cached by training. Do not fetch replacement weights.
    vision_config = SiglipVisionConfig.from_pretrained(config["model"],
                         revision=config["revision"], local_files_only=True)
    encoder = TemporalSiglip(SiglipVisionModel(vision_config), kind=config["kind"],
                             placement=config["placement"])
    encoder.checkpoint_revision = config["revision"]
    model = VideoRecognizer(encoder, config["classes"], config["readout"], config["seed"] + 1)
    model.load_state_dict(saved["model"], strict=True)
    info = dict(step=saved["step"], epoch=saved["epoch"], cursor=saved["cursor"],
                checkpoint_sha256=digest, checkpoint_selection="fixed optimizer step, not best selection",
                config=config)
    return model.to(device).eval(), info
