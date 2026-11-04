"""Task readouts, kept separate from the proposed temporal encoder.

The attentive probe is the MIT-licensed upstream RVM/4DS-style readout. It is
an evaluation component, not a claimed architectural contribution.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn

READOUT_COMMIT = "59249bf83311bc34bae277e2e8adec287ffe5d0f"
READOUT_SHA256 = "c90f3383d6c4398f9d8631b6345ca7da6f867f1b93bad73d62adfa120ff3914c"


class MeanReadout(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.classifier = nn.Linear(dim, classes)

    def forward(self, features):
        return self.classifier(features.mean(1))


def make_readout(dim, classes, kind):
    if kind == "mean":
        return MeanReadout(dim, classes)
    if kind != "attention":
        raise ValueError("Unknown readout")
    path = Path(__file__).parent / "vendor/frozen-video/models/rvm_modules/rvm_blocks.py"
    if not path.exists():
        raise FileNotFoundError("Run setup.sh to fetch the pinned reference readout")
    if hashlib.sha256(path.read_bytes()).hexdigest() != READOUT_SHA256:
        raise ValueError("Reference readout source differs from the pinned version")
    spec = importlib.util.spec_from_file_location("reference_rvm_blocks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RVMReadout(num_params=dim, input_dim=dim, num_classes=classes,
                             num_queries=1, num_heads=12, mlp_ratio=4,
                             add_temporal_posenc=False)


class VideoRecognizer(nn.Module):
    def __init__(self, encoder, classes=174, readout="attention", seed=43):
        super().__init__()
        self.encoder = encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.readout = make_readout(encoder.backbone.config.hidden_size, classes, readout)

    def forward(self, video):
        if self.encoder.kind == "none":
            # The independent-image baseline only consumes the same current
            # image; avoid recomputing unused history frames.
            video = video[:, -1:]
        return self.readout(self.encoder.encode_current(video))
