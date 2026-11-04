"""Published GatedMambaMix after a frozen SigLIP 2 spatial encoder.

This is a closest-prior baseline, not a proposed architecture. The temporal
core is imported unchanged from the pinned public implementation. Its public
inference configuration uses four layers, state size 16, and 16 spatial heads.
The backbone/preprocessing and output width match our SigLIP2 experiments.
"""
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_siglip import TemporalSiglip


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "vendor/frozen-video"
REFERENCE_COMMIT = "59249bf83311bc34bae277e2e8adec287ffe5d0f"
REFERENCE_FILES = {
    "models/ssm_modules/mamba_modules.py": "f1ea9f55ae640610c92429f85aa35d8bf2f2fd0e024c0bd9d6748f45b0b1c8a7",
    "models/rvm_modules/rvm_blocks.py": "c90f3383d6c4398f9d8631b6345ca7da6f867f1b93bad73d62adfa120ff3914c",
}


def reference_core():
    for name, expected in REFERENCE_FILES.items():
        if hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published GatedMambaMix source differs: {name}")
    # Upstream imports its RVM helper through the models namespace. Limit the
    # search-path change to importing the exact verified files.
    sys.path.insert(0, str(REFERENCE))
    try:
        module = importlib.import_module("models.ssm_modules.mamba_modules")
        helper = importlib.import_module("models.rvm_modules.rvm_blocks")
    finally:
        sys.path.pop(0)
    if (Path(module.__file__).resolve() != REFERENCE / "models/ssm_modules/mamba_modules.py"
            or Path(helper.__file__).resolve() != REFERENCE / "models/rvm_modules/rvm_blocks.py"):
        raise ValueError("Another models package shadowed the verified reference")
    return module.GatedMambaMixSeqCore


@dataclass
class GMMixStreamState:
    owner: int
    batch: int
    image_shape: tuple
    core: object

    @property
    def offset(self):
        return self.core.seqlen_offset

    @property
    def nbytes(self):
        count = sum(value.numel() * value.element_size()
                    for values in self.core.key_value_memory_dict.values() for value in values)
        lengths = self.core.lengths_per_sample
        return count + (0 if lengths is None else lengths.numel() * lengths.element_size())


class GatedMambaMixSiglip(TemporalSiglip):
    def __init__(self, backbone, layers=4, state_size=16, spatial_heads=16,
                 mamba_dropout=.1, chunk_size=512, kind="gmmix", placement="after_encoder"):
        if kind != "gmmix" or placement != "after_encoder":
            raise ValueError("Published GatedMambaMix is an after-encoder baseline")
        super().__init__(backbone, kind="none", freeze_backbone=True)
        self.kind = "gmmix"
        self.placement = "after_encoder"
        self.architecture = dict(kind=kind, placement=placement, layers=layers, state_size=state_size, spatial_heads=spatial_heads,
                                 mamba_dropout=mamba_dropout, chunk_size=chunk_size)
        self.core_config = dict(d_model=backbone.config.hidden_size, n_layers=layers, d_state=state_size,
            expand=2, mamba_dropout=mamba_dropout, spatial_num_heads=spatial_heads, spatial_mlp_ratio=4,
            gate_hidden=0, max_seqlen=4096,
            num_patches=(backbone.config.image_size // backbone.config.patch_size) ** 2, chunk_size=chunk_size)
        self.core = reference_core()(**self.core_config)
        self.train(self.training)

    def spatial_features(self, video):
        # Same FP32 SigLIP patch features and post-LayerNorm as the no-history
        # baseline. The pretrained attention pooler is not used.
        with torch.no_grad():
            return super()._encode(video)

    def forward(self, video):
        return self.core(self.spatial_features(video), return_sequence=True)

    @torch.no_grad()
    def step(self, image, state=None):
        if self.training:
            raise RuntimeError("Call eval() before streaming")
        if image.ndim != 4:
            raise ValueError("Expected [B,3,H,W]")
        if state is None:
            state = GMMixStreamState(id(self), len(image), tuple(image.shape[1:]),
                                    self.core.init_state(len(image), image.device))
        if (state.owner != id(self) or state.batch != len(image)
                or state.image_shape != tuple(image.shape[1:])):
            raise ValueError("State belongs to another model or input layout; reset it")
        tokens = self.spatial_features(image[:, None])[:, 0]
        output, state.core = self.core.step(tokens, state.core)
        return output, state

    @torch.no_grad()
    def encode_chunk(self, video, state=None):
        if self.training:
            raise RuntimeError("Call eval() before streaming")
        if video.ndim != 5 or video.shape[1] < 1:
            raise ValueError("Expected nonempty [B,T,3,H,W]")
        outputs = []
        for image in video.unbind(1):
            output, state = self.step(image, state)
            outputs.append(output)
        return torch.stack(outputs, 1), state

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=1, kind="gmmix_siglip", architecture=self.architecture,
            backbone_config=self.backbone.config.to_dict(), checkpoint_revision=getattr(self, "checkpoint_revision", None),
            reference_commit=REFERENCE_COMMIT, reference_files=REFERENCE_FILES,
            source_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ("gmmix_siglip.py", "temporal_siglip.py")},
            state_dict={key:value.detach().cpu() for key, value in self.state_dict().items()})
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get("kind") != "gmmix_siglip":
            raise ValueError("Expected a GatedMambaMix/SigLIP checkpoint")
        if payload["reference_commit"] != REFERENCE_COMMIT or payload["reference_files"] != REFERENCE_FILES:
            raise ValueError("Published core reference differs")
        for name, expected in payload["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Baseline inference source changed")
        model = cls(SiglipVisionModel(SiglipVisionConfig.from_dict(payload["backbone_config"])),
                    **payload["architecture"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.checkpoint_revision = payload["checkpoint_revision"]
        return model.to(device).eval()
