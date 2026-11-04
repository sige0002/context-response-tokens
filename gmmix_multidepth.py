"""Multi-depth SigLIP 2 features for the published GatedMambaMix core.

Additional baseline, separate from the pinned final-layer-only experiments.
The four residual BN/MLPs are loaded unchanged from the public DINO wrapper.
We apply the same feature-fusion recipe to SigLIP 2 layers 3, 6, 9, and 12.
This is a SigLIP adaptation, not reproduction of a released DINO checkpoint.

During training, upstream BatchNorm pools the flattened batch/time/token axes.
Only eval-mode inference is strictly causal and independent of clip grouping.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from gmmix_siglip import GatedMambaMixSiglip, ROOT, REFERENCE, REFERENCE_COMMIT, REFERENCE_FILES


ENRICHMENT_FILE = "models/image_encoder/dino_wrappers.py"
ENRICHMENT_SHA256 = "72a4ee993f09995f7b839e2c1d29f0b815c3c65c4b1a069f9c5e421e659a698e"
SOURCES = ("gmmix_multidepth.py", "gmmix_siglip.py", "temporal_siglip.py")


def reference_enrichment_module():
    path = REFERENCE / ENRICHMENT_FILE
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENRICHMENT_SHA256:
        raise ValueError("Published multi-depth feature source differs")
    spec = importlib.util.spec_from_file_location("reference_multidepth_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultiDepthGatedMambaMixSiglip(GatedMambaMixSiglip):
    def __init__(self, backbone, layers=4, state_size=16, spatial_heads=16,
                 mamba_dropout=.1, chunk_size=512, kind="gmmix_multidepth",
                 placement="after_encoder", feature_depths=None,
                 feature_normalization="frozen_post_layernorm", feature_precision="float32"):
        if (kind != "gmmix_multidepth" or feature_normalization != "frozen_post_layernorm"
                or feature_precision != "float32"):
            raise ValueError("Unsupported multi-depth baseline configuration")
        depth = backbone.config.num_hidden_layers
        expected = [depth // 4, depth // 2, 3 * depth // 4, depth]
        if min(expected) < 1 or len(set(expected)) != 4:
            raise ValueError("At least four distinct spatial layers are required")
        if feature_depths is not None and list(feature_depths) != expected:
            raise ValueError("Expected four equally spaced feature depths")
        super().__init__(backbone, layers=layers, state_size=state_size,
                         spatial_heads=spatial_heads, mamba_dropout=mamba_dropout,
                         chunk_size=chunk_size, placement=placement)
        self.kind = kind
        self.feature_depths = tuple(expected)
        enrichment = reference_enrichment_module()._IntermFeatMLP
        self.feature_mlps = nn.ModuleList(enrichment(backbone.config.hidden_size) for _ in expected)
        self.architecture.update(kind=kind, feature_depths=expected,
                                 feature_normalization=feature_normalization,
                                 feature_precision=feature_precision)
        self.train(self.training)

    def extract_depth_features(self, video):
        """Frozen FP32 features, returned before trainable depth enrichment."""
        if video.ndim != 5 or video.shape[1] < 1 or video.shape[2] != 3:
            raise ValueError("Expected nonempty [B,T,3,H,W] video")
        b, t, c, h, w = video.shape
        size = self.backbone.config.image_size
        if (h, w) != (size, size):
            raise ValueError(f"Use checkpoint preprocessing: expected {size}x{size}")
        vision = self.backbone.vision_model
        outputs = []
        with torch.no_grad(), torch.autocast(video.device.type, enabled=False):
            x = vision.embeddings(video.reshape(b * t, c, h, w).float())
            for depth, layer in enumerate(vision.encoder.layers, start=1):
                x = layer(x.float(), attention_mask=None)
                if depth in self.feature_depths:
                    outputs.append(vision.post_layernorm(x.float()))
        return outputs

    def spatial_features(self, video):
        b, t = video.shape[:2]
        frozen_features = self.extract_depth_features(video)
        # Keep this outside no_grad: these four public MLPs/BNs are trainable.
        # BatchNorm sees B*T*N observations, as in the upstream wrapper.
        # This is part of the spatial feature path. Keep enrichment FP32 along
        # with the frozen ViT; the caller's BF16 autocast still applies to GMMix.
        with torch.autocast(video.device.type, enabled=False):
            enriched = [mlp(value.float()) for mlp, value in zip(self.feature_mlps, frozen_features)]
            fused = sum(enriched) / len(enriched)
        return fused.reshape(b, t, *fused.shape[1:])

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=1, kind="gmmix_multidepth_siglip",
            architecture=self.architecture, backbone_config=self.backbone.config.to_dict(),
            checkpoint_revision=getattr(self, "checkpoint_revision", None),
            reference_commit=REFERENCE_COMMIT, reference_files=REFERENCE_FILES,
            enrichment_file=ENRICHMENT_FILE, enrichment_sha256=ENRICHMENT_SHA256,
            source_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
            state_dict={key:value.detach().cpu() for key,value in self.state_dict().items()})
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get("kind") != "gmmix_multidepth_siglip":
            raise ValueError("Expected a multi-depth GMMix/SigLIP checkpoint")
        if (payload["reference_commit"] != REFERENCE_COMMIT or payload["reference_files"] != REFERENCE_FILES
                or payload["enrichment_file"] != ENRICHMENT_FILE
                or payload["enrichment_sha256"] != ENRICHMENT_SHA256):
            raise ValueError("Published multi-depth baseline reference differs")
        if set(payload["source_sha256"]) != set(SOURCES):
            raise ValueError("Incomplete multi-depth inference source record")
        for name, expected in payload["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Multi-depth inference source changed: {name}")
        backbone = SiglipVisionModel(SiglipVisionConfig.from_dict(payload["backbone_config"]))
        model = cls(backbone, **payload["architecture"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.checkpoint_revision = payload["checkpoint_revision"]
        return model.to(device).eval()
