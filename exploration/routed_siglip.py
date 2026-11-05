"""Fixed-reference correspondence routing around temporal Mamba-3.

This control tests content addressing versus fixed image-coordinate states.
The anchor descriptors are initialized from frame zero and remain unchanged.
No object identity guarantee, visibility protection, new-slot birth or novelty
claim. Both training and inference keep the official four Mamba-3 states.
"""
from pathlib import Path

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_siglip import TemporalSiglip, siso_stream_step
from exploration.appearance_routing import AppearanceMetric, route


class RoutedAdapter(nn.Module):
    def __init__(self, dim, width=256, state_size=64, metric_hidden=128, metric_width=64,
                 routing="appearance"):
        super().__init__()
        if routing not in ("appearance", "coordinate"):
            raise ValueError("Unknown routing control")
        if width < 32 or width % 32:
            raise ValueError("Adapter width must be a positive multiple of 32")
        from mamba_ssm import Mamba3
        self.routing = routing
        self.norm, self.down = nn.LayerNorm(dim), nn.Linear(dim, width)
        self.up, self.gain = nn.Linear(width, dim), nn.Parameter(torch.zeros(dim))
        self.metric = AppearanceMetric(dim, metric_hidden, metric_width)
        self.mixer = Mamba3(d_model=width, d_state=state_size, headdim=64, expand=2, is_mimo=False)

    def forward(self, x, cache=None, offset=0, streaming=False):
        b, t, n, d = x.shape
        if t < 1 or (cache is not None and not streaming):
            raise ValueError("Nonempty clips; cached calls must be streaming")
        if (cache is None) != (offset == 0):
            raise ValueError("Cache and frame offset disagree")
        # Descriptors and softmax stay FP32. Recurrent projections follow the
        # caller's autocast, as in the fixed-coordinate baseline.
        with torch.autocast(x.device.type, enabled=False):
            descriptor = self.metric.encode(x.float())
            reference = descriptor[:, 0] if cache is None else cache[0]
            logits = self.metric.logits(reference[:, None], descriptor)
        u = self.down(self.norm(x))
        if self.routing == "appearance":
            routed, read = route(logits, u)
            if offset == 0:
                # Initial slot identity follows the image grid exactly.
                routed = torch.cat((u[:, :1], routed[:, 1:]), dim=1)
                eye = torch.eye(n, device=x.device, dtype=read.dtype).expand(b, 1, n, n)
                read = torch.cat((eye, read[:, 1:]), dim=1)
        else:
            routed, read = u, None
        sequence = routed.permute(0, 2, 1, 3).reshape(b*n, t, -1).contiguous()
        if streaming:
            y, states = siso_stream_step(self.mixer, sequence, None if cache is None else cache[1:])
            new_cache = (reference, *states)
        else:
            y, new_cache = self.mixer(sequence), None
        y = y.reshape(b, n, t, -1).permute(0, 2, 1, 3)
        if read is not None:
            y = read.to(y.dtype) @ y
        return x + self.gain * self.up(y), new_cache


class RoutedSiglip(TemporalSiglip):
    def __init__(self, backbone, every=4, width=256, state_size=64,
                 freeze_backbone=True, placement="interleaved", metric_hidden=128,
                 metric_width=64, routing="appearance"):
        # Reuse the unchanged spatial path, precision, ownership and cache API.
        super().__init__(backbone, kind="none", every=every, width=width,
                         state_size=state_size, freeze_backbone=freeze_backbone, placement=placement)
        if every > backbone.config.num_hidden_layers:
            raise ValueError("Insertion interval exceeds backbone depth")
        self.kind = "mamba"
        self.architecture = dict(every=every, width=width, state_size=state_size,
            freeze_backbone=freeze_backbone, placement=placement, metric_hidden=metric_hidden,
            metric_width=metric_width, routing=routing)
        self.adapters = nn.ModuleDict({str(i):RoutedAdapter(backbone.config.hidden_size, width,
            state_size, metric_hidden, metric_width, routing)
            for i in range(every-1, backbone.config.num_hidden_layers, every)})
        self.train(self.training)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version="routed_siglip_v1", architecture=self.architecture,
            backbone_config=self.backbone.config.to_dict(),
            checkpoint_revision=getattr(self, "checkpoint_revision", None),
            state_dict={k:v.detach().cpu() for k,v in self.state_dict().items()})
        temporary = path.with_name(path.name+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != "routed_siglip_v1":
            raise ValueError("Not a routed encoder checkpoint")
        backbone = SiglipVisionModel(SiglipVisionConfig.from_dict(payload["backbone_config"]))
        model = cls(backbone, **payload["architecture"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.checkpoint_revision = payload["checkpoint_revision"]
        return model.to(device).eval()
