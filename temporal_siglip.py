"""SigLIP with causal, per-coordinate temporal adapters inside its ViT.

This is an experimental adapter architecture, not a reproduction of MEM.
Inputs use the checkpoint's image processor. Cameras are separate batch items.
Streaming caches belong to one episode; discard them at episode boundaries.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

def siso_stream_step(mixer, u, cache):
    """Official Triton SISO kernel with state passing (avoids CuteDSL ABI).

    Mirrors Mamba3.forward's SISO projections; retains the returned four states.
    Input is [B,T,D]. No history replay or approximation of the recurrence.
    """
    from mamba_ssm.modules.mamba3 import heavy_tail_activation
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    b, t, _ = u.shape
    h, p, s, g = mixer.nheads, mixer.headdim, mixer.d_state, mixer.num_bc_heads
    z, x, B, C, dt, A, trap, angles = mixer.in_proj(u).split(
        [mixer.d_inner, mixer.d_inner, s*g, s*g, h, h, h, mixer.num_rope_angles], -1)
    dt = F.softplus(dt + mixer.dt_bias)
    adt = (-heavy_tail_activation(A.float())).clamp(max=-mixer.A_floor) * dt
    B = mixer.B_norm(B.reshape(b, t, g, s))
    C = mixer.C_norm(C.reshape(b, t, g, s))
    result = mamba3_siso_combined(
        Q=C, K=B, V=x.reshape(b, t, h, p),
        ADT=adt.transpose(1, 2), DT=dt.transpose(1, 2), Trap=trap.transpose(1, 2),
        Q_bias=mixer.C_bias.squeeze(1), K_bias=mixer.B_bias.squeeze(1),
        Angles=angles.unsqueeze(2).expand(-1, -1, h, -1).float(),
        D=mixer.D, Z=z.reshape(b, t, h, p), Input_States=cache,
        chunk_size=mixer.chunk_size, return_final_states=True)
    y, *states = result
    return mixer.out_proj(y.reshape(b, t, -1).to(x.dtype)), tuple(states)


class CausalAttention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, cache=None, offset=0, streaming=False):
        b, t, d = x.shape
        # Absolute frame index: same encoding in batched and streaming modes.
        pos = torch.arange(offset, offset + t, device=x.device).float()
        freq = torch.exp(-torch.arange(0, d, 2, device=x.device).float()
                         * (torch.log(torch.tensor(10000., device=x.device)) / d))
        angles = pos[:, None] * freq
        pe = torch.stack((angles.sin(), angles.cos() - 1), -1).flatten(-2)
        q, k, v = self.qkv(x + pe.to(x.dtype)).reshape(
            b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        mask = None
        if streaming:
            queries = offset + torch.arange(t, device=x.device)
            keys = torch.arange(k.shape[2], device=x.device)
            mask = keys[None, :] <= queries[:, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=not streaming)
        y = self.out(y.transpose(1, 2).reshape(b, t, d))
        return y, (k, v) if streaming else None


class TemporalAdapter(nn.Module):
    def __init__(self, dim, width=256, kind="mamba", state_size=64):
        super().__init__()
        if width < 32 or width % 32:
            raise ValueError("width must be a positive multiple of 32")
        self.kind = kind
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, width)
        self.up = nn.Linear(width, dim)
        # Exact pretrained-image path at initialization, with learnable gating.
        self.gain = nn.Parameter(torch.zeros(dim))
        if kind == "mamba":
            from mamba_ssm import Mamba3
            self.mixer = Mamba3(d_model=width, d_state=state_size,
                                headdim=64, expand=2, is_mimo=False)
        elif kind == "attention":
            self.mixer = CausalAttention(width)
        else:
            raise ValueError("kind must be mamba or attention")

    def forward(self, x, cache=None, offset=0, streaming=False):
        b, t, n, d = x.shape
        u = self.down(self.norm(x)).permute(0, 2, 1, 3).reshape(b * n, t, -1).contiguous()
        if self.kind == "attention":
            y, cache = self.mixer(u, cache, offset, streaming)
        elif streaming:
            y, cache = siso_stream_step(self.mixer, u, cache)
        else:
            y = self.mixer(u)
        update = self.up(y).reshape(b, n, t, d).permute(0, 2, 1, 3)
        return x + self.gain * update, cache


@dataclass
class StreamState:
    owner: int
    batch: int
    image_shape: tuple
    offset: int = 0
    layers: dict = field(default_factory=dict)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size()
                   for values in self.layers.values() for t in values)


class TemporalSiglip(nn.Module):
    def __init__(self, backbone, kind="mamba", every=4, width=256,
                 state_size=64, freeze_backbone=True, placement="interleaved"):
        super().__init__()
        if kind not in ("mamba", "attention", "none") or every < 1:
            raise ValueError("Invalid kind or insertion interval")
        self.backbone = backbone
        self.kind = kind
        if placement not in ("interleaved", "late"):
            raise ValueError("placement must be interleaved or late")
        self.placement = placement
        self.freeze_backbone = freeze_backbone
        self.architecture = dict(kind=kind, every=every, width=width,
                                 state_size=state_size, freeze_backbone=freeze_backbone,
                                 placement=placement)
        if kind != "none" and every > backbone.config.num_hidden_layers:
            raise ValueError("Insertion interval exceeds backbone depth: no temporal layer")
        if freeze_backbone:
            backbone.requires_grad_(False)
        depth = backbone.config.num_hidden_layers
        self.adapters = nn.ModuleDict({
            str(i): TemporalAdapter(backbone.config.hidden_size, width, kind, state_size)
            for i in range(every - 1, depth, every) if kind != "none"
        })
        self.train(self.training)

    @classmethod
    def from_pretrained(cls, name=DEFAULT_MODEL, revision=None, **kwargs):
        revision = revision or (DEFAULT_REVISION if name == DEFAULT_MODEL else "main")
        model = cls(SiglipVisionModel.from_pretrained(name, revision=revision), **kwargs)
        model.checkpoint_revision = revision
        return model

    def save_checkpoint(self, path):
        """Self-contained weights/config, excluding episode state and optimizer.

        Frozen weights are included so loading never silently substitutes a
        different backbone. Resume-training optimizer state belongs to a trainer.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=1, architecture=self.architecture,
                       backbone_config=self.backbone.config.to_dict(),
                       checkpoint_revision=getattr(self, "checkpoint_revision", None),
                       state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()})
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        """Load without network access. Returns eval mode with fresh memory."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("Unsupported temporal encoder checkpoint format")
        backbone = SiglipVisionModel(SiglipVisionConfig.from_dict(payload["backbone_config"]))
        model = cls(backbone, **payload["architecture"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.checkpoint_revision = payload["checkpoint_revision"]
        return model.to(device).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode(self, video, state=None):
        if video.ndim != 5 or video.shape[2] != 3 or video.shape[1] < 1:
            raise ValueError("Expected nonempty [B,T,3,H,W] video")
        b, t, c, h, w = video.shape
        size = self.backbone.config.image_size
        if (h, w) != (size, size):
            raise ValueError(f"Use checkpoint image processor: expected {size}x{size}")
        vision = self.backbone.vision_model
        # GB10 BF16 SigLIP exhibits noticeable batch-size-dependent error.
        # Keep pretrained spatial calculations FP32; temporal mixers use the
        # caller's BF16 autocast. This is part of the benchmark configuration.
        with torch.autocast(video.device.type, enabled=False):
            x = vision.embeddings(video.reshape(b * t, c, h, w).float())
        n, d = x.shape[1:]
        for i, layer in enumerate(vision.encoder.layers):
            with torch.autocast(video.device.type, enabled=False):
                x = layer(x.float(), attention_mask=None)
            key = str(i)
            if self.placement == "interleaved" and key in self.adapters:
                x = self._apply_temporal(key, x, b, t, n, d, state)
        # Placement-only control: identical modules, weights, parameter count,
        # and final normalization; all temporal updates occur after spatial layers.
        if self.placement == "late":
            for key in self.adapters:
                x = self._apply_temporal(key, x, b, t, n, d, state)
        with torch.autocast(video.device.type, enabled=False):
            return vision.post_layernorm(x.float()).reshape(b, t, n, d)

    def _apply_temporal(self, key, x, b, t, n, d, state):
        x, cache = self.adapters[key](
            x.reshape(b, t, n, d),
            cache=None if state is None else state.layers.get(key),
            offset=0 if state is None else state.offset,
            streaming=state is not None)
        if state is not None:
            state.layers[key] = cache
        return x.reshape(b * t, n, d)

    def forward(self, video):
        """Return all causal frame features [B,T,N,D]; use [:,-1] downstream."""
        return self._encode(video)

    def encode_current(self, video):
        """Differentiable current-frame representation [B,N,D] from a clip."""
        return self(video)[:, -1]

    @torch.no_grad()
    def encode_chunk(self, video, state=None):
        """Inference over new frames only. Return [B,T,N,D] and updated state.

        A chunk may contain one or more frames. State must belong to the same
        episode and batch ordering. Pass None to reset; no images are cached.
        """
        if self.training:
            raise RuntimeError("Call eval() before streaming")
        if video.ndim != 5 or video.shape[1] < 1:
            raise ValueError("Expected nonempty [B,T,3,H,W]")
        if state is None:
            state = StreamState(id(self), video.shape[0], tuple(video.shape[2:]))
        if (state.owner != id(self) or state.batch != video.shape[0]
                or state.image_shape != tuple(video.shape[2:])):
            raise ValueError("State belongs to another model or input layout; reset it")
        output = self._encode(video, state)
        state.offset += video.shape[1]
        return output, state

    @torch.no_grad()
    def step(self, image, state=None):
        """Inference only. Mutates state. Pass None to start/reset an episode."""
        if image.ndim != 4:
            raise ValueError("Expected [B,3,H,W]")
        output, state = self.encode_chunk(image[:, None], state)
        return output[:, 0], state
