"""A shared point-tracking probe for causal dense encoders.

The probe keeps a fixed appearance/coordinate descriptor from each query frame.
It reads only current-frame tokens afterwards. Queries never attend to each
other. This is an evaluation component, not a claimed encoder contribution or
an exact reproduction of the 4DS/RVM point head.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PointQueries:
    owner: int
    descriptors: torch.Tensor
    xy: torch.Tensor

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in (self.descriptors, self.xy))


class PointReadout(nn.Module):
    def __init__(self, input_dim=768, grid_size=14, dim=1024, heads=8, frequencies=16, hidden=512):
        super().__init__()
        if min(input_dim, grid_size, dim, heads, frequencies, hidden) < 1 or dim % heads:
            raise ValueError("Invalid point-readout configuration")
        self.config = dict(input_dim=input_dim, grid_size=grid_size, dim=dim,
                           heads=heads, frequencies=frequencies, hidden=hidden)
        self.input_norm = nn.LayerNorm(input_dim)
        self.project_tokens = nn.Linear(input_dim, dim)
        self.register_buffer("frequencies", 2. ** torch.arange(frequencies).float() * math.pi)
        axis = (torch.arange(grid_size).float() + .5) / grid_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid_xy", torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        self.project_position = nn.Linear(4 * frequencies, dim, bias=False)
        self.query_encoder = nn.Sequential(nn.Linear(input_dim + 4 * frequencies, hidden),
                                          nn.GELU(), nn.Linear(hidden, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def fourier(self, xy):
        angles = xy.float().unsqueeze(-1) * self.frequencies.float()
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    def validate_tokens(self, tokens):
        if (tokens.ndim != 3 or tokens.shape[1] != self.config["grid_size"]**2
                or tokens.shape[2] != self.config["input_dim"]):
            raise ValueError("Expected [batch, spatial-grid tokens, input dimension]")

    def sample(self, tokens, xy):
        """Bilinear sampling with patch centers at (index+0.5)/grid_size."""
        self.validate_tokens(tokens)
        if (xy.ndim != 3 or xy.shape[0] != tokens.shape[0] or xy.shape[-1] != 2
                or not torch.isfinite(xy).all() or ((xy < 0) | (xy > 1)).any()):
            raise ValueError("Queries must be finite normalized [batch, points, xy]")
        size = self.config["grid_size"]
        # Sampling locations remain FP32, including under BF16 autocast.
        with torch.autocast(tokens.device.type, enabled=False):
            # Match grid_sample's align_corners=False/border convention, using
            # gather so PyTorch can supply its deterministic CUDA backward.
            # CUDA grid_sample backward has no deterministic implementation.
            position = (xy.float() * size - .5).clamp(0, size - 1)
            lower = position.floor().long()
            upper = (lower + 1).clamp(max=size - 1)
            fraction = position - lower
            fx, fy = fraction.unbind(-1)
            sampled = 0.
            for xx, yy, weight in ((lower[..., 0], lower[..., 1], (1-fx)*(1-fy)),
                                   (upper[..., 0], lower[..., 1], fx*(1-fy)),
                                   (lower[..., 0], upper[..., 1], (1-fx)*fy),
                                   (upper[..., 0], upper[..., 1], fx*fy)):
                index = (yy * size + xx)[..., None].expand(-1, -1, tokens.shape[-1])
                sampled = sampled + torch.gather(tokens.float(), 1, index) * weight[..., None]
        return sampled

    def start_queries(self, tokens, xy):
        appearance = self.input_norm(self.sample(tokens, xy))
        descriptors = self.query_encoder(torch.cat((appearance, self.fourier(xy)), dim=-1))
        return PointQueries(id(self), descriptors, xy.clone())

    def append_queries(self, existing, new):
        if new.owner != id(self):
            raise ValueError("Queries belong to another readout")
        if existing is None:
            return new
        if (existing.owner != id(self) or new.owner != id(self)
                or existing.xy.shape[0] != new.xy.shape[0]):
            raise ValueError("Queries belong to another readout or batch")
        return PointQueries(id(self), torch.cat((existing.descriptors, new.descriptors), dim=1),
                            torch.cat((existing.xy, new.xy), dim=1))

    def predict(self, tokens, queries):
        self.validate_tokens(tokens)
        if queries.owner != id(self) or len(queries.xy) != len(tokens):
            raise ValueError("Queries belong to another readout or batch")
        if queries.xy.shape[1] == 0:
            return dict(points=tokens.new_empty(len(tokens), 0, 2),
                        visibility_logits=tokens.new_empty(len(tokens), 0))
        memory = self.project_tokens(self.input_norm(tokens))
        memory = memory + self.project_position(self.fourier(self.grid_xy)).unsqueeze(0)
        normalized = self.memory_norm(memory)
        query = queries.descriptors
        update, _ = self.attention(self.query_norm(query), normalized, normalized, need_weights=False)
        query = query + update
        query = query + self.ffn(self.ffn_norm(query))
        # Coordinates and supplied query positions must retain FP32 precision
        # even when feature/readout matrix products use BF16 autocast.
        raw = self.output(query).float()
        return dict(points=raw[..., :2].sigmoid(), visibility_logits=raw[..., 2])

    def forward(self, features, xy, query_frames):
        """Trainable clip interface; [B,T,N,D] -> predictions [B,Q,T,2].

        Future queries have no outputs until their supplied query frame. Each
        anchor samples only its own query frame, even in this batched interface.
        """
        if features.ndim != 4 or features.shape[1] < 1:
            raise ValueError("Expected nonempty [batch,time,tokens,channels]")
        b, t, n, d = features.shape
        if (xy.ndim != 3 or xy.shape[0] != b or xy.shape[1] < 1 or xy.shape[2] != 2
                or query_frames.shape != xy.shape[:2] or query_frames.dtype not in (torch.int32, torch.int64)
                or ((query_frames < 0) | (query_frames >= t)).any()):
            raise ValueError("Invalid query layout or frame indices")
        q = xy.shape[1]
        # Sample all query locations per frame, then gather their query times.
        # This avoids duplicating the whole feature map once per query.
        expanded_xy = xy[:, None].expand(-1, t, -1, -1).reshape(b * t, q, 2)
        samples = self.sample(features.reshape(b * t, n, d), expanded_xy).reshape(b, t, q, d)
        anchor = samples[torch.arange(b, device=xy.device)[:, None], query_frames,
                         torch.arange(q, device=xy.device)[None]]
        descriptors = self.query_encoder(torch.cat((self.input_norm(anchor), self.fourier(xy)), dim=-1))
        queries = PointQueries(id(self), descriptors[:, None].expand(-1, t, -1, -1).reshape(b * t, q, -1), expanded_xy)
        output = self.predict(features.reshape(b * t, n, d), queries)
        points = output["points"].reshape(b, t, q, 2).transpose(1, 2)
        logits = output["visibility_logits"].reshape(b, t, q).transpose(1, 2)
        times = torch.arange(t, device=xy.device)[None, None]
        active = times >= query_frames[..., None]
        at_query = times == query_frames[..., None]
        points = torch.where(at_query[..., None], xy[:, :, None], points)
        return dict(points=torch.where(active[..., None], points, 0.),
                    visibility_logits=torch.where(active, torch.where(at_query, 10., logits), -30.),
                    active=active)


@dataclass
class PointTrackerState:
    owner: int
    encoder_state: object
    queries: PointQueries | None = None

    @property
    def nbytes(self):
        return self.encoder_state.nbytes + (self.queries.nbytes if self.queries is not None else 0)


class PointTracker(nn.Module):
    def __init__(self, encoder, readout, freeze_encoder=True):
        super().__init__()
        self.encoder, self.readout = encoder, readout
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            encoder.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, video, query_xy, query_frames):
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(video)
        else:
            features = self.encoder(video)
        return self.readout(features, query_xy, query_frames)

    @torch.no_grad()
    def step(self, image, state=None, new_queries=None):
        """Online inference: supply only queries arriving on this frame.

        Query counts may grow; returned query indices are append order. A batch
        shares the number of new queries per step. Use batch 1 for independent
        episodes with different query arrival schedules. Reset with state=None.
        """
        if self.training:
            raise RuntimeError("Call eval() before streaming")
        if state is not None and state.owner != id(self):
            raise ValueError("Tracking state belongs to another model")
        tokens, encoder_state = self.encoder.step(image, None if state is None else state.encoder_state)
        if state is None:
            state = PointTrackerState(id(self), encoder_state)
        state.encoder_state = encoder_state
        added = 0
        if new_queries is not None and new_queries.shape[1]:
            queries = self.readout.start_queries(tokens, new_queries)
            state.queries = self.readout.append_queries(state.queries, queries)
            added = new_queries.shape[1]
        if state.queries is None:
            output = dict(points=tokens.new_empty(len(tokens), 0, 2), visibility_logits=tokens.new_empty(len(tokens), 0))
        else:
            output = self.readout.predict(tokens, state.queries)
            if added:
                output["points"][:, -added:] = new_queries
                output["visibility_logits"][:, -added:] = 10.
        return output, state


def point_tracking_loss(output, target_xy, occluded, query_frames):
    """Mean over clips of visible Huber and post-query visibility BCE.

    Equal clip weights make gradient accumulation equivalent to a larger
    batch even when clips have different visible-point counts.
    """
    prediction, logits = output["points"], output["visibility_logits"]
    if target_xy.shape != prediction.shape or occluded.shape != logits.shape or occluded.dtype != torch.bool:
        raise ValueError("Targets and predictions have different layouts")
    if query_frames.shape != logits.shape[:2]:
        raise ValueError("Invalid query-frame layout")
    active = torch.arange(logits.shape[2], device=logits.device)[None, None] > query_frames[..., None]
    visible = active & ~occluded
    active_counts = active.sum((1, 2))
    if (active_counts == 0).any():
        raise ValueError("No post-query supervision")
    safe_target = torch.where(visible[..., None], target_xy, prediction.detach())
    per_position = F.huber_loss(prediction.float(), safe_target.float(), delta=.05, reduction="none").mean(-1)
    position = ((per_position * visible).sum((1, 2)) / visible.sum((1, 2)).clamp_min(1)).mean()
    per_visibility = F.binary_cross_entropy_with_logits(logits.float(), (~occluded).float(), reduction="none")
    visibility = ((per_visibility * active).sum((1, 2)) / active_counts).mean()
    return dict(loss=100. * position + .1 * visibility, position=position, visibility=visibility)
