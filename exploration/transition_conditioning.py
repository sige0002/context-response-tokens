"""Causal transition-conditioned Mamba-3 input projections.

A matched appearance conditioner has identical parameters and computation.
Temporal differences are established prior art (TEA/TDN); this module is a
performance hypothesis, not a claim of a new temporal-difference principle.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn


class ConditionedProjection(nn.Module):
    def __init__(self, base, mode='difference', hidden=128):
        super().__init__()
        if mode not in ('difference', 'appearance') or type(hidden) is not int or hidden < 1:
            raise ValueError('Invalid conditioner configuration')
        self.base, self.mode = base, mode
        self.branch = nn.Sequential(nn.Linear(base.in_features, hidden, bias=False),
            nn.SiLU(), nn.Linear(hidden, base.out_features, bias=False))
        nn.init.zeros_(self.branch[-1].weight)
        self._active = False
        self._previous = self._last = None

    @property
    def weight(self): return self.base.weight

    @property
    def bias(self): return self.base.bias

    @contextmanager
    def context(self, previous=None):
        if self._active: raise RuntimeError('Overlapping episode execution is unsupported')
        self._active, self._previous, self._last = True, previous, None
        try:
            yield self
        finally:
            self._active, self._previous, self._last = False, None, None

    def forward(self, u):
        if not self._active: raise RuntimeError('Projection requires an explicit clip/episode context')
        if u.ndim != 3 or u.shape[1] < 1: raise ValueError('Expected nonempty [batch,time,width]')
        previous = u[:, :1] if self._previous is None else self._previous
        if previous.shape != u[:, :1].shape or previous.device != u.device or previous.dtype != u.dtype:
            raise ValueError('Previous observation layout/device/dtype changed; reset the episode')
        lagged = torch.cat((previous, u[:, :-1]), dim=1)
        condition = u-lagged if self.mode == 'difference' else u
        # Clone: retaining a view would retain the entire input clip's storage.
        self._last = u[:, -1:].detach().clone()
        return self.base(u) + self.branch(condition)


@dataclass
class TransitionState:
    owner: int
    base_state: object = None
    previous: dict = field(default_factory=dict)

    @property
    def nbytes(self):
        return (0 if self.base_state is None else self.base_state.nbytes) + sum(
            t.numel()*t.element_size() for t in self.previous.values())


class TransitionEncoder(nn.Module):
    def __init__(self, encoder, mode='difference', hidden=128, seed=701):
        super().__init__()
        if encoder.kind != 'mamba': raise ValueError('Requires the native Mamba encoder')
        self.encoder = encoder
        self.conditioning = dict(mode=mode, hidden=hidden, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for adapter in encoder.adapters.values():
                if isinstance(adapter.mixer.in_proj, ConditionedProjection):
                    raise ValueError('Encoder already conditioned')
                adapter.mixer.in_proj = ConditionedProjection(adapter.mixer.in_proj, mode, hidden)

    @property
    def backbone(self): return self.encoder.backbone

    @property
    def kind(self): return self.encoder.kind

    @property
    def projections(self):
        return {k: a.mixer.in_proj for k,a in self.encoder.adapters.items()}

    @contextmanager
    def _contexts(self, previous):
        from contextlib import ExitStack
        with ExitStack() as stack:
            active = {k: stack.enter_context(p.context(previous.get(k))) for k,p in self.projections.items()}
            yield active

    def forward(self, video):
        with self._contexts({}):
            return self.encoder(video)

    def encode_current(self, video): return self(video)[:, -1]

    @torch.no_grad()
    def encode_chunk(self, video, state=None):
        if self.training: raise RuntimeError('Call eval before streaming')
        if state is None: state = TransitionState(id(self))
        if state.owner != id(self): raise ValueError('State belongs to another encoder')
        if state.base_state is None:
            if state.previous: raise ValueError('Unexpected previous inputs in a fresh episode')
        elif set(state.previous) != set(self.projections):
            raise ValueError('Missing previous inputs; reset the episode')
        with self._contexts(state.previous) as active:
            features, base_state = self.encoder.encode_chunk(video, state.base_state)
            previous = {k: p._last for k,p in active.items()}
            if any(p is None for p in previous.values()): raise RuntimeError('A conditioner was not executed')
        state.base_state, state.previous = base_state, previous
        return features, state

    @torch.no_grad()
    def step(self, image, state=None):
        if image.ndim != 4: raise ValueError('Expected [batch,channels,height,width]')
        features, state = self.encode_chunk(image[:, None], state)
        return features[:, 0], state


def attach(model, mode, hidden=128, seed=701):
    model.encoder = TransitionEncoder(model.encoder, mode, hidden, seed)
    return model


def save_model(model, path, readout):
    from train_temporal import atomic_save
    base = model.encoder.encoder
    atomic_save(dict(format='transition-conditioned-video-v1',
        conditioning=model.encoder.conditioning, architecture=base.architecture,
        backbone_config=base.backbone.config.to_dict(),
        checkpoint_revision=getattr(base, 'checkpoint_revision', None),
        readout=dict(readout),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}), path)


def load_model(path, device='cpu'):
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from temporal_siglip import TemporalSiglip
    from temporal_readout import VideoRecognizer
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format'] != 'transition-conditioned-video-v1': raise ValueError('Wrong model format')
    base = TemporalSiglip(SiglipVisionModel(SiglipVisionConfig.from_dict(saved['backbone_config'])),
        **saved['architecture'])
    base.checkpoint_revision = saved['checkpoint_revision']
    head = saved['readout']
    model = VideoRecognizer(base, head['classes'], head['kind'], head['seed'])
    attach(model, **saved['conditioning'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(device).eval()
