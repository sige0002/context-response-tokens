"""Fixed-width appearance/context output around an existing temporal encoder.

The observation half is captured before the first temporal adapter. No joint
normalization mixes it back with memory. This guarantees history independence
of that half for fixed parameters, not information preservation or accuracy.
Multi-depth features and two-stream representations are not novelty claims.
"""
from pathlib import Path

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_siglip import TemporalSiglip, StreamState


class SeparatedProjection(nn.Module):
    def __init__(self, dim=768, observation_dim=384, mode='separated'):
        super().__init__()
        if mode not in ('separated', 'final') or not 0 < observation_dim < dim:
            raise ValueError('Invalid output partition')
        self.config = dict(dim=dim, observation_dim=observation_dim, mode=mode)
        def branch(width):
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width, bias=False), nn.LayerNorm(width))
        self.observation = branch(observation_dim)
        self.context = branch(dim-observation_dim)

    def forward(self, observation, context):
        if observation.shape != context.shape or context.shape[-1] != self.config['dim']:
            raise ValueError('Expected aligned observation/context feature grids')
        first = context if self.config['mode'] == 'final' else observation
        return torch.cat((self.observation(first), self.context(context)), dim=-1)


class SeparatedEncoder(nn.Module):
    def __init__(self, encoder, observation_dim=384, mode='separated', tap_depth=4,
                 temporal_precision='bfloat16', execution='framewise'):
        super().__init__()
        if encoder.placement != 'interleaved' or tap_depth < 1:
            raise ValueError('Expected interleaved encoder and positive tap depth')
        if tap_depth > len(encoder.backbone.vision_model.encoder.layers):
            raise ValueError('Tap exceeds backbone depth')
        # The tap is BEFORE a temporal insertion at this depth, but may not
        # follow any earlier temporal insertion.
        if any(int(key) < tap_depth-1 for key in encoder.adapters):
            raise ValueError('Observation tap is downstream of temporal mixing')
        if temporal_precision not in ('float32', 'bfloat16'):
            raise ValueError('Unknown temporal precision')
        if execution not in ('framewise', 'batched'):
            raise ValueError('Unknown execution order')
        self.encoder = encoder
        self.projection = SeparatedProjection(encoder.backbone.config.hidden_size, observation_dim, mode)
        self.config = dict(observation_dim=observation_dim, mode=mode, tap_depth=tap_depth,
                           temporal_precision=temporal_precision, execution=execution)

    @property
    def backbone(self):
        return self.encoder.backbone

    @property
    def kind(self):
        return self.encoder.kind

    def _capture(self, function, video, *args):
        captured = []
        handle = self.backbone.vision_model.encoder.layers[self.config['tap_depth']-1].register_forward_hook(
            lambda _module, _inputs, value: captured.append(value))
        try:
            with torch.autocast(video.device.type, dtype=torch.bfloat16,
                    enabled=video.device.type == 'cuda' and self.config['temporal_precision'] == 'bfloat16'):
                result = function(video, *args)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one spatial feature tap per encoder call')
        context = result[0] if isinstance(result, tuple) else result
        observation = captured[0].reshape_as(context)
        # Independent FP32 projections avoid adding a second shape-dependent
        # BF16 matrix product after the native temporal recurrence.
        with torch.autocast(video.device.type, enabled=False):
            output = self.projection(observation.float(), context.float())
        return (output, result[1]) if isinstance(result, tuple) else output

    def forward(self, video):
        if self.config['execution'] == 'batched':
            return self._capture(self.encoder, video)
        return self._framewise(video)[0]

    def _framewise(self, video, state=None):
        if video.ndim != 5 or video.shape[1] < 1 or video.shape[2] != 3:
            raise ValueError('Expected nonempty BTCHW video')
        if state is None:
            state = StreamState(id(self.encoder), video.shape[0], tuple(video.shape[2:]))
        if state.owner != id(self.encoder) or state.batch != video.shape[0] or state.image_shape != tuple(video.shape[2:]):
            raise ValueError('State belongs to another model or input layout')
        outputs = []
        for frame in video.unbind(1):
            # _encode has no inference-only decorator. Native initial/final
            # state gradients connect observations during framewise training.
            outputs.append(self._capture(self.encoder._encode, frame[:, None], state))
            state.offset += 1
        return torch.cat(outputs, 1), state

    def encode_current(self, video):
        return self(video)[:, -1]

    @torch.no_grad()
    def encode_chunk(self, video, state=None):
        if self.training:
            raise RuntimeError('Call eval() before streaming')
        if self.config['execution'] == 'batched':
            return self._capture(self.encoder.encode_chunk, video, state)
        return self._framewise(video, state)

    @torch.no_grad()
    def step(self, image, state=None):
        if image.ndim != 4:
            raise ValueError('Expected BCHW image')
        output, state = self.encode_chunk(image[:, None], state)
        return output[:, 0], state

    def save_checkpoint(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format='separated_encoder_v1', output_config=self.config,
            architecture=self.encoder.architecture, backbone_config=self.backbone.config.to_dict(),
            revision=getattr(self.encoder, 'checkpoint_revision', None),
            state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()})
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(payload, temporary); temporary.replace(path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['format'] != 'separated_encoder_v1':
            raise ValueError('Wrong encoder checkpoint format')
        backbone = SiglipVisionModel(SiglipVisionConfig.from_dict(data['backbone_config']))
        encoder = TemporalSiglip(backbone, **data['architecture'])
        encoder.checkpoint_revision = data['revision']
        model = cls(encoder, **data['output_config'])
        model.load_state_dict(data['state_dict'], strict=True)
        return model.to(device).eval()
