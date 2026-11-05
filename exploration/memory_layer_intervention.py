"""Keep every adapter's current computation; erase selected past states only."""
import torch


@torch.no_grad()
def stream_with_memory(encoder, video, retained_keys):
    """Start a fresh episode and preserve its absolute observation offset.

    Unretained adapters still execute on every frame and write a new state;
    that state is discarded before the following frame. Other adapters can
    carry history through their own states. This is a model intervention,
    not an independently trained architecture or a causal world experiment.
    """
    retained_keys = tuple(retained_keys)
    if (encoder.training or encoder.placement != 'interleaved'
            or len(set(retained_keys)) != len(retained_keys)
            or not set(retained_keys) <= set(encoder.adapters)):
        raise ValueError('Expected an eval interleaved encoder and unique valid state keys')
    if video.ndim != 5 or video.shape[1] == 0:
        raise ValueError('Expected a nonempty [B,T,C,H,W] video')
    state = None; records = []
    for image in video.unbind(1):
        if state is not None:
            for key in tuple(state.layers):
                if key not in retained_keys: del state.layers[key]
        features, state = encoder.step(image, state)
        records.append(dict(offset=state.offset, state_bytes=state.nbytes,
                            state_layer_keys=[k for k in encoder.adapters if k in state.layers]))
    return features, records
