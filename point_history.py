"""Online encoder-state interventions with shared, causal query anchors.

Full-history query descriptors are created only at query arrival, then kept
identical across branches. This isolates the ongoing feature-state path given
those anchors; reset branches are NOT completely history-free trackers.
Absolute frame counters are preserved when clearing caches. Each branch owns
its encoder state. No future target/occlusion argument exists in this API.
"""
from dataclasses import replace

import numpy as np
from PIL import Image
import torch

from temporal_siglip import StreamState


def clear_encoder_memory(encoder, state):
    if isinstance(state, StreamState):
        return replace(state, layers={})
    from gmmix_siglip import GMMixStreamState
    if isinstance(state, GMMixStreamState):
        core = encoder.core.init_state(state.batch, next(encoder.parameters()).device)
        core.seqlen_offset = state.offset
        return replace(state, core=core)
    raise TypeError("Unsupported encoder state; refusing to guess its reset semantics")


@torch.no_grad()
def track_history_interventions(tracker, video, query_xy, query_frames, processor, device, periods=(1, 24)):
    tracker.eval()
    if not periods or any(type(p) is not int or p < 1 for p in periods) or len(set(periods)) != len(periods):
        raise ValueError("Expected distinct positive reset periods")
    xy, first = np.asarray(query_xy, dtype=np.float32), np.asarray(query_frames)
    if (xy.ndim != 2 or xy.shape[1] != 2 or first.shape != xy.shape[:1] or not len(xy)
            or not np.isfinite(xy).all() or ((xy < 0) | (xy > 1)).any()
            or not np.issubdtype(first.dtype, np.integer) or ((first < 0) | (first >= len(video))).any()):
        raise ValueError("Invalid online query layout")
    conditions = {"full":None, **{f"reset_{period}":period for period in periods}}
    states = dict.fromkeys(conditions)
    result = {name:dict(predicted_xy=np.zeros((len(xy),len(video),2), dtype=np.float32),
                       visibility_logits=np.full((len(xy),len(video)), -30., dtype=np.float32),
                       state_bytes=[], resets=[]) for name in conditions}
    queries, order = None, []
    for t, image in enumerate(video):
        pixels = processor(images=Image.fromarray(image), return_tensors="pt").pixel_values.to(device)
        arriving = np.flatnonzero(first == t)
        new_xy = torch.from_numpy(xy[arriving])[None].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            full_tokens, states["full"] = tracker.encoder.step(pixels, states["full"])
            if len(arriving):
                new_queries = tracker.readout.start_queries(full_tokens, new_xy)
                queries = tracker.readout.append_queries(queries, new_queries)
                order.extend(arriving.tolist())
            for name, period in conditions.items():
                if period is None:
                    tokens = full_tokens
                else:
                    if t and t % period == 0:
                        states[name] = clear_encoder_memory(tracker.encoder, states[name])
                        result[name]["resets"].append(t)
                    tokens, states[name] = tracker.encoder.step(pixels, states[name])
                if states[name].offset != t + 1:
                    raise ValueError("Encoder reset changed the absolute frame counter")
                if queries is not None:
                    prediction = tracker.readout.predict(tokens, queries)
                    # Known query coordinates are returned exactly, as in the
                    # standard PointTracker API. Query instants are unscored.
                    if len(arriving):
                        prediction["points"][:, -len(arriving):] = new_xy
                        prediction["visibility_logits"][:, -len(arriving):] = 10.
                    result[name]["predicted_xy"][order,t] = prediction["points"][0].float().cpu().numpy()
                    result[name]["visibility_logits"][order,t] = prediction["visibility_logits"][0].float().cpu().numpy()
                result[name]["state_bytes"].append(dict(encoder=states[name].nbytes,
                    queries=0 if queries is None else queries.nbytes, absolute_frames=states[name].offset))
        if len({id(state) for state in states.values()}) != len(states):
            raise ValueError("Intervention branches share a mutable encoder state")
    if len(order) != len(xy):
        raise ValueError("Some queries were never admitted")
    for values in result.values():
        if not np.isfinite(values["predicted_xy"]).all() or not np.isfinite(values["visibility_logits"]).all():
            raise FloatingPointError("Nonfinite intervention predictions")
    return result
