from dataclasses import dataclass

import pytest
import torch
from torch import nn

from point_readout import PointReadout, PointTracker, point_tracking_loss


def make_readout():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    return PointReadout(input_dim=8, grid_size=2, dim=16, heads=2, frequencies=3, hidden=12)


def test_bilinear_query_coordinates_and_appearance():
    head = make_readout()
    # x increases along columns, y along rows; four exact patch centers.
    values = torch.tensor([0., 2., 4., 6.])[None, :, None].expand(1, 4, 8)
    centers = torch.tensor([[[.25, .25], [.75, .25], [.25, .75], [.75, .75], [.5, .5]]])
    sampled = head.sample(values, centers)
    torch.testing.assert_close(sampled[0, :, 0], torch.tensor([0., 2., 4., 6., 3.]))
    features = torch.randn(1, 4, 8)
    query = torch.tensor([[[.5, .5]]])
    first = head.start_queries(features, query)
    changed = features.clone()
    changed[:, :, 0] += 5
    second = head.start_queries(changed, query)
    assert not torch.allclose(first.descriptors, second.descriptors)
    with pytest.raises(ValueError, match="normalized"):
        head.start_queries(features, query + 2)
    with pytest.raises(ValueError, match="another readout"):
        head.append_queries(None, make_readout().start_queries(features, query))


def test_bilinear_sampler_matches_grid_sample_values_and_feature_gradients():
    head = make_readout()
    features = torch.randn(2, 4, 8, requires_grad=True)
    xy = torch.tensor([[[0., 0.], [1., 1.], [.5, .5], [.31, .62]],
                       [[.7, .2], [.2, .1], [.9, .6], [.25, .75]]])
    actual = head.sample(features, xy)
    expected = torch.nn.functional.grid_sample(features.transpose(1, 2).reshape(2, 8, 2, 2),
        xy.unsqueeze(2)*2-1, mode="bilinear", padding_mode="border", align_corners=False).squeeze(-1).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    direction = torch.randn_like(actual)
    ag = torch.autograd.grad((actual*direction).sum(), features)[0]
    eg = torch.autograd.grad((expected*direction).sum(), features)[0]
    torch.testing.assert_close(ag, eg, atol=1e-6, rtol=1e-6)


def test_queries_use_own_frame_and_do_not_mix_or_read_future():
    head = make_readout().eval()
    features = torch.randn(1, 5, 4, 8, requires_grad=True)
    xy = torch.tensor([[[.4, .3], [.7, .8]]])
    times = torch.tensor([[0, 2]])
    full = head(features, xy, times)
    # Evaluate every output using just its specified query appearance and the
    # relevant current frame, an independent formulation of the clip contract.
    for q in range(2):
        start = int(times[0, q])
        anchor = head.start_queries(features[:, start], xy[:, q:q + 1])
        for t in range(start + 1, 5):
            direct = head.predict(features[:, t], anchor)
            torch.testing.assert_close(full["points"][:, q:q + 1, t], direct["points"])
    altered = features.detach().clone()
    altered[:, 3:] = torch.randn_like(altered[:, 3:]) * 20
    changed = head(altered, xy, times)
    torch.testing.assert_close(full["points"][:, :, :3], changed["points"][:, :, :3])
    # A query arriving later must not influence the earlier query, including
    # through a shared query self-attention operation.
    other_xy = xy.clone()
    other_xy[:, 1] = torch.tensor([.1, .1])
    other_times = torch.tensor([[0, 4]])
    changed_queries = head(features, other_xy, other_times)
    torch.testing.assert_close(full["points"][:, 0], changed_queries["points"][:, 0])
    one_query = head(features, xy[:, :1], times[:, :1])
    torch.testing.assert_close(full["points"][:, :1], one_query["points"])
    full["points"][:, 0, 1].sum().backward()
    assert features.grad[:, :2].abs().sum() > 0
    assert features.grad[:, 2:].abs().sum() == 0
    assert not full["active"][0, 1, :2].any()
    assert full["points"][0, 1, :2].abs().sum() == 0


@dataclass
class FakeState:
    value: torch.Tensor

    @property
    def nbytes(self):
        return self.value.numel() * self.value.element_size()


class CausalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, video):
        return (video * self.scale).cumsum(1)

    def step(self, image, state=None):
        value = image * self.scale
        if state is not None:
            value = value + state.value
        return value, FakeState(value)


def test_tracker_clip_stream_query_arrival_reset_and_ownership():
    tracker = PointTracker(CausalEncoder(), make_readout()).eval()
    video = torch.randn(1, 5, 4, 8)
    xy = torch.tensor([[[.4, .3], [.7, .8], [.3, .6]]])
    times = torch.tensor([[1, 3, 3]])
    expected = tracker(video, xy, times)
    state = None
    for t in range(5):
        new = xy[:, times[0] == t]
        output, state = tracker.step(video[:, t], state, new_queries=new)
        active = times[0] <= t
        torch.testing.assert_close(output["points"], expected["points"][:, active, t], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(output["visibility_logits"], expected["visibility_logits"][:, active, t], atol=1e-6, rtol=1e-5)
    assert state.queries.descriptors.shape[1] == 3
    assert state.nbytes == state.encoder_state.nbytes + state.queries.nbytes
    reset, reset_state = tracker.step(video[:, 0])
    assert reset["points"].shape == (1, 0, 2)
    assert reset_state.queries is None
    other = PointTracker(CausalEncoder(), make_readout()).eval()
    with pytest.raises(ValueError, match="another model"):
        other.step(video[:, 0], state)
    tracker.train()
    assert not tracker.encoder.training
    with pytest.raises(RuntimeError, match="eval"):
        tracker.step(video[:, 0])


def test_loss_excludes_query_and_invisible_positions_and_trains_readout():
    head = make_readout()
    tracker = PointTracker(CausalEncoder(), head).train()
    video = torch.randn(1, 4, 4, 8)
    xy, times = torch.tensor([[[.4, .3]]]), torch.tensor([[1]])
    output = tracker(video, xy, times)
    target = torch.rand(1, 1, 4, 2)
    occluded = torch.tensor([[[False, False, False, True]]])
    loss = point_tracking_loss(output, target, occluded, times)
    altered = target.clone()
    altered[:, :, :2] = float("nan")
    altered[:, :, 3] = float("nan")
    torch.testing.assert_close(loss["loss"], point_tracking_loss(output, altered, occluded, times)["loss"])
    loss["loss"].backward()
    assert tracker.encoder.scale.grad is None
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert head.query_encoder[0].weight.grad.abs().sum() > 0
    assert head.attention.in_proj_weight.grad.abs().sum() > 0
    occluded.fill_(True)
    invisible = point_tracking_loss(output, target, occluded, times)
    assert invisible["position"] == 0 and torch.isfinite(invisible["loss"])
    with pytest.raises(ValueError, match="No post-query"):
        point_tracking_loss(output, target, occluded, torch.tensor([[3]]))


def test_autocast_preserves_supplied_query_coordinates_in_streaming():
    tracker = PointTracker(CausalEncoder(), make_readout()).eval()
    video = torch.randn(1, 3, 4, 8)
    xy = torch.tensor([[[.35123457, .45123457]]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full = tracker(video, xy, torch.tensor([[0]]))
        online, _ = tracker.step(video[:, 0], new_queries=xy)
    assert online["points"].dtype == torch.float32
    torch.testing.assert_close(online["points"], xy, atol=0, rtol=0)
    torch.testing.assert_close(full["points"][:, :, 0], xy, atol=0, rtol=0)


def test_loss_gradient_is_invariant_to_microbatch_partition_with_unequal_visibility():
    torch.manual_seed(23)
    points = torch.rand(2, 2, 5, 2, requires_grad=True)
    logits = torch.randn(2, 2, 5, requires_grad=True)
    target = torch.rand_like(points)
    occluded = torch.zeros(2, 2, 5, dtype=torch.bool)
    occluded[0, :, 2:] = True
    first = torch.tensor([[0, 1], [1, 0]])
    full = point_tracking_loss(dict(points=points, visibility_logits=logits), target, occluded, first)["loss"]
    full_grad = torch.autograd.grad(full, (points, logits))
    split = sum(point_tracking_loss(dict(points=points[i:i+1], visibility_logits=logits[i:i+1]),
                                    target[i:i+1], occluded[i:i+1], first[i:i+1])["loss"]
                for i in range(2)) / 2
    split_grad = torch.autograd.grad(split, (points, logits))
    torch.testing.assert_close(full, split, atol=1e-7, rtol=1e-6)
    for actual, expected in zip(split_grad, full_grad):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
