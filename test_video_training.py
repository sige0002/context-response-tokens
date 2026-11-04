from types import SimpleNamespace

import torch
from torch import nn

from temporal_readout import VideoRecognizer, make_readout
from train_video import learning_rate, parameter_groups


class FakeEncoder(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.backbone = SimpleNamespace(config=SimpleNamespace(hidden_size=48))
        self.seen_frames = None

    def encode_current(self, video):
        self.seen_frames = video.shape[1]
        return video[:, -1]


def test_current_only_probe_and_independent_initialization():
    torch.set_num_threads(2)
    model = VideoRecognizer(FakeEncoder("mamba"), classes=3, seed=42)
    altered = torch.randn(2, 5, 7, 48)
    original = altered.clone()
    altered[:, :-1] *= -3
    torch.testing.assert_close(model(original), model(altered), atol=0, rtol=0)
    # The readout has no separate access to the original video/history.
    torch.manual_seed(999)
    second = VideoRecognizer(FakeEncoder("none"), classes=3, seed=42)
    assert all(torch.equal(v, second.readout.state_dict()[k]) for k, v in model.readout.state_dict().items())
    torch.testing.assert_close(model(original), second(original), atol=0, rtol=0)
    assert second.encoder.seen_frames == 1
    # Preserve sensitivity to spatial evidence and differentiability.
    features = original[:, -1].detach().requires_grad_()
    model.readout(features).sum().backward()
    assert features.grad.abs().sum() > 0


def test_schedule_and_no_weight_decay_contract():
    assert learning_rate(0, 10, 2, 1e-4) == 5e-5
    assert learning_rate(1, 10, 2, 1e-4) == 1e-4
    assert abs(learning_rate(9, 10, 2, 1e-4) - 1e-7) < 1e-12
    head = make_readout(48, 3, "mean")
    head.register_parameter("special", nn.Parameter(torch.ones(2, 2)))
    head.special._no_weight_decay = True
    groups = parameter_groups(head, 0.1)
    regular, excluded = [{id(p) for p in group["params"]} for group in groups]
    assert id(head.classifier.weight) in regular
    assert id(head.classifier.bias) in excluded and id(head.special) in excluded
    assert not regular & excluded
    assert regular | excluded == {id(p) for p in head.parameters()}
