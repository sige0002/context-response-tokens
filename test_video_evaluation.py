import torch
from torch import nn

from benchmark_streaming import clone_state
from evaluate_video import evaluate_modes
from temporal_siglip import StreamState


class SumEncoder(nn.Module):
    def step(self, image, state=None):
        if state is None:
            state = StreamState(id(self), len(image), tuple(image.shape[1:]))
            prior = torch.zeros_like(image)
        else:
            prior = state.layers["sum"][0]
        total = prior + image
        state.layers["sum"] = (total,)
        state.offset += 1
        return total, state


class SumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SumEncoder()
        self.readout = nn.Identity()

    def forward(self, video):
        return video.sum(1)


def test_stream_resets_between_batches_and_matches_clip():
    # The second batch would be predicted incorrectly if the first batch's
    # state leaked into it. Different final batch size is also permitted.
    loader = [(torch.tensor([[[5., 0.], [5., 0.]], [[0., 3.], [0., 3.]]]), torch.tensor([0, 1])),
              (torch.tensor([[[0., 1.], [0., 1.]]]), torch.tensor([1]))]
    labels, predictions, loss, paired, sizes = evaluate_modes(SumClassifier(), loader, torch.device("cpu"))
    assert labels == [0, 1, 1]
    assert predictions["clip"] == predictions["stream"] == labels
    assert loss["clip"] == loss["stream"]
    assert paired["logit_max_abs"] == paired["logit_rmse"] == 0
    assert sizes["by_frame"] == [16, 16]


def test_benchmark_state_copies_do_not_grow_or_mutate_prefill():
    encoder = SumEncoder()
    _, base = encoder.step(torch.ones(2, 3))
    saved = base.layers["sum"][0].clone()
    for _ in range(5):
        copied = clone_state(base)
        copied.layers["sum"][0].add_(1)
        _, copied = encoder.step(torch.ones(2, 3), copied)
        assert copied.offset == 2 and base.offset == 1
        torch.testing.assert_close(base.layers["sum"][0], saved, atol=0, rtol=0)
    assert clone_state(None) is None
