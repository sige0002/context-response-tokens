"""Verify that interventions retain the current observation and recurrent meaning."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluate_video import streaming_logits
from exploration.evaluate_video_history import intervene, summarize


def test_history_changes_only_requested_observations():
    video = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1, 1)
    expected = {'full_stream': [1, 2, 3, 4], 'current_only': [4],
                'repeat_current': [4, 4, 4, 4], 'reverse_history': [3, 2, 1, 4]}
    for name, sequence in expected.items():
        assert intervene(video, name).flatten().tolist() == sequence
    assert video.flatten().tolist() == [1, 2, 3, 4]
    for name in expected:
        assert torch.equal(intervene(video[:, -1:], name), video[:, -1:])
    with pytest.raises(ValueError):
        intervene(video, 'reverse_all')


def test_stream_keeps_history_but_resets_between_calls():
    class Encoder:
        def step(self, image, state):
            old = 0 if state is None else state.value
            new = 10 * old + image.flatten(1)
            return new, SimpleNamespace(value=new, nbytes=4)
    model = SimpleNamespace(encoder=Encoder(), readout=lambda x: x)
    video = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1, 1)
    expected = {'full_stream': 1234, 'current_only': 4,
                'repeat_current': 4444, 'reverse_history': 3214}
    for name, result in expected.items():
        first, sizes = streaming_logits(model, intervene(video, name))
        second, _ = streaming_logits(model, intervene(video, name))
        assert first.item() == result and torch.equal(first, second)
        assert sizes == [4] * intervene(video, name).shape[1]


def test_paired_scores_separate_prediction_changes_from_correctness():
    targets = np.array([0, 0, 1, 1])
    full = np.array([[2, 0], [0, 2], [0, 2], [2, 0]], dtype=float)
    changed = np.array([[0, 2], [2, 0], [0, 2], [2, 0]], dtype=float)
    scores = summarize({'full_stream': full, 'current_only': changed}, targets, ['a','b','c','d'])
    assert scores['full_stream']['correct'] == 2
    row = scores['current_only']
    assert row['gained_vs_stream'] == row['lost_vs_stream'] == 1
    assert row['changed_predictions'] == 2 and row['delta_vs_stream_pp'] == 0
