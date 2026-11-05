import numpy as np
import pytest

from exploration.temporal_density import schedules, paired


def test_nested_anchors_and_no_new_information_in_hold_control():
    anchors = [2, 10, 18, 26, 34, 42, 50, 58]
    result = schedules(anchors)
    assert result['new_15'] == list(range(2, 59, 4))
    assert result['new_29'] == list(range(2, 59, 2))
    for factor, length in [(2, 15), (4, 29)]:
        dense, held = result[f'new_{length}'], result[f'held_{length}']
        assert dense[::factor] == held[::factor] == anchors
        assert set(held) == set(anchors)
        assert all(h <= d for h, d in zip(held, dense))
        assert held[-1] == dense[-1] == anchors[-1]
    assert result['new_29'][::2] == result['new_15']


def test_short_clips_do_not_invent_distinct_frames():
    result = schedules([0, 0, 0, 1, 1, 2, 2, 2])
    assert all(set(indices) <= {0, 1, 2} for indices in result.values())
    assert all(indices == sorted(indices) for indices in result.values())
    assert schedules([0]*8)['new_29'] == [0]*29
    with pytest.raises(ValueError): schedules([0, 1])
    with pytest.raises(ValueError): schedules([0, 2, 1, 3, 4, 5, 6, 7])


def test_paired_gain_and_loss_are_not_just_prediction_changes():
    targets = np.array([0, 0, 1, 1])
    reference = np.eye(3)[[0, 1, 1, 0]]
    changed = np.eye(3)[[1, 0, 1, 2]]
    score = paired(changed, reference, targets)
    assert score['correct'] == 2 and score['accuracy'] == .5
    assert score['gained'] == score['lost'] == 1
    assert score['changed'] == 3 and score['delta_pp'] == 0
