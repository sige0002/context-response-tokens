import pytest
from summarize_temporal import paired_interval


def test_paired_statistics_and_group_units():
    r = paired_interval([0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], ["a", "a", "b", "b"])
    assert r["groups"] == 2 and r["samples"] == 4
    assert r["delta_percentage_points"] == 100
    assert r["conditional_95_interval_pp"] == [100, 100]
    assert r["a_only_correct"] == 4 and r["b_only_correct"] == 0
    identical = paired_interval([0, 1], [0, 0], [0, 0], ["a", "b"])
    assert identical["conditional_95_interval_pp"] == [0, 0]
    with pytest.raises(ValueError):
        paired_interval([0], [], [0], ["a"])
