import numpy as np
import pytest

from exploration.inspect_temporal_decay import carry_windows, summarize


def test_observed_state_excludes_its_own_write_multiplier():
    a = np.array([-100., -.1, -.2, -.3]).reshape(4, 1, 1)
    np.testing.assert_allclose(carry_windows(a, 2).ravel(), [-.3, -.5])
    assert summarize(a, [3])['3']['carry_quantiles'][1] == pytest.approx(np.exp(-.6))


def test_coordinate_and_head_dependence_preserved():
    a = -np.arange(1, 13).reshape(3, 2, 2)/10
    np.testing.assert_allclose(carry_windows(a, 1), a[1:])
    np.testing.assert_allclose(carry_windows(a, 2)[0], a[1]+a[2])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), .1])
def test_invalid_decay_rejected(value):
    with pytest.raises(ValueError, match='nonpositive'):
        carry_windows(np.full((3, 1, 1), value), 1)


def test_unobserved_history_is_not_extrapolated():
    with pytest.raises(ValueError, match='Lag'):
        carry_windows(np.zeros((3, 1, 1)), 3)
