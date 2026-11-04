import copy

import pytest
import torch

from exploration.oracle_anchor_readout import paired_summary
from exploration.oracle_coordinate_readout import centers
from exploration.oracle_transport import FEATURES, basis, sample


def warp_at(xy, grid):
    indices, weights = basis(xy, grid)
    return torch.zeros(grid*grid, grid*grid, dtype=torch.float64).scatter_add_(1, indices, weights)


def test_coordinate_readout_agrees_with_repeated_integer_permutations():
    grid = 4
    xy = centers(grid)
    memory = torch.randn(grid*grid, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    original = memory.clone()
    repeated, coordinates = memory, xy
    for permutation in (torch.arange(16).roll(3), torch.arange(16).flip(0)):
        warp = torch.eye(16, dtype=torch.float64)[permutation]
        repeated = warp @ repeated
        coordinates = warp @ coordinates
        torch.testing.assert_close(sample(memory, coordinates, grid), repeated)
    torch.testing.assert_close(memory, original)


def test_half_cell_round_trip_erases_values_but_not_interior_coordinates():
    grid = 8
    xy = centers(grid)
    checker = ((torch.arange(grid).repeat(grid) % 2)*2-1).double()[:, None]
    offset = torch.tensor([.5/grid, 0.], dtype=torch.float64)
    forward, backward = warp_at(xy+offset, grid), warp_at(xy-offset, grid)
    repeated = backward @ (forward @ checker)
    coordinates = backward @ (forward @ xy)
    readout = sample(checker, coordinates, grid)
    interior = (xy[:, 0] > 2/grid) & (xy[:, 0] < 1-2/grid)
    torch.testing.assert_close(readout[interior], checker[interior])
    assert repeated[interior].abs().max() < 1e-12


@pytest.mark.parametrize("field,value", [("points", 5), ("target_energy", 9.), ("missing", None)])
def test_anchor_comparison_rejects_changed_query_cohort(field, value):
    score = dict(points=4, target_energy=8., normalized_mse=.5)
    features = {name: {"retained": {key: copy.deepcopy(score) for key in
        ("all_visible", "continuous", "after_occlusion", "moving_over_2px")}} for name in FEATURES}
    old = {"scene": {"rows": [dict(stride=1, frame=1, updates=1, methods={"bilinear_flow": features})]}}
    new = {"scene": {"rows": [dict(stride=1, frame=1, updates=1,
        methods={"anchor_bilinear": copy.deepcopy(features)})]}}
    target = new["scene"]["rows"][0]["methods"]["anchor_bilinear"][FEATURES[0]]["retained"]
    if field == "missing":
        target["all_visible"] = None
    else:
        target["all_visible"][field] = value
    with pytest.raises(ValueError, match="cohort|Selection"):
        paired_summary(old, new, dict(scenes=["scene"], frames=2, strides=[1]))
