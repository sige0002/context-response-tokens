import torch

from exploration.readout_diagnostic import initialize_head, parameter_digest


def test_same_parameters_offset_preserves_query_instead_of_predicting_center():
    absolute = initialize_head('absolute', input_dim=8, grid_size=2, dim=32, heads=4, frequencies=2, hidden=16)
    offset = initialize_head('offset', input_dim=8, grid_size=2, dim=32, heads=4, frequencies=2, hidden=16)
    assert parameter_digest(absolute) == parameter_digest(offset)
    feature = torch.randn(2, 4, 8)
    xy = torch.tensor([[[0., 0.], [.13, .82], [1., 1.]], [[.97, .1], [.45, .6], [.1, .2]]])
    for head in (absolute, offset):
        query = head.start_queries(feature, xy)
        actual = head.predict(torch.randn_like(feature), query)['points']
        expected = xy if head is offset else torch.full_like(xy, .5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_offset_is_trainable_when_target_moves_and_initial_xy_stays_fixed():
    head = initialize_head('offset', input_dim=8, grid_size=2, dim=32, heads=4, frequencies=2, hidden=16)
    feature = torch.randn(1, 4, 8)
    xy = torch.tensor([[[.2, .3], [.6, .7]]])
    query = head.start_queries(feature, xy)
    target = xy+torch.tensor([.1, -.1])
    optimizer = torch.optim.SGD(head.parameters(), lr=.02)
    before = (head.predict(feature, query)['points']-target).square().mean()
    before.backward()
    assert head.output[-1].weight.grad[:2].norm() > 0
    optimizer.step()
    query = head.start_queries(feature, xy)
    after = (head.predict(feature, query)['points']-target).square().mean()
    assert after < before
    torch.testing.assert_close(query.xy, xy, rtol=0, atol=0)
