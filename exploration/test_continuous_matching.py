import torch

from exploration.continuous_matching import continuous_match,field_sample
from exploration.oracle_transport import sample


def test_field_sampling_agrees_with_existing_coordinates_including_borders():
    torch.manual_seed(31)
    current=torch.randn(16,8)
    xy=torch.tensor([[[0.,0.],[1.,1.],[.125,.125]],[[.4,.7],[.8,.2],[1.1,-.1]]])
    actual=field_sample(current,xy,4)
    expected=sample(current,xy.reshape(-1,2),4).reshape_as(actual)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


def test_fractional_queries_follow_translated_appearance_without_query_coordinates():
    torch.manual_seed(4)
    grid=6
    original=torch.randn(grid,grid,32)
    xy=torch.tensor([[2.6875,2.3125],[1.625,3.375],[3.5,1.5]])/grid
    reference=sample(original.flatten(0,1),xy,grid)
    shifted=torch.roll(original,shifts=(1,-1),dims=(0,1)).flatten(0,1)
    expected=xy+torch.tensor([-1.,1.])/grid
    for normalized in (False,True):
        output=continuous_match(reference,shifted,grid,normalized)
        error=(output['field_top4']['points']-expected).abs().max()
        assert error<.04/grid
        assert (output['field_top4']['scores']>.9999).all()
        assert (output['field_top1']['scores']+2e-5>=output['center']['scores']).all()


def test_constant_field_cannot_invent_query_locations():
    feature=torch.tensor([[1.,2.,3.,4.]]).expand(16,-1)
    reference=feature[:3].clone()
    output=continuous_match(reference,feature,grid=4)
    for value in output.values():
        torch.testing.assert_close(value['points'],torch.full((3,2),.125),atol=0,rtol=0)
