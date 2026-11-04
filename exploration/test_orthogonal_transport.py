"""Operator tests only: these do not measure recognition or learned motion."""
import torch
import pytest

from exploration.orthogonal_transport import ObservationTransport,pair_rotate,transport


@pytest.mark.parametrize("shape",[(2,14,14,8),(1,5,7,3),(1,1,3,2)])
def test_roundtrip_and_distances_for_fixed_observation(shape):
    generator = torch.Generator().manual_seed(23)
    a = torch.randn(shape,generator=generator,dtype=torch.float64)
    b = torch.randn(shape,generator=generator,dtype=torch.float64)
    angles = torch.randn((*shape[:3],4),generator=generator,dtype=torch.float64)
    qa,qb = transport(a,angles),transport(b,angles)
    torch.testing.assert_close(transport(qa,angles,inverse=True),a,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close((qa-qb).square().sum(),(a-b).square().sum(),rtol=1e-12,atol=1e-12)


def test_observation_conditioned_operator_has_finite_gradients():
    torch.manual_seed(37)
    module = ObservationTransport(3,8).double()
    memory = torch.randn(1,3,5,4,dtype=torch.float64,requires_grad=True)
    previous,current = [torch.randn(1,3,5,3,dtype=torch.float64) for _ in range(2)]
    output,angles = module(memory,previous,current)
    torch.testing.assert_close(output,memory,rtol=0,atol=0)
    (output*torch.randn_like(output)).sum().backward()
    assert torch.isfinite(memory.grad).all()
    assert module.predict[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_transport_order_matters():
    memory = torch.tensor([[[[1.],[2.],[3.]]]],dtype=torch.float64)
    angles = torch.full((1,1,3),.6,dtype=torch.float64)
    ab = pair_rotate(pair_rotate(memory,angles,2,0),angles,2,1)
    ba = pair_rotate(pair_rotate(memory,angles,2,1),angles,2,0)
    assert not torch.allclose(ab,ba)


def test_repeated_half_cell_average_loses_an_alternating_detail():
    # A counterexample to lossless soft alignment, not a learned-model baseline.
    detail = torch.tensor([[[[1.],[-1.],[1.],[-1.]]]],dtype=torch.float64)
    averaged = .5*(detail+detail.roll(1,dims=2))
    assert averaged.norm() == 0
    angles = torch.full((1,1,4,4),.25,dtype=torch.float64)
    moved = detail
    for _ in range(32):
        moved = transport(moved,angles)
    torch.testing.assert_close(moved.norm(),detail.norm(),rtol=1e-12,atol=1e-12)
