import torch

from exploration.oracle_transport import basis,cross_gram,orthogonal_fit,sample,track_partition,operators


def test_bilinear_basis_matches_explicit_sampling_and_borders():
    xy=torch.tensor([[0.,0.],[.5,.5],[1.,1.]],dtype=torch.float64)
    values=torch.arange(4,dtype=torch.float64)[:,None]
    torch.testing.assert_close(sample(values,xy,2).flatten(),torch.tensor([0.,1.5,3.],dtype=torch.float64))
    indices,weights=basis(xy,2)
    matrix=torch.zeros(3,4,dtype=torch.float64).scatter_add_(1,indices,weights)
    torch.testing.assert_close(cross_gram(xy,xy,2),matrix.T@matrix/3)


def test_oracle_recovers_even_permutation_on_held_out_memory():
    centers=torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]],dtype=torch.float64)
    previous=centers[[1,2,0,3]]
    maps,_=operators(previous,centers,2,1e-8)
    memory=torch.randn(4,11,dtype=torch.float64,generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(maps['orthogonal_oracle']@memory,memory[[1,2,0,3]],atol=1e-6,rtol=1e-6)


def test_orthogonality_cannot_copy_one_location_into_two():
    # If two image locations need the same memory value, an isometry alone
    # cannot realize that semantic readout for arbitrary memory contents.
    target=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
    q=orthogonal_fit(target,0.)
    torch.testing.assert_close(q.T@q,torch.eye(2,dtype=torch.float64))
    assert float((q-target).square().sum()) > 1.


def test_fitting_and_query_tracks_are_disjoint_and_repeatable():
    import numpy as np
    a,b=track_partition(100,17,40,40)
    c,d=track_partition(100,17,40,40)
    assert not set(a)&set(b)
    assert np.array_equal(a,c) and np.array_equal(b,d)
