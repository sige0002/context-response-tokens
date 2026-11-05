import torch
from torch.nn import functional as F

from exploration.correlation_readout import CorrelationReadout, correlation_loss


def test_initial_output_matches_frozen_cosine_and_tracks_permuted_cells():
    torch.manual_seed(17)
    head = CorrelationReadout(dim=8, grid=2, hidden=8, local_dim=4)
    original = torch.randn(1,4,8)
    perm = torch.tensor([2,0,3,1])
    current = original[:,perm]
    xy = torch.tensor([[[.1,.1],[.7,.1],[.1,.7],[.7,.7]]])
    out = head(original,current,xy)
    normalized = F.normalize(F.layer_norm(original,(8,)),dim=-1)
    target = normalized @ F.normalize(F.layer_norm(current,(8,)),dim=-1).transpose(-2,-1)
    torch.testing.assert_close(out['cell_logits']/head.log_scale.exp(),target)
    expected_index = torch.argsort(perm)[None]
    assert torch.equal(out['cell_indices'],expected_index)
    centers = torch.stack((expected_index%2+.5, expected_index//2+.5),-1)/2
    torch.testing.assert_close(out['points'],centers,rtol=0,atol=0)


def test_invisible_coordinates_do_not_affect_loss_and_both_paths_get_gradients():
    torch.manual_seed(9)
    head = CorrelationReadout(dim=8,grid=2,hidden=8,local_dim=4)
    reference,current = torch.randn(1,3,8),torch.randn(1,4,8)
    xy = torch.rand(1,3,2)
    target = torch.tensor([[[.12,.33],[.7,.7],[.8,.22]]])
    visible = torch.tensor([[True,False,True]])
    a = correlation_loss(head,reference,current,xy,target,visible)
    changed = target.clone(); changed[:,1] = float('nan')
    b = correlation_loss(head,reference,current,xy,changed,visible)
    torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)
    a['loss'].backward()
    assert torch.isfinite(head.metric[-1].weight.grad).all()
    assert head.metric[-1].weight.grad.norm()>0
    assert head.local[-1].weight.grad[:2].norm()>0
