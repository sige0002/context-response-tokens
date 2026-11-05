import torch

from exploration.linear_correspondence_metric import LinearCorrespondenceMetric,metric_loss
from exploration.oracle_transport import sample


def test_identity_and_linear_transform_commute_with_subcell_sampling():
    torch.manual_seed(8)
    model=LinearCorrespondenceMetric(dim=8,rank=3)
    features=torch.randn(16,8); xy=torch.rand(11,2)
    torch.testing.assert_close(model(features),features,rtol=0,atol=0)
    with torch.no_grad(): model.up.weight.normal_()
    torch.testing.assert_close(model(sample(features,xy,4)),sample(model(features),xy,4),rtol=2e-5,atol=2e-6)


def test_visibility_masks_invalid_targets_and_loss_trains_the_projection():
    torch.manual_seed(11)
    model=LinearCorrespondenceMetric(dim=8,rank=3)
    reference=torch.randn(2,3,8); current=torch.randn(2,16,8)
    target=torch.rand(2,3,2); visible=torch.tensor([[True,False,True],[False,True,True]])
    original=metric_loss(model,reference,current,target,visible,4)
    changed=target.clone(); changed[~visible]=float('nan')
    torch.testing.assert_close(original,metric_loss(model,reference,current,changed,visible,4),rtol=0,atol=0)
    original.backward()
    assert torch.isfinite(model.up.weight.grad).all() and model.up.weight.grad.norm()>0
    assert torch.isfinite(model.log_scale.grad)
