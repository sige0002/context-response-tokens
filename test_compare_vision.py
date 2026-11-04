import torch
from compare_vision import SmallViT, metric


def test_confusion_metrics():
    perfect = metric(torch.diag(torch.tensor([4, 5])))
    assert perfect['miou'] == 1 and perfect['pixel_accuracy'] == 1
    # Always background: only its IoU is nonzero, equal to its true prevalence.
    c = torch.tensor([[0, 3], [0, 3]])
    got = metric(c)
    assert got['background_iou'] == .5
    assert got['miou'] == .25
    assert got['pet_iou'] == 0


def test_vit_spatial_output_and_gradients():
    for conv in (False, True):
        model = SmallViT(32, conv)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 128, 8, 8)
        out[:, 0].square().mean().backward()
        assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
