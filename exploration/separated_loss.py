"""Direct point-correspondence supervision; no feature teacher or distillation."""
import torch
from torch.nn import functional as F

from exploration.oracle_transport import basis, sample


def correspondence_loss(features, points, visible, grid=14):
    if features.ndim != 4 or features.shape[0] != 1 or features.shape[1] < 2:
        raise ValueError('Expected one multi-frame training clip')
    if points.shape != (*visible.shape, 2) or points.shape[1] != features.shape[1]:
        raise ValueError('Points must be [queries,time,xy]')
    if not visible[:, 0].all():
        raise ValueError('Queries must be initially visible')
    with torch.autocast(features.device.type, enabled=False):
        values = features[0].float()
        anchor = F.normalize(sample(values[0], points[:, 0], grid), dim=-1)
        current = F.normalize(values[1:], dim=-1)
        logits = 20.*(anchor[None] @ current.transpose(-2, -1))
        target = points[:, 1:].transpose(0, 1)
        valid = visible[:, 1:].transpose(0, 1)
        safe = torch.where(valid[..., None], target, 0.)
        indices, weights = basis(safe.reshape(-1, 2), grid)
        logp = logits.log_softmax(-1).reshape(-1, grid*grid)
        losses = -(logp.gather(-1, indices)*weights).sum(-1).reshape_as(valid)
        return (losses*valid).sum()/valid.sum().clamp_min(1)
