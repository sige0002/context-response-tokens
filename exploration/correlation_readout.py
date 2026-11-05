"""A correspondence-supervised readout control, not an encoder contribution.

Starts at normalized frozen-feature cosine matching. Learns an optional
residual metric, a within-cell displacement and visibility. Query anchors are
immutable; no recurrent state is learned or claimed by this head.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CorrelationReadout(nn.Module):
    def __init__(self, dim=768, grid=14, hidden=128, local_dim=64, adapt_metric=True):
        super().__init__()
        self.config = dict(dim=dim, grid=grid, hidden=hidden, local_dim=local_dim, adapt_metric=adapt_metric)
        self.metric = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.metric[-1].weight); nn.init.zeros_(self.metric[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.)))
        self.local_project = nn.Linear(dim, local_dim)
        # Reference/current descriptors, initial within-cell xy, peak/margin.
        self.local = nn.Sequential(nn.Linear(2*local_dim+4, hidden), nn.GELU(), nn.Linear(hidden, 3))
        with torch.no_grad():
            self.local[-1].weight[:2].zero_(); self.local[-1].bias[:2].zero_()
        if not adapt_metric:
            self.metric.requires_grad_(False)
            self.log_scale.requires_grad_(False)

    def normalized(self, x):
        return F.layer_norm(x.float(), (self.config['dim'],))

    def descriptor(self, x):
        x = self.normalized(x)
        return F.normalize(x+self.metric(x).float(), dim=-1)

    def scores(self, reference, current):
        return self.log_scale.clamp(0, math.log(100.)).exp() * (self.descriptor(reference) @ self.descriptor(current).transpose(-2,-1))

    def local_output(self, reference, current, xy, indices, logits):
        dim, grid = self.config['dim'], self.config['grid']
        chosen = current.gather(1, indices[..., None].expand(-1,-1,dim))
        query_feature = self.local_project(self.normalized(reference))
        current_feature = self.local_project(self.normalized(chosen))
        within_cell = (xy.float()*grid).remainder(1)-.5
        top = logits.topk(2, dim=-1).values
        evidence = torch.stack((top[...,0], top[...,0]-top[...,1]), -1)
        return self.local(torch.cat((query_feature, current_feature, within_cell, evidence), -1)).float()

    def forward(self, reference, current, xy):
        grid = self.config['grid']
        if current.ndim != 3 or current.shape[1:] != (grid*grid, self.config['dim']):
            raise ValueError('Expected [batch,grid*grid,dim] current features')
        if reference.shape[:2] != xy.shape[:2] or xy.shape[-1] != 2:
            raise ValueError('Invalid query layout')
        logits = self.scores(reference, current)
        indices = logits.argmax(-1)
        raw = self.local_output(reference, current, xy, indices, logits)
        centers = torch.stack((indices % grid+.5, indices//grid+.5), -1)/grid
        delta = .5*raw[...,:2].tanh()
        return dict(points=(centers+delta/grid).clamp(0,1), visibility_logits=raw[...,2],
                    cell_logits=logits, cell_indices=indices, offset=delta)


def correlation_loss(model, reference, current, query_xy, target_xy, visible):
    output = model(reference, current, query_xy)
    grid = model.config['grid']
    safe_target = torch.where(visible[...,None], target_xy.float(), 0.)
    cell = (safe_target*grid).floor().long().clamp(0,grid-1)
    indices = cell[...,1]*grid+cell[...,0]
    weights = visible.float()
    count = weights.sum().clamp_min(1)
    coarse = F.cross_entropy(output['cell_logits'].flatten(0,1), indices.flatten(), reduction='none').reshape_as(weights)
    coarse = (coarse*weights).sum()/count
    # Train refinement at the ground-truth cell. GT is never used by forward.
    raw = model.local_output(reference, current, query_xy, indices, output['cell_logits'])
    delta = .5*raw[...,:2].tanh()
    target_delta = safe_target*grid-(cell.float()+.5)
    offset = F.smooth_l1_loss(delta, target_delta, beta=.1, reduction='none').mean(-1)
    offset = (offset*weights).sum()/count
    visibility = F.binary_cross_entropy_with_logits(output['visibility_logits'], weights)
    return dict(loss=coarse+offset+.1*visibility, coarse=coarse, offset=offset, visibility=visibility)
