"""Parameter-free localization in the bilinear field of frozen patch features.

Search improves feature similarity, not necessarily tracking accuracy. The
only query information here is an immutable appearance descriptor: no initial
or current coordinates, motion prior, visibility label, or future observation.
"""
import torch
from torch.nn import functional as F


def descriptor(values, layer_norm=False):
    values=values.float()
    if layer_norm:
        values=F.layer_norm(values,(values.shape[-1],))
    return F.normalize(values,dim=-1)


def field_sample(current, xy, grid):
    """Sample raw features first; descriptor normalization must follow this."""
    if current.ndim!=2 or current.shape[0]!=grid*grid or xy.ndim!=3 or xy.shape[-1]!=2:
        raise ValueError('Expected [grid*grid,dim] features and [query,candidate,xy] coordinates')
    field=current.T.reshape(1,current.shape[-1],grid,grid).float()
    values=F.grid_sample(field,(2*xy.float()-1)[None],mode='bilinear',
                         padding_mode='border',align_corners=False)
    return values[0].permute(1,2,0)


def candidate_scores(reference, current, xy, grid, layer_norm):
    return (reference[:,None]*descriptor(field_sample(current,xy,grid),layer_norm)).sum(-1)


def _best(reference, current, candidates, grid, layer_norm):
    scores=candidate_scores(reference,current,candidates,grid,layer_norm)
    indices=scores.argmax(-1)
    rows=torch.arange(len(reference),device=reference.device)
    return candidates[rows,indices],scores[rows,indices]


@torch.no_grad()
def continuous_match(reference, current, grid=14, layer_norm=False,
                     topk=4, refinement_steps=(.125,.0625,.03125)):
    if reference.ndim!=2 or current.ndim!=2 or reference.shape[-1]!=current.shape[-1]:
        raise ValueError('Expected query and current feature matrices')
    if current.shape[0]!=grid*grid or not 1<=topk<=grid*grid:
        raise ValueError('Invalid spatial grid or candidate count')
    if not torch.isfinite(reference).all() or not torch.isfinite(current).all():
        raise ValueError('Nonfinite features')
    ref=descriptor(reference,layer_norm)
    logits=ref@descriptor(current,layer_norm).T
    best=logits.argmax(-1)
    center=torch.stack((best%grid+.5,best//grid+.5),-1)/grid
    # Stable ordering preserves the first argmax when coarse similarities tie.
    selected=logits.argsort(dim=-1,descending=True,stable=True)[:,:topk]
    centers=torch.stack((selected%grid+.5,selected//grid+.5),-1)/grid
    axis=torch.tensor([-.5,-.25,0.,.25,.5],device=current.device)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    offsets=torch.stack((xx,yy),-1).reshape(-1,2)/grid
    rows=torch.arange(len(ref),device=current.device)
    outputs={'center':dict(points=center,scores=logits[rows,best])}
    for name,count in (('field_top1',1),('field_top4',topk)):
        candidates=(centers[:,:count,None]+offsets[None,None]).flatten(1,2).clamp(0,1)
        # Prefer the old position on an exact tie, including border plateaus.
        candidates=torch.cat((center[:,None],candidates),1)
        position,score=_best(ref,current,candidates,grid,layer_norm)
        for step in refinement_steps:
            axis=torch.tensor([-step,0.,step],device=current.device)
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            offsets_local=torch.stack((xx,yy),-1).reshape(-1,2)/grid
            candidates=(position[:,None]+offsets_local[None]).clamp(0,1)
            candidates=torch.cat((position[:,None],candidates),1)
            position,score=_best(ref,current,candidates,grid,layer_norm)
        if (score+2e-5<outputs['center']['scores']).any():
            raise AssertionError('Refinement lost the retained coarse candidate')
        outputs[name]=dict(points=position,scores=score)
    return outputs
