"""Identity-initialized linear probe; learns from correspondence annotations."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from exploration.continuous_matching import descriptor
from exploration.oracle_transport import basis


class LinearCorrespondenceMetric(nn.Module):
    def __init__(self,dim=768,rank=32):
        super().__init__()
        self.config=dict(dim=dim,rank=rank)
        self.down=nn.Linear(dim,rank,bias=False)
        self.up=nn.Linear(rank,dim,bias=False)
        nn.init.zeros_(self.up.weight)
        self.log_scale=nn.Parameter(torch.tensor(math.log(20.)))

    def forward(self,x):
        return x+self.up(self.down(x))

    def scores(self,reference,current):
        return self.log_scale.clamp(0,math.log(100.)).exp()*(descriptor(self(reference),True)@descriptor(self(current),True).transpose(-2,-1))


def metric_loss(model,reference,current,target,visible,grid=14):
    logits=model.scores(reference,current)
    safe=torch.where(visible[...,None],target,0.)
    indices,weights=basis(safe.reshape(-1,2),grid)
    log_probability=logits.log_softmax(-1).reshape(-1,grid*grid)
    values=-(log_probability.gather(-1,indices)*weights).sum(-1).reshape_as(visible)
    return (values*visible).sum()/visible.sum().clamp_min(1)
