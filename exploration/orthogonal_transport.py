"""Exploratory spatial memory transport, independent of the running models.

This is an operator prototype, not a trained vision encoder or a Mamba-3 kernel.
For fixed angles, transport is orthogonal. Angles may be learned from current
and previous observations; they must not depend on the memory when invoking
the conditional distance-preservation statement.
"""
import torch
from torch import nn


def pair_rotate(memory, angles, axis, offset, inverse=False):
    """Rotate disjoint adjacent pairs, leaving unmatched boundary cells intact.

    memory: [B,H,W,C]; angles: [B,H,W], per pair uses its first cell's angle.
    No periodic wrap-around or spatial interpolation is performed.
    """
    if memory.ndim != 4 or angles.shape != memory.shape[:3]:
        raise ValueError("Expected memory [B,H,W,C] and angles [B,H,W]")
    if axis not in (1,2) or offset not in (0,1):
        raise ValueError("Spatial axis and pair offset required")
    count = (memory.shape[axis]-offset)//2
    if count <= 0:
        return memory
    x = memory.movedim(axis,1)
    theta = angles.movedim(axis,1)[:,offset:offset+2*count:2]
    if inverse:
        theta = -theta
    cosine,sine = theta.cos()[...,None],theta.sin()[...,None]
    a,b = x[:,offset:offset+2*count:2],x[:,offset+1:offset+2*count:2]
    middle = torch.stack((cosine*a-sine*b,sine*a+cosine*b),dim=2).flatten(1,2)
    return torch.cat((x[:,:offset],middle,x[:,offset+2*count:]),dim=1).movedim(1,axis)


def transport(memory, angles, inverse=False):
    """One sweep: horizontal even/odd pairs, then vertical even/odd pairs."""
    if angles.shape != (*memory.shape[:3],4):
        raise ValueError("Expected four angle maps per grid")
    operations = ((2,0),(2,1),(1,0),(1,1))
    order = range(3,-1,-1) if inverse else range(4)
    for index in order:
        axis,offset = operations[index]
        memory = pair_rotate(memory,angles[...,index],axis,offset,inverse)
    return memory


class ObservationTransport(nn.Module):
    """Learn rotation angles from two causal observations, not fixed flow rules.

    The human-selected constraint is orthogonality and a local pair topology.
    This does not claim to learn the operator family or guarantee correspondence.
    """
    def __init__(self,observation_dim,hidden_dim=32):
        super().__init__()
        self.predict = nn.Sequential(nn.Conv2d(2*observation_dim,hidden_dim,3,padding=1),
                                     nn.GELU(),nn.Conv2d(hidden_dim,4,1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self,memory,previous,current):
        if previous.shape != current.shape or previous.shape[:3] != memory.shape[:3]:
            raise ValueError("Observation grids must match memory")
        inputs = torch.cat((previous,current),dim=-1).permute(0,3,1,2)
        angles = self.predict(inputs).permute(0,2,3,1)
        return transport(memory,angles),angles
