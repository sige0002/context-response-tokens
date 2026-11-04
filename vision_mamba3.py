"""Small image encoder built from official Mamba-3 SISO blocks.

Row-major/column-major forward/reverse scans share weights. This is a prototype
using established cross-scan ideas, not a reproduction of VMamba or visionMamba3.
Run on CUDA with torch.autocast('cuda', dtype=torch.bfloat16).
"""
import torch
from torch import nn
from mamba_ssm import Mamba3


def cross_scan(x):
    """BHWC -> (B*4, HW, C). Each direction gets an independent SSM state."""
    b,h,w,c=x.shape
    rows=x.reshape(b,h*w,c)
    cols=x.transpose(1,2).reshape(b,h*w,c)
    return torch.stack([rows,rows.flip(1),cols,cols.flip(1)],1).reshape(b*4,h*w,c)


def cross_merge(sequences,height,width):
    """Undo each scan's coordinates, then average, preserving the spatial grid."""
    b4,length,c=sequences.shape
    if b4%4 or length!=height*width:raise ValueError('Invalid scan dimensions')
    y=sequences.reshape(b4//4,4,length,c)
    row=y[:,0].reshape(-1,height,width,c)
    rev=y[:,1].flip(1).reshape(-1,height,width,c)
    col=y[:,2].reshape(-1,width,height,c).transpose(1,2)
    colrev=y[:,3].flip(1).reshape(-1,width,height,c).transpose(1,2)
    return (row+rev+col+colrev)*.25


class VisionBlock(nn.Module):
    def __init__(self,width,state):
        super().__init__()
        self.local=nn.Conv2d(width,width,3,padding=1,groups=width)
        self.norm=nn.LayerNorm(width)
        self.mixer=Mamba3(d_model=width,d_state=state,headdim=64,expand=2,is_mimo=False)
        self.norm_ff=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.scale_mix=nn.Parameter(torch.full((width,),.01))
        self.scale_ff=nn.Parameter(torch.full((width,),.01))

    def forward(self,x):
        x=x+self.local(x.permute(0,3,1,2)).permute(0,2,3,1)
        _,h,w,_=x.shape
        update=cross_merge(self.mixer(cross_scan(self.norm(x))),h,w)
        x=x+self.scale_mix*update
        return x+self.scale_ff*self.ff(self.norm_ff(x))


class VisionMamba3(nn.Module):
    """RGB BCHW -> dense BCHW features at approximately1/4 input resolution."""
    def __init__(self,width=128,depth=2,state=64):
        super().__init__()
        if width<32 or width%32:raise ValueError('width must be a positive multiple of32')
        if depth<1:raise ValueError('depth must be positive')
        self.width,self.depth,self.state=width,depth,state
        self.stem=nn.Sequential(nn.Conv2d(3,width//2,3,stride=2,padding=1),
            nn.GroupNorm(8,width//2),nn.GELU(),
            nn.Conv2d(width//2,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.GELU())
        self.blocks=nn.Sequential(*(VisionBlock(width,state) for _ in range(depth)))
        self.norm=nn.LayerNorm(width)

    def forward(self,image):
        if image.ndim!=4 or image.shape[1]!=3:raise ValueError('Expected BCHW RGB image')
        x=self.stem(image).permute(0,2,3,1)
        return self.norm(self.blocks(x)).permute(0,3,1,2).contiguous()
