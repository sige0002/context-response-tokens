"""Native MIMO adapters with a SISO-preserving mathematical initialization.

Only rank zero initially writes and reads the SSM. Extra ranks have random
B/C projections and zero input/output gains so their gains can learn. Native
BF16 kernel and rotary-layout differences need separate numerical checks.
Training/clip inference uses the pinned root environment; native step uses
the documented TVM-FFI overlay. This module changes neither environment.
"""
from dataclasses import dataclass

import torch
from torch import nn


def rotary_permutation(size):
    """Convert pairwise half-RoPE coordinates into MIMO's split-half layout."""
    if size < 4 or size % 4: raise ValueError('State size must be divisible by four')
    quarter=size//4
    return torch.cat((torch.arange(0,2*quarter,2),torch.arange(2*quarter,3*quarter),
        torch.arange(1,2*quarter,2),torch.arange(3*quarter,4*quarter)))


@torch.no_grad()
def promote_core(base,rank=4,chunk_size=8,seed=701):
    from mamba_ssm import Mamba3
    if (base.is_mimo or base.is_outproj_norm or base.num_bc_heads!=1
            or base.rotary_dim_divisor!=4 or base.in_proj.weight.device.type!='cpu'):
        raise ValueError('Expected a CPU SISO core with one B/C group, half RoPE, no output norm')
    if rank!=4 or chunk_size!=8:
        raise ValueError('Only the examined rank-four, chunk-eight configuration is supported')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new=Mamba3(d_model=base.d_model,d_state=base.d_state,expand=base.expand,
            headdim=base.headdim,ngroups=1,rope_fraction=.5,A_floor=base.A_floor,
            is_mimo=True,mimo_rank=rank,chunk_size=chunk_size,dtype=base.in_proj.weight.dtype)
    n,h,d=base.d_state,base.nheads,base.d_inner
    old=base.in_proj.weight.split([d,d,n,n,h,h,h,base.num_rope_angles],0)
    target=new.in_proj.weight.split([d,d,rank*n,rank*n,h,h,h,base.num_rope_angles],0)
    for i in (0,1,4,5,6,7): target[i].copy_(old[i])
    order=rotary_permutation(n)
    for i in (2,3): target[i].reshape(rank,n,base.d_model)[0].copy_(old[i][order])
    for name in ('B','C'):
        getattr(new,name+'_bias')[:,0].copy_(getattr(base,name+'_bias')[:,0,order])
        getattr(new,name+'_norm').weight.copy_(getattr(base,name+'_norm').weight[order])
    for name in ('D','dt_bias'): getattr(new,name).copy_(getattr(base,name))
    new.out_proj.weight.copy_(base.out_proj.weight)
    new.mimo_x.zero_();new.mimo_x[:,0].fill_(1.)
    new.mimo_o.zero_();new.mimo_o[:,0].fill_(1.)
    new.mimo_z.fill_(1.)
    return new


class MimoAdapter(nn.Module):
    def __init__(self,base,rank=4,chunk_size=8,seed=701):
        super().__init__()
        self.norm,self.down,self.up,self.gain=base.norm,base.down,base.up,base.gain
        self.mixer=promote_core(base.mixer,rank,chunk_size,seed)

    def forward(self,x,cache=None,offset=0,streaming=False):
        b,t,n,d=x.shape
        u=self.down(self.norm(x)).permute(0,2,1,3).reshape(b*n,t,-1).contiguous()
        if streaming:
            if self.training: raise RuntimeError('Call eval before native streaming')
            if cache is None: cache=self.mixer.allocate_inference_cache(b*n,t,dtype=u.dtype)
            expected=[(b*n,self.mixer.nheads,self.mixer.num_rope_angles),
                (b*n,self.mixer.nheads,self.mixer.headdim,self.mixer.d_state),
                (b*n,self.mixer.mimo_rank,self.mixer.nheads,self.mixer.d_state),
                (b*n,self.mixer.nheads,self.mixer.headdim)]
            dtypes=[torch.float32,torch.float32,u.dtype,u.dtype]
            if len(cache)!=4 or any(tuple(v.shape)!=s or v.dtype!=dt or v.device!=u.device
                    for v,s,dt in zip(cache,expected,dtypes)):
                raise ValueError('Native MIMO state layout/device/dtype changed; reset the episode')
            outputs=[]
            for frame in u.unbind(1):
                value,*_=self.mixer.step(frame,*cache)
                outputs.append(value)
            y=torch.stack(outputs,1)
        else:
            y=self.mixer(u)
        update=self.up(y).reshape(b,n,t,d).permute(0,2,1,3)
        return x+self.gain*update,cache


@dataclass
class MimoState:
    owner:int
    base_state:object=None

    @property
    def nbytes(self): return 0 if self.base_state is None else self.base_state.nbytes


class MimoEncoder(nn.Module):
    def __init__(self,base,rank=4,chunk_size=8,seed=701):
        super().__init__()
        if base.kind!='mamba': raise ValueError('Expected SISO Mamba encoder')
        self.encoder=base;self.mimo_config=dict(rank=rank,chunk_size=chunk_size,seed=seed)
        for key,adapter in list(base.adapters.items()):
            if isinstance(adapter,MimoAdapter): raise ValueError('Encoder is already MIMO')
            base.adapters[key]=MimoAdapter(adapter,rank,chunk_size,seed+int(key))
        self.train(base.training)

    @property
    def backbone(self): return self.encoder.backbone

    @property
    def kind(self): return 'mamba'

    def forward(self,video): return self.encoder(video)
    def encode_current(self,video): return self(video)[:,-1]

    @torch.no_grad()
    def encode_chunk(self,video,state=None):
        """Stream new observations with a fixed one-image execution shape.

        Batching images through SigLIP/down-projections can change rounding
        before the BF16 recurrence. Use the same execution for every chunk
        partition; clip training still uses the separate batched forward.
        """
        if self.training: raise RuntimeError('Call eval before streaming')
        if video.ndim!=5 or video.shape[1]<1:
            raise ValueError('Expected nonempty [batch,time,channels,height,width]')
        if state is None: state=MimoState(id(self))
        if not isinstance(state,MimoState) or state.owner!=id(self):
            raise ValueError('State belongs to another encoder; reset the episode')
        outputs=[]
        for frame in video.split(1,dim=1):
            features,state.base_state=self.encoder.encode_chunk(frame,state.base_state)
            outputs.append(features)
        return torch.cat(outputs,dim=1),state

    @torch.no_grad()
    def step(self,image,state=None):
        if image.ndim!=4: raise ValueError('Expected [batch,channels,height,width]')
        features,state=self.encode_chunk(image[:,None],state)
        return features[:,0],state


def attach(model,**kwargs):
    model.encoder=MimoEncoder(model.encoder,**kwargs)
    return model


def save_model(model,path,readout):
    from train_temporal import atomic_save
    base=model.encoder.encoder
    atomic_save(dict(format='native-mimo-video-v1',mimo=model.encoder.mimo_config,
        architecture=base.architecture,backbone_config=base.backbone.config.to_dict(),
        checkpoint_revision=getattr(base,'checkpoint_revision',None),readout=dict(readout),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)


def load_model(path,device='cpu'):
    from transformers import SiglipVisionConfig,SiglipVisionModel
    from temporal_siglip import TemporalSiglip
    from temporal_readout import VideoRecognizer
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['format']!='native-mimo-video-v1': raise ValueError('Wrong model format')
    base=TemporalSiglip(SiglipVisionModel(SiglipVisionConfig.from_dict(saved['backbone_config'])),
        **saved['architecture'])
    base.checkpoint_revision=saved['checkpoint_revision'];r=saved['readout']
    model=VideoRecognizer(base,r['classes'],r['kind'],r['seed'])
    attach(model,**saved['mimo']);model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval()
