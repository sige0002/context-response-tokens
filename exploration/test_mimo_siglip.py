import math

import pytest
import torch
from torch.nn import functional as F

from exploration.mimo_fp64_reference import reference as mimo_reference
from exploration.mimo_siglip import MimoState,attach,load_model,promote_core,save_model


def siso_reference(sample,weights):
    """Direct rank-one CPU recurrence; no migration or native kernel calls."""
    w={k:v.double() for k,v in weights.items()};b,t,_=sample.shape
    z,x,k,q,dt,a,trap,angles=F.linear(sample.double(),w['in_proj.weight']).split(
        [512,512,64,64,8,8,8,16],-1)
    x=x.reshape(b,t,8,64);z=z.reshape(b,t,8,64)
    def norm(value,name):
        value=value.reshape(b,t,1,64)
        return value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)*w[name]
    k=norm(k,'B_norm.weight')+w['B_bias'][:,0]
    q=norm(q,'C_norm.weight')+w['C_bias'][:,0]
    dt=F.softplus(dt+w['dt_bias'])
    a=-(a.clamp_min(0)+torch.reciprocal(1-a.clamp_max(0)))
    a=a.clamp(max=-1e-4)
    state=torch.zeros(b,8,64,64,dtype=torch.float64)
    theta=torch.zeros(b,8,16,dtype=torch.float64)
    prev_k=torch.zeros(b,8,64,dtype=torch.float64)
    prev_x=torch.zeros(b,8,64,dtype=torch.float64);out=[]
    for i in range(t):
        theta=theta+math.pi*angles[:,i].tanh()[:,None]*dt[:,i,:,None]
        cos=F.pad(theta.cos(),(0,16),value=1.);sin=F.pad(theta.sin(),(0,16))
        def rotate(v):
            pair=v.reshape(b,8,32,2)
            p0,p1=pair[...,0],pair[...,1]
            return torch.stack((p0*cos-p1*sin,p0*sin+p1*cos),-1).reshape(b,8,64)
        ki,qi=rotate(k[:,i]),rotate(q[:,i]);alpha=(dt[:,i]*a[:,i]).exp()
        gate=trap[:,i].sigmoid()
        state=(alpha[:,:,None,None]*state
            +((1-gate)*dt[:,i]*alpha)[:,:,None,None]*prev_x[:,:,:,None]*prev_k[:,:,None,:]
            +(gate*dt[:,i])[:,:,None,None]*x[:,i,:,:,None]*ki[:,:,None,:])
        value=torch.einsum('bhpn,bhn->bhp',state,qi)+w['D'][None,:,None]*x[:,i]
        out.append(F.linear((value*F.silu(z[:,i])).reshape(b,512),w['out_proj.weight']))
        prev_k,prev_x=ki,x[:,i]
    return torch.stack(out,1)


@pytest.mark.parametrize('seed,length',[(3,1),(17,7),(701,17)])
def test_migration_preserves_complete_recurrence_with_nonuniform_norms(seed,length):
    from mamba_ssm import Mamba3
    torch.set_num_threads(2);torch.manual_seed(seed)
    base=Mamba3(d_model=256,d_state=64,expand=2,headdim=64,is_mimo=False)
    with torch.no_grad():
        for name in ('B','C'):
            getattr(base,name+'_norm').weight.uniform_(.4,1.7)
            getattr(base,name+'_bias').normal_()
    initial={k:v.clone() for k,v in base.state_dict().items()}
    promoted=promote_core(base)
    sample=torch.randn(2,length,256)
    expected=siso_reference(sample,base.state_dict())
    actual,_=mimo_reference(sample,promoted.state_dict())
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    assert all(torch.equal(initial[k],v) for k,v in base.state_dict().items())
    with pytest.raises(ValueError): promote_core(promoted)


def test_extra_ranks_can_start_learning_from_the_preserving_initialization():
    from mamba_ssm import Mamba3
    torch.set_num_threads(2);torch.manual_seed(9)
    core=promote_core(Mamba3(d_model=256,d_state=64,headdim=64,is_mimo=False))
    sample=torch.randn(2,7,256)
    # Unwrap only the reference's no_grad decorator for a CPU autograd check.
    value,_=mimo_reference.__wrapped__(sample,dict(core.named_parameters()))
    value.square().mean().backward()
    assert core.mimo_x.grad[:,1:].abs().sum()>0
    assert core.mimo_o.grad[:,1:].abs().sum()>0
    with torch.no_grad():
        core.mimo_x.add_(core.mimo_x.grad,alpha=-.01)
        core.mimo_o.add_(core.mimo_o.grad,alpha=-.01)
    core.zero_grad()
    value,_=mimo_reference.__wrapped__(sample,dict(core.named_parameters()))
    value.square().mean().backward()
    # The extra B/C projection rows must cease to be dead after gate updates.
    grad=core.in_proj.weight.grad
    assert grad[1024+64:1024+256].abs().sum()>0
    assert grad[1280+64:1280+256].abs().sum()>0


def test_self_contained_save_restore_and_episode_owner(tmp_path):
    from transformers import SiglipVisionConfig,SiglipVisionModel
    from temporal_siglip import TemporalSiglip,StreamState
    from temporal_readout import VideoRecognizer
    torch.set_num_threads(2);torch.manual_seed(5)
    config=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=4,image_size=32,patch_size=16)
    base=TemporalSiglip(SiglipVisionModel(config))
    model=VideoRecognizer(base,7,'mean',43).eval();attach(model)
    assert not model.encoder.training
    assert not any(p.requires_grad for p in model.encoder.backbone.parameters())
    path=tmp_path/'mimo.pt';save_model(model,path,dict(classes=7,kind='mean',seed=43))
    reloaded=load_model(path)
    assert model.state_dict().keys()==reloaded.state_dict().keys()
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    image=torch.zeros(1,1,3,32,32)
    with pytest.raises(ValueError,match='another encoder'):
        reloaded.encoder.encode_chunk(image,MimoState(id(model.encoder)))
    with pytest.raises(ValueError,match='another encoder'):
        model.encoder.encode_chunk(image,StreamState(id(base),1,(3,32,32)))
    model.train()
    with pytest.raises(RuntimeError,match='eval'):
        model.encoder.encode_chunk(image)
