"""Independent CPU/FP64 recurrence for the fixed width-256, rank-four probe.

Implements Mamba-3's exponential-trapezoidal state update and half-split
rotary convention without calling native projections, normalization, or SSM
kernels. This small random-input comparison does not qualify a vision model.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from video_checkpoint import file_hash


@torch.no_grad()
def reference(sample, weights):
    w={k:v.double() for k,v in weights.items()};u=sample.double()
    assert w['in_proj.weight'].shape==(1576,256)
    b,t,_=u.shape
    z,x,k,q,dd_dt,dd_a,trap,angles=F.linear(u,w['in_proj.weight']).split(
        [512,512,256,256,8,8,8,16],-1)
    z=z.reshape(b,t,8,64);x=x.reshape(b,t,8,64)
    def norm(v,name):
        v=v.reshape(b,t,4,1,64)
        return v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)*w[name]
    k=norm(k,'B_norm.weight').expand(-1,-1,-1,8,-1)+w['B_bias'].permute(1,0,2)
    q=norm(q,'C_norm.weight').expand(-1,-1,-1,8,-1)+w['C_bias'].permute(1,0,2)
    dt=F.softplus(dd_dt+w['dt_bias'])
    a=-(1+dd_a.clamp_min(0)+torch.reciprocal(1-dd_a.clamp_max(0))-1)
    a=a.clamp(max=-1e-4);gate=trap.sigmoid()
    state=torch.zeros(b,8,64,64,dtype=torch.float64)
    angle=torch.zeros(b,8,16,dtype=torch.float64)
    prev_k=torch.zeros(b,4,8,64,dtype=torch.float64)
    prev_x=torch.zeros(b,8,64,dtype=torch.float64);outputs=[]
    mx=w['mimo_x'].permute(1,0,2);mz=w['mimo_z'].permute(1,0,2)
    mo=w['mimo_o'].permute(1,0,2)
    for i in range(t):
        angle=angle+math.pi*angles[:,i].tanh()[:,None,:]*dt[:,i,:,None]
        cos=F.pad(angle.cos(),(0,16),value=1.)[:,None]
        sin=F.pad(angle.sin(),(0,16),value=0.)[:,None]
        def rotate(value):
            first,second=value.split(32,-1)
            return torch.cat((first*cos-second*sin,first*sin+second*cos),-1)
        ki,qi=rotate(k[:,i]),rotate(q[:,i]);vi=x[:,i,None]*mx
        alpha=torch.exp(a[:,i]*dt[:,i]);beta=(1-gate[:,i])*dt[:,i]*alpha
        gamma=gate[:,i]*dt[:,i]
        old=torch.einsum('brhp,brhn->bhpn',prev_x[:,None]*mx,prev_k)
        new=torch.einsum('brhp,brhn->bhpn',vi,ki)
        state=alpha[:,:,None,None]*state+beta[:,:,None,None]*old+gamma[:,:,None,None]*new
        y=torch.einsum('bhpn,brhn->brhp',state,qi)+w['D'][None,None,:,None]*vi
        y=y*F.silu(z[:,i,None]*mz)
        y=(y*mo).sum(1).reshape(b,512)
        outputs.append(F.linear(y,w['out_proj.weight']))
        prev_k,prev_x=ki,x[:,i]
    return torch.stack(outputs,1),(angle,state,prev_k,prev_x)


def metric(actual,expected):
    difference=actual.double()-expected
    return dict(max_abs=float(difference.abs().max()),
        relative_l2=float(difference.norm()/expected.norm().clamp_min(1e-12)))


def run(fixture,decode,output):
    if output.exists(): raise FileExistsError('Preserve previous comparisons')
    saved=torch.load(fixture,map_location='cpu',weights_only=True)
    prior=json.loads((fixture.parent/'summary.json').read_text())
    record=json.loads((decode/'summary.json').read_text())
    if (prior['fixture_sha256']!=file_hash(fixture) or record['fixture_sha256']!=file_hash(fixture)
            or not record['completed'] or record['streamed_sha256']!=file_hash(decode/'streamed.pt')):
        raise ValueError('Changed or incomplete native artifacts')
    torch.set_num_threads(2)
    expected,states=reference(saved['input'],saved['state_dict'])
    actual=torch.load(decode/'streamed.pt',map_location='cpu',weights_only=True)
    result=dict(fixture_sha256=file_hash(fixture),decode_summary_sha256=file_hash(decode/'summary.json'),
        source_sha256=file_hash(__file__),clip_vs_fp64=metric(saved['output'],expected),
        stream_vs_fp64=metric(actual['output'],expected),
        final_states={name:metric(a,e) for name,a,e in
            zip(('angle','ssm','previous_key','previous_value'),actual['states'],states)},
        input_shape=list(saved['input'].shape),full_visual_model_qualified=False,
        scope='Two random sequences of eight observations. CPU/FP64 full-block recurrence compared to native BF16 clip and streaming paths; no gradient-reference, accuracy, or timing claim.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--decode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.fixture,a.decode,a.output)
