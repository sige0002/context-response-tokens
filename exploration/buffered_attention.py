"""Inference-only preallocated KV control for the existing temporal attention.

Same projections, positions, masking, and full-history attention; no pruning,
quantization, trained-parameter changes, or modification of training sources.
Capacity is an explicit finite bound, not history-independent memory.
"""
import torch
from torch import nn
from torch.nn import functional as F

from temporal_siglip import CausalAttention


class BufferedAttention(CausalAttention):
    def __init__(self, source, capacity):
        nn.Module.__init__(self)
        if type(source) is not CausalAttention or type(capacity) is not int or capacity < 1:
            raise ValueError('Expected the original attention and positive integer capacity')
        self.heads=source.heads
        # Reuse the very same parameter objects, preserving state_dict names.
        self.qkv=source.qkv; self.out=source.out; self.capacity=capacity
        self.train(source.training)

    def forward(self,x,cache=None,offset=0,streaming=False):
        if not streaming:
            return super().forward(x,cache,offset,streaming)
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Buffered KV updates require eval and no-grad/inference mode')
        b,t,d=x.shape
        if type(offset) is not int or offset<0 or t<1 or offset+t>self.capacity:
            raise ValueError('Invalid offset or preallocated attention capacity exceeded')
        if cache is None:
            if offset!=0: raise ValueError('A fresh buffer must start at observation zero')
        elif len(cache)!=3 or cache[2].device.type!='cpu' or cache[2].numel()!=1 or int(cache[2])!=offset:
            raise ValueError('Buffered cache cursor does not match stream offset')
        # Keep exactly the original positional encoding and Q/K/V computation.
        pos=torch.arange(offset,offset+t,device=x.device).float()
        freq=torch.exp(-torch.arange(0,d,2,device=x.device).float()
                       *(torch.log(torch.tensor(10000.,device=x.device))/d))
        angles=pos[:,None]*freq
        pe=torch.stack((angles.sin(),angles.cos()-1),-1).flatten(-2)
        q,k,v=self.qkv(x+pe.to(x.dtype)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        shape=(b,self.heads,self.capacity,d//self.heads)
        if cache is None:
            cache=(torch.empty(shape,device=k.device,dtype=k.dtype),
                   torch.empty(shape,device=v.device,dtype=v.dtype),torch.zeros((),dtype=torch.int64))
        elif any(tuple(z.shape)!=shape or z.dtype!=k.dtype or z.device!=k.device for z in cache[:2]):
            raise ValueError('KV shape, precision or device differs from the allocated buffer')
        end=offset+t
        cache[0][:,:,offset:end].copy_(k); cache[1][:,:,offset:end].copy_(v)
        keys=torch.arange(end,device=x.device)
        queries=offset+torch.arange(t,device=x.device)
        mask=keys[None,:]<=queries[:,None]
        # Unwritten capacity is never passed to attention, even as masked slots.
        y=F.scaled_dot_product_attention(q,cache[0][:,:,:end],cache[1][:,:,:end],attn_mask=mask,is_causal=False)
        cache[2].fill_(end)
        y=self.out(y.transpose(1,2).reshape(b,t,d))
        return y,cache


def variants(encoder,capacity):
    if encoder.kind!='attention': raise ValueError('Only the temporal-attention baseline is supported')
    original={key:a.mixer for key,a in encoder.adapters.items()}
    buffered={key:BufferedAttention(m,capacity) for key,m in original.items()}
    return original,buffered


def select(encoder,mixers):
    if set(mixers)!=set(encoder.adapters): raise ValueError('Adapter layer set changed')
    for key,mixer in mixers.items(): encoder.adapters[key].mixer=mixer


@torch.inference_mode()
def compare_streams(encoder,images,capacity,readout=None):
    """Compare both implementations frame by frame, outside any timed region."""
    original,buffered=variants(encoder,capacity)
    states={'dynamic':None,'buffered':None}; rows=[]
    device=next(encoder.parameters()).device
    try:
        for frame,image in enumerate(images):
            outputs={}; predictions={}
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                for name,mixers in (('dynamic',original),('buffered',buffered)):
                    select(encoder,mixers)
                    features,states[name]=encoder.step(image.to(device),states[name])
                    outputs[name]=features.float()
                    if not torch.isfinite(features).all():
                        raise FloatingPointError('Nonfinite features during implementation comparison')
                    if readout is not None:
                        predictions[name]=readout(features).argmax(-1).cpu().tolist()
            diff=outputs['dynamic']-outputs['buffered']
            rows.append(dict(frame=frame,exact=bool(torch.equal(outputs['dynamic'],outputs['buffered'])),
                max_abs=float(diff.abs().max()),rmse=float(diff.square().mean().sqrt()),
                states={k:s.nbytes for k,s in states.items()},
                readout_predictions=predictions))
        if not rows: raise ValueError('No comparison images')
        return rows
    finally:
        select(encoder,original)
