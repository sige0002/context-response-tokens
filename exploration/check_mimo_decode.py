"""Probe native single-step MIMO in an isolated uv dependency overlay.

Uses exact weights, inputs and native clip outputs saved by check_mimo_native.
No recognition or independent mathematical-reference claim.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch

from video_checkpoint import file_hash


def run(fixture,output):
    if output.exists(): raise FileExistsError('Use a new output')
    output.mkdir(parents=True)
    saved=torch.load(fixture,map_location='cpu',weights_only=True)
    report=json.loads((fixture.parent/'summary.json').read_text())
    if report['fixture_sha256']!=file_hash(fixture): raise ValueError('Changed clip fixture')
    from mamba_ssm.modules.mamba3 import Mamba3
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model=Mamba3(d_model=256,d_state=64,headdim=64,expand=2,is_mimo=True,
        mimo_rank=4,chunk_size=saved['mimo_chunk_size']).cuda().eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    sample=saved['input'].cuda();cache=model.allocate_inference_cache(len(sample),8,dtype=torch.bfloat16)
    result=dict(fixture=str(fixture),fixture_sha256=file_hash(fixture),
        versions={p:importlib.metadata.version(p) for p in
            ('torch','tilelang','nvidia-cutlass-dsl','apache-tvm-ffi')},
        source_sha256=file_hash(__file__),full_visual_model_qualified=False,
        scope=__doc__)
    try:
        outputs=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for frame in sample.unbind(1):
                value,*_=model.step(frame,*cache)
                outputs.append(value)
            actual=torch.stack(outputs,1).float().cpu()
        if not torch.isfinite(actual).all(): raise FloatingPointError('Nonfinite streaming output')
        error=actual-saved['output'].float()
        result.update(completed=True,max_abs=float(error.abs().max()),
            relative_l2=float(error.norm()/saved['output'].float().norm().clamp_min(1e-12)),
            allocated_cache_bytes=sum(t.numel()*t.element_size() for t in cache))
        torch.save(dict(output=actual,states=[t.detach().cpu() for t in cache]),output/'streamed.pt')
        result['streamed_sha256']=file_hash(output/'streamed.pt')
    except Exception as exc:
        result.update(completed=False,error_type=type(exc).__name__,error=str(exc))
        (output/'error.txt').write_text(traceback.format_exc())
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.fixture,a.output)
