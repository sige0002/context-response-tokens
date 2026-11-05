"""Check native Mamba-3 MIMO execution without changing the vision experiment.

Random inputs and weights: this is an execution probe, not a recognition,
speed, independent-reference, or full visual-encoder qualification.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch

from video_checkpoint import file_hash


def run(output):
    if output.exists():
        raise FileExistsError('Preserve prior probes; choose a new output')
    output.mkdir(parents=True)
    from mamba_ssm.modules.mamba3 import Mamba3

    torch.set_num_threads(2)
    torch.manual_seed(701)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [__file__, 'vendor/mamba/mamba_ssm/modules/mamba3.py',
        'vendor/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py',
        'vendor/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py',
        'vendor/mamba/mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py', 'uv.lock']
    result = dict(sources={p:file_hash(p) for p in sources},
        versions={p:importlib.metadata.version(p) for p in
            ('torch','tilelang','nvidia-cutlass-dsl','apache-tvm-ffi')},
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        input_shape=[2,8,256], seed=701, full_visual_model_qualified=False,
        scope=__doc__, stages={})

    def save():
        (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    models={}
    for mode in ('siso','mimo'):
        models[mode]=Mamba3(d_model=256,d_state=64,headdim=64,expand=2,
            is_mimo=mode=='mimo',mimo_rank=4,chunk_size=16 if mode=='mimo' else 64)
        cache=models[mode].allocate_inference_cache(1,8,dtype=torch.bfloat16)
        result[mode]=dict(parameters=sum(p.numel() for p in models[mode].parameters()),
            allocated_cache_bytes_one_patch=sum(t.numel()*t.element_size() for t in cache),
            cache_shapes=[list(t.shape) for t in cache],cache_dtypes=[str(t.dtype) for t in cache])
    model=models['mimo'].cuda()
    sample=torch.randn(*result['input_shape'],device='cuda')
    save()
    stage='forward'
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            full=model(sample)
            loss=full.float().square().mean()
        torch.cuda.synchronize()
        if not torch.isfinite(full).all(): raise FloatingPointError('Nonfinite forward output')
        result['stages'][stage]=dict(completed=True,output_shape=list(full.shape),loss=float(loss.detach()))
        save();print(json.dumps(dict(stage=stage,completed=True)),flush=True)
        stage='backward'
        loss.backward();torch.cuda.synchronize()
        missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if missing or nonfinite: raise ValueError(f'Invalid gradients: missing={missing}, nonfinite={nonfinite}')
        result['stages'][stage]=dict(completed=True,parameters_with_grad=sum(p.grad is not None for p in model.parameters()))
        save();print(json.dumps(dict(stage=stage,completed=True)),flush=True)
        stage='native_step'
        model.eval()
        cache=model.allocate_inference_cache(2,8,dtype=torch.bfloat16)
        outputs=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for frame in sample.unbind(1):
                out,*_=model.step(frame,*cache)
                outputs.append(out)
            streamed=torch.stack(outputs,1).float()
        torch.cuda.synchronize()
        if not torch.isfinite(streamed).all(): raise FloatingPointError('Nonfinite streaming output')
        error=streamed-full.detach().float()
        result['stages'][stage]=dict(completed=True,max_abs=float(error.abs().max()),
            relative_l2=float(error.norm()/full.detach().float().norm().clamp_min(1e-12)),
            note='Comparison of two native paths only; no independent recurrence check.')
    except Exception as exc:
        result['stages'][stage]=dict(completed=False,error_type=type(exc).__name__,error=str(exc))
        (output/'error.txt').write_text(traceback.format_exc())
    save()
    print(json.dumps(dict(stages=result['stages'],full_visual_model_qualified=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(Path(parser.parse_args().output))
