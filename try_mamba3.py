"""Minimal runnable Mamba-3 SISO demo on CUDA; random weights, not an accuracy benchmark."""
import argparse
import json
import time
from pathlib import Path

import torch
from mamba_ssm import Mamba3


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--state',type=int,default=64)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--train-steps',type=int,default=0,
                   help='Optional fixed-batch fitting demo; not generalization or visual recognition')
    p.add_argument('--output',type=Path,default=Path('results/smoke.json'))
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(4);torch.manual_seed(42)
    model=Mamba3(d_model=a.width,d_state=a.state,headdim=64,is_mimo=False,
                 device='cuda',dtype=torch.bfloat16)
    x=torch.randn(a.batch,a.length,a.width,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    start=time.perf_counter()
    y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    loss=y.float().square().mean();loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.cuda.synchronize();first_seconds=time.perf_counter()-start
    print('Forward/backward PASS:',tuple(y.shape),flush=True)
    report=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),variant='Mamba-3 SISO',
        parameters=sum(p.numel() for p in model.parameters()),shape=list(y.shape),
        dtype='BF16 projections with constructor-preserved FP32 state parameters',
        first_forward_backward_seconds=first_seconds,finite_output=True,finite_gradients=True,
        scope='Random initialization, GPU execution check. No pretrained recognition ability.')
    if a.train_steps:
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        inputs=x.detach();target=torch.zeros_like(inputs);target[:,1:]=inputs[:,:-1]
        losses=[]
        for step in range(a.train_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(inputs)
            loss=(prediction.float()-target.float()).square().mean()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        report['fixed_batch_delay_fit']=dict(steps=a.train_steps,initial_loss=losses[0],last_loss=losses[-1],
            scope='Optimization demonstration on one fixed batch, not held-out performance',losses=losses)
        print('Fixed-batch fit:',losses[0],'->',losses[-1],flush=True)
    model.eval()
    with torch.inference_mode():
        for _ in range(10):model(x.detach())
        torch.cuda.synchronize();times=[]
        for _ in range(30):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();model(x.detach());end.record();end.synchronize()
            times.append(begin.elapsed_time(end))
    report['warm_forward_median_ms']=sorted(times)[len(times)//2]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
