"""Compare actual Triton forward/backward with the official FP32 reference.

Small shapes, including a non-chunk-aligned length. Assertions are enabled here;
some comparisons in the upstream test file only print errors.
"""
import copy
import importlib.util
import json
from pathlib import Path

import torch
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


def main():
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('official_reference',
        Path(__file__).parent/'vendor/mamba/tests/ops/triton/test_mamba3_siso.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    results=[]
    for length in (64,197):
        torch.manual_seed(42)
        inputs=ref.create_mamba3_siso_inputs(batch=2,seqlen=length,nheads=4,nheads_qk=1,
            headdim_qk=64,headdim_v=64,dtype=torch.bfloat16,device='cuda',has_D=True,has_Z=True,
            has_input_states=False,requires_grad=True)
        reference=copy.deepcopy(inputs)
        args=['Q','K','V','ADT','DT','Trap','Q_bias','K_bias','Angles','D','Z']
        expected,_=ref.mamba3_siso_fwd_ref(*[reference[k] for k in args],dtype=torch.float32)
        actual=mamba3_siso_combined(*[inputs[k] for k in args],chunk_size=64)
        def error(a,b):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            return float((torch.linalg.vector_norm(a.float()-b.float())/torch.linalg.vector_norm(b.float()).clamp_min(1e-12)).detach())
        forward=error(actual,expected)
        assert forward<.05,(length,'forward',forward)
        gradient=torch.randn_like(expected)
        actual.backward(gradient.to(actual.dtype));expected.backward(gradient)
        errors={k:error(inputs[k].grad,reference[k].grad) for k in args}
        assert max(errors.values())<.1,(length,errors)
        result=dict(length=length,forward_relative_l2=forward,gradient_relative_l2=errors)
        results.append(result);print(json.dumps(result),flush=True)
    dest=Path('results/reference.json');dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(dict(scope='BF16 kernel vs official FP32 reference, seed42, '
        'batch2 heads4 groups1 state64 headdim64; forward tolerance .05, gradient tolerance .1',
        results=results),indent=2)+'\n')


if __name__=='__main__':main()
