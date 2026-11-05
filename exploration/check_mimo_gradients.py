"""Compare native MIMO gradients with an independent CPU/FP64 recurrence.

Two small numerical cases use the first trained temporal core, random hidden
inputs, and one fixed linear output objective. This is not an image benchmark,
a general proof of kernel correctness, or a speed measurement.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import torch

from exploration.mimo_fp64_reference import metric, reference
from train_video import configure_reproducibility
from video_checkpoint import file_hash


def gradient_metrics(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    result = metric(a, e)
    result.update(reference_norm=float(e.norm()), actual_norm=float(a.norm()))
    result['cosine'] = (float(torch.dot(a,e)/(a.norm()*e.norm()))
                        if a.norm()>0 and e.norm()>0 else None)
    return result


def native_gradients(core, sample, target):
    core.zero_grad(set_to_none=True)
    u = sample.cuda().detach().requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        value = core(u)
    loss = (value.float()*target.cuda()).mean()
    loss.backward()
    result = {}
    for name, parameter in list(core.named_parameters()) + [('input', u)]:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f'Missing/nonfinite gradient: {name}')
        result[name] = parameter.grad.detach().cpu().clone()
    return value.detach().float().cpu(), result


def run(fixture, output):
    if output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(2)
    configure_reproducibility(701)
    summary = json.loads((fixture/'summary.json').read_text())
    if file_hash(fixture/'mimo_model.pt') != summary['model_sha256']:
        raise ValueError('Trained export changed')
    saved = torch.load(fixture/'mimo_model.pt', map_location='cpu', weights_only=True)
    from mamba_ssm import Mamba3
    core = Mamba3(d_model=256, d_state=64, expand=2, headdim=64,
                  is_mimo=True, mimo_rank=4, chunk_size=8)
    prefix = 'encoder.encoder.adapters.3.mixer.'
    weights = {k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)}
    core.load_state_dict(weights, strict=True)
    del saved
    sample = torch.randn(2,8,256)
    target = torch.randn_like(sample)
    cases = {'preserving_initialization':core}
    active = copy.deepcopy(core)
    with torch.no_grad():
        active.mimo_x[:,1:].uniform_(-.1,.1)
        active.mimo_o[:,1:].uniform_(-.1,.1)
    cases['extra_ranks_active'] = active
    sources = [__file__, 'exploration/mimo_fp64_reference.py', 'train_video.py', 'uv.lock',
        'vendor/mamba/mamba_ssm/modules/mamba3.py',
        'vendor/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py']
    # This tolerance is a diagnostic alarm, not an accuracy/correctness theorem.
    plan = dict(input_shape=list(sample.shape), seed=701, trained_layer=4,
        checkpoint_step=11000, export_sha256=summary['model_sha256'],
        objective='Mean elementwise product of the complete core output and a fixed random target.',
        cases=list(cases), native_precision='BF16 autocast with FP32 parameters', reference_precision='FP64',
        alarm_relative_l2=.1, alarm_reference_norm_floor=1e-8,
        sources={p:file_hash(p) for p in sources},
        versions={p:importlib.metadata.version(p) for p in ('torch','tilelang','apache-tvm-ffi')},
        scope=__doc__)
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    reports, artifacts = {}, {}
    for name, module in cases.items():
        params = {k:v.detach().double().requires_grad_(True) for k,v in module.named_parameters()}
        u = sample.double().requires_grad_(True)
        expected, _ = reference.__wrapped__(u, params)
        (expected*target.double()).mean().backward()
        ref_grads = {k:v.grad.detach() for k,v in params.items()}
        ref_grads['input'] = u.grad.detach()
        native = copy.deepcopy(module).cuda()
        actual, grads = native_gradients(native, sample, target)
        repeated, repeated_grads = native_gradients(native, sample, target)
        checks = {k:gradient_metrics(grads[k], ref_grads[k]) for k in ref_grads}
        alarms = [k for k,v in checks.items() if v['reference_norm']>plan['alarm_reference_norm_floor']
                  and v['relative_l2']>plan['alarm_relative_l2']]
        report = dict(output=metric(actual, expected.detach()), gradients=checks,
            alarm_parameters=alarms, within_diagnostic_thresholds=not alarms,
            forward_replay_exact=torch.equal(actual, repeated),
            gradient_replay={k:gradient_metrics(repeated_grads[k], grads[k]) for k in grads},
            gradient_replay_exact=all(torch.equal(grads[k],repeated_grads[k]) for k in grads))
        reports[name] = report
        artifacts[name] = dict(weights=module.state_dict(), output_native=actual,
            output_reference=expected.detach(), gradients_native=grads,
            gradients_reference=ref_grads, gradients_native_repeat=repeated_grads)
        print(json.dumps(dict(case=name, within_diagnostic_thresholds=not alarms,
            worst_gradient_relative_l2=max(v['relative_l2'] for v in checks.values()),
            gradient_replay_exact=report['gradient_replay_exact'])), flush=True)
        del native
    torch.save(dict(input=sample,target=target,cases=artifacts),output/'gradients.pt')
    if any(file_hash(p)!=h for p,h in plan['sources'].items()):
        raise ValueError('Source changed during diagnostic')
    result = dict(completed=True,plan_sha256=file_hash(output/'plan.json'),
        artifact_sha256=file_hash(output/'gradients.pt'),results=reports,
        full_visual_model_qualified=False,recognition_accuracy_tested=False,
        scope=plan['scope'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.fixture,a.output)
