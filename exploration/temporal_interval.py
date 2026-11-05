"""Fixed-weight integration-interval intervention; no training-code changes.

The original eight anchors have seven unit intervals. Subdivide each into
factor intervals, retaining the first observation's original integration step.
This is a nominal sampling-density normalization, not physical timestamps.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from exploration.temporal_density import (
    NAMES, density_inputs, paired, read, summarize, verify as verify_density, write,
)
from exploration.evaluate_video_history import model_hash
from train_video import configure_reproducibility
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips


def interval_weights(factor):
    if type(factor) is not int or factor not in (1, 2, 4):
        raise ValueError('Expected preregistered subdivision factor 1, 2 or 4')
    return [1.] + [1./factor] * (7*factor)


def scale_arguments(arguments, factor):
    """Scale decay, integration weights and rotary phase via DT and ADT.

Native code computes the trapezoidal write coefficients from DT and the
rotary phase from Angles * DT. Angles are rates and remain unchanged.
"""
    weights = interval_weights(factor)
    dt, adt = arguments['DT'], arguments['ADT']
    if dt.ndim != 3 or dt.shape != adt.shape or dt.shape[-1] != len(weights):
        raise ValueError('Sampling schedule and native kernel length differ')
    scale = dt.new_tensor(weights)[None, None, :]
    return dict(arguments, DT=dt*scale, ADT=adt*scale)


@contextmanager
def interval_kernel(factor, qualification=False):
    """Scope an argument-only wrapper to this diagnostic process.

Uses the unchanged official Mamba3.forward and official SISO CUDA kernel.
Applies only to the clip-forward entry point, not the separate stream helper.
This is single-threaded experimental instrumentation, not a serving API.
"""
    import mamba_ssm.modules.mamba3 as native
    interval_weights(factor)
    original = native.mamba3_siso_combined
    if getattr(original, '_interval_diagnostic', False):
        raise RuntimeError('Nested interval instrumentation is unsupported')
    checks = []

    def wrapped(**kwargs):
        scaled = scale_arguments(kwargs, factor)
        if qualification:
            if factor != 1:
                raise ValueError('Identity qualification requires unit scale')
            expected = original(**dict(kwargs, return_final_states=True))
            actual = original(**dict(scaled, return_final_states=True))
            errors = [float((a.float()-b.float()).abs().max()) for a, b in zip(actual, expected)]
            if len(errors) != 5 or any(e != 0 for e in errors):
                raise ValueError(f'Unit scale changed native output/state: {errors}')
            checks.append(dict(output_and_four_states_max_abs=errors))
            return actual if kwargs['return_final_states'] else actual[0]
        checks.append(dict(length=scaled['DT'].shape[-1]))
        return original(**scaled)

    wrapped._interval_diagnostic = True
    native.mamba3_siso_combined = wrapped
    try:
        yield checks
    finally:
        native.mamba3_siso_combined = original


def scores(values, controls, targets):
    return dict(vs_baseline=summarize(values, targets)['vs_baseline'],
        normalized_vs_unscaled={n: paired(values[n], controls[n], targets) for n in NAMES},
        new_vs_held=summarize(values, targets)['new_vs_held'])


def verify(output):
    plan, summary = read(output/'plan.json'), read(output/'summary.json')
    prior = Path(plan['density_result'])
    verified = verify_density(prior)
    if verified['summary_sha256'] != plan['density_summary_sha256']:
        raise ValueError('Changed density control')
    for name in ('plan.json', 'inputs.json', 'logits.npz'):
        if file_hash(output/name) != summary[name.split('.')[0]+'_sha256']:
            raise ValueError('Changed result artifact')
    for path, digest in plan['sources'].items():
        if file_hash(path) != digest: raise ValueError('Changed source')
    if file_hash(plan['checkpoint']) != plan['checkpoint_info']['checkpoint_sha256']:
        raise ValueError('Changed checkpoint')
    if read(output/'inputs.json') != read(prior/'inputs.json'):
        raise ValueError('Input schedule changed')
    with np.load(output/'logits.npz') as saved, np.load(prior/'logits.npz') as old:
        for key in ('ids', 'targets'):
            if not np.array_equal(saved[key], old[key]): raise ValueError('Population changed')
        targets = saved['targets'].copy()
        values = {n: saved[n].copy() for n in NAMES}
        controls = {n: old[n].copy() for n in NAMES}
    for n, a in values.items():
        if a.shape != (348, 174) or not np.isfinite(a).all(): raise ValueError('Invalid logits')
        if summary['predictions'][n] != a.argmax(-1).tolist(): raise ValueError('Predictions differ')
    if not np.array_equal(values['baseline_8'], controls['baseline_8']):
        raise ValueError('Unit-scale full baseline differs')
    if summary['scores'] != scores(values, controls, targets): raise ValueError('Score recount differs')
    if not summary['weights_unchanged']: raise ValueError('Weights changed')
    if len(summary['identity_checks']) != 3 or any(
        r['output_and_four_states_max_abs'] != [0.]*5 for r in summary['identity_checks']):
        raise ValueError('Native qualification missing')
    if summary['baseline_replays'] != [0., 0.]: raise ValueError('Baseline repeat failed')
    return dict(passed=True, predictions_recounted=1740, summary_sha256=file_hash(output/'summary.json'),
        scope='CPU artifact checks, original-label recount through verified density controls; no independent GPU replay.')


@torch.inference_mode()
def run(args):
    from transformers import SiglipImageProcessor
    output, prior = Path(args.output), Path(args.density_result)
    if output.exists(): raise FileExistsError('Use a fresh output; no partial reuse')
    checked = verify_density(prior)
    density_plan = read(prior/'plan.json')
    checkpoint = Path(density_plan['checkpoint'])
    torch.set_num_threads(2); configure_reproducibility(42)
    # Establish a CUDA context before allocating a second model in UMA memory.
    torch.empty(1, device='cuda')
    model, info = load_video_checkpoint(checkpoint, 'cpu')
    if info != density_plan['checkpoint_info'] or info['step'] != 11000:
        raise ValueError('Expected identical fixed 11000-update model')
    model.requires_grad_(False).eval()
    if model.encoder.kind != 'mamba' or len(model.encoder.adapters) != 3:
        raise ValueError('Expected three native SISO adapters')
    processor = SiglipImageProcessor.from_pretrained(info['config']['model'],
        revision=info['config']['revision'], local_files_only=True)
    dataset = RawVideoClips(info['config']['val'], processor, frames=8, classes=174)
    if file_hash(dataset.path) != info['config']['val_sha256']: raise ValueError('Manifest changed')
    sources = list(density_plan['sources']) + [__file__, 'exploration/test_temporal_interval.py',
        'vendor/mamba/mamba_ssm/modules/mamba3.py',
        'vendor/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py',
        'vendor/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py',
        'vendor/mamba/mamba_ssm/ops/triton/mamba3/angle_dt.py']
    plan = dict(checkpoint=str(checkpoint.resolve()), checkpoint_info=info,
        density_result=str(prior.resolve()), density_summary_sha256=checked['summary_sha256'],
        sources={str(Path(p)): file_hash(p) for p in sources}, subset=density_plan['subset'],
        factors={n: (int(n.split('_')[1])-1)//7 for n in NAMES},
        interval_weights={str(f): interval_weights(f) for f in (1, 2, 4)},
        hypotheses='Normalize learned integration intervals when adding observations; no sweep or retraining.',
        mode='Same batch-one causal clip forward as completed unscaled controls',
        limitations=density_plan['limitations']+[
            'Nominal anchor-relative integration units, not actual elapsed seconds.',
            'The known acquisition schedule sets the factor, not observed recognition accuracy.',
            'No guarantee of rate invariance: input-dependent gates and trapezoidal writes still change.',
            'Shared GPU; no speed claim. Post-hoc diagnostic, not evidence of new SOTA architecture.'])
    output.mkdir(parents=True); write(output/'plan.json', plan)
    old_inputs = read(prior/'inputs.json')
    with np.load(prior/'logits.npz') as saved:
        controls = {n: saved[n].copy() for n in NAMES}
        targets, ids = saved['targets'].copy(), saved['ids'].copy()
    before = model_hash(model); model.cuda()
    values = {n: [] for n in NAMES}; records = []; replays = []; identity = []
    for number, item in enumerate(plan['subset']):
        row = dataset.rows[item['index']]
        if row['id'] != item['id'] or row['label'] != item['label'] or file_hash(row['video']) != item['video_sha256']:
            raise ValueError('Source video or target changed')
        inputs, metadata, conditions = density_inputs(dataset, item['index'])
        record = dict(baseline=metadata, conditions=conditions)
        if record != old_inputs[number]: raise ValueError('Exact RGB intervention changed')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if number == 0:
                baseline = model(inputs['baseline_8'][None].cuda())[0].float().cpu().numpy()
                with interval_kernel(1, qualification=True) as checks:
                    qualified = model(inputs['baseline_8'][None].cuda())[0].float().cpu().numpy()
                if not np.array_equal(baseline, qualified) or len(checks) != 3:
                    raise ValueError('Whole-model native identity check failed')
                identity = checks; write(output/'qualification.json', dict(passed=True, checks=identity))
            for name in NAMES:
                with interval_kernel(plan['factors'][name]) as calls:
                    logits = model(inputs[name][None].cuda())[0].float().cpu().numpy()
                if len(calls) != 3 or not np.isfinite(logits).all(): raise ValueError('Missing intervention/nonfinite')
                if name == 'baseline_8' and not np.array_equal(logits, controls[name][number]):
                    raise ValueError('Unit-scale baseline changed from completed density run')
                values[name].append(logits)
            if number in (0, 347):
                baseline = model(inputs['baseline_8'][None].cuda())[0].float().cpu().numpy()
                delta = float(np.abs(baseline-values['baseline_8'][-1]).max())
                if delta != 0: raise ValueError('Instrumentation leaked outside context')
                replays.append(delta)
        records.append(record)
        with (output/'progress.jsonl').open('a') as handle:
            import json
            handle.write(json.dumps(dict(completed=number+1, id=item['id']))+'\n')
        if (number+1)%24 == 0 or number == 347:
            print(f'completed={number+1}/348', flush=True)
    if model_hash(model) != before: raise ValueError('Weights changed')
    for p, digest in plan['sources'].items():
        if file_hash(p) != digest: raise ValueError('Source changed while running')
    values = {n: np.stack(v) for n, v in values.items()}
    np.savez_compressed(output/'logits.npz', targets=targets, ids=ids, **values)
    write(output/'inputs.json', records)
    summary = dict(plan_sha256=file_hash(output/'plan.json'), inputs_sha256=file_hash(output/'inputs.json'),
        logits_sha256=file_hash(output/'logits.npz'), model_state_sha256=before, weights_unchanged=True,
        identity_checks=identity, baseline_replays=replays, scores=scores(values, controls, targets),
        predictions={n: v.argmax(-1).tolist() for n,v in values.items()}, limitations=plan['limitations'])
    write(output/'summary.json', summary)
    print({n: r['correct'] for n,r in summary['scores']['vs_baseline'].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-result', default='results/temporal_density_step11000_v1/evaluation_v2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        result = verify(Path(args.output)); write(Path(args.output)/'verification.json', result)
        print(result)
    else: run(args)


if __name__ == '__main__': main()
