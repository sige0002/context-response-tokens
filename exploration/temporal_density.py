"""Nested observation-density intervention, with a matched held-frame control.

The classifier, final image, original eight observations and their anchor
positions are fixed. Intermediate frames are either newly observed or replaced
by the preceding anchor. This is fixed-weight inference, not density training.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from transformers import SiglipImageProcessor

from exploration.evaluate_video_history import model_hash, tensor_hash
from train_video import configure_reproducibility
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips, frame_to_rgb, temporal_indices


NAMES = ('baseline_8', 'new_15', 'held_15', 'new_29', 'held_29')


def schedules(anchors):
    if (len(anchors) != 8 or any(type(i) is not int or i < 0 for i in anchors)
            or any(a > b for a, b in zip(anchors, anchors[1:]))):
        raise ValueError('Expected eight nonnegative, ordered integer anchors')
    result = {'baseline_8': list(anchors)}
    for factor in (2, 4):
        dense, held = [anchors[0]], [anchors[0]]
        for a, b in zip(anchors, anchors[1:]):
            dense.extend(a + (b-a)*j//factor for j in range(1, factor))
            held.extend([a]*(factor-1))
            dense.append(b); held.append(b)
        result[f'new_{len(dense)}'] = dense
        result[f'held_{len(held)}'] = held
    return result


def paired(values, reference, targets):
    pred, base = values.argmax(-1), reference.argmax(-1)
    hit, old = pred == targets, base == targets
    delta = (hit.astype(float)-old.astype(float))*100
    rng = np.random.default_rng(17)
    strata = [np.flatnonzero(targets == c) for c in np.unique(targets)]
    draws = np.concatenate([rng.choice(s, (2000, len(s))) for s in strata], axis=1)
    return dict(correct=int(hit.sum()), total=len(targets), accuracy=float(hit.mean()),
        delta_pp=float(delta.mean()), gained=int((hit & ~old).sum()),
        lost=int((~hit & old).sum()), changed=int((pred != base).sum()),
        bootstrap_95_pp=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())


def summarize(values, targets):
    return dict(vs_baseline={n: paired(values[n], values['baseline_8'], targets) for n in NAMES},
        new_vs_held={str(t): paired(values[f'new_{t}'], values[f'held_{t}'], targets) for t in (15, 29)})


def read(path): return json.loads(Path(path).read_bytes())


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def density_inputs(dataset, index):
    row = dataset.rows[index]
    with av.open(str(row['video'])) as container:
        container.streams.video[0].thread_count = 1
        decoded = list(container.decode(video=0))
    if not decoded:
        raise ValueError('Empty source video')
    anchors = temporal_indices(len(decoded), 8)
    indices = schedules(anchors)
    unique = sorted(set(indices['new_29']))
    images = [frame_to_rgb(decoded[i]) for i in unique]
    width, height = images[0].size
    if any(im.size != (width, height) for im in images):
        raise ValueError('Changing source resolution')
    pixels = dataset.processor(images=images, return_tensors='pt').pixel_values
    positions = {i: k for k, i in enumerate(unique)}
    inputs = {name: pixels[[positions[i] for i in order]] for name, order in indices.items()}
    baseline = inputs['baseline_8']
    metadata = dict(id=row['id'], indices=anchors, crop=[0, 0, width, height],
        timestamps=[decoded[i].time for i in anchors], decoded_frames=len(decoded), epoch=0,
        label=row['label'], pixels_sha256=tensor_hash(baseline),
        current_pixels_sha256=tensor_hash(baseline[-1]))
    conditions = {}
    for name, video in inputs.items():
        if not torch.equal(video[-1], baseline[-1]):
            raise ValueError('Final RGB changed')
        conditions[name] = dict(indices=indices[name], timestamps=[decoded[i].time for i in indices[name]],
            pixels_sha256=tensor_hash(video), distinct_frames=len(set(indices[name])),
            new_distinct_frames=len(set(indices[name])-set(anchors)))
    return inputs, metadata, conditions


def verify(output):
    plan, summary = read(output/'plan.json'), read(output/'summary.json')
    for name, key in [('plan.json', 'plan_sha256'), ('inputs.json', 'inputs_sha256'), ('logits.npz', 'logits_sha256')]:
        if file_hash(output/name) != summary[key]: raise ValueError('Changed result artifact')
    for path, digest in plan['sources'].items():
        if file_hash(path) != digest: raise ValueError('Changed source')
    for name, digest in plan['prior_hashes'].items():
        if file_hash(Path(plan['prior'])/name) != digest: raise ValueError('Changed prior inputs')
    if file_hash(plan['checkpoint']) != plan['checkpoint_info']['checkpoint_sha256']:
        raise ValueError('Changed checkpoint')
    config = plan['checkpoint_info']['config']; manifest = Path(config['val'])
    if file_hash(manifest) != config['val_sha256']: raise ValueError('Changed manifest')
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    inputs, prior = read(output/'inputs.json'), read(Path(plan['prior'])/'inputs.json')
    with np.load(output/'logits.npz') as saved:
        ids, targets = saved['ids'].tolist(), saved['targets']
        if (len(ids) != 348 or len(set(ids)) != 348 or ids != [r['id'] for r in plan['subset']]
                or targets.tolist() != [rows[r['index']]['label'] for r in plan['subset']]):
            raise ValueError('Population or original labels changed')
        values = {n: saved[n].copy() for n in NAMES}
    for n, v in values.items():
        if v.shape != (348, 174) or not np.isfinite(v).all(): raise ValueError('Invalid logits')
        if summary['predictions'][n] != v.argmax(-1).tolist(): raise ValueError('Prediction recount differs')
    for i, record in enumerate(inputs):
        if record['baseline'] != prior[i]: raise ValueError('Original eight RGB inputs changed')
        expected = schedules(prior[i]['indices'])
        for name in NAMES:
            if record['conditions'][name]['indices'] != expected[name]: raise ValueError('Sampling changed')
    if summary['scores'] != summarize(values, targets): raise ValueError('Score recount differs')
    if (not summary['weights_unchanged'] or len(summary['baseline_replays']) != 2
            or any(r['max_abs'] != 0 for r in summary['baseline_replays'])):
        raise ValueError('Weight preservation or repeated forward failed')
    return dict(passed=True, summary_sha256=file_hash(output/'summary.json'),
        predictions_recounted=348*len(NAMES),
        scope='Source and artifact verification plus original-label recount; no independent GPU forward or source-video re-decode.')


@torch.inference_mode()
def run(args):
    output, prior, checkpoint = Path(args.output), Path(args.prior), Path(args.checkpoint)
    if output.exists(): raise FileExistsError('Use a fresh output, no partial reuse')
    torch.set_num_threads(2); configure_reproducibility(42)
    old_plan, old_summary, checked = read(prior/'plan.json'), read(prior/'summary.json'), read(prior/'verification.json')
    if not checked['passed'] or checked['summary_sha256'] != file_hash(prior/'summary.json'):
        raise ValueError('Prior cohort not verified')
    for name, key in [('plan.json', 'plan_sha256'), ('inputs.json', 'inputs_sha256')]:
        if file_hash(prior/name) != old_summary[key]: raise ValueError('Prior artifacts changed')
    model, info = load_video_checkpoint(checkpoint, 'cpu')
    if info['step'] != 11000 or model.encoder.kind != 'mamba': raise ValueError('Expected fixed 11000-update Mamba')
    model.requires_grad_(False).eval()
    processor = SiglipImageProcessor.from_pretrained(info['config']['model'],
        revision=info['config']['revision'], local_files_only=True)
    dataset = RawVideoClips(info['config']['val'], processor, frames=8, classes=174)
    if file_hash(dataset.path) != info['config']['val_sha256']: raise ValueError('Manifest changed')
    sources = [__file__, 'exploration/test_temporal_density.py', 'exploration/evaluate_video_history.py',
        'temporal_siglip.py', 'temporal_readout.py', 'video_data.py', 'video_checkpoint.py',
        'train_video.py', 'pyproject.toml', 'uv.lock']
    plan = dict(checkpoint=str(checkpoint.resolve()), checkpoint_info=info,
        prior=str(prior.resolve()), subset=old_plan['subset'], conditions=list(NAMES),
        prior_hashes={n: file_hash(prior/n) for n in ('plan.json', 'summary.json', 'verification.json', 'inputs.json')},
        sources={str(Path(p)): file_hash(p) for p in sources}, baseline_replay_indices=[0, 347],
        mode='batch-one causal clip inference for all conditions',
        hypotheses='New intermediate images versus holding the previous anchor, at identical update counts.',
        limitations=['Exploratory, previously inspected balanced 348-clip subset; one interim model.',
            'Eight-observation training; longer and differently spaced sequences are distribution changes.',
            'A negative result does not prove extra observations are useless after retraining.',
            'No claim of a unique bottleneck, new architecture, SOTA, or long-term memory.',
            'First and last observations, classifier and model weights fixed; no additional early history.',
            'Shared GPU with main training; no latency or throughput claim.'])
    output.mkdir(parents=True); write(output/'plan.json', plan)
    old_inputs = read(prior/'inputs.json'); before = model_hash(model)
    values = {n: [] for n in NAMES}; records = []; replays = []
    model.cuda()
    for number, item in enumerate(plan['subset']):
        row = dataset.rows[item['index']]
        if row['id'] != item['id'] or row['label'] != item['label'] or file_hash(row['video']) != item['video_sha256']:
            raise ValueError('Source video/label drift')
        inputs, metadata, conditions = density_inputs(dataset, item['index'])
        if metadata != old_inputs[number]: raise ValueError('Original RGB/indices changed')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for name in NAMES:
                logits = model(inputs[name][None].cuda())[0].float().cpu().numpy()
                if not np.isfinite(logits).all(): raise FloatingPointError('Nonfinite prediction')
                values[name].append(logits)
            if number in plan['baseline_replay_indices']:
                repeated = model(inputs['baseline_8'][None].cuda())[0].float().cpu().numpy()
                delta = float(np.abs(repeated-values['baseline_8'][-1]).max())
                replays.append(dict(index=number, id=item['id'], max_abs=delta))
                if delta != 0: raise ValueError('Repeated forward mismatch')
        records.append(dict(baseline=metadata, conditions=conditions))
        with (output/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(completed=number+1, id=item['id']))+'\n')
        if (number+1)%24 == 0 or number == 347: print(json.dumps(dict(completed=number+1, total=348)), flush=True)
    if model_hash(model) != before: raise ValueError('Weights changed')
    for p, digest in plan['sources'].items():
        if file_hash(p) != digest: raise ValueError('Source changed while running')
    values = {n: np.stack(v) for n, v in values.items()}
    targets = np.array([r['label'] for r in plan['subset']], dtype=np.int64)
    ids = np.array([r['id'] for r in plan['subset']])
    np.savez_compressed(output/'logits.npz', targets=targets, ids=ids, **values)
    write(output/'inputs.json', records)
    summary = dict(plan_sha256=file_hash(output/'plan.json'), inputs_sha256=file_hash(output/'inputs.json'),
        logits_sha256=file_hash(output/'logits.npz'), model_state_sha256=before, weights_unchanged=True,
        baseline_replays=replays, scores=summarize(values, targets),
        predictions={n: v.argmax(-1).tolist() for n, v in values.items()}, limitations=plan['limitations'])
    write(output/'summary.json', summary)
    print(json.dumps(summary['scores']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', default='results/video_history_intervention_seed17')
    parser.add_argument('--checkpoint', default='results/temporal_density_step11000_v1/checkpoint.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        result = verify(Path(args.output)); write(Path(args.output)/'verification.json', result)
        print(json.dumps(result))
    else: run(args)


if __name__ == '__main__': main()
