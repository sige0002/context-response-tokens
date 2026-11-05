"""Recount matched bounded/full-attention/Mamba delayed-action probes on CPU."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from exploration.delayed_action_data import sha
from exploration.delayed_action_probe import evaluate


def read(path):
    return json.loads(Path(path).read_text())


def paired_counts(left, right, labels):
    if len(left) != len(right) or len(left) != len(labels) or not len(labels):
        raise ValueError('Paired prediction populations differ')
    a, b, y = np.asarray(left), np.asarray(right), np.asarray(labels)
    ca, cb = a == y, b == y
    gained = int((ca & ~cb).sum())
    lost = int((~ca & cb).sum())
    return dict(left_correct=int(ca.sum()), right_correct=int(cb.sum()),
                left_only_correct=gained, right_only_correct=lost,
                difference_pp=100 * (gained - lost) / len(y), samples=len(y))


def verify_probe(directory, reference, data_hash, window=None, expected_data=None):
    cache, probe = directory / 'cache', directory / 'probe'
    plan, receipt = read(cache / 'plan.json'), read(cache / 'summary.json')
    info = plan['checkpoint_info']
    if expected_data is not None and plan['data'] != expected_data:
        raise ValueError('Cached cohort differs from the original data plan')
    checkpoint = Path(reference['run']) / 'last.pt'
    if (receipt['scope'] != 'complete_cache' or receipt['plan_sha256'] != sha(cache / 'plan.json')
            or receipt['features_sha256'] != sha(cache / 'features.npz')
            or receipt['inputs_sha256'] != sha(cache / 'inputs.json')
            or receipt['weights_unchanged'] is not True
            or plan['data_plan_sha256'] != data_hash or info['config'] != reference['config']
            or info['step'] != reference['config']['steps']
            or Path(plan['checkpoint_path']).resolve() != checkpoint.resolve()
            or info['checkpoint_sha256'] != sha(checkpoint)):
        raise ValueError('Cache does not match the completed reference experiment')
    if window is not None:
        policy = dict(window=window, includes_current=True, absolute_positions=True,
                      reset_preserves_absolute_time=True, maximal_input_support=1 + 3 * (window - 1))
        if plan.get('window_policy') != policy or receipt.get('window_policy') != policy:
            raise ValueError('Window policy differs')
    for path, digest in plan['sources'].items():
        if sha(path) != digest:
            raise ValueError('Feature source changed: ' + path)
    measured = read(probe / 'summary.json')
    if (measured['cache_summary_sha256'] != sha(cache / 'summary.json')
            or measured['readouts_sha256'] != sha(probe / 'readouts.npz')
            or measured['source_sha256'] != sha('exploration/delayed_action_probe.py')
            or measured['validation_used_for_fitting'] is not False
            or measured['encoder_updated'] is not False):
        raise ValueError('Probe provenance differs')
    with np.load(cache / 'features.npz') as features, np.load(probe / 'readouts.npz') as weights:
        recounted, _ = evaluate(features, plan['data'])
        if recounted != measured['results']:
            raise ValueError('CPU probe refit does not reproduce reported predictions')
        for condition in plan['data']['conditions']:
            for split in ('train', 'validation'):
                x = features[split + '_' + condition]
                predicted = ((x - weights[condition + '_mean']) @ weights[condition + '_weight']
                             + weights[condition + '_bias']).argmax(-1)
                expected = np.array([r['predictions'] for r in recounted[condition][split]]).T
                if not np.array_equal(predicted, expected):
                    raise ValueError('Saved probe weights do not reproduce predictions')
        labels = features['validation_labels'].copy()
        ids = features['validation_ids'].tolist()
    return dict(results=recounted, labels=labels, ids=ids, inputs_sha256=receipt['inputs_sha256'],
                checkpoint_sha256=info['checkpoint_sha256'], cache_summary_sha256=sha(cache / 'summary.json'),
                probe_summary_sha256=sha(probe / 'summary.json'))


def verify_pair_inputs(left, right, same_checkpoint=False):
    if (left['inputs_sha256'] != right['inputs_sha256'] or left['ids'] != right['ids']
            or not np.array_equal(left['labels'], right['labels'])):
        raise ValueError('Compared models used different observations or target ordering')
    if same_checkpoint and left['checkpoint_sha256'] != right['checkpoint_sha256']:
        raise ValueError('Bounded and full-history attention used different weights')


def aggregate(experiment_path):
    plan = read(experiment_path)
    if sha(plan['comparison_plan']) != plan['comparison_plan_sha256']:
        raise ValueError('Original comparison plan changed')
    for path, digest in plan['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Prepared experiment source changed: ' + path)
    baseline = read(plan['comparison_plan'])
    if sha(plan['data_plan']) != plan['data_plan_sha256']:
        raise ValueError('Original delayed-action data plan changed')
    data = read(plan['data_plan'])
    baseline_root = Path(plan['comparison_plan']).parent
    if read(baseline_root / 'status.json')['status'] != 'complete':
        raise ValueError('Original delayed-action comparison is incomplete')
    original_summary = read(baseline_root / 'summary.json')
    if original_summary['plan_sha256'] != sha(plan['comparison_plan']):
        raise ValueError('Original delayed-action summary uses a different plan')
    expected = {(seed, window) for seed in plan['seeds'] for window in plan['windows']}
    if len(plan['runs']) != len(expected) or {(r['seed'], r['window']) for r in plan['runs']} != expected:
        raise ValueError('Prepared seed/window grid is incomplete or duplicated')
    references = {(r['model'], r['seed']): r for r in baseline['runs']}
    full, records, inputs = {}, {}, {}
    for seed in plan['seeds']:
        for kind in ('mamba_interleaved', 'attention_interleaved'):
            row = references[kind, seed]
            result = verify_probe(baseline_root / row['name'], row, plan['data_plan_sha256'], expected_data=data)
            if original_summary['runs'][row['name']]['probe_summary_sha256'] != result['probe_summary_sha256']:
                raise ValueError('Original delayed-action aggregate changed')
            full[kind, seed] = result
            records[row['name']] = {k: result[k] for k in ('cache_summary_sha256', 'probe_summary_sha256', 'checkpoint_sha256')}
        verify_pair_inputs(full['mamba_interleaved', seed], full['attention_interleaved', seed])
    paired = {}
    for row in plan['runs']:
        seed, window = row['seed'], row['window']
        reference = references['attention_interleaved', seed]
        if row['reference_row'] != reference:
            raise ValueError('Prepared bounded-control reference changed')
        bounded = verify_probe(Path(row['output']), reference, plan['data_plan_sha256'], window, data)
        verify_pair_inputs(full['attention_interleaved', seed], bounded, same_checkpoint=True)
        inputs[str(seed)] = bounded['inputs_sha256']
        name = f'attention_window{window}_seed{seed}'
        records[name] = {k: bounded[k] for k in ('cache_summary_sha256', 'probe_summary_sha256', 'checkpoint_sha256')}
        for condition, splits in bounded['results'].items():
            for index, target in enumerate(splits['validation']):
                key = (condition, window, target['delay_observations'])
                entry = dict(seed=seed, window_accuracy=target['accuracy'])
                for kind, label in (('mamba_interleaved', 'mamba_minus_window'),
                                    ('attention_interleaved', 'full_attention_minus_window')):
                    other = full[kind, seed]['results'][condition]['validation'][index]
                    if other['delay_observations'] != target['delay_observations']:
                        raise ValueError('Comparison delay grids differ')
                    entry[label] = paired_counts(other['predictions'], target['predictions'], bounded['labels'])
                paired.setdefault(key, []).append(entry)
    rows = []
    for (condition, window, delay), values in sorted(paired.items()):
        values.sort(key=lambda r: r['seed'])
        if [r['seed'] for r in values] != plan['seeds']:
            raise ValueError('Missing or duplicate result seed')
        stats = {}
        for label in ('mamba_minus_window', 'full_attention_minus_window'):
            differences = [r[label]['difference_pp'] for r in values]
            stats[label] = dict(differences_pp=differences, mean_pp=mean(differences), sample_sd_pp=stdev(differences))
        rows.append(dict(condition=condition, window=window, delay_observations=delay,
                         seed_results=values, **stats))
    return dict(experiment_plan_sha256=sha(experiment_path), baseline_summary_sha256=sha(baseline_root / 'summary.json'),
                run_artifacts=records, input_sha256_by_seed=inputs, comparisons=rows,
                scope='All prepared seeds/windows; paired predictions on the same synthetic delayed-action cohort',
                limits=plan['limitations'] + ['Three seeds and two validation suffix blocks do not establish statistical significance or natural-video SOTA.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.plan)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(verified_runs=len(result['run_artifacts']), comparisons=len(result['comparisons']))))
