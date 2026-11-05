"""Cache the unchanged delayed-action cohort under a bounded attention policy.

Requires successful real-RGB GPU qualification and final scheduled attention
weights. Uses the original stream feature extractor and ridge-probe format.
No training, readout fitting, sampling changes, or queue changes occur here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from exploration.buffered_attention import select
from exploration.cache_delayed_action import stream_features
from exploration.check_resettable_window_attention import checked_plan, require_completed_chain
from exploration.delayed_action_data import sha, verify_plan
from exploration.evaluate_video_history import model_hash, tensor_hash
from exploration.resettable_window_attention import variants
from exploration.run_delayed_action import verify as verify_comparison
from train_video import configure_reproducibility
from video_checkpoint import load_video_checkpoint
from video_data import RawVideoClips


CONDITIONS = ['full_memory', 'suffix_only', 'saved_evidence']


def matching_reference(comparison, checkpoint, info):
    rows = [row for row in comparison['runs'] if row['model'] == 'attention_interleaved'
            and (Path(row['run']) / 'last.pt').resolve() == checkpoint.resolve()]
    if len(rows) != 1 or rows[0]['config'] != info['config']:
        raise ValueError('Checkpoint is not the exact scheduled full-history attention control')
    return rows[0]


def require_qualification(root):
    plan = checked_plan(root)
    if not (root / 'summary.json').exists():
        raise ValueError('Completed real-RGB GPU qualification is required')
    result = json.loads((root / 'summary.json').read_text())
    if (result.get('passed') is not True or result.get('weights_unchanged') is not True
            or result.get('plan_sha256') != sha(root / 'plan.json')
            or result.get('rows') != plan['expected_rows']
            or result.get('rows_sha256') != sha(root / 'rows.jsonl')):
        raise ValueError('GPU qualification is failed, incomplete or changed')
    rows = [json.loads(line) for line in (root / 'rows.jsonl').read_text().splitlines() if line]
    expected = {(precision, window, origin, t)
                for precision in plan['tolerances'] for window in plan['windows']
                for origin in plan['origins'] for t in range(origin, plan['fixture_plan']['frames'])}
    observed = [(r['precision'], r['window'], r['origin'], r['absolute_observation']) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('GPU qualification coverage differs')
    for row in rows:
        limits = plan['tolerances'][row['precision']]
        if (not 0 <= row['max_abs'] <= limits['max_abs']
                or not 0 <= row['relative_l2'] <= limits['relative_l2'] or row['passed'] is not True):
            raise ValueError('GPU qualification numerical alarm is not satisfied')
    return plan


def collect_block(encoder, block, reader, delays):
    """Same evidence, suffix, current-token features and reset as the baseline."""
    if not delays or delays[0] != 0 or len(delays) < 2:
        raise ValueError('Expected zero and positive observation delays')
    suffix = torch.cat([reader(row) for row in block['suffix']])[:max(delays)]
    if len(suffix) != max(delays):
        raise ValueError('Incomplete shared suffix')
    values = {name: [] for name in CONDITIONS}
    labels, ids, states = [], [], []
    shared = None
    for row in block['evidence']:
        prefix = reader(row)
        if len(prefix) != 8:
            raise ValueError('Evidence must retain the original eight observations')
        full, sizes = stream_features(encoder, prefix, suffix, delays)
        if shared is None:
            shared, reset_sizes = stream_features(encoder, prefix, suffix, delays, True)
        values['full_memory'].append(full)
        values['suffix_only'].append(shared.copy())
        values['saved_evidence'].append(np.broadcast_to(full[0], full.shape).copy())
        labels.append(row['label'])
        ids.append(row['id'])
        states.append(dict(block=block['block'], id=row['id'], full_bytes=sizes,
                           suffix_only_bytes=reset_sizes,
                           current_rgb_sha256=[tensor_hash(prefix[-1])]
                           + [tensor_hash(suffix[d - 1]) for d in delays[1:]]))
    if not labels:
        raise ValueError('Empty evidence block')
    if len({tuple(row['current_rgb_sha256'][1:]) for row in states}) != 1:
        raise ValueError('Current RGB depends on the evidence identity')
    return ({name: np.stack(rows) for name, rows in values.items()},
            dict(labels=np.array(labels), ids=np.array(ids),
                 blocks=np.full(len(labels), block['block'], dtype=np.int64)), states)


@torch.inference_mode()
def extract(data_path, checkpoint, qualification, comparison_path, window, output):
    if output.exists():
        raise FileExistsError('Use a fresh cache directory; preserve partial outputs')
    qualified = require_qualification(qualification)
    if window not in qualified['windows']:
        raise ValueError('Window was not included in the fixed GPU qualification')
    require_completed_chain(qualified)
    torch.set_num_threads(2)
    configure_reproducibility(42)
    data = json.loads(data_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    verify_comparison(comparison)
    verify_plan(data)
    if data['conditions'] != CONDITIONS:
        raise ValueError('Expected the unchanged three-condition delayed probe')
    # This is the cohort already scheduled for full-history attention and Mamba.
    fixture_plan = qualified['fixture_plan']
    if (sha(data_path) != fixture_plan['data_plan_sha256']
            or sha(data_path) != comparison['data_plan_sha256']):
        raise ValueError('Delayed cohort differs from the prepared qualification input')
    model, info = load_video_checkpoint(checkpoint, 'cpu')
    reference_row = matching_reference(comparison, checkpoint, info)
    config = info['config']
    if (info['step'] != 20000 or config['steps'] != 20000 or config['frames'] != 8
            or config['kind'] != 'attention' or config['placement'] != 'interleaved'
            or config['model'] != fixture_plan['model'] or config['revision'] != fixture_plan['revision']
            or config['seed'] not in (42, 43, 44)):
        raise ValueError('Expected final scheduled attention weights under the fixed comparison protocol')
    for split, key in (('train', 'train'), ('validation', 'val')):
        record = data['manifests'][split]
        if sha(record['path']) != record['sha256'] or record['sha256'] != config[key + '_sha256']:
            raise ValueError('Corpus or split manifest changed')
    model.requires_grad_(False).eval()
    before = model_hash(model)
    processor = SiglipImageProcessor.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    sources = ['exploration/cache_windowed_delayed_action.py', 'exploration/test_windowed_delayed_action.py',
               'exploration/run_delayed_action.py',
               'exploration/cache_delayed_action.py', 'exploration/delayed_action_data.py',
               'exploration/delayed_action_probe.py', 'exploration/resettable_window_attention.py',
               'exploration/check_resettable_window_attention.py', 'exploration/evaluate_video_history.py',
               'exploration/buffered_attention.py', 'temporal_siglip.py', 'temporal_readout.py',
               'video_data.py', 'video_checkpoint.py', 'train_video.py', 'uv.lock', 'pyproject.toml']
    plan = dict(data=data, data_plan_sha256=sha(data_path), checkpoint_info=info,
                reference_row=reference_row, comparison_plan=str(comparison_path.resolve()),
                comparison_plan_sha256=sha(comparison_path),
                checkpoint_path=str(checkpoint.resolve()), qualification=False,
                gpu_qualification=str(qualification.resolve()), gpu_qualification_summary_sha256=sha(qualification / 'summary.json'),
                conditions=data['conditions'], sources={p: sha(p) for p in sources},
                window_policy=dict(window=window, includes_current=True, absolute_positions=True,
                                   reset_preserves_absolute_time=True, maximal_input_support=1 + 3 * (window - 1)),
                representation='Mean of current tokens; unchanged original stream_features; no cache exposed to probe',
                suffix_reset='Empty K/V at absolute observation eight; no fictitious prefix images',
                timing_scope='No timing measurement', encoder_updated=False)
    output.mkdir(parents=True)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    model.cuda()
    original, bounded = variants(model.encoder, window)
    select(model.encoder, bounded)
    features, inputs, states = {}, [], []
    try:
        for split in ('train', 'validation'):
            dataset = RawVideoClips(data['manifests'][split]['path'], processor, frames=8, classes=data['classes'])
            index = {r['id']: i for i, r in enumerate(dataset.rows)}

            def reader(row):
                i = index[row['id']]
                actual = dataset.rows[i]
                if any(actual[k] != row[k] for k in ('id', 'group', 'label', 'sha256', 'bytes')):
                    raise ValueError('Source row changed')
                if sha(actual['video']) != row['sha256']:
                    raise ValueError('Source video bytes changed')
                pixels, target, metadata = dataset.read(i)
                if target != row['label']:
                    raise ValueError('Evidence target changed')
                inputs.append(dict(split=split, video_sha256=row['sha256'], pixels_sha256=tensor_hash(pixels), **metadata))
                return pixels

            pieces = {name: [] for name in CONDITIONS + ['labels', 'ids', 'blocks']}
            for block in data['splits'][split]:
                values, identities, cache_states = collect_block(model.encoder, block, reader, data['delays'])
                for name, value in {**values, **identities}.items():
                    pieces[name].append(value)
                states.extend(dict(split=split, **row) for row in cache_states)
                with (output / 'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(split=split, block=block['block'], completed=sum(len(v) for v in pieces['labels']))) + '\n')
                print(json.dumps(dict(split=split, block=block['block'])), flush=True)
            features.update({split + '_' + name: np.concatenate(values) for name, values in pieces.items()})
        if model_hash(model) != before:
            raise ValueError('Encoder/classifier weights changed during feature extraction')
    finally:
        select(model.encoder, original)
    expected_bytes = 602112 * window + 48
    if any(set(r['full_bytes'] + r['suffix_only_bytes']) != {expected_bytes} for r in states):
        raise ValueError('Unexpected bounded state capacity')
    if sha(checkpoint) != info['checkpoint_sha256']:
        raise ValueError('Final checkpoint changed during extraction')
    if sha(comparison_path) != plan['comparison_plan_sha256']:
        raise ValueError('Original full-history comparison plan changed during extraction')
    for path, digest in plan['sources'].items():
        if sha(path) != digest:
            raise ValueError('Source changed during extraction: ' + path)
    np.savez_compressed(output / 'features.npz', **features)
    (output / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    summary = dict(scope='complete_cache', plan_sha256=sha(output / 'plan.json'),
                   features_sha256=sha(output / 'features.npz'), inputs_sha256=sha(output / 'inputs.json'),
                   weights_unchanged=True, model_state_sha256=before, states=states,
                   current_rgb_independent_of_target=True, scores=None, limits=data['limits'],
                   window_policy=plan['window_policy'], qualification_summary_sha256=plan['gpu_qualification_summary_sha256'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(scope=summary['scope'], window=window, state_bytes=expected_bytes,
                          train_examples=len(features['train_labels']), validation_examples=len(features['validation_labels']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/delayed_action_memory_v1/data_plan.json'))
    parser.add_argument('--qualification', type=Path, default=Path('results/resettable_window_attention_qualification_v1'))
    parser.add_argument('--comparison', type=Path, default=Path('results/delayed_action_memory_v1/comparison_v2/plan.json'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--window', type=int, choices=(8, 32, 128), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.data, args.checkpoint, args.qualification, args.comparison, args.window, args.output)
