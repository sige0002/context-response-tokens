"""Synthetic reporting fixtures do not measure trained-model performance."""
from copy import deepcopy
import csv
import hashlib
import json
from statistics import mean, stdev

import pytest

from paper import windowed_tables as tables


@pytest.fixture
def bundle():
    comparisons = []
    for condition in tables.CONDITIONS:
        for window in tables.WINDOWS:
            for delay in tables.DELAYS:
                if condition == 'suffix_only' and delay == 0:
                    continue
                seeds = []
                for seed in tables.SEEDS:
                    right = 2 if condition == 'suffix_only' else 145 + seed - 42 + tables.WINDOWS.index(window)
                    value = dict(seed=seed, window_accuracy=right / 348)
                    for i, contrast in enumerate(tables.CONTRASTS):
                        left = 2 if condition == 'suffix_only' else 160 + seed - 42 + i
                        value[contrast] = dict(left_correct=left, right_correct=right,
                            left_only_correct=max(0, left - right), right_only_correct=max(0, right - left),
                            samples=348, difference_pp=100 * (left - right) / 348)
                    seeds.append(value)
                row = dict(condition=condition, window=window, delay_observations=delay, seed_results=seeds)
                for contrast in tables.CONTRASTS:
                    values = [s[contrast]['difference_pp'] for s in seeds]
                    row[contrast] = dict(differences_pp=values, mean_pp=mean(values), sample_sd_pp=stdev(values))
                comparisons.append(row)
    records = {}
    for seed in tables.SEEDS:
        for model in ('mamba_interleaved', 'attention_interleaved', 'attention_window8', 'attention_window32', 'attention_window128'):
            core = 'mamba' if model == 'mamba_interleaved' else 'attention'
            records[f'{model}_seed{seed}'] = dict(checkpoint_sha256=hashlib.sha256(f'{core}{seed}'.encode()).hexdigest(),
                cache_summary_sha256='a' * 64, probe_summary_sha256='b' * 64)
    costs = []
    for selection, frames in tables.VIDEOS.items():
        for model in tables.VARIANTS:
            bins = []
            for lo, hi in ((0, 0), (1, 7), (8, 31), (32, 127), (128, 255), (256, None)):
                if lo >= frames:
                    continue
                n = min(frames - 1, hi if hi is not None else frames - 1) - lo + 1
                bins.append(dict(history_min=lo, history_max=hi, unique_frames=n, samples=5 * n,
                    **{metric: dict(p50=10.) for metric in ('gpu_ms', 'model_wall_ms', 'pipeline_wall_ms')}))
            state = (79027200 if model == 'mamba_interleaved' else 602112 * frames + 24
                     if model == 'attention_buffered' else 602112 * int(model.removeprefix('attention_window')) + 48)
            costs.append(dict(model=model, video=dict(selection=selection, frames=frames, id=selection),
                measurement=dict(frames_per_pass=frames, repeats=5, bins=bins, paused_compute_pids=[],
                    hardware=dict(gpu='synthetic fixture', torch='test', cuda='test'),
                    max_state_bytes=state, torch_peak_allocated_bytes=state + 1_000_000_000)))
    return dict(format_version=1, experiment_sha256=tables.EXPERIMENT_SHA, cost_specification_sha256=tables.COST_SPEC_SHA,
        recognition=dict(comparisons=comparisons, run_artifacts=records), costs=costs,
        cost_checkpoint_sha256={m: records[f'{m}_seed42']['checkpoint_sha256'] for m in ('mamba_interleaved', 'attention_interleaved')},
        submission_ready=False, fixture_only=True)


def test_all_tables_keep_diagnostic_population_cost_units_and_per_seed_differences(bundle, tmp_path):
    output = tmp_path / 'review'
    evidence = tables.write_output(bundle, output)
    assert len(evidence['outputs']) == 10
    assert evidence['submission_ready'] is False
    assert evidence['timing_bins']['longest'] == [256, 1336]
    assert '41.95' in (output / 'windowed_full_memory_scores.tex').read_text()  # Mean 146/348 in percent.
    assert '75.37' in (output / 'windowed_cost_longest.tex').read_text()  # Mamba state bytes / 2**20.
    assert '0.57' in (output / 'windowed_suffix_only_scores.tex').read_text()
    with (output / 'windowed_seed_differences.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 126 and {r['samples'] for r in rows} == {'348'}
    assert {r['seed'] for r in rows} == {'42', '43', '44'}
    assert all(float(r['mamba_minus_window_pp']) >= 0 for r in rows)
    assert json.loads((output / 'windowed_result_bundle.json').read_text()) == bundle
    with pytest.raises(FileExistsError):
        tables.write_output(bundle, output)


@pytest.mark.parametrize('fault', ['condition', 'seed', 'population', 'sign', 'paired', 'reference',
                                 'chance', 'spread', 'weights', 'cost_weights', 'repeats', 'cost_video',
                                 'cost_bin', 'state', 'nan', 'ready'])
def test_misleading_or_inconsistent_bundles_cannot_produce_tables(bundle, tmp_path, fault):
    row = bundle['recognition']['comparisons'][0]
    value = row['seed_results'][0]
    if fault == 'condition':
        bundle['recognition']['comparisons'].pop()
    elif fault == 'seed':
        row['seed_results'][-1] = deepcopy(row['seed_results'][0])
    elif fault == 'population':
        value['mamba_minus_window']['samples'] = 24777
    elif fault == 'sign':
        value['mamba_minus_window']['difference_pp'] *= -1
    elif fault == 'paired':
        value['mamba_minus_window']['left_only_correct'] += 1
    elif fault == 'reference':
        # Keep this row internally consistent while changing the supposedly
        # shared Mamba reference in only the W8 comparison.
        pair = value['mamba_minus_window']
        pair['left_correct'] += 1
        pair['left_only_correct'] += 1
        pair['difference_pp'] = 100 * (pair['left_correct'] - pair['right_correct']) / 348
        differences = [s['mamba_minus_window']['difference_pp'] for s in row['seed_results']]
        row['mamba_minus_window'] = dict(differences_pp=differences, mean_pp=mean(differences), sample_sd_pp=stdev(differences))
    elif fault == 'chance':
        row = next(r for r in bundle['recognition']['comparisons'] if r['condition'] == 'suffix_only')
        row['seed_results'][0]['mamba_minus_window'].update(left_correct=3, left_only_correct=1, difference_pp=100 / 348)
    elif fault == 'spread':
        row['mamba_minus_window']['sample_sd_pp'] += 1
    elif fault == 'weights':
        bundle['recognition']['run_artifacts']['attention_window128_seed42']['checkpoint_sha256'] = 'c' * 64
    elif fault == 'cost_weights':
        bundle['cost_checkpoint_sha256']['mamba_interleaved'] = 'c' * 64
    elif fault == 'repeats':
        bundle['costs'][0]['measurement']['repeats'] = 3
    elif fault == 'cost_video':
        bundle['costs'][0]['video']['id'] = 'another input'
    elif fault == 'cost_bin':
        bundle['costs'][0]['measurement']['bins'][-1]['history_min'] += 1
    elif fault == 'state':
        bundle['costs'][0]['measurement']['max_state_bytes'] -= 16
    elif fault == 'nan':
        bundle['costs'][0]['measurement']['bins'][0]['gpu_ms']['p50'] = float('nan')
    else:
        bundle['submission_ready'] = True
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        tables.write_output(bundle, output)
    assert not output.exists()


def test_collection_refuses_incomplete_job_before_importing_training_dependencies(tmp_path):
    recognition, cost = tmp_path / 'recognition', tmp_path / 'cost'
    recognition.mkdir()
    cost.mkdir()
    (recognition / 'status.json').write_text('{"status":"waiting_for_mimo_pilot"}')
    with pytest.raises(ValueError, match='recognition control is incomplete'):
        tables.collect(recognition, cost)
