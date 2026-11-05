"""Synthetic fixtures test reporting invariants, never recognition performance."""
from copy import deepcopy
import json
import statistics

import pytest

from paper.final_tables import (LABELS, REPLICATED, VARIANTS, VIDEOS, PLAN_SHA256,
                                collect, validate, write_output)


@pytest.fixture
def bundle():
    runs = []
    for i, model in enumerate(LABELS):
        for seed in ((42,43,44) if model in REPLICATED else (42,)):
            count = 10000 + i*100 + (seed-42)*i
            runs.append(dict(model=model, seed=seed, correct=count, samples=24777,
                accuracy=count/24777, steps=20000, frames=8, trainable_parameters=1000000*(i+1),
                checkpoint_sha256=f'{i*100+seed:064x}', evaluation_sha256={
                    **{'final_clip_stream.json':'a'*64},
                    **({'final_history_ablations.json':'b'*64} if model != 'none_interleaved' else {})},
                clip_stream_disagreements=i))
    costs = {}
    for name, models in (('costs',LABELS),('buffered_costs',VARIANTS)):
        costs[name] = []
        for model in models:
            for selection, frames in VIDEOS.items():
                bins = []
                for low, high in ((0,0),(1,7),(8,31),(32,127),(128,255),(256,None)):
                    end = min(frames-1, high if high is not None else frames-1)
                    if end < low: continue
                    bins.append(dict(history_min=low, history_max=high, unique_frames=end-low+1,
                        samples=3*(end-low+1), gpu_ms={'p50':3.}, model_wall_ms={'p50':4.},
                        pipeline_wall_ms={'p50':5.}))
                costs[name].append(dict(model=model, video=dict(selection=selection,frames=frames,id=selection),
                    measurement=dict(bins=bins,frames_per_pass=frames,repeats=3,paused_compute_pids=[],
                        hardware=dict(gpu='synthetic fixture only',torch='fixture',cuda='fixture'),
                        max_state_bytes=1000000,torch_peak_allocated_bytes=2000000)))
    values = {(r['model'],r['seed']):r['accuracy'] for r in runs}
    comparisons = []
    for other in REPLICATED[1:]:
        delta = [100*(values['mamba_interleaved',s]-values[other,s]) for s in (42,43,44)]
        comparisons.append(dict(other=other,seed_order=[42,43,44],differences_pp=delta,
            mean_pp=statistics.mean(delta),sample_sd_pp=statistics.stdev(delta)))
    conditional = {}
    counts = {(r['model'],r['seed']):r['correct'] for r in runs}
    for other in set(LABELS)-{'mamba_interleaved'}:
        a,b = values['mamba_interleaved',42],values[other,42]
        delta = counts['mamba_interleaved',42]-counts[other,42]
        interval = [a-b-.001,a-b+.001]
        conditional[other] = dict(a_mean=a,b_mean=b,difference=a-b,difference_percentage_points=100*(a-b),
            conditional_95_interval=interval,conditional_95_interval_pp=[100*v for v in interval],
            examples=24777,clusters=24777,bootstrap_repeats=10000,bootstrap_seed=17,
            a_only_correct=max(0,delta),b_only_correct=max(0,-delta))
    return dict(format_version=1,primary_plan_sha256=PLAN_SHA256,runs=runs,
        seed_comparisons=comparisons,single_seed_comparisons=conditional,**costs,submission_ready=False,fixture_only=True)


def test_render_keeps_seed_scores_cost_units_and_scope(bundle, tmp_path):
    output = tmp_path/'tables'
    evidence = write_output(bundle, output)
    assert len(evidence['tables']) == 10
    assert evidence['submission_ready'] is False
    assert evidence['timing_bins'] == dict(shortest=[32,51],median=[128,244],longest=[256,1336])
    assert '40.36' in (output/'comparison_final.tex').read_text()  # 10000/24777 * 100.
    assert '0.95' in (output/'costs_longest.tex').read_text()  # bytes to MiB, not MB.
    assert 'Attn. buffered' in (output/'buffered_costs_longest.tex').read_text()
    assert 'Mamba minus' in (output/'seed_differences_final.tex').read_text()
    assert json.loads((output/'final_result_bundle.json').read_text()) == bundle
    with pytest.raises(FileExistsError): write_output(bundle, output)


@pytest.mark.parametrize('change', ('interim','partial','wrong_accuracy','nan','missing_seed','duplicate_seed','wrong_sign','unpaired_count'))
def test_plausible_looking_incomplete_or_inconsistent_recognition_rejected(bundle, change):
    if change == 'interim': bundle['runs'][0]['steps'] = 8000
    elif change == 'partial': bundle['runs'][0]['samples'] = 348
    elif change == 'wrong_accuracy': bundle['runs'][0]['accuracy'] += .01
    elif change == 'nan': bundle['runs'][0]['accuracy'] = float('nan')
    elif change == 'missing_seed': bundle['runs'].pop()
    elif change == 'duplicate_seed': bundle['runs'][-1] = deepcopy(bundle['runs'][-2])
    elif change == 'wrong_sign': bundle['seed_comparisons'][0]['mean_pp'] *= -1
    elif change == 'unpaired_count': next(iter(bundle['single_seed_comparisons'].values()))['a_only_correct'] += 1
    with pytest.raises(ValueError): validate(bundle)


@pytest.mark.parametrize('change', ('prefix','partial','hardware','paused_gpu','missing_buffered','nonfinite_timing','other_video','shifted_bin'))
def test_unfair_or_incomplete_cost_comparisons_rejected(bundle, change):
    entry = bundle['costs'][0]
    if change == 'prefix': entry['measurement']['frames_per_pass'] = 8
    elif change == 'partial': entry['measurement']['bins'].pop()
    elif change == 'hardware': entry['measurement']['hardware']['gpu'] = 'different GPU'
    elif change == 'paused_gpu': entry['measurement']['paused_compute_pids'] = [1234]
    elif change == 'missing_buffered': bundle['buffered_costs'].pop()
    elif change == 'nonfinite_timing': entry['measurement']['bins'][0]['gpu_ms']['p50'] = float('nan')
    elif change == 'other_video': entry['video']['id'] = 'different RGB sequence'
    elif change == 'shifted_bin': entry['measurement']['bins'][-1]['history_min'] += 1
    with pytest.raises(ValueError): validate(bundle)


def test_incomplete_live_protocol_rejects_before_weights_are_loaded(tmp_path):
    priority=tmp_path/'priority'; buffered=tmp_path/'buffered'
    priority.mkdir(); buffered.mkdir()
    (priority/'status.json').write_text('{"status":"waiting_for_video_comparisons"}')
    (buffered/'status.json').write_text('{"status":"complete"}')
    # No plans, weights, datasets, or training-library imports are needed to reject.
    with pytest.raises(ValueError, match='priority queue is incomplete'):
        collect(priority,buffered)
