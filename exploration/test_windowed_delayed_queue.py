import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from exploration import aggregate_windowed_delayed_action as agg
from exploration import run_windowed_delayed_action as queue
from exploration.delayed_action_probe import evaluate
from exploration.test_delayed_action import fixture as feature_fixture


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_paired_count_sign_is_reference_minus_window():
    result = agg.paired_counts([0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1])
    assert result == dict(left_correct=3, right_correct=2, left_only_correct=2,
                          right_only_correct=1, difference_pp=25., samples=4)
    with pytest.raises(ValueError):
        agg.paired_counts([0], [], [0])


def probe_fixture(tmp_path):
    data, arrays = feature_fixture()
    cache, probe, model = (tmp_path / p for p in ('cache', 'probe', 'model'))
    for p in (cache, probe, model):
        p.mkdir()
    checkpoint = model / 'last.pt'
    checkpoint.write_bytes(b'controlled test checkpoint')
    reference = dict(run=str(model), config=dict(steps=20000, seed=42))
    plan = dict(data=data, data_plan_sha256='fixed cohort', checkpoint_path=str(checkpoint), sources={},
                checkpoint_info=dict(step=20000, config=reference['config'], checkpoint_sha256=agg.sha(checkpoint)))
    put(cache / 'plan.json', plan)
    np.savez_compressed(cache / 'features.npz', **arrays)
    put(cache / 'inputs.json', [])
    receipt = dict(scope='complete_cache', plan_sha256=agg.sha(cache / 'plan.json'),
                   features_sha256=agg.sha(cache / 'features.npz'), inputs_sha256=agg.sha(cache / 'inputs.json'),
                   weights_unchanged=True)
    put(cache / 'summary.json', receipt)
    results, weights = evaluate(arrays, data)
    np.savez_compressed(probe / 'readouts.npz', **weights)
    put(probe / 'summary.json', dict(cache_summary_sha256=agg.sha(cache / 'summary.json'),
        readouts_sha256=agg.sha(probe / 'readouts.npz'), source_sha256=agg.sha('exploration/delayed_action_probe.py'),
        validation_used_for_fitting=False, encoder_updated=False, results=results))
    return reference, data


def test_probe_refit_and_saved_readout_replay(tmp_path):
    reference, data = probe_fixture(tmp_path)
    result = agg.verify_probe(tmp_path, reference, 'fixed cohort', expected_data=data)
    assert result['results']['full_memory']['validation'][-1]['accuracy'] == 1.
    assert result['results']['suffix_only']['validation'][-1]['accuracy'] == .5


def test_changed_readout_is_rejected_even_if_its_file_hash_is_updated(tmp_path):
    reference, data = probe_fixture(tmp_path)
    path = tmp_path / 'probe/readouts.npz'
    with np.load(path) as saved:
        weights = {k: saved[k].copy() for k in saved.files}
    weights['full_memory_bias'][0] = 100
    np.savez_compressed(path, **weights)
    summary = agg.read(tmp_path / 'probe/summary.json')
    summary['readouts_sha256'] = agg.sha(path)
    put(tmp_path / 'probe/summary.json', summary)
    with pytest.raises(ValueError, match='Saved probe weights'):
        agg.verify_probe(tmp_path, reference, 'fixed cohort', expected_data=data)


@pytest.mark.parametrize('field', ['inputs_sha256', 'ids', 'labels', 'checkpoint_sha256'])
def test_model_pair_requires_identical_inputs_targets_and_attention_weights(field):
    left = dict(inputs_sha256='rgb', ids=['a', 'b'], labels=np.array([0, 1]), checkpoint_sha256='weights')
    right = copy.deepcopy(left)
    right[field] = dict(inputs_sha256='other', ids=['b', 'a'], labels=np.array([1, 0]),
                        checkpoint_sha256='other weights')[field]
    with pytest.raises(ValueError):
        agg.verify_pair_inputs(left, right, same_checkpoint=True)


def aggregate_fixture(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline'
    rows = [dict(name=f'{kind}_seed{seed}', model=kind, seed=seed, run=f'run/{kind}/{seed}', config={})
            for seed in (42, 43, 44) for kind in ('mamba_interleaved', 'attention_interleaved')]
    put(baseline / 'plan.json', dict(runs=rows))
    put(baseline / 'status.json', dict(status='complete'))
    put(baseline / 'summary.json', dict(plan_sha256=agg.sha(baseline / 'plan.json'),
        runs={row['name']: dict(probe_summary_sha256=row['name']) for row in rows}))
    put(tmp_path / 'data.json', {})
    runs = [dict(seed=row['seed'], window=w, reference_row=row,
                 output=str(tmp_path / f"attention_window{w}_seed{row['seed']}"))
            for row in rows if row['model'] == 'attention_interleaved' for w in (8, 32, 128)]
    plan = dict(comparison_plan=str(baseline / 'plan.json'), comparison_plan_sha256=agg.sha(baseline / 'plan.json'),
                data_plan=str(tmp_path / 'data.json'), data_plan_sha256=agg.sha(tmp_path / 'data.json'),
                source_sha256={}, seeds=[42, 43, 44], windows=[8, 32, 128], runs=runs, limitations=[])
    path = tmp_path / 'experiment.json'
    put(path, plan)

    def verified(directory, reference, data_hash, window=None, expected_data=None):
        pred = ([0, 0, 1, 1] if window is not None else
                [0, 1, 0, 1] if reference['model'] == 'mamba_interleaved' else [0, 1, 1, 1])
        return dict(results=dict(full_memory=dict(validation=[dict(delay_observations=d, predictions=pred,
            accuracy=sum(a == b for a, b in zip(pred, [0, 1, 0, 1])) / 4) for d in (0, 128)])),
            labels=np.array([0, 1, 0, 1]), ids=['a', 'b', 'c', 'd'], inputs_sha256='rgb',
            checkpoint_sha256=reference['model'] + str(reference['seed']),
            cache_summary_sha256=directory.name, probe_summary_sha256=directory.name)
    monkeypatch.setattr(agg, 'verify_probe', verified)
    return path, plan


def test_aggregation_keeps_every_seed_and_window_with_correct_units(tmp_path, monkeypatch):
    path, plan = aggregate_fixture(tmp_path, monkeypatch)
    result = agg.aggregate(path)
    assert len(result['run_artifacts']) == 15
    assert len(result['comparisons']) == 6
    for row in result['comparisons']:
        assert [x['seed'] for x in row['seed_results']] == [42, 43, 44]
        assert row['mamba_minus_window'] == dict(differences_pp=[50., 50., 50.], mean_pp=50., sample_sd_pp=0.)
        assert row['full_attention_minus_window']['mean_pp'] == 25.
    plan['runs'][-1] = plan['runs'][0]
    put(path, plan)
    with pytest.raises(ValueError, match='incomplete or duplicated'):
        agg.aggregate(path)


def queue_fixture(tmp_path):
    scheduler, pilot, baseline, qualification = (tmp_path / p for p in ('scheduler', 'pilot', 'baseline', 'qualification'))
    put(scheduler / 'status.json', dict(status='released_to_pilot'))
    put(pilot / 'plan.json', dict(scope='fixture'))
    put(pilot / 'summary.json', dict(plan_sha256=queue.sha(pilot / 'plan.json')))
    put(pilot / 'status.json', dict(status='complete', summary_sha256=queue.sha(pilot / 'summary.json')))
    put(baseline / 'plan.json', {})
    put(baseline / 'summary.json', dict(plan_sha256=queue.sha(baseline / 'plan.json')))
    put(baseline / 'status.json', dict(status='complete'))
    put(qualification / 'plan.json', {})
    experiment = tmp_path / 'experiment.json'
    put(experiment, dict(comparison_plan=str(baseline / 'plan.json'),
        comparison_plan_sha256=queue.sha(baseline / 'plan.json'),
        gpu_qualification_plan_sha256=queue.sha(qualification / 'plan.json')))
    return dict(experiment=str(experiment), scheduler=str(scheduler), pilot=str(pilot),
                qualification=str(qualification), pins={str(experiment): queue.sha(experiment)})


@pytest.mark.parametrize('failure', ['scheduler_waiting', 'pilot_failed', 'pilot_result_changed', 'baseline_failed'])
def test_predecessor_exit_alone_cannot_start_gpu_work(tmp_path, failure):
    plan = queue_fixture(tmp_path)
    queue.verify_dependencies(plan)
    if failure == 'scheduler_waiting':
        put(Path(plan['scheduler']) / 'status.json', dict(status='waiting_for_final_memory_layers'))
    elif failure == 'pilot_failed':
        put(Path(plan['pilot']) / 'status.json', dict(status='failed'))
    elif failure == 'pilot_result_changed':
        put(Path(plan['pilot']) / 'summary.json', {})
    else:
        put(tmp_path / 'baseline/status.json', dict(status='failed'))
    with pytest.raises(ValueError):
        queue.verify_dependencies(plan)


def wait_file(path, child):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        if child.poll() is not None:
            raise AssertionError('Controlled process exited before readiness')
        time.sleep(.01)
    raise AssertionError('Controlled process did not become ready')


def test_cancelling_waiter_never_signals_predecessor(tmp_path):
    plan = queue_fixture(tmp_path)
    ready = tmp_path / 'ready'
    predecessor = subprocess.Popen([sys.executable, '-c',
        'import sys,time; from pathlib import Path; Path(sys.argv[1]).write_text("ready"); time.sleep(30)',
        str(ready), 'exploration.run_mimo_after_comparisons', '--output', plan['scheduler'], '--pilot', plan['pilot']])
    waiter = None
    try:
        wait_file(ready, predecessor)
        with pytest.raises(ValueError, match='different experiment'):
            queue.bind_scheduler(predecessor.pid, tmp_path / 'wrong', Path(plan['pilot']))
        output = tmp_path / 'waiting'
        output.mkdir()
        plan_path = tmp_path / 'queue_plan.json'
        put(plan_path, plan)
        waiter = subprocess.Popen([sys.executable, '-c',
            'import sys; from pathlib import Path; from exploration.run_windowed_delayed_action import execute,read; '
            'execute(read(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))',
            str(plan_path), str(output), str(predecessor.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_file(output / 'status.json', waiter)
        assert queue.read(output / 'status.json')['status'] == 'waiting_for_mimo_pilot'
        waiter.terminate()
        waiter.wait(timeout=5)
        assert waiter.returncode != 0
        assert queue.read(output / 'status.json')['status'] == 'stopped'
        assert predecessor.poll() is None
    finally:
        for child in (waiter, predecessor):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
