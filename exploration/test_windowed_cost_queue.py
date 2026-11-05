import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from exploration import run_windowed_cost as queue
from exploration import verify_windowed_cost as verifier
from exploration.benchmark_windowed_attention import REPEATS, VARIANTS, WINDOWS, expected_state_bytes, traversal_order
from benchmark_real_video import summarize_rows
from encoder_costs import paired_costs
from point_experiment import read_json, write_json
from video_checkpoint import file_hash


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def artifact_fixture(tmp_path):
    """Synthetic timing records exercise recounting; these are not GPU results."""
    directory, qualification = tmp_path / 'measured', tmp_path / 'qualification'
    video = dict(selection='shortest', id='synthetic', frames=3, shard='test', video_sha256='fixed RGB')
    qualified = dict(tolerances=dict(bf16=dict(max_abs=.125, relative_l2=.005)))
    put(qualification / 'summary.json', qualified)
    checkpoints = {}
    for name in ('mamba_interleaved', 'attention_interleaved'):
        run = tmp_path / name
        run.mkdir()
        (run / 'last.pt').write_bytes(('test weights ' + name).encode())
        checkpoints[name] = dict(run=str(run), config=dict(seed=42, steps=20000, kind=name))
    spec = dict(comparisons=[dict(video=video)], qualification=str(qualification),
                checkpoints=checkpoints, priority_plan_sha256='priority', source_sha256={})
    predecessor = dict(queue='previous', plan_sha256='p', summary_sha256='s')
    plan = dict(video=video, frames=3, variants=list(VARIANTS), windows=list(WINDOWS),
        batch_size=1, repeats=REPEATS, warmup_passes=1, output_tokens=[196, 768], classifier_timed=False,
        measured_order=[list(traversal_order(i)) for i in range(REPEATS)], predecessor=predecessor,
        priority_plan_sha256='priority', source_sha256={}, full_attention_max_abs_tolerance=1e-5,
        window_tolerances=qualified['tolerances']['bf16'],
        qualification_summary_sha256=file_hash(qualification / 'summary.json'),
        checkpoints={name: dict(config=row['config'], step=20000,
            checkpoint_sha256=file_hash(Path(row['run']) / 'last.pt')) for name, row in checkpoints.items()},
        parameter_counts={name: dict(count=100, bytes=400) for name in checkpoints},
        gpu='synthetic-test', torch='test', cuda='test')
    put(directory / 'plan.json', plan)
    correctness = dict(plan_sha256=file_hash(directory / 'plan.json'),
        full_attention=[dict(frame=i, exact=True, max_abs=0., rmse=0.,
            states=dict(dynamic=602112 * (i + 1), buffered=602112 * 3 + 24)) for i in range(3)],
        windows={str(w): [dict(frame=i, max_abs=0., relative_l2=0.,
            state_bytes=expected_state_bytes(f'attention_window{w}', 3)) for i in range(3)] for w in WINDOWS})
    put(directory / 'correctness.json', correctness)
    entries = []
    for number, variant in enumerate(VARIANTS, 1):
        folder = directory / variant
        folder.mkdir()
        rows = [dict(repeat=r, history_frames=i, absolute_frames_after=i + 1,
            gpu_ms=float(number), model_wall_ms=float(number + 1), pipeline_wall_ms=float(number + 2),
            state_bytes=expected_state_bytes(variant, 3),
            torch_allocated_before_bytes=1000 + expected_state_bytes(variant, 3),
            torch_allocated_after_bytes=1001 + expected_state_bytes(variant, 3),
            torch_peak_allocated_bytes=1002 + number + expected_state_bytes(variant, 3),
            torch_reserved_after_bytes=2000 + expected_state_bytes(variant, 3))
            for r in range(REPEATS) for i in range(3)]
        (folder / 'timings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        summary = dict(**summarize_rows(rows, 3, REPEATS), parameter_count=100, parameter_bytes=400,
            hardware={k: plan[k] for k in ('gpu', 'torch', 'cuda')}, repeated_final_outputs_equal=True,
            final_output_sha256_by_pass=['a' * 64] * REPEATS, paused_compute_pids=[],
            process_memory_after_pass=[dict(mib=None, status='unreported')] * REPEATS,
            timings_sha256=file_hash(folder / 'timings.jsonl'))
        put(folder / 'summary.json', summary)
        entries.append(dict(model=variant, video=video, measurement=summary))
    put(directory / 'summary.json', dict(plan_sha256=file_hash(directory / 'plan.json'),
        correctness_sha256=file_hash(directory / 'correctness.json'), weights_unchanged=True,
        measurements=entries, **paired_costs(entries)))
    return directory, spec, predecessor, qualified


def test_raw_recount_preserves_five_variants_and_ratio_direction(tmp_path):
    fixture = artifact_fixture(tmp_path)
    actual = verifier.verify_artifacts(fixture[0], fixture[1], 'shortest', fixture[2], fixture[3])
    assert len(actual) == 5
    ratios = {r['other']: r for r in paired_costs(actual)['comparisons']}
    assert ratios['attention_buffered']['bins'][0]['mamba_over_other_gpu_median'] == .5
    assert ratios['attention_window128']['bins'][0]['mamba_over_other_gpu_median'] == .2


@pytest.mark.parametrize('fault', ['checkpoint', 'rotation', 'tolerance', 'full_missing', 'window_alarm',
                                 'duplicate_variant', 'missing_frame', 'metric', 'capacity', 'allocation',
                                 'pass_record', 'ratio'])
def test_recount_rejects_semantic_mismatches_even_with_refreshed_hashes(tmp_path, fault):
    directory, spec, predecessor, qualified = artifact_fixture(tmp_path)
    plan = read_json(directory / 'plan.json')
    report = read_json(directory / 'summary.json')
    checked = read_json(directory / 'correctness.json')
    entry = report['measurements'][-1]
    folder = directory / entry['model']
    summary = entry['measurement']
    if fault == 'checkpoint':
        plan['checkpoints']['attention_interleaved']['config']['seed'] = 43
    elif fault == 'rotation':
        plan['measured_order'][-1] = plan['measured_order'][0]
    elif fault == 'tolerance':
        plan['window_tolerances']['max_abs'] = 10
    elif fault == 'full_missing':
        checked['full_attention'].pop()
    elif fault == 'window_alarm':
        checked['windows']['128'][0]['relative_l2'] = .006
    elif fault == 'duplicate_variant':
        report['measurements'][-1] = copy.deepcopy(report['measurements'][0])
    elif fault in ('missing_frame', 'capacity', 'allocation'):
        rows = [json.loads(s) for s in (folder / 'timings.jsonl').read_text().splitlines()]
        if fault == 'missing_frame':
            rows.pop()
        elif fault == 'capacity':
            rows[-1]['state_bytes'] -= 16
            summary.update(summarize_rows(rows, 3, REPEATS))
        else:
            rows[-1]['torch_allocated_after_bytes'] = 1
            summary.update(summarize_rows(rows, 3, REPEATS))
        (folder / 'timings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        summary['timings_sha256'] = file_hash(folder / 'timings.jsonl')
    elif fault == 'metric':
        summary['bins'][0]['gpu_ms']['p50'] *= .01
    elif fault == 'pass_record':
        summary['final_output_sha256_by_pass'].pop()
    else:
        report['comparisons'][0]['bins'][0]['mamba_over_other_gpu_median'] = 100
    put(folder / 'summary.json', summary)
    put(directory / 'plan.json', plan)
    checked['plan_sha256'] = file_hash(directory / 'plan.json')
    put(directory / 'correctness.json', checked)
    report['plan_sha256'] = checked['plan_sha256']
    report['correctness_sha256'] = file_hash(directory / 'correctness.json')
    put(directory / 'summary.json', report)
    with pytest.raises(ValueError):
        verifier.verify_artifacts(directory, spec, 'shortest', predecessor, qualified)


def test_current_prepared_plan_matches_fixed_sources_and_videos():
    plan = verifier.verify_specification('results/windowed_cost_plan_v1.json')
    assert [row['video']['frames'] for row in plan['comparisons']] == [52, 245, 1337]


@pytest.mark.parametrize('partial', [False, True])
def test_serial_launch_collects_all_videos_or_preserves_partial_output(tmp_path, monkeypatch, partial):
    fixture_root = tmp_path / 'fixture'
    fixture_root.mkdir()
    fixture = artifact_fixture(fixture_root)
    entries = verifier.verify_artifacts(fixture[0], fixture[1], 'shortest', fixture[2], fixture[3])
    output = tmp_path / 'queue'
    output.mkdir()
    put(output / 'plan.json', {})
    spec = dict(qualification='qualified', comparisons=[])
    for selection in verifier.SELECTIONS:
        directory = tmp_path / selection
        command = [sys.executable, '-c',
            'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/"summary.json").write_text("{}");',
            str(directory)]
        spec['comparisons'].append(dict(video=dict(selection=selection), output=str(directory), command=command))
    if partial:
        directory = Path(spec['comparisons'][0]['output'])
        directory.mkdir()
        (directory / 'partial-evidence').write_text('preserve')
    def exited(*args):
        reader, writer = os.pipe()
        os.close(writer)
        return reader
    def verified(directory, specification, selection):
        result = copy.deepcopy(entries)
        for row in result:
            row['video'].update(id=selection, selection=selection)
        return result
    monkeypatch.setattr(queue, 'verify_plan', lambda plan: spec)
    monkeypatch.setattr(queue, 'bind_process', exited)
    monkeypatch.setattr(queue, 'require_predecessor', lambda *args: {})
    monkeypatch.setattr(queue, 'assert_exclusive', lambda: None)
    monkeypatch.setattr(queue, 'verify_report', verified)
    plan = dict(predecessor='previous', specification='prepared')
    if partial:
        with pytest.raises(ValueError, match='Partial cost output'):
            queue.execute(plan, output, os.getpid())
        assert (directory / 'partial-evidence').read_text() == 'preserve'
        status = read_json(output / 'status.json')
        assert status['status'] == 'failed' and status['operations'] == []
    else:
        queue.execute(plan, output, os.getpid())
        status, report = read_json(output / 'status.json'), read_json(output / 'summary.json')
        assert status['status'] == 'complete'
        assert status['completed'] == list(verifier.SELECTIONS)
        assert [r['selection'] for r in status['operations']] == list(verifier.SELECTIONS)
        assert all(r['returncode'] == 0 for r in status['operations'])
        assert len(report['measurements']) == 15 and len(report['comparisons']) == 12
        assert status['summary_sha256'] == file_hash(output / 'summary.json')


def wait_ready(path, process, predicate=lambda value: True):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists() and predicate(read_json(path)):
            return
        if process.poll() is not None:
            raise AssertionError('Controlled process exited before readiness')
        time.sleep(.02)
    raise AssertionError('Controlled process did not become ready')


def test_cancelling_bound_waiter_preserves_predecessor_and_creates_no_measurement(tmp_path):
    predecessor_dir = tmp_path / 'predecessor'
    spec = read_json('results/windowed_cost_plan_v1.json')
    ready = tmp_path / 'ready.json'
    predecessor = subprocess.Popen([sys.executable, '-c',
        'import sys,time; from pathlib import Path; Path(sys.argv[1]).write_text("{}"); time.sleep(30)',
        str(ready), 'exploration.run_windowed_delayed_action', '--output', str(predecessor_dir)])
    waiter = None
    try:
        wait_ready(ready, predecessor)
        with pytest.raises(ValueError, match='different output'):
            queue.bind_process(predecessor.pid, 'exploration.run_windowed_delayed_action', '--output', tmp_path / 'wrong')
        output = tmp_path / 'waiter'
        output.mkdir()
        spec_path = tmp_path / 'spec.json'
        put(spec_path, spec)
        # Only specification verification is replaced; real pidfd binding,
        # signals, cleanup and status writes execute in the child process.
        command = ('import sys; from pathlib import Path; import exploration.run_windowed_cost as q; '
                   'spec=q.read_json(sys.argv[1]); q.verify_plan=lambda plan: spec; '
                   'q.execute(dict(predecessor=sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]))')
        waiter = subprocess.Popen([sys.executable, '-c', command, str(spec_path), str(predecessor_dir),
            str(output), str(predecessor.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_ready(output / 'status.json', waiter, lambda value: value['status'] == 'waiting_for_windowed_recognition')
        waiter.terminate()
        waiter.wait(timeout=5)
        assert waiter.returncode != 0
        status = read_json(output / 'status.json')
        assert status['status'] == 'stopped' and status['operations'] == []
        assert predecessor.poll() is None
        assert not (output / 'summary.json').exists()
    finally:
        for process in (waiter, predecessor):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
