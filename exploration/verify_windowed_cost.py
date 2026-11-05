"""Recount bounded-attention costs from saved frame measurements on CPU.

Verification does not replay GPU inference. Numerical and timing observations
remain empirical records from the separately executed measurement process.
"""
import json
from pathlib import Path

from benchmark_real_video import summarize_rows
from encoder_costs import paired_costs
from exploration.benchmark_windowed_attention import (
    REPEATS, VARIANTS, WINDOWS, expected_state_bytes, require_predecessor,
    select_protocol, traversal_order, verify_numerics,
)
from exploration.cache_windowed_delayed_action import require_qualification
from point_experiment import read_json
from temporal_priority import verify_video
from video_checkpoint import file_hash


SELECTIONS = ('shortest', 'median', 'longest')


def verify_specification(path):
    spec = read_json(path)
    for name, digest in spec['source_sha256'].items():
        if file_hash(name) != digest:
            raise ValueError('Prepared measurement source changed: ' + name)
    for name in ('priority', 'qualification', 'after_queue'):
        if file_hash(Path(spec[name]) / 'plan.json') != spec[name + '_plan_sha256']:
            raise ValueError('Prepared upstream plan changed: ' + name)
    if (spec['variants'] != list(VARIANTS) or spec['windows'] != list(WINDOWS)
            or spec['repeats'] != REPEATS or spec['warmup_passes'] != 1
            or spec['batch_size'] != 1 or spec['input_size'] != 224
            or spec['output_tokens'] != [196, 768]
            or spec['measured_order'] != [list(traversal_order(r)) for r in range(REPEATS)]
            or [r['video']['selection'] for r in spec['comparisons']] != list(SELECTIONS)):
        raise ValueError('Prepared cost protocol or video grid differs')
    for row in spec['comparisons']:
        selection = row['video']['selection']
        checkpoints, _, video = select_protocol(spec['priority'], selection)
        if checkpoints != spec['checkpoints'] or row['video'] != video:
            raise ValueError('Prepared cost uses different checkpoints or RGB videos')
        if row['expected_state_bytes'] != {v: expected_state_bytes(v, video['frames']) for v in VARIANTS}:
            raise ValueError('Prepared state capacity differs')
        command = ['uv', 'run', '--locked', 'python', '-m', 'exploration.benchmark_windowed_attention',
                   '--selection', selection, '--output', row['output']]
        if row['command'] != command:
            raise ValueError('Prepared command differs')
    return spec


def verify_artifacts(directory, spec, selection, predecessor, qualified):
    """Verify one video using independently verified dependencies and a spec."""
    expected = [r for r in spec['comparisons'] if r['video']['selection'] == selection]
    if len(expected) != 1:
        raise ValueError('Cost video is missing or duplicated')
    video = expected[0]['video']
    plan, report = read_json(directory / 'plan.json'), read_json(directory / 'summary.json')
    plan_hash = file_hash(directory / 'plan.json')
    if (report['plan_sha256'] != plan_hash
            or report['correctness_sha256'] != file_hash(directory / 'correctness.json')
            or report.get('weights_unchanged') is not True):
        raise ValueError('Cost completion artifacts are incomplete or changed')
    fixed = dict(video=video, frames=video['frames'], variants=list(VARIANTS), windows=list(WINDOWS),
        batch_size=1, repeats=REPEATS, warmup_passes=1, output_tokens=[196, 768], classifier_timed=False,
        measured_order=[list(traversal_order(r)) for r in range(REPEATS)], predecessor=predecessor,
        priority_plan_sha256=spec['priority_plan_sha256'], source_sha256=spec['source_sha256'],
        full_attention_max_abs_tolerance=1e-5, window_tolerances=qualified['tolerances']['bf16'],
        qualification_summary_sha256=file_hash(Path(spec['qualification']) / 'summary.json'))
    if any(plan.get(k) != value for k, value in fixed.items()):
        raise ValueError('Measured cost protocol differs from the prepared controls')
    if set(plan['checkpoints']) != set(spec['checkpoints']):
        raise ValueError('Cost checkpoint set differs')
    for name, reference in spec['checkpoints'].items():
        info = plan['checkpoints'][name]
        if (info['config'] != reference['config'] or info['step'] != 20000
                or info['checkpoint_sha256'] != file_hash(Path(reference['run']) / 'last.pt')):
            raise ValueError('Cost measurement checkpoint differs')
    for path, digest in plan['source_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Measured source changed: ' + path)
    checked = read_json(directory / 'correctness.json')
    full = checked['full_attention']
    if (checked['plan_sha256'] != plan_hash
            or [r['frame'] for r in full] != list(range(video['frames']))
            or set(checked['windows']) != {str(w) for w in WINDOWS}):
        raise ValueError('Framewise numerical comparison coverage differs')
    for row in full:
        if (not 0 <= row['max_abs'] <= 1e-5 or not 0 <= row['rmse'] <= row['max_abs']
                or row['states'] != dict(dynamic=602112 * (row['frame'] + 1),
                                        buffered=602112 * video['frames'] + 24)):
            raise ValueError('Full-attention numerical or capacity evidence differs')
    for window in WINDOWS:
        verify_numerics(checked['windows'][str(window)], video['frames'], plan['window_tolerances'],
                        expected_state_bytes(f'attention_window{window}', video['frames']))
    measurements = report['measurements']
    if len(measurements) != len(VARIANTS) or {m['model'] for m in measurements} != set(VARIANTS):
        raise ValueError('Five measured variants are incomplete or duplicated')
    for entry in measurements:
        variant = entry['model']
        folder = directory / variant
        summary = read_json(folder / 'summary.json')
        if (entry['video'] != video or entry['measurement'] != summary
                or summary['timings_sha256'] != file_hash(folder / 'timings.jsonl')):
            raise ValueError('Raw timing or measurement summary changed')
        observed = [json.loads(s) for s in (folder / 'timings.jsonl').read_text().splitlines() if s]
        recounted = summarize_rows(observed, video['frames'], REPEATS)
        if any(summary.get(k) != value for k, value in recounted.items()):
            raise ValueError('Reported cost does not match raw-frame recount')
        if any(r['state_bytes'] != expected_state_bytes(variant, video['frames']) for r in observed):
            raise ValueError('Observed state capacity differs')
        name = 'mamba_interleaved' if variant == 'mamba_interleaved' else 'attention_interleaved'
        parameters = plan['parameter_counts'][name]
        if (summary['hardware'] != {k: plan[k] for k in ('gpu', 'torch', 'cuda')}
                or summary['parameter_count'] != parameters['count']
                or summary['parameter_bytes'] != parameters['bytes']
                or summary['paused_compute_pids'] != []):
            raise ValueError('Measured hardware, parameters or exclusivity differs')
        cpu_metadata = 0 if variant == 'mamba_interleaved' else 24 if variant == 'attention_buffered' else 48
        minimum_live = parameters['bytes'] + expected_state_bytes(variant, video['frames']) - cpu_metadata
        if any(r['torch_allocated_after_bytes'] < minimum_live for r in observed):
            raise ValueError('Allocated memory excludes live parameters or GPU state')
        outputs = summary['final_output_sha256_by_pass']
        if (len(outputs) != REPEATS or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in outputs)
                or summary['repeated_final_outputs_equal'] != (len(set(outputs)) == 1)
                or len(summary['process_memory_after_pass']) != REPEATS):
            raise ValueError('Measured pass output or memory records are incomplete')
    if any(report.get(k) != value for k, value in paired_costs(measurements).items()):
        raise ValueError('Paired cost ratios differ from measured costs')
    return measurements


def verify_report(directory, specification, selection):
    spec = verify_specification(specification)
    predecessor = require_predecessor(Path(spec['after_queue']), Path(spec['qualification']))
    qualified = require_qualification(Path(spec['qualification']))
    for row in spec['checkpoints'].values():
        verify_video(row)
    return verify_artifacts(directory, spec, selection, predecessor, qualified)
