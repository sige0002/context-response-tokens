"""Export the completed bounded-memory controls without mixing score protocols.

Collection revalidates saved predictions and timings, without GPU replay.
Portable rendering verifies internal consistency, not artifact authenticity.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev

from paper.build_tables import tabular
from paper.final_tables import read, sha


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SHA = 'ffe182b668c7518782aeadad67bf30f4038c8f4d20a8a4ce1b178a3552c7c94e'
COST_SPEC_SHA = '6a805a94880f760a016852bd79c2402d8172dc750ab0f09319b5fe0987cf5cf2'
WINDOWS, SEEDS, DELAYS = (8, 32, 128), (42, 43, 44), (0, 1, 8, 32, 128)
CONDITIONS = ('full_memory', 'suffix_only', 'saved_evidence')
VIDEOS = dict(shortest=52, median=245, longest=1337)
VARIANTS = dict(mamba_interleaved='Mamba', attention_buffered='Full attn.',
                attention_window8='Attn. W8', attention_window32='Attn. W32', attention_window128='Attn. W128')
CONTRASTS = ('mamba_minus_window', 'full_attention_minus_window')


def collect(recognition, cost_queue):
    recognition, cost_queue = Path(recognition).resolve(), Path(cost_queue).resolve()
    for name, directory in (('recognition', recognition), ('cost', cost_queue)):
        state = read(directory / 'status.json')
        if state.get('status') != 'complete':
            raise ValueError(f'{name} control is incomplete; no result tables may be generated')
        if state.get('summary_sha256') != sha(directory / 'summary.json'):
            raise ValueError('Completed control summary changed')
    if Path.cwd() != ROOT:
        raise ValueError('Collect from the repository root')
    import torch
    from exploration.aggregate_windowed_delayed_action import aggregate
    from exploration.run_windowed_delayed_action import verify_dependencies
    from exploration.run_windowed_cost import verify_plan
    from exploration.verify_windowed_cost import verify_report
    from encoder_costs import paired_costs

    torch.set_num_threads(2)
    rp, cp = read(recognition / 'plan.json'), read(cost_queue / 'plan.json')
    if sha(rp['experiment']) != EXPERIMENT_SHA or sha(cp['specification']) != COST_SPEC_SHA:
        raise ValueError('Controls use a different predeclared experiment')
    spec = verify_plan(cp)
    if Path(spec['after_queue']).resolve() != recognition:
        raise ValueError('Cost and recognition use different experiments')
    verify_dependencies(rp)
    recounted = aggregate(Path(rp['experiment']))
    if read(recognition / 'summary.json') != dict(queue_plan_sha256=sha(recognition / 'plan.json'), **recounted):
        raise ValueError('Delayed-action summary differs from prediction recount')
    costs, measured_checkpoints, artifacts, sources = [], {}, {}, {}

    def record(path):
        path = Path(path).resolve()
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sources[name] = sha(path)

    for row in spec['comparisons']:
        directory, selection = Path(row['output']), row['video']['selection']
        costs += verify_report(directory, Path(cp['specification']), selection)
        artifacts[selection] = sha(directory / 'summary.json')
        checkpoint_hashes = {k: v['checkpoint_sha256'] for k, v in read(directory / 'plan.json')['checkpoints'].items()}
        if measured_checkpoints and checkpoint_hashes != measured_checkpoints:
            raise ValueError('Cost videos use different checkpoints')
        measured_checkpoints = checkpoint_hashes
        for name in ('plan.json', 'summary.json', 'correctness.json'):
            record(directory / name)
        for variant in VARIANTS:
            for name in ('summary.json', 'timings.jsonl'):
                record(directory / variant / name)
    expected = dict(queue_plan_sha256=sha(cost_queue / 'plan.json'), measurement_summary_sha256=artifacts,
                    measurements=costs, **paired_costs(costs))
    if read(cost_queue / 'summary.json') != expected:
        raise ValueError('Cost aggregate differs from raw-frame recount')
    for directory in (recognition, cost_queue):
        for name in ('plan.json', 'status.json', 'summary.json'):
            record(directory / name)
    for path in (rp['experiment'], cp['specification'], __file__):
        record(path)
    for path in set(rp['pins']) | set(cp['sources']) | set(spec['source_sha256']):
        record(path)
    bundle = dict(format_version=1, experiment_sha256=EXPERIMENT_SHA, cost_specification_sha256=COST_SPEC_SHA,
        recognition=recounted, costs=costs, cost_checkpoint_sha256=measured_checkpoints, sources=sources,
        submission_ready=False, scope='Controlled delayed-action probes (348 validation videos) and separate real-RGB costs',
        limits=['Not full SSv2 validation, natural continuous-video memory, or robot success.',
                'Three encoder seeds; timing uses only final seed42 and five correlated traversals.',
                'The maximum delay of 128 does not exceed the three-layer W128 input support of 382 observations.',
                'PyTorch peak allocation excludes driver/context and other non-PyTorch allocations.',
                'Collection rechecks saved results; no independent GPU replay.'])
    validate(bundle)
    return bundle


def validate(bundle):
    if (bundle['format_version'] != 1 or bundle['experiment_sha256'] != EXPERIMENT_SHA
            or bundle['cost_specification_sha256'] != COST_SPEC_SHA or bundle['submission_ready'] is not False):
        raise ValueError('Bounded-control bundle protocol differs')
    rows = bundle['recognition']['comparisons']
    expected = {(c, w, d) for c in CONDITIONS for w in WINDOWS for d in DELAYS if c != 'suffix_only' or d > 0}
    actual = [(r['condition'], r['window'], r['delay_observations']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate diagnostic conditions')
    references = {}
    for row in rows:
        if [r['seed'] for r in row['seed_results']] != list(SEEDS):
            raise ValueError('Missing or reordered encoder seeds')
        differences = {k: [] for k in CONTRASTS}
        for value in row['seed_results']:
            if not math.isfinite(value['window_accuracy']):
                raise ValueError('Nonfinite diagnostic accuracy')
            for contrast in CONTRASTS:
                pair = value[contrast]
                if pair['samples'] != 348:
                    raise ValueError('Diagnostic must use its original 348-video cohort')
                counts = [pair[k] for k in ('left_correct', 'right_correct', 'left_only_correct', 'right_only_correct')]
                if any(type(v) is not int or not 0 <= v <= 348 for v in counts):
                    raise ValueError('Invalid paired prediction counts')
                left, right, gained, lost = counts
                both = left - gained
                if both < 0 or right - lost != both or both + gained + lost > 348:
                    raise ValueError('Paired correct/incorrect transitions differ')
                difference = 100 * (left - right) / 348
                if (not math.isfinite(pair['difference_pp']) or abs(pair['difference_pp'] - difference) > 1e-12
                        or abs(value['window_accuracy'] - right / 348) > 1e-12):
                    raise ValueError('Diagnostic percentages or contrast sign differ from counts')
                if row['condition'] == 'suffix_only' and (left != 2 or right != 2):
                    raise ValueError('Class-independent suffix must retain its exact chance control')
                key = (row['condition'], row['delay_observations'], value['seed'], contrast)
                if key in references and references[key] != left:
                    raise ValueError('Reference scores change across window comparisons')
                references[key] = left
                differences[contrast].append(difference)
        for contrast, values in differences.items():
            stats = row[contrast]
            if (len(stats['differences_pp']) != 3
                    or any(not math.isfinite(x) or abs(x - y) > 1e-12 for x, y in zip(stats['differences_pp'], values))
                    or not math.isfinite(stats['mean_pp']) or abs(stats['mean_pp'] - mean(values)) > 1e-12
                    or not math.isfinite(stats['sample_sd_pp']) or abs(stats['sample_sd_pp'] - stdev(values)) > 1e-12):
                raise ValueError('Training-seed summaries differ from paired counts')
    records = bundle['recognition']['run_artifacts']
    models = ('mamba_interleaved', 'attention_interleaved', 'attention_window8', 'attention_window32', 'attention_window128')
    if set(records) != {f'{m}_seed{s}' for m in models for s in SEEDS}:
        raise ValueError('Missing diagnostic model/seed artifacts')
    for seed in SEEDS:
        full = records[f'attention_interleaved_seed{seed}']['checkpoint_sha256']
        if any(records[f'attention_window{w}_seed{seed}']['checkpoint_sha256'] != full for w in WINDOWS):
            raise ValueError('Attention windows use different learned weights')
    for model in ('mamba_interleaved', 'attention_interleaved'):
        if bundle['cost_checkpoint_sha256'][model] != records[f'{model}_seed42']['checkpoint_sha256']:
            raise ValueError('Cost and diagnostic use different seed42 weights')
    costs = bundle['costs']
    keys = [(r['model'], r['video']['selection']) for r in costs]
    if len(keys) != 15 or set(keys) != {(m, v) for m in VARIANTS for v in VIDEOS}:
        raise ValueError('Missing or duplicate five-variant cost measurements')
    hardware, videos = None, {}
    for entry in costs:
        model, video, measured = entry['model'], entry['video'], entry['measurement']
        selection, frames = video['selection'], video['frames']
        if (frames != VIDEOS[selection] or measured['frames_per_pass'] != frames
                or measured['repeats'] != 5 or measured['paused_compute_pids']):
            raise ValueError('Cost scope, traversal count or GPU exclusivity differs')
        if selection in videos and videos[selection] != video:
            raise ValueError('Cost variants use different videos')
        videos[selection] = video
        if hardware is not None and hardware != measured['hardware']:
            raise ValueError('Cost variants use different hardware/software')
        hardware = measured['hardware']
        bins = [(lo, hi, min(frames - 1, hi if hi is not None else frames - 1) - lo + 1)
                for lo, hi in ((0, 0), (1, 7), (8, 31), (32, 127), (128, 255), (256, None)) if lo < frames]
        if [(b['history_min'], b['history_max'], b['unique_frames'], b['samples']) for b in measured['bins']] != [(a, b, n, 5 * n) for a, b, n in bins]:
            raise ValueError('Incomplete cost history bins')
        for b in measured['bins']:
            for metric in ('gpu_ms', 'model_wall_ms', 'pipeline_wall_ms'):
                if not math.isfinite(b[metric]['p50']) or b[metric]['p50'] <= 0:
                    raise ValueError('Invalid cost latency')
        expected_state = (79027200 if model == 'mamba_interleaved' else 602112 * frames + 24
                          if model == 'attention_buffered' else 602112 * int(model.removeprefix('attention_window')) + 48)
        if measured['max_state_bytes'] != expected_state or measured['torch_peak_allocated_bytes'] < expected_state:
            raise ValueError('Recorded cost memory is inconsistent with state capacity')


def render(bundle):
    validate(bundle)
    lookup = {(r['condition'], r['window'], r['delay_observations']): r for r in bundle['recognition']['comparisons']}
    output, csv_rows = {}, []
    def spread(values):
        return f'{mean(values):.2f} $\\pm$ {stdev(values):.2f}'
    for condition in CONDITIONS:
        scores, differences = [], []
        for delay in DELAYS:
            if condition == 'suffix_only' and delay == 0:
                continue
            anchor = lookup[condition, 8, delay]['seed_results']
            reference = [spread([100 * r[k]['left_correct'] / 348 for r in anchor]) for k in CONTRASTS]
            windows = [spread([100 * r['window_accuracy'] for r in lookup[condition, w, delay]['seed_results']]) for w in WINDOWS]
            scores.append((delay, *reference, *windows))
            for window in WINDOWS:
                row = lookup[condition, window, delay]
                differences.append((delay, window, *[f"{row[k]['mean_pp']:+.2f} $\\pm$ {row[k]['sample_sd_pp']:.2f}" for k in CONTRASTS]))
                for seed in row['seed_results']:
                    csv_rows.append(dict(condition=condition, delay_observations=delay, window=window, seed=seed['seed'],
                        samples=348, window_accuracy_percent=100 * seed['window_accuracy'],
                        mamba_minus_window_pp=seed['mamba_minus_window']['difference_pp'],
                        full_attention_minus_window_pp=seed['full_attention_minus_window']['difference_pp']))
        output[f'windowed_{condition}_scores.tex'] = tabular('@{}rrrrrr@{}', 'Delay & Mamba & Full attn. & W8 & W32 & W128', scores)
        output[f'windowed_{condition}_differences.tex'] = tabular('@{}rrrr@{}', 'Delay & W & Mamba $-$ W (pp) & Full attn. $-$ W (pp)', differences)
    for selection in VIDEOS:
        values = {r['model']: r['measurement'] for r in bundle['costs'] if r['video']['selection'] == selection}
        rows = []
        for model, label in VARIANTS.items():
            measured = values[model]
            last = measured['bins'][-1]
            rows.append((label, f"{last['gpu_ms']['p50']:.2f}", f"{last['pipeline_wall_ms']['p50']:.2f}",
                         f"{measured['max_state_bytes']/2**20:.2f}", f"{measured['torch_peak_allocated_bytes']/2**20:.1f}"))
        output[f'windowed_cost_{selection}.tex'] = tabular('@{}lrrrr@{}', 'Encoder & GPU ms & Pipe ms & State MiB & Peak MiB', rows)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    output['windowed_seed_differences.csv'] = stream.getvalue()
    return output


def write_output(bundle, output):
    rendered = render(bundle)
    text = json.dumps(bundle, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'windowed_result_bundle.json').write_text(text)
    for name, contents in rendered.items():
        (output / name).write_text(contents)
    evidence = dict(bundle_sha256=sha(output / 'windowed_result_bundle.json'), renderer_sha256=sha(__file__),
        outputs={name: sha(output / name) for name in rendered}, submission_ready=False,
        timing_bins=dict(shortest=[32, 51], median=[128, 244], longest=[256, 1336]),
        scope='Review tables; original manuscript and full-population recognition tables are unchanged')
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--recognition', type=Path, default=ROOT / 'results/windowed_delayed_action_queue_v1')
    c.add_argument('--cost', type=Path, default=ROOT / 'results/windowed_attention_cost_queue_v1')
    r = sub.add_parser('render')
    r.add_argument('--bundle', type=Path, required=True)
    for p in (c, r):
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bundle = collect(args.recognition, args.cost) if args.command == 'collect' else read(args.bundle)
    print(json.dumps(write_output(bundle, args.output), indent=2))
