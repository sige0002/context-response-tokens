"""Export verified final comparisons and generate reviewable LaTeX tables.

Collection reuses the experiment's CPU validators and needs local checkpoints,
manifests and raw measurements. Rendering an exported bundle only checks its
internal consistency; it does not repeat GPU inference or authenticate evidence.
Neither operation declares the manuscript submission-ready.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from paper.build_tables import tabular


ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = '2cc915986c3024e1f96370934c8157a93628fd6830b5cf797b4ab70cd122dd41'
LABELS = {
    'none_interleaved': 'Image only',
    'attention_interleaved': 'Temporal attn.',
    'mamba_interleaved': 'Mamba inter.',
    'mamba_late': 'Mamba late',
    'gmmix_after_encoder': 'GMMix final',
    'gmmix_multidepth_after_encoder': 'GMMix multi.',
}
REPLICATED = ('mamba_interleaved', 'attention_interleaved', 'mamba_late', 'gmmix_multidepth_after_encoder')
VARIANTS = ('attention_dynamic', 'attention_buffered', 'mamba_interleaved')
VARIANT_LABELS = dict(attention_dynamic='Attn. dynamic', attention_buffered='Attn. buffered',
                      mamba_interleaved='Mamba inter.')
VIDEOS = dict(shortest=52, median=245, longest=1337)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require_complete(priority, buffered):
    for name, directory in (('priority', priority), ('buffered attention', buffered)):
        if read(directory/'status.json').get('status') != 'complete':
            raise ValueError(f'{name} queue is incomplete; no final tables may be generated')
    if sha(priority/'plan.json') != PLAN_SHA256:
        raise ValueError('Not the predeclared final comparison plan')


def collect(priority, buffered):
    """Revalidate authoritative artifacts; do not trust completed status alone."""
    priority, buffered = Path(priority).resolve(), Path(buffered).resolve()
    require_complete(priority, buffered)  # Fast refusal before loading any weights.
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run final collection from the repository root')
    import torch
    from temporal_priority import verify_plan, summarize_videos, summarize_replications
    from encoder_costs import verify_measurement, paired_costs
    from exploration.run_buffered_cost import verify_plan as verify_buffered_plan, verify_report

    torch.set_num_threads(2)
    plan = read(priority/'plan.json'); buffer_plan = read(buffered/'plan.json')
    if (Path(buffer_plan['priority']).resolve() != priority
            or buffer_plan['priority_plan_sha256'] != PLAN_SHA256):
        raise ValueError('Attention implementation control belongs to a different comparison')
    verify_plan(plan); verify_buffered_plan(buffer_plan)
    base = plan['encoders'] + [plan['multidepth']]
    six = summarize_videos(base)
    three = summarize_replications(plan)
    if six != read(priority/'six_video_summary.json') or three != read(priority/'three_seed_video_summary.json'):
        raise ValueError('Published recognition summary differs from revalidated predictions')
    sources = {}

    def record(path):
        path = Path(path).resolve()
        try: name = str(path.relative_to(ROOT))
        except ValueError: name = str(path)
        sources[name] = sha(path)

    for name in plan['source_sha256']:
        record(ROOT/name)
    for name in buffer_plan['sources']:
        record(ROOT/name)
    for directory, names in ((priority, ('plan.json', 'status.json', 'six_video_summary.json', 'three_seed_video_summary.json', 'six_costs.json')),
                              (buffered, ('plan.json', 'status.json', 'summary.json'))):
        for name in names: record(directory/name)
    runs = []
    base_by_name = {r['name']: r for r in six['runs']}
    for entry in base + plan['replications']:
        model = entry.get('model', entry['name']); config = entry['config']; directory = Path(entry['run'])
        report = read(directory/'final_clip_stream.json')
        predictions = report['ablations']['full']['modes']['stream']['clips']
        correct = sum(p['prediction'] == p['label'] for p in predictions)
        checkpoint = report['checkpoint_sha256']
        expected = base_by_name[model] if config['seed'] == 42 else three['models'][model][str(config['seed'])]
        accuracy = expected['stream_accuracy'] if config['seed'] == 42 else expected['accuracy']
        if (checkpoint != expected['checkpoint_sha256'] or len(predictions) != 24777
                or abs(correct/24777-accuracy) > 1e-12):
            raise ValueError('Final recognition count or checkpoint differs')
        progress = [json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines() if s]
        parameters = {p['trainable_parameters'] for p in progress if p.get('event') == 'start'}
        if len(parameters) != 1:
            raise ValueError('Missing or inconsistent trainable parameter count')
        runs.append(dict(model=model, seed=config['seed'], correct=correct, samples=24777,
            accuracy=accuracy, steps=config['steps'], frames=config['frames'],
            trainable_parameters=parameters.pop(), checkpoint_sha256=checkpoint,
            evaluation_sha256=expected['evaluation_sha256'],
            clip_stream_disagreements=report['ablations']['full']['clip_vs_stream']['prediction_disagreements']))
        for name in ('config.json', 'last.pt', 'progress.jsonl', *expected['evaluation_sha256']):
            record(directory/name)
        if sha(directory/'last.pt') != checkpoint:
            raise ValueError('Final checkpoint changed during collection')
        for name in ('train', 'val'): record(config[name])
    measurements = []
    cost = plan['cost']
    for entry in base:
        for video in cost['videos']:
            directory = priority/'costs'/entry['name']/video['selection']
            checked = verify_measurement(directory, entry['config'],
                dict(checkpoint_sha256=base_by_name[entry['name']]['checkpoint_sha256'], step=20000),
                video, cost['repeats'], cost['warmup_passes'], cost['extraction_sha256'])
            measurements.append(dict(model=entry['name'], video=video, measurement=checked))
            for name in ('plan.json', 'summary.json', 'timings.jsonl'): record(directory/name)
    if dict(measurements=measurements, **paired_costs(measurements)) != read(priority/'six_costs.json'):
        raise ValueError('Cost summary differs from revalidated raw frame timings')
    buffer_measurements = []
    for selection in VIDEOS:
        directory = buffered/selection
        buffer_measurements += verify_report(directory, priority, selection)
        for name in ('plan.json', 'correctness.json', 'summary.json'): record(directory/name)
        for variant in VARIANTS:
            for name in ('summary.json', 'timings.jsonl'): record(directory/variant/name)
    buffer_expected = dict(plan_sha256=sha(buffered/'plan.json'), measurements=buffer_measurements,
                           **paired_costs(buffer_measurements))
    if buffer_expected != read(buffered/'summary.json'):
        raise ValueError('Buffered attention aggregate differs from verified implementations')
    bundle = dict(format_version=1, primary_plan_sha256=PLAN_SHA256, runs=runs,
        costs=measurements, buffered_costs=buffer_measurements,
        seed_comparisons=three['comparisons'], single_seed_comparisons=six['comparisons'],
        sources=sources, collector_sha256=sha(__file__), submission_ready=False,
        scope='Final SSv2 validation and fixed RGB costs only; no natural long-term-memory or robot-success result.',
        limits=['Validation inspected during development; not a held-out test.',
                'Primary four models have three seeds; image-only and final-layer GMMix have one seed.',
                'Timing repeats are correlated, not uncertainty across independent deployments.',
                'PyTorch allocator peak is not total system or device memory.',
                'GMMix uses the documented convolution fallback; attention buffering is an implementation control.',
                'Collection rechecks saved predictions and measurements, without new GPU execution.'])
    validate(bundle)
    return bundle


def validate(bundle):
    """Validate portable bundle consistency, not the original raw artifact chain."""
    if bundle['format_version'] != 1 or bundle['primary_plan_sha256'] != PLAN_SHA256:
        raise ValueError('Final bundle protocol differs')
    expected = {(m, 42) for m in LABELS} | {(m, s) for m in REPLICATED for s in (43, 44)}
    rows = bundle['runs']; keys = [(r['model'], r['seed']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Incomplete or duplicate model/seed grid')
    for row in rows:
        if (row['steps'] != 20000 or row['frames'] != 8 or row['samples'] != 24777
                or type(row['correct']) is not int or not 0 <= row['correct'] <= row['samples']
                or not math.isfinite(row['accuracy'])
                or abs(row['accuracy']-row['correct']/row['samples']) > 1e-12):
            raise ValueError('Expected final full-population prediction counts')
        if (type(row['trainable_parameters']) is not int or row['trainable_parameters'] <= 0
                or type(row['clip_stream_disagreements']) is not int
                or not 0 <= row['clip_stream_disagreements'] <= row['samples']):
            raise ValueError('Invalid parameter or disagreement count')
        expected_reports = {'final_clip_stream.json'}
        if row['model'] != 'none_interleaved': expected_reports.add('final_history_ablations.json')
        if set(row['evaluation_sha256']) != expected_reports:
            raise ValueError('Missing final recognition/history evidence')
        for digest in (row['checkpoint_sha256'], *row['evaluation_sha256'].values()):
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid evidence digest')
    accuracies = {(r['model'], r['seed']): r['accuracy'] for r in rows}
    if set(bundle['single_seed_comparisons']) != set(LABELS)-{'mamba_interleaved'}:
        raise ValueError('Incomplete fixed-weight comparisons')
    for other, entry in bundle['single_seed_comparisons'].items():
        a, b = accuracies['mamba_interleaved',42], accuracies[other,42]
        expected_values = dict(a_mean=a, b_mean=b, difference=a-b, difference_percentage_points=100*(a-b))
        if any(not math.isfinite(entry[k]) or abs(entry[k]-v)>1e-10 for k,v in expected_values.items()):
            raise ValueError('Fixed-weight contrast differs from prediction counts')
        interval = entry['conditional_95_interval']
        if (len(interval)!=2 or not all(math.isfinite(v) and -1<=v<=1 for v in interval)
                or interval[0]>interval[1] or entry['conditional_95_interval_pp'] != [100*v for v in interval]
                or entry['examples']!=24777 or not 2<=entry['clusters']<=24777
                or entry['bootstrap_repeats']!=10000 or entry['bootstrap_seed']!=17):
            raise ValueError('Invalid conditional interval or resampling protocol')
        only_a, only_b = entry['a_only_correct'], entry['b_only_correct']
        if (type(only_a) is not int or type(only_b) is not int or min(only_a,only_b)<0
                or only_a+only_b>24777 or abs((only_a-only_b)/24777-(a-b))>1e-12):
            raise ValueError('Paired correct/incorrect transitions differ')
    if Counter(c['other'] for c in bundle['seed_comparisons']) != Counter(REPLICATED[1:]):
        raise ValueError('Incomplete seed comparisons')
    for entry in bundle['seed_comparisons']:
        delta = [100*(accuracies['mamba_interleaved',s]-accuracies[entry['other'],s]) for s in (42,43,44)]
        expected_values = dict(seed_order=[42,43,44], differences_pp=delta,
                               mean_pp=statistics.mean(delta), sample_sd_pp=statistics.stdev(delta))
        if any(entry[k] != v for k, v in expected_values.items()):
            raise ValueError('Seed comparison sign, order or spread differs from raw counts')
    hardware = None; video_identities = {}
    for field, models in (('costs', LABELS), ('buffered_costs', VARIANTS)):
        entries = bundle[field]
        keys = [(e['model'], e['video']['selection']) for e in entries]
        if len(keys) != len(models)*3 or set(keys) != {(m,s) for m in models for s in VIDEOS}:
            raise ValueError('Incomplete or duplicate model/video cost grid')
        for entry in entries:
            video, measured = entry['video'], entry['measurement']
            selection = video['selection']
            if selection in video_identities and video != video_identities[selection]:
                raise ValueError('Models use different cost videos')
            video_identities[selection] = video
            if (video['frames'] != VIDEOS[video['selection']] or measured['frames_per_pass'] != video['frames']
                    or measured['repeats'] != 3 or measured['paused_compute_pids']):
                raise ValueError('Cost history length, repeats or GPU exclusivity differs')
            if hardware is None: hardware = measured['hardware']
            if measured['hardware'] != hardware:
                raise ValueError('Mixed hardware/software in cost comparison')
            expected_bins = []
            for low, high in ((0,0),(1,7),(8,31),(32,127),(128,255),(256,None)):
                end = min(video['frames']-1, high if high is not None else video['frames']-1)
                if end >= low: expected_bins.append((low, high, end-low+1, 3*(end-low+1)))
            actual_bins = [(b['history_min'],b['history_max'],b['unique_frames'],b['samples']) for b in measured['bins']]
            if actual_bins != expected_bins:
                raise ValueError('Incomplete history bins')
            for b in measured['bins']:
                for metric in ('gpu_ms', 'model_wall_ms', 'pipeline_wall_ms'):
                    value = b[metric]['p50']
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError('Invalid latency')
            if (type(measured['max_state_bytes']) is not int or measured['max_state_bytes'] < 0
                    or type(measured['torch_peak_allocated_bytes']) is not int or measured['torch_peak_allocated_bytes'] <= 0):
                raise ValueError('Invalid memory size')


def tables(bundle):
    validate(bundle)
    runs = {(r['model'],r['seed']): r for r in bundle['runs']}
    output = {'comparison_final.tex': tabular('lrrr', 'Encoder & Trainable M & Top-1 (\\%) & Disagree', [
        (label, f"{runs[m,42]['trainable_parameters']/1e6:.2f}", f"{100*runs[m,42]['accuracy']:.2f}",
         runs[m,42]['clip_stream_disagreements']) for m,label in LABELS.items()])}
    output['seeds_final.tex'] = tabular('lrrrr', 'Encoder & 42 & 43 & 44 & Mean $\\pm$ s.d.', [
        (LABELS[m], *[f'{100*runs[m,s]["accuracy"]:.2f}' for s in (42,43,44)],
         f'{statistics.mean(100*runs[m,s]["accuracy"] for s in (42,43,44)):.2f} $\\pm$ '
         f'{statistics.stdev(100*runs[m,s]["accuracy"] for s in (42,43,44)):.2f}') for m in REPLICATED])
    output['seed_differences_final.tex'] = tabular('lrr', 'Mamba minus & Mean (pp) & Sample s.d. (pp)',
        [(LABELS[r['other']], f"{r['mean_pp']:+.2f}", f"{r['sample_sd_pp']:.2f}") for r in bundle['seed_comparisons']])
    output['conditional_differences_final.tex'] = tabular('@{}lrr@{}', 'Mamba minus & Difference (pp) & Conditional 95\\% (pp)',
        [(LABELS[m], f"{r['difference_percentage_points']:+.2f}",
          f"[{r['conditional_95_interval_pp'][0]:+.2f}, {r['conditional_95_interval_pp'][1]:+.2f}]")
         for m,r in bundle['single_seed_comparisons'].items()])
    for field, labels in (('costs', LABELS), ('buffered_costs', VARIANT_LABELS)):
        pairs = {(r['model'],r['video']['selection']):r['measurement'] for r in bundle[field]}
        for selection in VIDEOS:
            rows = []
            for model, label in labels.items():
                measured = pairs[model,selection]; late = measured['bins'][-1]
                rows.append((label, f"{late['gpu_ms']['p50']:.2f}", f"{late['pipeline_wall_ms']['p50']:.2f}",
                    f"{measured['max_state_bytes']/2**20:.2f}", f"{measured['torch_peak_allocated_bytes']/2**20:.1f}"))
            output[f'{field}_{selection}.tex'] = tabular('@{}lrrrr@{}', 'Encoder & GPU ms & Pipe ms & State MiB & Peak MiB', rows)
    return output


def write_output(bundle, output):
    rendered = tables(bundle)
    bundle_text = json.dumps(bundle, indent=2, allow_nan=False)+'\n'
    timing_bins = {}
    for selection in VIDEOS:
        measurements = [e['measurement'] for e in bundle['costs'] if e['video']['selection'] == selection]
        bounds = [(m['bins'][-1]['history_min'], m['bins'][-1]['history_max']) for m in measurements]
        if len(set(bounds)) != 1:
            raise ValueError('Cost tables cannot combine different history bins')
        low, high = bounds[0]
        timing_bins[selection] = [low, min(VIDEOS[selection]-1, high) if high is not None else VIDEOS[selection]-1]
    output.mkdir(parents=True, exist_ok=False)
    for name, content in rendered.items(): (output/name).write_text(content)
    (output/'final_result_bundle.json').write_text(bundle_text)
    evidence = dict(bundle_sha256=sha(output/'final_result_bundle.json'), renderer_sha256=sha(__file__),
        tables={name:sha(output/name) for name in rendered}, submission_ready=False,
        interpretation='Tables for review; main.tex is not automatically rewritten.',
        timing_bins=timing_bins)
    (output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--priority', type=Path, default=ROOT/'results/temporal_priority_seed42_rgbfix')
    c.add_argument('--buffered', type=Path, default=ROOT/'results/buffered_attention_cost_control')
    r = sub.add_parser('render'); r.add_argument('--bundle', type=Path, required=True)
    for p in (c,r): p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    bundle = collect(args.priority, args.buffered) if args.command == 'collect' else read(args.bundle)
    print(json.dumps(write_output(bundle,args.output), indent=2))


if __name__ == '__main__':
    main()
