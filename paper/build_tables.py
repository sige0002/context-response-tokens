"""Generate draft tables from public audited artifacts; no model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'training': 'results/ssv2_implementation_audit/rgbfix_full_run_metadata_audit_step9000.json',
    'history': 'results/video_history_intervention_seed17/summary.json',
    'history_verification': 'results/video_history_intervention_seed17/verification.json',
    'history_plan': 'results/video_history_intervention_seed17/plan.json',
    'memory': 'results/memory_layer_intervention_seed17_v2/summary.json',
    'memory_verification': 'results/memory_layer_intervention_seed17_v2/verification.json',
    'memory_plan': 'results/memory_layer_intervention_seed17_v2/plan.json',
    'comparison_plan': 'results/temporal_priority_seed42_rgbfix/plan.json',
}
LABELS = {
    'none_interleaved': 'Image only',
    'attention_interleaved': 'Interleaved temporal attention',
    'mamba_interleaved': 'Interleaved Mamba-3',
    'mamba_late': 'Late Mamba-3',
    'gmmix_after_encoder': 'Final-layer GMMix',
    'gmmix_multidepth_after_encoder': 'Multi-depth GMMix',
}
HISTORY = {
    'full_clip': 'Original history, clip',
    'full_stream': 'Original history, stream',
    'current_only': 'Current image only',
    'repeat_current': 'Current image repeated',
    'reverse_history': 'Past observations reversed',
}
MEMORY = {
    'full_stream': 'None',
    'clear_4': 'After block 4',
    'clear_8': 'After block 8',
    'clear_12': 'After block 12',
    'reset_all': 'All three layers',
}


def collect_memory(data, paths):
    """Check published provenance and recount labels, without GPU execution."""
    memory, plan, verified = (data[k] for k in ('memory', 'memory_plan', 'memory_verification'))
    history = data['history']
    if not verified['passed'] or verified['summary_sha256'] != sha(paths['memory']):
        raise ValueError('Memory summary lacks its matching recount')
    if memory['plan_sha256'] != sha(paths['memory_plan']):
        raise ValueError('Memory protocol hash changed')
    if plan['checkpoint_info'] != data['history_plan']['checkpoint_info']:
        raise ValueError('Memory checkpoint differs from history diagnostic')
    if plan['subset'] != data['history_plan']['subset']:
        raise ValueError('Memory cohort differs from history diagnostic')
    for key, filename in [('history', 'summary.json'), ('history_plan', 'plan.json'),
                          ('history_verification', 'verification.json')]:
        if plan['prior_hashes'][filename] != sha(paths[key]):
            raise ValueError('Memory baseline provenance changed')
    if (not memory['weights_unchanged'] or not history['weights_unchanged']
            or memory['model_state_sha256'] != history['model_state_sha256']
            or memory['inputs_sha256'] != history['inputs_sha256']
            or memory['inputs_sha256'] != plan['prior_hashes']['inputs.json']):
        raise ValueError('Memory weights or inputs changed')
    kept = {'clear_4': ['7', '11'], 'clear_8': ['3', '11'], 'clear_12': ['3', '7']}
    reused = {'full_stream': 'full_stream', 'reset_all': 'current_only'}
    if plan['retained_keys'] != kept or plan['reused_baselines'] != reused:
        raise ValueError('Memory intervention definition changed')
    if set(memory['scores']) != set(MEMORY) or verified['predictions_recounted'] != 1740:
        raise ValueError('Incomplete memory comparison')
    cohort = [(r['id'], r['label']) for r in plan['subset']]
    base = memory['scores']['full_stream']
    for name, row in memory['scores'].items():
        records = row['predictions']
        if [(r['id'], r['target']) for r in records] != cohort or row['total'] != 348:
            raise ValueError('Memory prediction cohort or labels changed')
        if any(type(r['prediction']) is not int or not 0 <= r['prediction'] < 174 for r in records):
            raise ValueError('Invalid memory prediction class')
        correct = sum(r['target'] == r['prediction'] for r in records)
        if correct != row['correct'] or correct != verified['counts'][name]:
            raise ValueError('Memory counts do not match predictions')
        expected = {'original_label_top1': correct / 348,
                    'delta_vs_stream_pp': 100 * (correct - base['correct']) / 348}
        for key, value in expected.items():
            if not math.isfinite(row[key]) or abs(row[key] - value) > 1e-10:
                raise ValueError('Memory accuracy or delta does not match predictions')
        if name in reused and records != history['scores'][reused[name]]['predictions']:
            raise ValueError('Reused memory baseline predictions changed')
    if set(memory['state_by_frame']) != set(kept):
        raise ValueError('Missing memory state trace')
    for frames in memory['state_by_frame'].values():
        if (len(frames) != 8 or any(r['offset'] != i + 1 or r['state_bytes'] != 79027200
                or r['state_layer_keys'] != ['3', '7', '11'] for i, r in enumerate(frames))):
            raise ValueError('Memory state trace does not match the claimed intervention')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    paths = {name: root / path for name, path in SOURCES.items()}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    training = data['training']
    if not all(training['sources_match'].values()) or any(training['cross_split_overlap'].values()):
        raise ValueError('Training provenance audit did not pass')
    curve = training['validation']
    if [r['step'] for r in curve] != list(range(1000, 9001, 1000)):
        raise ValueError('This draft explicitly uses the audited 9000-step snapshot')
    for row in curve:
        if row['samples'] != 24777 or abs(row['correct']/row['samples']-row['accuracy']) > 1e-12:
            raise ValueError('Wrong validation population or count')
    history = data['history']; verified = data['history_verification']; plan = data['history_plan']
    if not verified['passed'] or verified['summary_sha256'] != sha(paths['history']):
        raise ValueError('History summary lacks its matching recount')
    if history['plan_sha256'] != sha(paths['history_plan']) or plan['checkpoint_info']['step'] != 4000:
        raise ValueError('History checkpoint or protocol changed')
    if set(history['scores']) != set(HISTORY) or len(plan['subset']) != 348:
        raise ValueError('Incomplete history comparison')
    for name, row in history['scores'].items():
        records = row['predictions']
        count = sum(r['target'] == r['prediction'] for r in records)
        if len(records) != 348 or count != row['correct'] or count != verified['counts'][name]['correct']:
            raise ValueError('History counts do not match saved predictions')
        if abs(row['original_label_top1'] - count/348) > 1e-12:
            raise ValueError('History accuracy does not match its prediction count')
        if len({r['id'] for r in records}) != 348:
            raise ValueError('Duplicated history examples')
    collect_memory(data, paths)
    comparisons = data['comparison_plan']
    rows = comparisons['encoders'] + [comparisons['multidepth']]
    if len(rows) != 6 or {r['name'] for r in rows} != set(LABELS):
        raise ValueError('Wrong comparison grid')
    if any(r['config']['steps'] != 20000 or r['config']['frames'] != 8 for r in rows):
        raise ValueError('Comparison budget changed')
    return data, {str(path.relative_to(root)): sha(path) for path in paths.values()}


def tabular(columns, header, rows):
    return '\\begin{tabular}{'+columns+'}\n\\hline\n'+header+' \\\\\n\\hline\n'+\
        '\n'.join(' & '.join(map(str, row))+' \\\\' for row in rows)+'\n\\hline\n\\end{tabular}\n'


def build(root, output):
    data, hashes = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    training = tabular('rrr', 'Updates & Correct / 24,777 & Top-1 (\\%)',
        [(f"{r['step']:,}", f"{r['correct']:,}", f"{100*r['accuracy']:.2f}") for r in data['training']['validation']])
    history = tabular('lrr', 'History intervention & Correct / 348 & Top-1 (\\%)',
        [(label, data['history']['scores'][key]['correct'],
          f"{100*data['history']['scores'][key]['original_label_top1']:.2f}") for key, label in HISTORY.items()])
    comparisons = tabular('lc', 'Final comparison, 20,000 updates & Streaming top-1',
        [(label, '--') for label in LABELS.values()])
    memory = tabular('lrrr', 'Erased history & Correct & Top-1 (\\%) & $\\Delta$ (pp)',
        [(label, data['memory']['scores'][key]['correct'],
          f"{100*data['memory']['scores'][key]['original_label_top1']:.2f}",
          f"{data['memory']['scores'][key]['delta_vs_stream_pp']:+.2f}")
         for key, label in MEMORY.items()])
    # Deliberately no fallback from interim to final cells.
    generated = {'training.tex': training, 'history.tex': history,
                 'memory_layers.tex': memory, 'comparison_pending.tex': comparisons}
    for name, contents in generated.items():
        (output/name).write_text(contents)
    evidence = dict(sources=hashes, generator_sha256=sha(Path(__file__)),
        generated={name: sha(output/name) for name in generated},
        final_comparison_status='not populated; final trained comparisons are required',
        history_scope='4000 updates, balanced 348-clip previously inspected validation subset',
        memory_scope='Same 4000-update checkpoint and 348 clips; all adapters run, only selected past states erased; two baselines reused',
        learning_curve_scope='interim clip validation on all 24777 clips; one seed',
        submission_ready=False)
    (output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'paper/generated'))
    args = parser.parse_args()
    result = build(ROOT, Path(args.output))
    print(json.dumps(result, indent=2))
