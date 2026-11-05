"""Motion/occlusion/history strata for the completed, fixed-budget pilot.

Compare the separated output with both the final-feature control and the
image-only multi-depth control. Reused validation scenes are exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from exploration.oracle_transport import digest


def strata(truth, visible):
    movement = np.linalg.norm((truth[:, 1:]-truth[:, :1])*256, axis=-1)
    valid = visible[:, 1:]
    continuous = np.logical_and.accumulate(visible, axis=1)
    masks = dict(all_visible=valid, moving_gt2=valid & (movement > 2), moving_gt16=valid & (movement > 16),
        after_occlusion=valid & ~continuous[:, 1:], reappearance=valid & ~visible[:, :-1])
    age = np.arange(1, visible.shape[1])[None]
    for lo, hi in [(1, 7), (8, 15), (16, 23)]:
        masks[f'age_{lo}_{hi}'] = valid & (age >= lo) & (age <= hi)
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args(); root = Path(args.run)
    plan = json.loads((root/'plan.json').read_text()); config = plan['config']
    rows = []; records = {}
    for arm in config['arms']:
        records[arm] = digest(root/(arm+'.json'))
        record = json.loads((root/(arm+'.json')).read_text())
        verification = json.loads((root/('verification_'+arm+'.json')).read_text())
        assert verification['status'] == 'passed' and verification['record_sha256'] == records[arm]
        for step, score in record['scores'].items():
            folder = root/arm/step
            for name, expected in score['predictions_sha256'].items():
                path = folder/name; assert digest(path) == expected
                scene, condition = path.stem.split('_', 1)
                with np.load(path) as data:
                    truth, visible = data['target_xy'], data['visible']
                    masks = strata(truth, visible)
                    for method in ('field_top4', 'static_query'):
                        errors = np.linalg.norm((data[method][:, 1:]-truth[:, 1:])*256, axis=-1)
                        for label, mask in masks.items():
                            selected = errors[mask]; count = len(selected)
                            rows.append(dict(arm=arm, step=int(step), scene=scene, condition=condition, method=method,
                                stratum=label, points=count, mean_error=float(selected.mean()) if count else None,
                                **{f'pck_{q}': float((selected < q).mean()) if count else None for q in (4, 8, 16)}))
    summary = []
    keys = ('arm', 'step', 'condition', 'method', 'stratum')
    for values in sorted({tuple(row[k] for k in keys) for row in rows}):
        selected = [row for row in rows if tuple(row[k] for k in keys) == values and row['points']]
        summary.append(dict(zip(keys, values), scenes=len(selected), points=sum(r['points'] for r in selected),
            **{key: float(np.mean([r[key] for r in selected])) if selected else None for key in ('mean_error', 'pck_4', 'pck_8', 'pck_16')}))
    comparisons = []
    final = config['steps']; rng = np.random.default_rng(17)
    contrasts = [
        ('separated_memory', 'full', 'final_memory', 'full'),
        ('separated_memory', 'full', 'separated_image', 'full'),
        ('separated_memory', 'full', 'separated_memory', 'reset1'),
        ('final_memory', 'full', 'final_memory', 'reset1')]
    index = {(r['arm'], r['condition'], r['scene'], r['stratum']): r for r in rows
             if r['step'] == final and r['method'] == 'field_top4'}
    for a, ac, b, bc in contrasts:
        for label in sorted({r['stratum'] for r in rows}):
            chosen = [scene for scene in config['val'] if index[(a, ac, scene, label)]['points'] and index[(b, bc, scene, label)]['points']]
            if not chosen: continue
            draw = rng.integers(len(chosen), size=(2000, len(chosen)))
            for metric in ('pck_4', 'pck_16'):
                difference = np.array([index[(a, ac, scene, label)][metric]-index[(b, bc, scene, label)][metric] for scene in chosen])*100
                comparisons.append(dict(a=a, a_condition=ac, b=b, b_condition=bc, stratum=label, metric=metric,
                    scenes=len(chosen), mean_percentage_points=float(difference.mean()),
                    exploratory_95_interval=np.percentile(difference[draw].mean(-1), [2.5, 97.5]).tolist()))
    report = dict(source_sha256=digest(__file__), plan_sha256=digest(root/'plan.json'), records_sha256=records,
        rows=rows, summary=summary, comparisons=comparisons,
        uncertainty='2000 paired scene bootstrap draws, seed17, unadjusted exploratory intervals. Eight repeatedly inspected scenes; no unseen-test claim.',
        limits='Motion, prior occlusion and history-age strata are post-training diagnostic analyses. Age bands change both memory length and physical time; they do not isolate forgetting. Always-visible readout; query anchor is retained in reset and image controls.')
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in summary:
        if row['step'] == final and row['method'] == 'field_top4': print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
