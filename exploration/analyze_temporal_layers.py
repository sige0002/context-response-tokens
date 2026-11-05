"""Report all masks and paired layer effects; never select a winning mask.

Intervals are exploratory percentile bootstraps over videos or rendered
scenes. They are not simultaneous intervals or independent test evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration.oracle_transport import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args(); root = Path(args.run)
    plan = json.loads((root/'plan.json').read_text())
    result = json.loads((root/'summary.json').read_text())
    assert digest(root/'plan.json') == result['plan_sha256']
    masks = list(plan['config']['masks']); scenes = plan['config']['val_scenes']
    points = result['points']
    rows = [row for row in points['summary'] if row['task'] == 'motion' and row['method'] == 'field_top4']
    indexed = {(row['mask'], row['stratum']): row for row in rows}
    tables = []
    for mask in masks:
        all_points = indexed[(mask, 'all_visible')]; moving = indexed[(mask, 'moving_gt16')]
        video = result['videos']['scores'][mask]
        tables.append(dict(mask=mask, active_depths=[int(key)+1 for key in plan['config']['masks'][mask]],
            video_accuracy_percent=video['accuracy']*100, video_correct=video['correct'], video_total=video['total'],
            pck4_percent=all_points['pck_4']*100, pck16_percent=all_points['pck_16']*100,
            moving_gt16_pck16_percent=moving['pck_16']*100,
            aj_percent=all_points['full_sequence_metrics']['average_jaccard']*100,
            state_bytes=points['states'][scenes[0]][mask]['state_bytes']))
    path = root/'video_logits.npz'; assert digest(path) == result['videos']['logits_sha256']
    with np.load(path) as raw:
        video_values = {mask: (raw[mask].argmax(-1) == raw['targets']).astype(float) for mask in masks}
        full_predictions = raw['111'].argmax(-1)
        target = raw['targets'].copy()
    checkpoint_path = Path(plan['probe_run'])/'checkpoint.pt'
    assert digest(checkpoint_path) == plan['checkpoint']['checkpoint_sha256']
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert saved['config'] == plan['checkpoint']['config'] and saved['step'] == plan['checkpoint']['step']
    old_validation = next(row['validation'] for row in saved['history'] if row['step'] == saved['step'])
    indices = [row['index'] for row in plan['video_subset']]
    reference_predictions = np.array(old_validation['predictions'])[indices]
    differences = np.flatnonzero(reference_predictions != full_predictions)
    reference = dict(checkpoint_sha256=plan['checkpoint']['checkpoint_sha256'],
        full_validation_accuracy=old_validation['accuracy'],
        subset_reference_correct=int((reference_predictions == target).sum()),
        subset_repeated_correct=int((full_predictions == target).sum()),
        subset_samples=len(target), different_predictions=len(differences),
        differences=[dict(id=plan['video_subset'][i]['id'], label=int(target[i]),
            reference=int(reference_predictions[i]), repeated=int(full_predictions[i])) for i in differences],
        scope='Same pinned checkpoint; stored full-validation predictions use batch4, intervention rerun uses batch1. This subset has a different class mix and sampling error from the full validation.')
    del saved
    point_rows = [row for row in points['rows'] if row['task'] == 'motion' and row['method'] == 'field_top4']
    by_scene = {(row['mask'], row['scene'], row['stratum']): row for row in point_rows}
    values = dict(video_accuracy=video_values)
    values['pck4'] = {mask: np.array([by_scene[(mask, scene, 'all_visible')]['pck_4'] for scene in scenes]) for mask in masks}
    values['moving_gt16_pck16'] = {mask: np.array([by_scene[(mask, scene, 'moving_gt16')]['pck_16'] for scene in scenes]) for mask in masks}
    rng = np.random.default_rng(17)
    selections = {name: rng.integers(len(value['000']), size=(2000, len(value['000']))) for name, value in values.items()}
    effects = []
    for position, depth in enumerate((4, 8, 12)):
        off = [mask for mask in masks if mask[position] == '0']
        pairs = [(mask, mask[:position]+'1'+mask[position+1:]) for mask in off]
        for name, value in values.items():
            paired = np.stack([value[on]-value[off] for off, on in pairs]).mean(0)
            interval = np.percentile(paired[selections[name]].mean(-1)*100, [2.5, 97.5])
            effects.append(dict(depth=depth, metric=name, enabling_effect_percentage_points=float(paired.mean()*100),
                bootstrap_95_interval_percentage_points=interval.tolist(),
                unit='video' if name=='video_accuracy' else 'scene', units=len(paired), pairs=pairs))
    report = dict(source_sha256=digest(__file__), summary_sha256=digest(root/'summary.json'),
        tables=tables, layer_effects=effects, reference_validation_check=reference,
        interpretation='Effect of enabling one trained layer averaged over all four settings of the other layers. Same fixed full-model classifier. Descriptive sensitivity, not trained-architecture superiority.',
        uncertainty='2000 paired bootstrap draws, seed17. Video or scene sampling units. Exploratory, unadjusted intervals; repeated validation data.')
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
