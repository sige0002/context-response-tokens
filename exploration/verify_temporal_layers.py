"""Recount layer-intervention metrics and regenerate the paired video inputs.

This does not rerun the GPU encoders or the video classifier. The main run
separately requires both endpoint GPU features and point predictions to equal
the preceding independently saved experiment exactly.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from transformers import SiglipImageProcessor

from exploration.oracle_transport import digest
from video_data import RawVideoClips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args(); root = Path(args.run)
    torch.set_num_threads(2)
    plan = json.loads((root/'plan.json').read_text()); config = plan['config']
    report = json.loads((root/'summary.json').read_text()); points = report['points']
    assert digest(root/'plan.json') == report['plan_sha256']
    for source, expected in plan['sources'].items():
        assert digest(source) == expected, source
    probe = Path(plan['probe_run'])
    assert digest(probe/'plan.json') == plan['probe_plan_sha256']
    assert digest(probe/'summary.json') == plan['probe_summary_sha256']
    assert digest(probe/'checkpoint.pt') == report['checkpoint_sha256'] == plan['checkpoint']['checkpoint_sha256']
    old = json.loads((probe/'plan.json').read_text()); old_result = json.loads((probe/'summary.json').read_text())
    cache = Path(old['cache']); receipt = json.loads((cache/'summary.json').read_text())
    assert digest(cache/'summary.json') == old['cache_summary_sha256']
    arrays = {}; files = 0
    for scene in config['val_scenes']:
        original_path = cache/(scene+'.pt')
        assert digest(original_path) == receipt['scenes'][scene]['sha256']
        original = torch.load(original_path, map_location='cpu', weights_only=True)
        candidates = torch.where(original['visible'][:, 0])[0]
        generator = torch.Generator().manual_seed(config['seed']+int(scene))
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:config['evaluation_queries']]]
        truth = original['points'][chosen].numpy(); visible = original['visible'][chosen].numpy()
        feature_path = root/(scene+'_features.pt')
        assert digest(feature_path) == points['features_sha256'][scene]
        current = torch.load(feature_path, map_location='cpu', weights_only=True)
        old_feature_path = probe/(scene+'_features.pt')
        assert digest(old_feature_path) == old_result['features_sha256'][scene]
        previous = torch.load(old_feature_path, map_location='cpu', weights_only=True)
        assert torch.equal(current['selected'], chosen) and current['frames'] == original['frames']
        assert current['pixels_sha256'] == previous['pixels_sha256']
        for label, condition in [('000', 'siglip_only'), ('111', 'mamba_full')]:
            assert torch.equal(current['features'][label], previous['features'][condition]['12'])
        for label, keys in config['masks'].items():
            state = points['states'][scene][label]
            assert state['frames'] == len(current['frames']) and state['active_keys'] == keys
            assert state['state_bytes'] == len(keys)*points['states'][scene]['111']['state_bytes']//3
            feature = current['features'][label]
            assert feature.shape == (24, 196, 768) and torch.isfinite(feature).all()
            path = root/(scene+'_'+label+'.npz')
            assert digest(path) == points['predictions_sha256'][path.name]
            with np.load(path) as raw:
                data = {k: raw[k] for k in raw.files}
            assert np.array_equal(data['target_xy'], truth) and np.array_equal(data['visible'], visible)
            assert np.array_equal(data['selected'], chosen.numpy())
            assert np.array_equal(data['static_query'], np.broadcast_to(truth[:, :1], truth.shape))
            if label in ('000', '111'):
                condition = 'siglip_only' if label == '000' else 'mamba_full'
                previous_path = probe/(scene+'_'+condition+'_12.npz')
                assert digest(previous_path) == old_result['predictions_sha256'][previous_path.name]
                with np.load(previous_path) as prior:
                    for method in config['methods']:
                        assert np.array_equal(data[method], prior[method])
            arrays[(scene, label)] = data; files += 1
        del previous, current, original
    for row in points['rows']:
        data = arrays[(row['scene'], row['mask'])]
        truth, visible = data['target_xy'], data['visible']
        squared = (((data[row['method']]-truth)*256)**2).sum(-1)
        if row['task'] == 'identity':
            errors = np.sqrt(squared[:, 0])
        else:
            mask = visible[:, 1:].copy()
            movement = np.sqrt((((truth[:, 1:]-truth[:, :1])*256)**2).sum(-1))
            if row['stratum'] == 'moving_gt2': mask &= movement > 2
            if row['stratum'] == 'moving_gt16': mask &= movement > 16
            errors = np.sqrt(squared[:, 1:])[mask]
        assert len(errors) == row['points']
        assert abs(float(errors.mean())-row['mean_error']) < 2e-5
        for threshold in (1, 2, 4, 8, 16):
            assert float((errors < threshold).mean()) == row[f'pck_{threshold}']
        metrics = row['full_sequence_metrics']
        if metrics:
            valid = visible[:, 1:]; count = int(valid.sum()); total = valid.size
            js = []; pcks = []
            for threshold in (1, 2, 4, 8, 16):
                correct = int(((squared[:, 1:] < threshold**2) & valid).sum())
                j = correct/(count+total-correct); pck = correct/count
                assert abs(j-metrics[f'jaccard_{threshold}']) < 1e-12
                assert abs(pck-metrics[f'pts_within_{threshold}']) < 1e-12
                js.append(j); pcks.append(pck)
            assert abs(float(np.mean(js))-metrics['average_jaccard']) < 1e-12
            assert abs(float(np.mean(pcks))-metrics['average_pts_within_thresh']) < 1e-12
            assert abs(count/total-metrics['occlusion_accuracy']) < 1e-12
    for summary in points['summary']:
        keys = ('mask', 'depth', 'method', 'task', 'stratum')
        rows = [row for row in points['rows'] if all(row[k] == summary[k] for k in keys) and row['points']]
        assert len(rows) == summary['scenes'] and sum(r['points'] for r in rows) == summary['points']
        for key in ('mean_error', 'pck_1', 'pck_2', 'pck_4', 'pck_8', 'pck_16'):
            assert float(np.mean([r[key] for r in rows])) == summary[key]
        if summary['full_sequence_metrics']:
            for key, value in summary['full_sequence_metrics'].items():
                assert float(np.mean([r['full_sequence_metrics'][key] for r in rows])) == value
    checkpoint_config = plan['checkpoint']['config']
    assert digest(plan['validation_manifest']) == plan['validation_manifest_sha256'] == checkpoint_config['val_sha256']
    processor = SiglipImageProcessor.from_pretrained(checkpoint_config['model'],
        revision=checkpoint_config['revision'], local_files_only=True)
    dataset = RawVideoClips(plan['validation_manifest'], processor, frames=8, classes=174, training=False)
    # Rebuild the stratified selection from the full manifest, before predictions.
    rng = random.Random(config['seed']); selected = []
    for label in range(174):
        candidates = sorted([i for i, row in enumerate(dataset.rows) if row['label'] == label], key=lambda i: dataset.rows[i]['id'])
        selected.extend(rng.sample(candidates, config['video_per_class']))
    assert selected == [row['index'] for row in plan['video_subset']]
    assert digest(root/'video_inputs.json') == report['videos']['inputs_sha256']
    inputs = json.loads((root/'video_inputs.json').read_text())
    assert len(inputs) == len(selected) == 348
    assert Counter(row['label'] for row in inputs) == {label: 2 for label in range(174)}
    for item, saved in zip(plan['video_subset'], inputs, strict=True):
        assert digest(dataset.rows[item['index']]['video']) == item['video_sha256']
        pixels, target, metadata = dataset.read(item['index'])
        metadata.update(label=target, pixels_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest())
        assert metadata == saved and target == item['label'] and metadata['id'] == item['id']
    assert digest(root/'video_logits.npz') == report['videos']['logits_sha256']
    with np.load(root/'video_logits.npz') as raw:
        target = np.array([row['label'] for row in inputs])
        assert np.array_equal(target, raw['targets'])
        full = raw['111'].argmax(-1) == target
        for label in config['masks']:
            value = raw[label]; assert value.shape == (348, 174) and np.isfinite(value).all()
            correct = value.argmax(-1) == target
            score = dict(correct=int(correct.sum()), total=348, accuracy=float(correct.mean()),
                gained_vs_full=int((correct & ~full).sum()), lost_vs_full=int((~correct & full).sum()))
            assert score == report['videos']['scores'][label]
    verification = dict(status='passed', source_sha256=digest(__file__), summary_sha256=digest(root/'summary.json'),
        point_prediction_files=files, independently_recounted_point_rows=len(points['rows']),
        video_inputs_regenerated=len(inputs), video_predictions_recounted=348*8,
        endpoint_features_and_point_predictions_exactly_match_prior_run=True,
        scope='Point metrics and video predictions recounted; original video inputs regenerated. Saved endpoint GPU features matched preceding run. GPU encoders, classifier and nonendpoint point matching not rerun here.')
    (root/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
