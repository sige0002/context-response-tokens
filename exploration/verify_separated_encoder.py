"""Independently recount completed pilot arms and verify portable exports.

Final CPU matching is rerun on all saved feature tensors. This tool does not
rerun GPU encoders; exported backbone/adapter/projection tensors are compared
directly with the pinned source and final training checkpoint.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from exploration.continuous_matching import continuous_match
from exploration.oracle_transport import digest, sample
from exploration.readout_diagnostic import parameter_digest
from exploration.separated_encoder import SeparatedProjection


def recount(row, data):
    truth, visible = data['target_xy'], data['visible']
    squared = (((data[row['method']]-truth)*256)**2).sum(-1)
    if row['task'] == 'identity':
        errors = np.sqrt(squared[:, 0])
    else:
        movement = np.sqrt((((truth[:, 1:]-truth[:, :1])*256)**2).sum(-1))
        mask = visible[:, 1:].copy()
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--arm', choices=['final_memory', 'separated_memory', 'separated_image'])
    args = parser.parse_args(); root = Path(args.run)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    plan = json.loads((root/'plan.json').read_text()); config = plan['config']
    arms = [args.arm] if args.arm else config['arms']
    assert all((root/(arm+'.json')).exists() for arm in arms), 'Requested arms must have completed'
    for source, expected in plan['sources'].items():
        assert digest(source) == expected, source
    probe = Path(plan['probe_run']); prior = json.loads((probe/'plan.json').read_text())
    assert digest(probe/'plan.json') == plan['probe_plan_sha256']
    assert digest(probe/'checkpoint.pt') == plan['source_checkpoint_sha256']
    original = Path(plan['original_cache']); receipt = json.loads((original/'summary.json').read_text())
    assert digest(original/'summary.json') == plan['original_cache_summary_sha256']
    cache = {}
    for scene in config['train']+config['val']:
        path = original/(scene+'.pt'); assert digest(path) == receipt['scenes'][scene]['sha256']
        values = torch.load(path, map_location='cpu', weights_only=True)
        cache[scene] = {key: values[key] for key in ('points', 'visible', 'frames')}
    del values
    rng = random.Random(config['seed']); generator = torch.Generator().manual_seed(config['seed'])
    regenerated = []
    for _ in range(config['steps']):
        scene = rng.choice(config['train']); data = cache[scene]
        candidates = torch.where(data['visible'][:, 0])[0]
        selected = candidates[torch.randperm(len(candidates), generator=generator)[:config['queries']]]
        frames = [0]+sorted(rng.sample(range(1, len(data['frames'])), config['clip_frames']-1))
        regenerated.append(dict(scene=scene, queries=selected.tolist(), frames=frames))
    assert digest(root/'schedule.json') == plan['schedule_sha256']
    assert regenerated == json.loads((root/'schedule.json').read_text())
    source = torch.load(probe/'checkpoint.pt', map_location='cpu', weights_only=True)
    assert source['step'] == 4000 and source['config'] == prior['checkpoint']['config']
    original_weights = source['model']; del source
    input_hashes = json.loads((root/'input_pixels.json').read_text())
    torch.manual_seed(config['seed'])
    initialization = SeparatedProjection(768, config['observation_dim'])
    initial_hash = parameter_digest(initialization)
    for arm in arms:
        record = json.loads((root/(arm+'.json')).read_text())
        assert record['projection_initial_sha256'] == initial_hash
        assert digest(root/(arm+'_encoder.pt')) == record['exported_encoder_sha256']
        exported = torch.load(root/(arm+'_encoder.pt'), map_location='cpu', weights_only=True)
        assert exported['format'] == 'separated_encoder_v1'
        assert exported['output_config']['mode'] == ('final' if arm == 'final_memory' else 'separated')
        assert exported['output_config']['execution'] == 'framewise'
        for name, value in original_weights.items():
            if name.startswith('encoder.backbone.'):
                assert torch.equal(value, exported['state_dict'][name]), name
        files = calls = rows = 0
        for step, score in record['scores'].items():
            folder = root/arm/step
            assert json.loads((folder/'metrics.json').read_text()) == score
            assert digest(folder/'trainable.pt') == score['checkpoint_sha256']
            trained = torch.load(folder/'trainable.pt', map_location='cpu', weights_only=True)
            assert trained['source_checkpoint_sha256'] == plan['source_checkpoint_sha256'] and trained['step'] == int(step)
            assert trained['config'] == exported['output_config']
            assert trained['architecture'] == exported['architecture']
            assert sum(value.numel() for section in ('projection', 'adapters') for value in trained[section].values()) == record['trainable_parameters']
            if int(step) == 0:
                for key, value in initialization.state_dict().items():
                    assert torch.equal(value, trained['projection'][key])
                if arm != 'separated_image':
                    for name, value in trained['adapters'].items():
                        assert torch.equal(value, original_weights['encoder.adapters.'+name])
            if int(step) == config['steps']:
                for prefix in ('projection', 'adapters'):
                    export_prefix = 'projection.' if prefix == 'projection' else 'encoder.adapters.'
                    for name, value in trained[prefix].items():
                        assert torch.equal(value, exported['state_dict'][export_prefix+name])
                assert any(not torch.equal(value, initialization.state_dict()[name]) for name, value in trained['projection'].items())
                if arm != 'separated_image':
                    assert any(not torch.equal(value, original_weights['encoder.adapters.'+name]) for name, value in trained['adapters'].items())
            by_key = {}
            for scene in config['val']:
                original = cache[scene]
                candidates = torch.where(original['visible'][:, 0])[0]
                generator = torch.Generator().manual_seed(config['seed']+int(scene))
                selected = candidates[torch.randperm(len(candidates), generator=generator)[:config['evaluation_queries']]]
                truth = original['points'][selected].numpy(); visible = original['visible'][selected].numpy()
                feature_path = folder/(scene+'_features.pt')
                assert digest(feature_path) == score['features_sha256'][scene]
                payload = torch.load(feature_path, map_location='cpu', weights_only=True)
                assert torch.equal(payload['selected'], selected) and payload['pixels_sha256'] == input_hashes[scene]
                features = payload['features']
                assert set(features) == ({'full'} if arm == 'separated_image' else {'full', 'reset1'})
                if arm != 'separated_image':
                    assert torch.equal(features['full'][0], features['reset1'][0])
                    difference = float((features['full'][..., :384]-features['reset1'][..., :384]).abs().max())
                    assert difference == score['observation_full_reset_max_abs'][scene]
                    if arm == 'separated_memory': assert difference == 0
                for condition, current in features.items():
                    assert current.shape == (24, 196, 768) and torch.isfinite(current).all()
                    path = folder/(scene+'_'+condition+'.npz')
                    assert digest(path) == score['predictions_sha256'][path.name]
                    with np.load(path) as raw: data = {key: raw[key] for key in raw.files}
                    assert np.array_equal(data['selected'], selected.numpy())
                    assert np.array_equal(data['target_xy'], truth) and np.array_equal(data['visible'], visible)
                    assert np.array_equal(data['static_query'], np.broadcast_to(truth[:, :1], truth.shape))
                    if int(step) == config['steps']:
                        anchor = sample(current[0], torch.from_numpy(truth[:, 0]), config['grid'])
                        for frame in range(len(current)):
                            repeated = continuous_match(anchor, current[frame], config['grid'], False,
                                config['topk'], config['refinement_steps'])
                            for method, value in repeated.items():
                                assert np.array_equal(value['points'].numpy(), data[method][:, frame])
                                assert np.array_equal(value['scores'].numpy(), data[method+'_score'][:, frame])
                            calls += 1
                    by_key[(scene, condition)] = data; files += 1
            for row in score['rows']:
                recount(row, by_key[(row['scene'], row['condition'])]); rows += 1
            for summary in score['summary']:
                keys = ('condition', 'depth', 'method', 'task', 'stratum')
                selected_rows = [r for r in score['rows'] if all(r[key] == summary[key] for key in keys) and r['points']]
                assert len(selected_rows) == summary['scenes'] and sum(r['points'] for r in selected_rows) == summary['points']
                for key in ('mean_error', 'pck_1', 'pck_2', 'pck_4', 'pck_8', 'pck_16'):
                    assert float(np.mean([r[key] for r in selected_rows])) == summary[key]
                if summary['full_sequence_metrics']:
                    for key, value in summary['full_sequence_metrics'].items():
                        assert float(np.mean([r['full_sequence_metrics'][key] for r in selected_rows])) == value
        report = dict(status='passed', arm=arm, source_sha256=digest(__file__), record_sha256=digest(root/(arm+'.json')),
            plan_sha256=digest(root/'plan.json'), prediction_files=files, recounted_rows=rows,
            final_cpu_matching_calls=calls, training_schedule_reproduced=True, initial_weights_match=True,
            exported_weights_match_final_training=True, frozen_backbone_exactly_matches_source=True,
            scope='All curves recounted, final CPU matching rerun, and complete export tensors verified. GPU encoder and RGB preprocessing not rerun here.')
        (root/('verification_'+arm+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
