"""Prospective scene holdout for the three fixed-budget pilot exports.

Select prepared scenes from the existing validation split, excluding the
correspondence cache and earlier oracle scenes. Freeze the cohort before
evaluating any export. No training or best-checkpoint selection takes place.
"""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor

from kubric_data import load_points
from exploration.oracle_transport import digest
from exploration.readout_diagnostic import parameter_digest
from exploration.separated_encoder import SeparatedEncoder
from exploration.train_separated_encoder import evaluate


def select_cohort(rgb_root, excluded, count=32):
    ids = [f'{i:04d}' for i in range(5869)]
    random.Random(42).shuffle(ids)
    candidates = sorted(scene for scene in ids[:300] if scene not in excluded
                        and (rgb_root/scene/'metadata.json').is_file())
    random.Random(17).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError('Not enough prepared validation scenes outside the excluded cohorts')
    return sorted(candidates[:count])


def load_rgb_and_tracks(plan, processor):
    root = Path(plan['rgb_root']); config = plan['config']; data = {}; inputs = {}
    for scene in config['val']:
        folder = root/scene
        assert digest(folder/'metadata.json') == plan['metadata_sha256'][scene]
        metadata = json.loads((folder/'metadata.json').read_text())
        for name in ('images.zip', 'points.npz'):
            assert digest(folder/name) == metadata['files'][name]['sha256']
        with zipfile.ZipFile(folder/'images.zip') as archive:
            images = [Image.open(io.BytesIO(archive.read(f'{i:03d}.png'))).convert('RGB') for i in config['frames']]
        pixels = processor(images=images, return_tensors='pt').pixel_values
        points, occluded = load_points(folder/'points.npz')
        selected = np.random.default_rng(config['seed']+int(scene)).permutation(len(points))[:4096]
        points = points[selected][:, config['frames']]
        visible = ~occluded[selected][:, config['frames']] & np.isfinite(points).all(-1)
        visible &= (points[..., 0] >= 0) & (points[..., 0] <= metadata['native_width']-1)
        visible &= (points[..., 1] >= 0) & (points[..., 1] <= metadata['native_height']-1)
        points = points / np.array([metadata['native_width'], metadata['native_height']], dtype=np.float32)
        points = np.where(visible[..., None], points, 0)
        assert int(visible[:, 0].sum()) >= config['evaluation_queries']
        pixel_hash = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        data[scene] = dict(pixels=pixels, pixels_sha256=pixel_hash,
            points=torch.from_numpy(points.copy()), visible=torch.from_numpy(visible.copy()))
        inputs[scene] = dict(pixels_sha256=pixel_hash, track_indices=selected.tolist(),
            normalized_points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
            visible_sha256=hashlib.sha256(visible.tobytes()).hexdigest())
    return data, inputs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Execute an unstarted fixed cohort, not partial evaluation resume')
    args = parser.parse_args(); pilot = Path(args.pilot_run).resolve(); output = Path(args.output).resolve()
    prior = json.loads((pilot/'plan.json').read_text())
    probe = Path(prior['probe_run']); probe_plan = json.loads((probe/'plan.json').read_text())
    assert digest(probe/'plan.json') == prior['probe_plan_sha256']
    rgb_root = Path(probe_plan['rgb_root'])
    excluded = set(prior['config']['train']+prior['config']['val'])
    oracle = Path('results/oracle_transport_seed17/plan.json')
    oracle_plan = json.loads(oracle.read_text())
    # The diagnostic's actual scene list, rather than a hard-coded ID filter.
    oracle_scenes = oracle_plan['config']['scenes']
    excluded.update(oracle_scenes)
    sources = [Path(__file__), Path('exploration/separated_encoder.py'),
        Path('exploration/train_separated_encoder.py'), Path('exploration/continuous_matching.py'),
        Path('exploration/evaluate_continuous_matching.py'), Path('exploration/readout_diagnostic.py'),
        Path('exploration/oracle_transport.py'), Path('kubric_data.py'), Path('temporal_siglip.py'),
        Path('tapvid_metrics.py'), Path('pyproject.toml'), Path('uv.lock')]
    if args.resume:
        plan = json.loads((output/'plan.json').read_text())
        assert plan['pilot_plan_sha256'] == digest(pilot/'plan.json')
        assert plan['oracle_plan_sha256'] == digest(oracle)
        assert plan['excluded_scenes'] == sorted(excluded)
        assert not (output/'input_pixels.json').exists(), 'Evaluation already started'
        # Keep the frozen cohort even if data preparation added more scenes.
        for path, expected in plan['sources'].items(): assert digest(path) == expected
    else:
        cohort = select_cohort(rgb_root, excluded)
        config = dict(prior['config'], val=cohort, train=[], frames=list(range(0, 96, 4)))
        config['scope'] = '32 prospective scenes from the existing Kubric validation split, excluded from prior pilot and oracle scenes; one seed, no training'
        config['limits'] = 'Rendered RGB, same source dataset as training. Not unseen-domain real-video evaluation or an independent final test split. Always-visible correspondence readout.'
        plan = dict(pilot_run=str(pilot), pilot_plan_sha256=digest(pilot/'plan.json'),
            oracle_plan_sha256=digest(oracle), rgb_root=str(rgb_root), excluded_scenes=sorted(excluded),
            config=config, metadata_sha256={scene:digest(rgb_root/scene/'metadata.json') for scene in cohort},
            sources={str(path):digest(path) for path in sources})
        output.mkdir(parents=True, exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(dict(status='cohort_fixed', scenes=plan['config']['val'],
            excludes_prior_cohorts=True, output=str(output))), flush=True)
        return
    config = plan['config']
    assert all((pilot/('verification_'+arm+'.json')).is_file() for arm in config['arms'])
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.empty(1, device='cuda')
    processor = SiglipImageProcessor.from_pretrained(probe_plan['config']['model'],
        revision=probe_plan['config']['revision'], local_files_only=True)
    data, inputs = load_rgb_and_tracks(plan, processor)
    (output/'input_pixels.json').write_text(json.dumps(inputs, indent=2)+'\n')
    results = {}; started = time.monotonic()
    for arm in config['arms']:
        record = json.loads((pilot/(arm+'.json')).read_text())
        verification = json.loads((pilot/('verification_'+arm+'.json')).read_text())
        assert verification['status'] == 'passed' and verification['record_sha256'] == digest(pilot/(arm+'.json'))
        export = pilot/(arm+'_encoder.pt')
        assert digest(export) == record['exported_encoder_sha256']
        model = SeparatedEncoder.load_checkpoint(export, 'cuda'); before = parameter_digest(model)
        results[arm] = dict(metrics=evaluate(model, data, config, output/arm),
            exported_encoder_sha256=digest(export), pilot_record_sha256=digest(pilot/(arm+'.json')))
        assert parameter_digest(model) == before
        (output/(arm+'.json')).write_text(json.dumps(results[arm], indent=2)+'\n')
        print(json.dumps(dict(arm=arm, status='complete', scenes=len(config['val']))), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    report = dict(plan_sha256=digest(output/'plan.json'), results=results, elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
