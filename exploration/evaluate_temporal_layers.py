"""Exhaustive 3-layer inference intervention on one fixed SSv2 checkpoint.

Eight masks are fixed before results. Paired short-video classification and
rendered-scene point correspondence measure sensitivity of the *trained*
model; there is no fitting, best-mask adoption, or architecture superiority
claim. The classifier is the original full-model classifier in every mask.
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

from video_checkpoint import load_video_checkpoint
from video_data import RawVideoClips
from exploration.continuous_matching import continuous_match
from exploration.evaluate_continuous_matching import score_scene, aggregate
from exploration.oracle_transport import digest, sample
from exploration.readout_diagnostic import parameter_digest, select_queries
from exploration.temporal_layer_intervention import active_adapters, layer_masks


def video_subset(rows, seed, per_class, classes):
    rng = random.Random(seed)
    chosen = []
    for label in range(classes):
        candidates = sorted([i for i, row in enumerate(rows) if row['label'] == label],
                            key=lambda i: rows[i]['id'])
        if len(candidates) < per_class:
            raise ValueError(f'Too few videos in class {label}')
        chosen.extend(rng.sample(candidates, per_class))
    return chosen


def log(output, **record):
    with (output/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)


@torch.inference_mode()
def point_pass(model, processor, plan, prior, output):
    encoder = model.encoder
    probe = Path(plan['probe_run']); cache = Path(prior['cache']); root = Path(prior['rgb_root'])
    previous = json.loads((probe/'summary.json').read_text())
    receipt = json.loads((cache/'summary.json').read_text())
    assert digest(cache/'summary.json') == prior['cache_summary_sha256']
    rows = []; feature_files = {}; predictions = {}; states = {}; original_matches = {}
    for scene in plan['config']['val_scenes']:
        path = cache/(scene+'.pt'); assert digest(path) == receipt['scenes'][scene]['sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        selected = select_queries(data, scene, plan['config']['evaluation_queries'], plan['config']['seed'])
        truth, visible = data['points'][selected], data['visible'][selected]
        archive_path = root/scene/'images.zip'
        assert digest(archive_path) == prior['images'][scene]['images_sha256']
        with zipfile.ZipFile(archive_path) as archive:
            images = [Image.open(io.BytesIO(archive.read(f'{frame:03d}.png'))).convert('RGB') for frame in data['frames']]
        pixels = processor(images=images, return_tensors='pt').pixel_values
        old_features_path = probe/(scene+'_features.pt')
        assert digest(old_features_path) == previous['features_sha256'][scene]
        old_features = torch.load(old_features_path, map_location='cpu', weights_only=True)
        assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == old_features['pixels_sha256']
        features = {}; scene_states = {}
        for label, keys in plan['config']['masks'].items():
            with active_adapters(encoder, keys):
                state = None; current = []; sizes = []
                for frame, image in enumerate(pixels):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        value, state = encoder.step(image[None].cuda(), state)
                    current.append(value[0].float().cpu())
                    assert state.offset == frame+1 and set(state.layers) == set(keys)
                    assert all(len(native) == 4 for native in state.layers.values())
                    sizes.append(state.nbytes)
                assert len(set(sizes)) == 1
                features[label] = torch.stack(current)
                scene_states[label] = dict(frames=len(current), active_keys=keys, state_bytes=sizes[0])
        # Same native GPU forward as earlier: both endpoints must be exact.
        for label, condition in [('000', 'siglip_only'), ('111', 'mamba_full')]:
            assert torch.equal(features[label], old_features['features'][condition]['12']), (scene, label)
        original_matches[scene] = True; states[scene] = scene_states
        feature_path = output/(scene+'_features.pt')
        torch.save(dict(features=features, selected=selected, frames=data['frames'],
                        pixels_sha256=old_features['pixels_sha256']), feature_path)
        feature_files[scene] = digest(feature_path)
        for label, current in features.items():
            anchor = sample(current[0], truth[:, 0], plan['config']['grid'])
            coordinates = {name: [] for name in plan['config']['methods'] if name != 'static_query'}
            scores = {name: [] for name in coordinates}
            for frame in range(len(current)):
                values = continuous_match(anchor, current[frame], plan['config']['grid'], True,
                    plan['config']['topk'], plan['config']['refinement_steps'])
                for method, result in values.items():
                    coordinates[method].append(result['points'].numpy())
                    scores[method].append(result['scores'].numpy())
            arrays = {k: np.stack(v, 1) for k, v in coordinates.items()}
            arrays['static_query'] = np.broadcast_to(truth[:, :1].numpy(), truth.shape).copy()
            if label in ('000', '111'):
                condition = 'siglip_only' if label == '000' else 'mamba_full'
                with np.load(probe/(scene+'_'+condition+'_12.npz')) as old:
                    for method, values in arrays.items():
                        assert np.array_equal(values, old[method]), (scene, label, method)
            saved = dict(target_xy=truth.numpy(), visible=visible.numpy(), selected=selected.numpy(), **arrays)
            saved.update({k+'_score': np.stack(v, 1) for k, v in scores.items()})
            name = scene+'_'+label+'.npz'; np.savez_compressed(output/name, **saved)
            predictions[name] = digest(output/name)
            rows.extend(dict(row, mask=label) for row in score_scene(scene, 12, truth.numpy(), visible.numpy(), arrays))
        log(output, event='points', scene=scene)
        del features, current, anchor, pixels, images, old_features, data; gc.collect()
    summary = []
    for label in plan['config']['masks']:
        summary.extend(dict(row, mask=label) for row in aggregate([r for r in rows if r['mask']==label]))
    return dict(rows=rows, summary=summary, features_sha256=feature_files,
                predictions_sha256=predictions, states=states, endpoints_exactly_reproduced=original_matches)


@torch.inference_mode()
def video_pass(model, dataset, plan, output):
    records = []; logits = {label: [] for label in plan['config']['masks']}
    for number, item in enumerate(plan['video_subset']):
        pixels, target, metadata = dataset.read(item['index'])
        assert metadata['id'] == item['id'] and target == item['label']
        assert digest(dataset.rows[item['index']]['video']) == item['video_sha256']
        metadata.update(label=target, pixels_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest())
        batch = pixels[None].cuda()
        for label, keys in plan['config']['masks'].items():
            with active_adapters(model.encoder, keys), torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(batch)
            logits[label].append(result[0].float().cpu().numpy())
        records.append(metadata)
        if (number+1)%24 == 0 or number+1 == len(plan['video_subset']):
            log(output, event='videos', videos=number+1, total=len(plan['video_subset']))
    values = {label: np.stack(value) for label, value in logits.items()}
    targets = np.array([r['label'] for r in records], dtype=np.int64)
    np.savez_compressed(output/'video_logits.npz', targets=targets, **values)
    (output/'video_inputs.json').write_text(json.dumps(records, indent=2)+'\n')
    full = values['111'].argmax(-1) == targets
    scores = {}
    for label, value in values.items():
        correct = value.argmax(-1) == targets
        scores[label] = dict(correct=int(correct.sum()), total=len(targets), accuracy=float(correct.mean()),
                            gained_vs_full=int((correct & ~full).sum()), lost_vs_full=int((~correct & full).sum()))
    return dict(scores=scores, logits_sha256=digest(output/'video_logits.npz'),
                inputs_sha256=digest(output/'video_inputs.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    probe, output = Path(args.probe_run).resolve(), Path(args.output).resolve()
    prior = json.loads((probe/'plan.json').read_text())
    checkpoint_path = probe/'checkpoint.pt'
    assert digest(checkpoint_path) == prior['checkpoint']['checkpoint_sha256']
    if not args.plan_only:
        torch.empty(1, device='cuda')
    model, info = load_video_checkpoint(checkpoint_path, 'cpu')
    model.requires_grad_(False).eval()
    assert info == prior['checkpoint']
    assert tuple(model.encoder.adapters) == ('3', '7', '11')
    processor = SiglipImageProcessor.from_pretrained(info['config']['model'],
        revision=info['config']['revision'], local_files_only=True)
    dataset = RawVideoClips(info['config']['val'], processor, frames=8, classes=174, training=False)
    assert digest(dataset.path) == info['config']['val_sha256']
    chosen = video_subset(dataset.rows, 17, 2, 174)
    subset = [dict(index=i, id=dataset.rows[i]['id'], label=dataset.rows[i]['label'],
                   video_sha256=digest(dataset.rows[i]['video'])) for i in chosen]
    config = dict(seed=17, masks={label: list(keys) for label, keys in layer_masks(model.encoder.adapters).items()},
        val_scenes=prior['config']['val'], evaluation_queries=128, grid=14, topk=4,
        methods=prior['config']['methods'], refinement_steps=prior['config']['refinement_steps'],
        video_per_class=2, video_frames=8,
        readouts='Fixed original full-model SSv2 classifier; fixed continuous cosine point readout, no fitting',
        scope='All eight inference-time layer deletions of the same 4000-update model; not separately trained architectures',
        limits='Small stratified video subset and eight repeatedly inspected rendered scenes. Layer removal changes feature distribution. No best mask selected for production.',
        precision='Spatial FP32 TF32 off, temporal and video head BF16; point matching CPU FP32')
    sources = [Path(__file__), Path('exploration/temporal_layer_intervention.py'),
        Path('exploration/continuous_matching.py'), Path('exploration/evaluate_continuous_matching.py'),
        Path('exploration/oracle_transport.py'), Path('exploration/readout_diagnostic.py'),
        Path('temporal_siglip.py'), Path('temporal_readout.py'), Path('video_checkpoint.py'),
        Path('video_data.py'), Path('tapvid_metrics.py'), Path('pyproject.toml'), Path('uv.lock')]
    plan = dict(config=config, checkpoint=info, probe_run=str(probe),
        probe_plan_sha256=digest(probe/'plan.json'), probe_summary_sha256=digest(probe/'summary.json'),
        validation_manifest=str(dataset.path), validation_manifest_sha256=digest(dataset.path),
        video_subset=subset, sources={str(p): digest(p) for p in sources})
    if args.resume:
        if json.loads((output/'plan.json').read_text()) != plan or (output/'progress.jsonl').exists():
            raise ValueError('Changed fixed plan or already started')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(dict(step=info['step'], masks=config['masks'], videos=len(subset), scenes=len(config['val_scenes']))), flush=True)
        return
    before = parameter_digest(model); model.cuda(); started = time.monotonic()
    points = point_pass(model, processor, plan, prior, output)
    (output/'points.json').write_text(json.dumps(points, indent=2)+'\n')
    videos = video_pass(model, dataset, plan, output)
    assert parameter_digest(model) == before
    summary = dict(plan_sha256=digest(output/'plan.json'), checkpoint_sha256=digest(checkpoint_path),
        weights_unchanged=True, elapsed_seconds=time.monotonic()-started, points=points, videos=videos)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(event='complete', seconds=summary['elapsed_seconds'], video_scores=videos['scores'])), flush=True)


if __name__ == '__main__':
    main()
