"""Matched RGB training of the output factorization and temporal adapters.

Three arms distinguish a protected early feature path from a plain final-layer
projection, and distinguish memory from early features alone. The original
SigLIP 2 weights stay frozen, but Mamba weights are trained in the two temporal
arms. Correspondence labels supervise the encoder directly; no teacher model.
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

from temporal_siglip import TemporalSiglip
from video_checkpoint import load_video_checkpoint
from exploration.continuous_matching import continuous_match
from exploration.evaluate_continuous_matching import score_scene, aggregate
from exploration.oracle_transport import digest, sample
from exploration.readout_diagnostic import parameter_digest, select_queries
from exploration.separated_encoder import SeparatedEncoder
from exploration.separated_loss import correspondence_loss


def load_cache(prior, include_pixels):
    directory = Path(prior['cache']); receipt = json.loads((directory/'summary.json').read_text())
    assert digest(directory/'summary.json') == prior['cache_summary_sha256']
    original = json.loads((directory/'plan.json').read_text())
    config = original['config']; cache = {}
    processor = SiglipImageProcessor.from_pretrained(prior['config']['model'],
        revision=prior['config']['revision'], local_files_only=True) if include_pixels else None
    for scene in config['train']+config['val']:
        path = directory/(scene+'.pt'); assert digest(path) == receipt['scenes'][scene]['sha256']
        data = torch.load(path, map_location='cpu', weights_only=True); data.pop('features')
        if include_pixels:
            folder = Path(prior['rgb_root'])/scene
            metadata = json.loads((folder/'metadata.json').read_text())
            assert digest(folder/'metadata.json') == receipt['scenes'][scene]['metadata_sha256']
            assert digest(folder/'images.zip') == metadata['files']['images.zip']['sha256']
            with zipfile.ZipFile(folder/'images.zip') as archive:
                rgb = [Image.open(io.BytesIO(archive.read(f'{frame:03d}.png'))).convert('RGB') for frame in data['frames']]
            data['pixels'] = processor(images=rgb, return_tensors='pt').pixel_values
            data['pixels_sha256'] = hashlib.sha256(data['pixels'].numpy().tobytes()).hexdigest()
        cache[scene] = data
    return cache, config


def make_schedule(cache, config):
    rng = random.Random(config['seed']); generator = torch.Generator().manual_seed(config['seed'])
    schedule = []
    for _ in range(config['steps']):
        scene = rng.choice(config['train']); data = cache[scene]
        candidates = torch.where(data['visible'][:, 0])[0]
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:config['queries']]]
        assert len(chosen) == config['queries']
        frames = [0]+sorted(rng.sample(range(1, len(data['frames'])), config['clip_frames']-1))
        schedule.append(dict(scene=scene, queries=chosen.tolist(), frames=frames))
    return schedule


def build(probe, arm, config):
    recognizer, info = load_video_checkpoint(probe/'checkpoint.pt')
    base = recognizer.encoder; del recognizer
    if arm == 'separated_image':
        base = TemporalSiglip(base.backbone, kind='none').eval()
        base.checkpoint_revision = info['config']['revision']
    torch.manual_seed(config['seed'])
    model = SeparatedEncoder(base, observation_dim=config['observation_dim'],
        mode='final' if arm == 'final_memory' else 'separated', temporal_precision=config['temporal_precision'],
        execution=config['execution'])
    return model.cuda(), info


def event(output, **value):
    with (output/'training.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    print(json.dumps(value), flush=True)


@torch.no_grad()
def evaluate(model, cache, config, folder):
    model.eval(); folder.mkdir(parents=True)
    conditions = ['full'] if model.kind == 'none' else ['full', 'reset1']
    rows = []; predictions = {}; features_hash = {}; state_bytes = {}; invariance = {}
    for scene in config['val']:
        data = cache[scene]; chosen = select_queries(data, scene, config['evaluation_queries'], config['seed'])
        truth, visible = data['points'][chosen], data['visible'][chosen]
        features = {}; state_bytes[scene] = {}
        for condition in conditions:
            state = None; values = []; sizes = []
            for frame, image in enumerate(data['pixels']):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    current, state = model.step(image[None].cuda(), None if condition == 'reset1' else state)
                assert state.offset == (1 if condition == 'reset1' else frame+1)
                values.append(current[0].float().cpu()); sizes.append(state.nbytes)
            assert len(set(sizes)) == 1
            features[condition] = torch.stack(values); state_bytes[scene][condition] = sizes[-1]
        if 'reset1' in features:
            assert torch.equal(features['full'][0], features['reset1'][0])
            width = config['observation_dim']
            difference = float((features['full'][..., :width]-features['reset1'][..., :width]).abs().max())
            if model.config['mode'] == 'separated': assert difference == 0
            invariance[scene] = difference
        feature_path = folder/(scene+'_features.pt')
        torch.save(dict(features=features, selected=chosen, pixels_sha256=data['pixels_sha256']), feature_path)
        features_hash[scene] = digest(feature_path)
        for condition, current in features.items():
            anchor = sample(current[0], truth[:, 0], config['grid'])
            coordinates = {name: [] for name in config['methods'] if name != 'static_query'}
            scores = {name: [] for name in coordinates}
            for frame in range(len(current)):
                result = continuous_match(anchor, current[frame], config['grid'], False,
                    config['topk'], config['refinement_steps'])
                for method, value in result.items():
                    coordinates[method].append(value['points'].numpy()); scores[method].append(value['scores'].numpy())
            arrays = {key: np.stack(value, 1) for key, value in coordinates.items()}
            arrays['static_query'] = np.broadcast_to(truth[:, :1].numpy(), truth.shape).copy()
            saved = dict(target_xy=truth.numpy(), visible=visible.numpy(), selected=chosen.numpy(), **arrays)
            saved.update({key+'_score': np.stack(value, 1) for key, value in scores.items()})
            name = scene+'_'+condition+'.npz'; np.savez_compressed(folder/name, **saved)
            predictions[name] = digest(folder/name)
            # depth=0 denotes the exported, mixed-depth 768-wide encoder output.
            rows.extend(dict(row, condition=condition) for row in score_scene(scene, 0, truth.numpy(), visible.numpy(), arrays))
    summary = []
    for condition in conditions:
        summary.extend(dict(row, condition=condition) for row in aggregate([r for r in rows if r['condition']==condition]))
    return dict(rows=rows, summary=summary, predictions_sha256=predictions, features_sha256=features_hash,
        state_bytes=state_bytes, observation_full_reset_max_abs=invariance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Start an unchanged unexecuted plan, not partial-training resume')
    args = parser.parse_args(); probe = Path(args.probe_run).resolve(); output = Path(args.output).resolve()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    prior = json.loads((probe/'plan.json').read_text())
    assert digest(probe/'checkpoint.pt') == prior['checkpoint']['checkpoint_sha256']
    if not args.plan_only: torch.empty(1, device='cuda')
    cache, original = load_cache(prior, not args.plan_only)
    config = dict(seed=17, arms=['final_memory', 'separated_memory', 'separated_image'],
        steps=1024, checkpoints=[0, 256, 1024], queries=64, clip_frames=8, evaluation_queries=128,
        train=original['train'], val=original['val'], observation_dim=384, output_dim=768, grid=14,
        adapter_lr=1e-4, projection_lr=1e-3, weight_decay=1e-4, gradient_clip=1.,
        temporal_precision='bfloat16', projection_precision='float32', execution='framewise',
        methods=prior['config']['methods'], topk=4, refinement_steps=prior['config']['refinement_steps'],
        loss='Visible correspondence CE on bilinear target cells, fixed temperature20; no feature teacher or visibility head',
        train_scope='Update native Mamba adapters and output projections from RGB; original SigLIP 2 backbone frozen',
        schedule='Frame0 query plus seven randomly selected later observations, shared across all arms',
        evaluation='Full and reset1 streaming, 24 observations at original stride4. Fixed L2 cosine continuous readout without joint LayerNorm; always-visible prediction',
        scope='Pilot of encoder topology, one seed, fixed budget; no architecture novelty or superiority claim',
        limits='24 training / 8 repeatedly inspected rendered scenes. New output projections require downstream classification retraining; classification preservation not tested. Not the formal real-video benchmark.')
    assert not set(config['train']) & set(config['val'])
    schedule = make_schedule(cache, config); schedule_text = json.dumps(schedule, indent=2)+'\n'
    sources = [Path(__file__), Path('exploration/separated_encoder.py'), Path('exploration/separated_loss.py'),
        Path('exploration/continuous_matching.py'), Path('exploration/evaluate_continuous_matching.py'),
        Path('exploration/readout_diagnostic.py'), Path('exploration/oracle_transport.py'), Path('temporal_siglip.py'),
        Path('video_checkpoint.py'), Path('temporal_readout.py'), Path('tapvid_metrics.py'), Path('pyproject.toml'), Path('uv.lock')]
    plan = dict(config=config, probe_run=str(probe), probe_plan_sha256=digest(probe/'plan.json'),
        source_checkpoint_sha256=digest(probe/'checkpoint.pt'), original_cache=prior['cache'],
        original_cache_summary_sha256=prior['cache_summary_sha256'],
        schedule_sha256=hashlib.sha256(schedule_text.encode()).hexdigest(), sources={str(p): digest(p) for p in sources})
    if args.resume:
        if json.loads((output/'plan.json').read_text()) != plan or (output/'training.jsonl').exists():
            raise ValueError('Changed plan or already-started pilot')
        assert digest(output/'schedule.json') == plan['schedule_sha256']
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        (output/'schedule.json').write_text(schedule_text)
    if args.plan_only:
        print(json.dumps(dict(arms=config['arms'], updates_per_arm=1024, source_step=prior['checkpoint']['step'], output_tokens=[196, 768])), flush=True)
        return
    (output/'input_pixels.json').write_text(json.dumps({scene: data['pixels_sha256'] for scene, data in cache.items()}, indent=2)+'\n')
    projection_initial = None; results = {}; started = time.monotonic()
    for arm in config['arms']:
        model, provenance = build(probe, arm, config)
        assert provenance == prior['checkpoint']
        assert parameter_digest(model.backbone) == prior['frozen_backbone_sha256']
        initial = parameter_digest(model.projection)
        if projection_initial is not None: assert initial == projection_initial
        projection_initial = initial
        initial_adapters = parameter_digest(model.encoder.adapters)
        optimizer = torch.optim.AdamW([
            dict(params=list(model.projection.parameters()), lr=config['projection_lr']),
            dict(params=list(model.encoder.adapters.parameters()), lr=config['adapter_lr'])], weight_decay=config['weight_decay'])
        scores = {}
        for step in range(config['steps']+1):
            if step in config['checkpoints']:
                folder = output/arm/str(step)
                scores[str(step)] = evaluate(model, cache, config, folder)
                checkpoint = folder/'trainable.pt'
                torch.save(dict(config=model.config, architecture=model.encoder.architecture,
                    projection=model.projection.state_dict(), adapters=model.encoder.adapters.state_dict(),
                    source_checkpoint_sha256=plan['source_checkpoint_sha256'], step=step), checkpoint)
                scores[str(step)]['checkpoint_sha256'] = digest(checkpoint)
                (folder/'metrics.json').write_text(json.dumps(scores[str(step)], indent=2)+'\n')
                event(output, event='evaluation', arm=arm, step=step,
                    summary=[r for r in scores[str(step)]['summary'] if r['method']=='field_top4' and r['task']=='motion'])
            if step == config['steps']: break
            item = schedule[step]; data = cache[item['scene']]
            chosen = torch.tensor(item['queries']); frames = item['frames']
            pixels = data['pixels'][frames][None].cuda()
            truth = data['points'][chosen][:, frames].cuda(); visible = data['visible'][chosen][:, frames].cuda()
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = correspondence_loss(model(pixels), truth, visible, config['grid'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite correspondence loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%64 == 0:
                event(output, event='train', arm=arm, step=step+1, loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started)
        assert parameter_digest(model.backbone) == prior['frozen_backbone_sha256']
        assert parameter_digest(model.projection) != initial
        if model.kind != 'none': assert parameter_digest(model.encoder.adapters) != initial_adapters
        model.save_checkpoint(output/(arm+'_encoder.pt'))
        results[arm] = dict(scores=scores, projection_initial_sha256=initial,
            initial_adapters_sha256=initial_adapters, final_adapters_sha256=parameter_digest(model.encoder.adapters),
            frozen_backbone_unchanged=True, exported_encoder_sha256=digest(output/(arm+'_encoder.pt')),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        (output/(arm+'.json')).write_text(json.dumps(results[arm], indent=2)+'\n')
        del model, optimizer, loss; gc.collect(); torch.cuda.empty_cache()
    report = dict(plan_sha256=digest(output/'plan.json'), results=results, elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    event(output, event='complete', seconds=report['elapsed_seconds'])


if __name__ == '__main__':
    main()
