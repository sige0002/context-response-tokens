"""Separate coordinate-readout learning from temporal-encoder learning.

This is a diagnostic of the shared point head, not a new visual encoder. All
features are frozen SigLIP 2 final features from rendered Kubric RGB. The offset
control explicitly supplies the known initial coordinate as an identity prior.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from transformers import SiglipVisionModel

from point_readout import PointReadout, PointQueries, point_tracking_loss
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from exploration.oracle_transport import digest


class QueryOffsetReadout(PointReadout):
    def predict(self, tokens, queries):
        result = super().predict(tokens, queries)
        # A zero coordinate logit produces zero offset, exactly. This retains
        # FP32 query positions and adds no trainable parameters or extra memory.
        result['points'] = (queries.xy.float()+2*(result['points']-.5)).clamp(0, 1)
        return result


def initialize_head(kind, seed=17, **kwargs):
    torch.manual_seed(seed)
    cls = PointReadout if kind == 'absolute' else QueryOffsetReadout
    if kind not in ('absolute', 'offset'):
        raise ValueError('Unknown head')
    head = cls(**kwargs)
    # Both heads receive identical weights. The coordinate output starts at
    # the center for absolute prediction and the query location for offsets.
    with torch.no_grad():
        head.output[-1].weight[:2].zero_()
        head.output[-1].bias[:2].zero_()
    return head


def parameter_digest(head):
    h = hashlib.sha256()
    for name, value in head.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def select_queries(data, scene, count, seed):
    candidates = torch.where(data['visible'][:, 0])[0]
    g = torch.Generator().manual_seed(seed+int(scene))
    chosen = candidates[torch.randperm(len(candidates), generator=g)[:count]]
    if len(chosen) != count:
        raise ValueError('Too few evaluation queries')
    return chosen


@torch.no_grad()
def evaluate(head, cache, scenes, task, config):
    head.eval()
    rows = []
    for scene in scenes:
        data = cache[scene]
        selected = select_queries(data, scene, config['evaluation_queries'], config['seed'])
        truth, visible = data['points'][selected], data['visible'][selected]
        feature = data['features']
        if task == 'identity':
            feature = feature[:1].expand_as(feature)
            truth = truth[:, :1].expand_as(truth)
            visible = torch.ones_like(visible)
        xy = truth[:, 0].cuda()[None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            queries = head.start_queries(feature[:1].cuda(), xy)
        predictions = []
        for start in range(1, len(feature), 4):
            tokens = feature[start:start+4].cuda()
            q = PointQueries(id(head), queries.descriptors.expand(len(tokens), -1, -1),
                             xy.expand(len(tokens), -1, -1))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = head.predict(tokens, q)['points']
            predictions.append(pred.float().cpu())
        prediction = torch.cat(predictions).transpose(0, 1)
        displacement = torch.linalg.vector_norm((truth[:, 1:]-truth[:, :1])*256, dim=-1)
        for method, value in (('model', prediction), ('static_query', truth[:, :1].expand_as(truth[:, 1:]))):
            distance = torch.linalg.vector_norm((value-truth[:, 1:])*256, dim=-1)
            for stratum, mask in dict(all_visible=visible[:, 1:],
                    moving_gt2=visible[:, 1:] & (displacement > 2),
                    moving_gt16=visible[:, 1:] & (displacement > 16)).items():
                errors = distance[mask]
                count = len(errors)
                rows.append(dict(scene=scene, method=method, stratum=stratum, points=count,
                    mean_error=float(errors.mean()) if count else None,
                    **{f'pck_{q}':float((errors < q).float().mean()) if count else None for q in (4, 8, 16)}))
    summary = []
    for method in ('model', 'static_query'):
        for stratum in ('all_visible', 'moving_gt2', 'moving_gt16'):
            selected = [r for r in rows if r['method'] == method and r['stratum'] == stratum and r['points']]
            summary.append(dict(method=method, stratum=stratum, scenes=len(selected),
                points=sum(r['points'] for r in selected),
                **{k:sum(r[k] for r in selected)/len(selected) if selected else None
                   for k in ('mean_error', 'pck_4', 'pck_8', 'pck_16')}))
    return dict(rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Execute unchanged plan, not partial-training resume')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    directory, output = Path(args.cache).resolve(), Path(args.output).resolve()
    cache_plan = json.loads((directory/'plan.json').read_text())
    cache_summary = json.loads((directory/'summary.json').read_text())
    if cache_summary['plan_sha256'] != digest(directory/'plan.json'):
        raise ValueError('Cache plan mismatch')
    config = dict(seed=17, tasks={'identity':256, 'motion':1024}, heads=['absolute', 'offset'],
        checkpoints=[0, 256, 1024], batch=1, clip_frames=4, queries=32, evaluation_queries=128,
        lr=1e-4, weight_decay=1e-4, train=cache_plan['config']['train'], val=cache_plan['config']['val'],
        backbone=DEFAULT_MODEL, revision=DEFAULT_REVISION, depth=12,
        feature_transform='Pretrained final post-layernorm, FP32; then frozen',
        head_initialization='Same seed and parameters; zero both coordinate output rows for both heads',
        scope='Diagnostic point-head training only. No Mamba, encoder updates, or distillation. Identity repeats the first selected image; motion uses cached original frames at stride 4.',
        limits='24 training / 8 previously inspected validation scenes, one seed, fixed short budgets. PCK uses known visibility only, not AJ. Concurrent GPU learning excludes latency comparisons.')
    plan = dict(config=config, cache=str(directory), cache_summary_sha256=digest(directory/'summary.json'),
        sources={str(p):digest(p) for p in (Path(__file__), Path('point_readout.py'),
            Path('temporal_siglip.py'), Path('exploration/oracle_transport.py'), Path('pyproject.toml'), Path('uv.lock'))})
    if args.resume:
        if json.loads((output/'plan.json').read_text()) != plan or (output/'training.jsonl').exists():
            raise ValueError('Changed plan or already-started training')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    if args.plan_only:
        print('Four diagnostic arms fixed before results', flush=True)
        return
    torch.empty(1, device='cuda')
    backbone = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION,
                                               local_files_only=True).eval()
    norm = backbone.vision_model.post_layernorm
    norm.requires_grad_(False)
    norm_hash = parameter_digest(norm)
    del backbone
    cache = {}
    for scene in config['train']+config['val']:
        path = directory/(scene+'.pt')
        if digest(path) != cache_summary['scenes'][scene]['sha256']:
            raise ValueError('Cache bytes changed')
        data = torch.load(path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            feature = norm(data['features']['12']).detach()
        cache[scene] = dict(features=feature, points=data['points'], visible=data['visible'])
        del data
    del norm
    started, results, initial_hash = time.monotonic(), {}, None
    for task, steps in config['tasks'].items():
        for kind in config['heads']:
            name = task+'_'+kind
            head = initialize_head(kind, config['seed']).cuda()
            actual = parameter_digest(head)
            if initial_hash is not None and actual != initial_hash:
                raise ValueError('Initial head parameters differ')
            initial_hash = actual
            optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            rng = random.Random(config['seed'])
            generator = torch.Generator().manual_seed(config['seed'])
            scores = {}
            for step in range(steps+1):
                if step in config['checkpoints']:
                    scores[str(step)] = evaluate(head, cache, config['val'], task, config)
                    print(json.dumps(dict(event='evaluate', arm=name, step=step,
                                          summary=scores[str(step)]['summary'])), flush=True)
                if step == steps:
                    break
                head.train()
                scene = rng.choice(config['train'])
                data = cache[scene]
                start = rng.randrange(len(data['features'])-config['clip_frames']+1)
                candidates = torch.where(data['visible'][:, start])[0]
                selected = candidates[torch.randperm(len(candidates), generator=generator)[:config['queries']]]
                if len(selected) != config['queries']:
                    raise ValueError('Too few training queries')
                end = start+config['clip_frames']
                features = data['features'][start:end]
                points = data['points'][selected, start:end]
                visible = data['visible'][selected, start:end]
                if task == 'identity':
                    features = features[:1].expand_as(features)
                    points = points[:, :1].expand_as(points)
                    visible = torch.ones_like(visible)
                features, points, occluded = features[None].cuda(), points[None].cuda(), (~visible)[None].cuda()
                query_frames = torch.zeros(1, len(selected), dtype=torch.long, device='cuda')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = head(features, points[:, :, 0], query_frames)
                    loss = point_tracking_loss(prediction, points, occluded, query_frames)
                if not torch.isfinite(loss['loss']):
                    raise ValueError('Nonfinite loss')
                loss['loss'].backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizer.step()
                if (step+1) % 64 == 0:
                    record = dict(arm=name, step=step+1, loss=float(loss['loss'].detach()),
                        position=float(loss['position'].detach()), visibility=float(loss['visibility'].detach()),
                        elapsed_seconds=time.monotonic()-started)
                    with (output/'training.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    print(json.dumps(record), flush=True)
            destination = output/(name+'.pt')
            torch.save(dict(kind=kind, task=task, config=head.config, state_dict=head.state_dict()), destination)
            results[name] = dict(scores=scores, initial_parameters_sha256=initial_hash,
                checkpoint_sha256=digest(destination), parameters=sum(p.numel() for p in head.parameters()))
            (output/(name+'.json')).write_text(json.dumps(results[name], indent=2)+'\n')
            del head, optimizer, prediction, loss
            gc.collect(); torch.cuda.empty_cache()
    report = dict(plan_sha256=digest(output/'plan.json'), post_layernorm_sha256=norm_hash,
                  results=results, elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
