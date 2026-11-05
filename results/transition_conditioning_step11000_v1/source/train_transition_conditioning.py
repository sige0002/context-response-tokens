"""Matched warm-start pilot: transition versus appearance conditioning.

Only SSv2 training labels train parameters. No distillation or validation
selection. Both arms use the same initial weights, sample order and budget.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor

from exploration.evaluate_video_history import model_hash
from exploration.temporal_density import paired, read, write
from exploration.transition_conditioning import attach, load_model, save_model
from train_temporal import check_split
from train_video import configure_reproducibility, learning_rate, parameter_groups
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips


SOURCES = [__file__, 'exploration/transition_conditioning.py', 'exploration/test_transition_conditioning.py',
    'temporal_siglip.py', 'temporal_readout.py', 'video_data.py', 'video_checkpoint.py',
    'train_temporal.py', 'train_video.py', 'pyproject.toml', 'uv.lock']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def schedule(rows, steps, batch, seed):
    if steps*batch > len(rows): raise ValueError('Pilot schedule must have no repeated training videos')
    indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))[:steps*batch].tolist()
    return [dict(index=i, id=rows[i]['id'], label=rows[i]['label']) for i in indices]


def validation_subset(rows, excluded, per_class, seed):
    selected = []
    for label in range(174):
        candidates = [dict(index=i, id=r['id'], label=label) for i,r in enumerate(rows)
            if r['label'] == label and r['id'] not in excluded]
        candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        if len(candidates) < per_class: raise ValueError('Insufficient validation population')
        selected.extend(candidates[:per_class])
    return sorted(selected, key=lambda r:r['index'])


def datasets(info):
    c = info['config']
    processor = SiglipImageProcessor.from_pretrained(c['model'], revision=c['revision'], local_files_only=True)
    train = RawVideoClips(c['train'], processor, 8, 174, training=True, seed=701)
    val = RawVideoClips(c['val'], processor, 8, 174)
    check_split(train, val)
    for split, data in [('train', train), ('val', val)]:
        if file_hash(data.path) != c[split+'_sha256']: raise ValueError('Changed data manifest')
    return train, val


def prepare(output):
    if output.exists(): raise FileExistsError('Use a new pilot directory')
    checkpoint = Path('results/temporal_density_step11000_v1/checkpoint.pt').resolve()
    model, info = load_video_checkpoint(checkpoint, 'cpu')
    if info['step'] != 11000: raise ValueError('Expected fixed 11000-update source')
    del model
    train, val = datasets(info)
    previous = read('results/video_history_intervention_seed17/plan.json')['subset']
    planned = schedule(train.rows, 512, 16, 701)
    validation = validation_subset(val.rows, {r['id'] for r in previous}, 4, 701)
    plan = dict(checkpoint=str(checkpoint), checkpoint_info=info, seed=701,
        conditioning=dict(hidden=128, seed=701), arms=['appearance', 'difference'],
        steps=512, batch_size=4, accumulation=4, learning_rate=3e-5, warmup=32, weight_decay=1e-4,
        frames=8, evaluation_steps=[0,512], optimizer='fresh AdamW in both arms',
        training_schedule_sha256=digest(planned), train_count=len(planned), validation=validation,
        readout=dict(classes=174, kind='attention', seed=43),
        sources={str(Path(p)):file_hash(p) for p in SOURCES},
        hypothesis='Whether one-observation feature differences improve native Mamba-3 input projections beyond a matched current-feature MLP.',
        trainable='Existing temporal adapters, new conditioning branches, common classification readout; SigLIP2 stays frozen.',
        limitations=['Warm-start local architecture pilot, not the original from-scratch final comparison.',
            'Temporal difference is established prior art; no novelty/SOTA claim.',
            'Four videos per class (696 total), excluding the earlier 348-video diagnostic; full validation has been inspected in aggregate.',
            'Fixed final step, no best checkpoint selection; one seed and one short training budget.',
            'Clip inference is the pilot accuracy endpoint. Streaming numerical checks do not replace full streaming evaluation.',
            'Same GPU as main training; no latency comparison.'])
    output.mkdir(parents=True); write(output/'plan.json', plan)
    write(output/'training_schedule.json', planned)
    return plan


def checked_plan(output):
    plan = read(output/'plan.json')
    for path, sha in plan['sources'].items():
        if file_hash(path) != sha: raise ValueError(f'Changed pilot source: {path}')
    if file_hash(plan['checkpoint']) != plan['checkpoint_info']['checkpoint_sha256']:
        raise ValueError('Changed source weights')
    if digest(read(output/'training_schedule.json')) != plan['training_schedule_sha256']:
        raise ValueError('Changed sample order')
    return plan


class RecordedSubset(Dataset):
    def __init__(self, dataset, subset): self.dataset, self.subset = dataset, subset
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        item = self.subset[i]
        values, label, metadata = self.dataset.read(item['index'])
        if metadata['id'] != item['id'] or label != item['label']: raise ValueError('Changed sample identity')
        return values, label, json.dumps(metadata, sort_keys=True, separators=(',', ':'))


@torch.inference_mode()
def evaluate(model, data, subset):
    model.eval(); logits = []; input_hash = hashlib.sha256(); pixels_hash = hashlib.sha256()
    for images, labels, records in DataLoader(RecordedSubset(data, subset), batch_size=4, num_workers=0):
        for r in records: input_hash.update((r+'\n').encode())
        pixels_hash.update(images.contiguous().numpy().tobytes())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values = model(images.cuda()).float().cpu().numpy()
        if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite evaluation')
        logits.append(values)
    logits = np.concatenate(logits)
    targets = np.array([r['label'] for r in subset])
    return logits, dict(correct=int((logits.argmax(-1)==targets).sum()), total=len(targets),
        accuracy=float((logits.argmax(-1)==targets).mean()), input_metadata_sha256=input_hash.hexdigest(),
        input_pixels_sha256=pixels_hash.hexdigest())


def qualify(output):
    plan = checked_plan(output); target = output/'qualification'
    if target.exists(): raise FileExistsError('Qualification already started; preserve evidence')
    target.mkdir()
    torch.set_num_threads(2); configure_reproducibility(plan['seed'])
    model, info = load_video_checkpoint(plan['checkpoint'], 'cpu')
    train, val = datasets(info); model.cuda().eval()
    sample = val.read(plan['validation'][0]['index'])[0][None].cuda()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): baseline = model(sample).float()
    # Attach on CPU so identical branch initialization is independent of GPU RNG.
    model.cpu(); attach(model, 'difference', **plan['conditioning']); model.cuda().eval()
    before_backbone = model_hash(model.encoder.backbone)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): identity = model(sample).float()
    identity_error = float((identity-baseline).abs().max())
    if identity_error != 0: raise ValueError('Zero branch does not preserve native model')
    opt = torch.optim.AdamW(parameter_groups(model, plan['weight_decay']), lr=plan['learning_rate'])
    rows = read(output/'training_schedule.json')[:4]
    images = torch.stack([train.read(r['index'])[0] for r in rows]).cuda()
    labels = torch.tensor([r['label'] for r in rows], device='cuda')
    losses = []
    for _ in range(3):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16): loss = nn.functional.cross_entropy(model(images).float(), labels)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
        losses.append(float(loss.detach()))
    if model_hash(model.encoder.backbone) != before_backbone: raise ValueError('Frozen SigLIP2 changed')
    learned = {k:float(p.branch[-1].weight.abs().sum()) for k,p in model.encoder.projections.items()}
    if not all(v > 0 for v in learned.values()): raise ValueError('A conditioning branch did not train')
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        features = model.encoder(sample); altered = sample.clone(); altered[:,4:] = sample[:,4:].flip(1)
        causal = model.encoder(altered)
        causal_error = float((causal[:,:4]-features[:,:4]).abs().max())
        if causal_error != 0: raise ValueError('Future observations changed earlier features')
        state = None; chunks = []
        for a,b in [(0,3),(3,4),(4,8)]:
            feature,state = model.encoder.encode_chunk(sample[:,a:b],state); chunks.append(feature)
        streamed = torch.cat(chunks,1)
        stream_error = float((streamed-features).abs().max())
        # Same BF16 native recurrence as existing baseline; no bitwise chunk guarantee.
        if not torch.allclose(streamed,features,atol=.125,rtol=.005):
            raise ValueError(f'Chunk equivalence exceeds preset numerical tolerance: {stream_error}')
        expected = model(sample).float().cpu()
    saved = target/'model.pt'; save_model(model,saved,plan['readout'])
    reloaded = load_model(saved,'cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): actual = reloaded(sample).float().cpu()
    reload_error = float((expected-actual).abs().max())
    if reload_error != 0: raise ValueError('Export/reload changed the model')
    receipt = dict(passed=True,plan_sha256=file_hash(output/'plan.json'), sources=plan['sources'],
        native_initial_logit_max_abs=identity_error, losses=losses, branch_weight_absolute_sums=learned,
        frozen_backbone_unchanged=True, future_prefix_max_abs=causal_error,
        clip_chunk_feature_max_abs=stream_error, clip_chunk_atol=.125, clip_chunk_rtol=.005,
        state_bytes=state.nbytes, added_previous_input_bytes=state.nbytes-state.base_state.nbytes,
        export_reload_logit_max_abs=reload_error,
        scope='One preset validation clip for numerical checks; four training clips for three training updates. No accuracy claim.')
    write(target/'verification.json',receipt); print(json.dumps(receipt),flush=True)


def run(output):
    plan = checked_plan(output); qualification = read(output/'qualification/verification.json')
    if not qualification['passed'] or qualification['plan_sha256'] != file_hash(output/'plan.json'):
        raise ValueError('Matching native qualification is required')
    target = output/'training'
    if target.exists(): raise FileExistsError('Pilot already started; do not silently restart')
    target.mkdir(); torch.set_num_threads(2)
    planned = read(output/'training_schedule.json'); all_results = {}; first_initial = None
    targets = np.array([r['label'] for r in plan['validation']]); final_logits = {}
    for arm in plan['arms']:
        configure_reproducibility(plan['seed'])
        model,info = load_video_checkpoint(plan['checkpoint'],'cpu'); train,val = datasets(info)
        attach(model,arm,**plan['conditioning']); before = model_hash(model)
        if first_initial is None: first_initial = before
        if first_initial != before: raise ValueError('Arms do not start with identical parameter values')
        backbone_before = model_hash(model.encoder.backbone)
        model.cuda(); folder=target/arm;folder.mkdir()
        initial,initial_score = evaluate(model,val,plan['validation'])
        if all_results and not np.array_equal(initial,np.load(target/plan['arms'][0]/'predictions.npz')['initial']):
            raise ValueError('Zero-initialized arms differ before training')
        write(folder/'initial.json',initial_score)
        optimizer = torch.optim.AdamW(parameter_groups(model,plan['weight_decay']),lr=plan['learning_rate'])
        loader = DataLoader(RecordedSubset(train,planned),batch_size=plan['batch_size'],num_workers=2,
            multiprocessing_context='spawn',generator=torch.Generator().manual_seed(plan['seed']))
        running = hashlib.sha256(); pixels_hash = hashlib.sha256()
        steps=0; started=time.monotonic(); interval_loss=0.; count=0
        optimizer.zero_grad(set_to_none=True)
        for batch_index,(images,labels,records) in enumerate(loader):
            for record in records: running.update((record+'\n').encode())
            pixels_hash.update(images.contiguous().numpy().tobytes())
            model.train();images=images.cuda();labels=labels.cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=nn.functional.cross_entropy(model(images).float(),labels)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training')
            (loss/plan['accumulation']).backward();interval_loss+=float(loss.detach());count+=1
            if (batch_index+1)%plan['accumulation']: continue
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            lr=learning_rate(steps,plan['steps'],plan['warmup'],plan['learning_rate'])
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            if steps%32==0:
                record=dict(arm=arm,step=steps,loss=interval_loss/count,lr=lr,elapsed_seconds=time.monotonic()-started)
                with (output/'progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True);interval_loss=count=0
        if steps!=plan['steps']: raise ValueError('Incomplete update budget')
        if model_hash(model.encoder.backbone)!=backbone_before: raise ValueError('Frozen SigLIP2 changed')
        final,final_score=evaluate(model,val,plan['validation']);final_logits[arm]=final
        np.savez_compressed(folder/'predictions.npz',targets=targets,initial=initial,final=final)
        model_path=folder/'model.pt';save_model(model,model_path,plan['readout'])
        reloaded=load_model(model_path,'cuda');replayed,replay_score=evaluate(reloaded,val,plan['validation'])
        if not np.array_equal(final,replayed): raise ValueError('Full exported-model replay mismatch')
        result=dict(initial=initial_score,final=final_score,vs_initial=paired(final,initial,targets),
            initial_model_sha256=before,final_model_sha256=model_hash(model),frozen_backbone_unchanged=True,
            training_input_metadata_sha256=running.hexdigest(),training_pixels_sha256=pixels_hash.hexdigest(),steps=steps,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            checkpoint_sha256=file_hash(model_path),predictions_sha256=file_hash(folder/'predictions.npz'),
            all_validation_export_replay_max_abs=0.,replay_score=replay_score)
        write(folder/'summary.json',result);all_results[arm]=result
        for field in ('training_input_metadata_sha256','training_pixels_sha256'):
            if len({r[field] for r in all_results.values()})!=1:
                raise ValueError(f'Training inputs differ across arms: {field}')
        del model,reloaded,optimizer,loss,images,labels;gc.collect();torch.cuda.empty_cache()
    checked_plan(output)
    summary=dict(plan_sha256=file_hash(output/'plan.json'),arms=all_results,
        difference_vs_appearance=paired(final_logits['difference'],final_logits['appearance'],targets),
        limitations=plan['limitations'])
    write(output/'summary.json',summary);print(json.dumps(summary['difference_vs_appearance']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan-only',action='store_true');mode.add_argument('--qualify',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args();output=Path(args.output)
    if args.plan_only:
        plan=prepare(output);print(json.dumps(dict(steps=plan['steps'],arms=plan['arms'],train=plan['train_count'],validation=len(plan['validation']))))
    elif args.qualify: qualify(output)
    else: run(output)


if __name__=='__main__': main()
