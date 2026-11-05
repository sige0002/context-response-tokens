"""Fixed-budget SISO versus MIMO development pilot, with separate decode envs.

This is warm-start architecture exploration. The running 20k experiment and
its scheduled comparisons are unchanged. The pilot is not a SOTA comparison.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import SiglipVisionConfig, SiglipVisionModel

from exploration.evaluate_video_history import model_hash
from exploration.mimo_siglip import load_model as load_mimo, save_model as save_mimo
from exploration.temporal_density import paired, read, write
from exploration.train_transition_conditioning import (
    RecordedSubset, datasets, digest, evaluate, schedule, validation_subset)
from temporal_readout import VideoRecognizer
from temporal_siglip import TemporalSiglip
from train_temporal import atomic_save
from train_video import configure_reproducibility, learning_rate, parameter_groups
from video_checkpoint import file_hash, load_video_checkpoint


SOURCES = ['exploration/mimo_pilot.py','exploration/run_mimo_pilot.py',
    'exploration/test_mimo_pilot.py','exploration/mimo_siglip.py',
    'exploration/train_transition_conditioning.py','exploration/transition_conditioning.py',
    'exploration/temporal_density.py','exploration/evaluate_video_history.py',
    'temporal_siglip.py','temporal_readout.py','train_temporal.py','train_video.py',
    'video_checkpoint.py','video_data.py','pyproject.toml','uv.lock']
ARMS = ('siso','mimo')


def save_siso(model, path, readout):
    base = model.encoder
    atomic_save(dict(format='siso-video-pilot-v1', architecture=base.architecture,
        backbone_config=base.backbone.config.to_dict(),
        checkpoint_revision=getattr(base,'checkpoint_revision',None),readout=dict(readout),
        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),path)


def load_siso(path, device='cpu'):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['format']!='siso-video-pilot-v1': raise ValueError('Wrong SISO export format')
    base = TemporalSiglip(SiglipVisionModel(SiglipVisionConfig.from_dict(saved['backbone_config'])),
                          **saved['architecture'])
    base.checkpoint_revision=saved['checkpoint_revision'];r=saved['readout']
    model=VideoRecognizer(base,r['classes'],r['kind'],r['seed'])
    model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval()


def initial_model(plan, arm):
    if arm=='mimo': return load_mimo(plan['mimo_initial'])
    if arm=='siso': return load_video_checkpoint(plan['checkpoint'],'cpu')[0]
    raise ValueError('Unknown arm')


def prepare(output):
    if output.exists(): raise FileExistsError('Use a new pilot directory')
    torch.set_num_threads(2)
    migration=Path('results/mimo_trained_migration_cpu_v2')
    mp=read(migration/'plan.json');ms=read(migration/'summary.json')
    clip=Path('results/mimo_visual_clip_v2/summary.json')
    stream=Path('results/mimo_visual_stream_v2/summary.json')
    gradient=Path('results/mimo_gradient_reference_v1/summary.json')
    grad_verify=gradient.parent/'verification.json'
    if not ms['passed'] or ms['model_sha256']!=file_hash(migration/'mimo_model.pt'):
        raise ValueError('Migration export is unqualified or changed')
    if mp['sources']['exploration/mimo_siglip.py']!=file_hash('exploration/mimo_siglip.py'):
        raise ValueError('Qualified MIMO implementation changed')
    if not read(clip)['completed'] or not read(stream)['completed']:
        raise ValueError('Native image qualification incomplete')
    if not read(stream)['stages']['native_forward']['alternate_chunk_states_exact']:
        raise ValueError('Streaming partition qualification failed')
    if (not read(gradient)['completed'] or
        not all(v['within_diagnostic_thresholds'] for v in read(gradient)['results'].values()) or
        not read(grad_verify)['passed_recount']):
        raise ValueError('Gradient qualification failed')
    checkpoint=Path(mp['checkpoint']).resolve()
    model,info=load_video_checkpoint(checkpoint,'cpu')
    if info['step']!=11000 or info['checkpoint_sha256']!=mp['checkpoint_sha256']:
        raise ValueError('Expected the qualified fixed 11k source')
    siso_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    model=load_mimo(migration/'mimo_model.pt')
    mimo_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    train,val=datasets(info)
    old_history=Path('results/video_history_intervention_seed17/plan.json')
    old_conditioner=Path('results/transition_conditioning_step11000_v2/plan.json')
    excluded={r['id'] for r in read(old_history)['subset']}|{r['id'] for r in read(old_conditioner)['validation']}
    planned=schedule(train.rows,512,16,701)
    subset=validation_subset(val.rows,excluded,4,709)
    qualifications=[migration/'plan.json',migration/'summary.json',clip,stream,
                    clip.parent/'plan.json',stream.parent/'plan.json',gradient,grad_verify]
    plan=dict(checkpoint=str(checkpoint),checkpoint_info=info,
        mimo_initial=str((migration/'mimo_model.pt').resolve()),mimo_initial_sha256=ms['model_sha256'],
        arms=list(ARMS),steps=512,batch_size=4,accumulation=4,seed=701,frames=8,
        learning_rate=3e-5,warmup=32,weight_decay=1e-4,
        evaluation_steps=[0,512],evaluation_batch_size=4,validation=subset,
        validation_selection_seed=709,excluded_validation_ids=sorted(excluded),
        prior_cohort_hashes={str(p):file_hash(p) for p in (old_history,old_conditioner)},
        training_schedule_sha256=digest(planned),train_count=len(planned),
        readout=dict(classes=174,kind='attention',seed=43),
        trainable_parameters=dict(siso=siso_parameters,mimo=mimo_parameters),
        qualification_hashes={str(p):file_hash(p) for p in qualifications},
        environments={mode:read(path.parent/'plan.json')['versions']
                      for mode,path in [('train',clip),('stream',stream)]},
        sources={p:file_hash(p) for p in SOURCES},
        primary_endpoint='Final one-observation streaming top-1 on 696 class-balanced development videos.',
        promotion_rule='Final MIMO minus SISO >=1 percentage point, paired stratified-bootstrap 95% lower bound >0, and final gap larger than initial gap. This only motivates a larger comparison.',
        hypothesis='Whether rank-four native Mamba-3 improves temporal recognition over matched SISO continuation.',
        predecessor='results/ssv2_full_mamba_interleaved_seed42_rgbfix',
        limitations=['Fixed 11k warm start, one seed, 512 updates, 8192 unique training videos; no distillation.',
            'The 696 videos exclude two earlier diagnostic cohorts, but full validation aggregates were already inspected. This is development, not an untouched test.',
            'Equal updates, data, augmentations and optimizer settings; MIMO has extra parameters and computation, so this is not a matched-FLOP comparison.',
            'Numerical SISO/MIMO initialization differences are measured at step zero; identical initial predictions are not assumed.',
            'Frozen SigLIP2; temporal adapters and the common readout train. No best-checkpoint selection.',
            'Clip/training run in the locked root environment. Streaming runs in a separate TVM-FFI 0.1.10 overlay.',
            'May share the GPU with later baseline training; no speed or SOTA claim.',
            'Existing MIMO is not itself a novel research contribution.'])
    output.mkdir(parents=True)
    write(output/'plan.json',plan);write(output/'training_schedule.json',planned)
    return plan


def checked_plan(output):
    plan=read(output/'plan.json')
    for mapping in ('sources','qualification_hashes','prior_cohort_hashes'):
        for p,h in plan[mapping].items():
            if file_hash(p)!=h: raise ValueError('Pinned pilot input changed: '+p)
    if (file_hash(plan['checkpoint'])!=plan['checkpoint_info']['checkpoint_sha256'] or
        file_hash(plan['mimo_initial'])!=plan['mimo_initial_sha256'] or
        digest(read(output/'training_schedule.json'))!=plan['training_schedule_sha256']):
        raise ValueError('Pilot weights or schedule changed')
    return plan


def train_arm(output,arm):
    plan=checked_plan(output);folder=output/arm
    versions={p:importlib.metadata.version(p) for p in plan['environments']['train']}
    if versions!=plan['environments']['train']: raise ValueError('Training environment differs from qualification')
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(2);configure_reproducibility(plan['seed'])
    model=initial_model(plan,arm);before=model_hash(model)
    backbone_before=model_hash(model.encoder.backbone);readout_before=model_hash(model.readout)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=plan['trainable_parameters'][arm]:
        raise ValueError('Trainable parameter count changed')
    train,val=datasets(plan['checkpoint_info']);model.cuda()
    initial,initial_score=evaluate(model,val,plan['validation'])
    # Align stochastic training after potentially different construction/evaluation paths.
    configure_reproducibility(plan['seed'])
    optimizer=torch.optim.AdamW(parameter_groups(model,plan['weight_decay']),lr=plan['learning_rate'])
    loader=DataLoader(RecordedSubset(train,read(output/'training_schedule.json')),
        batch_size=plan['batch_size'],num_workers=2,multiprocessing_context='spawn',
        generator=torch.Generator().manual_seed(plan['seed']))
    metadata=hashlib.sha256();pixels=hashlib.sha256();steps=0;count=0;loss_sum=0.;started=time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for i,(images,labels,records) in enumerate(loader):
        for record in records: metadata.update((record+'\n').encode())
        pixels.update(images.contiguous().numpy().tobytes())
        model.train();images=images.cuda();labels=labels.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=nn.functional.cross_entropy(model(images).float(),labels)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
        (loss/plan['accumulation']).backward();count+=1;loss_sum+=float(loss.detach())
        if (i+1)%plan['accumulation']: continue
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        lr=learning_rate(steps,plan['steps'],plan['warmup'],plan['learning_rate'])
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
        if steps%32==0:
            record=dict(arm=arm,step=steps,loss=loss_sum/count,lr=lr,elapsed_seconds=time.monotonic()-started)
            with (output/'progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);count=0;loss_sum=0.
    if steps!=plan['steps'] or model_hash(model.encoder.backbone)!=backbone_before:
        raise ValueError('Incomplete budget or changed backbone')
    final,final_score=evaluate(model,val,plan['validation'])
    (save_mimo if arm=='mimo' else save_siso)(model,folder/'model.pt',plan['readout'])
    reloaded=(load_mimo if arm=='mimo' else load_siso)(folder/'model.pt','cuda')
    replay,replay_score=evaluate(reloaded,val,plan['validation'])
    if not np.array_equal(final,replay): raise ValueError('Export replay changed clip logits')
    targets=np.array([r['label'] for r in plan['validation']])
    np.savez_compressed(folder/'clip.npz',initial=initial,final=final,targets=targets)
    checked_plan(output)
    write(folder/'training.json',dict(plan_sha256=file_hash(output/'plan.json'),arm=arm,steps=steps,
        initial=initial_score,final=final_score,replay=replay_score,clip_export_replay_exact=True,
        initial_model_hash=before,initial_backbone_hash=backbone_before,initial_readout_hash=readout_before,
        final_model_hash=model_hash(model),frozen_backbone_unchanged=True,
        trainable_parameters=plan['trainable_parameters'][arm],
        training_metadata_sha256=metadata.hexdigest(),training_pixels_sha256=pixels.hexdigest(),
        model_sha256=file_hash(folder/'model.pt'),clip_sha256=file_hash(folder/'clip.npz'),versions=versions))


def stream_arm(output,arm):
    plan=checked_plan(output);folder=output/arm
    versions={p:importlib.metadata.version(p) for p in plan['environments']['stream']}
    if versions!=plan['environments']['stream']: raise ValueError('Decode environment differs from qualification')
    if (folder/'stream.npz').exists() or (folder/'streaming.json').exists():
        raise FileExistsError('Streaming stage already started; preserve evidence')
    trained=read(folder/'training.json')
    if trained['model_sha256']!=file_hash(folder/'model.pt'): raise ValueError('Final model changed')
    torch.set_num_threads(2);configure_reproducibility(plan['seed'])
    _,val=datasets(plan['checkpoint_info'])
    model=initial_model(plan,arm).cuda()
    initial,initial_score=evaluate(model,val,plan['validation'],streaming=True)
    del model;gc.collect();torch.cuda.empty_cache()
    model=(load_mimo if arm=='mimo' else load_siso)(folder/'model.pt','cuda')
    final,final_score=evaluate(model,val,plan['validation'],streaming=True)
    expected_hash=model_hash(model)
    del model;gc.collect();torch.cuda.empty_cache()
    model=(load_mimo if arm=='mimo' else load_siso)(folder/'model.pt','cuda')
    replay,replay_score=evaluate(model,val,plan['validation'],streaming=True)
    if not np.array_equal(final,replay) or model_hash(model)!=expected_hash:
        raise ValueError('Export replay changed streaming logits or weights')
    targets=np.array([r['label'] for r in plan['validation']])
    np.savez_compressed(folder/'stream.npz',initial=initial,final=final,targets=targets)
    checked_plan(output)
    write(folder/'streaming.json',dict(plan_sha256=file_hash(output/'plan.json'),arm=arm,
        initial=initial_score,final=final_score,replay=replay_score,stream_export_replay_exact=True,
        model_sha256=trained['model_sha256'],stream_sha256=file_hash(folder/'stream.npz'),versions=versions))


def aggregate(output):
    plan=checked_plan(output);records={};arrays={};targets=np.array([r['label'] for r in plan['validation']])
    input_signatures=[]
    for arm in ARMS:
        folder=output/arm;t=read(folder/'training.json');s=read(folder/'streaming.json')
        if t['versions']!=plan['environments']['train'] or s['versions']!=plan['environments']['stream']:
            raise ValueError('Recorded execution environments differ from plan')
        if t['steps']!=plan['steps'] or not t['frozen_backbone_unchanged']:
            raise ValueError('Budget or backbone mismatch')
        if t['model_sha256']!=s['model_sha256'] or t['model_sha256']!=file_hash(folder/'model.pt'):
            raise ValueError('Export mismatch')
        if not t['clip_export_replay_exact'] or not s['stream_export_replay_exact']:
            raise ValueError('Export replay incomplete')
        arrays[arm]={}
        for mode,record in [('clip',t),('stream',s)]:
            if (record['plan_sha256']!=file_hash(output/'plan.json') or
                    record[mode+'_sha256']!=file_hash(folder/(mode+'.npz'))):
                raise ValueError('Evaluation artifact changed')
            with np.load(folder/(mode+'.npz')) as data:
                if not np.array_equal(data['targets'],targets): raise ValueError('Targets changed')
                for stage in ('initial','final'):
                    logits=data[stage].copy();score=record[stage]
                    if logits.shape!=(len(targets),174) or not np.isfinite(logits).all():
                        raise ValueError('Invalid evaluation logits')
                    correct=int((logits.argmax(-1)==targets).sum())
                    if correct!=score['correct'] or score['total']!=len(targets) or score['accuracy']!=correct/len(targets):
                        raise ValueError('Accuracy recount differs')
                    arrays[arm][mode+'_'+stage]=logits
            for stage in ('initial','final','replay'):
                input_signatures.append(tuple(record[stage][k] for k in ('input_metadata_sha256','input_pixels_sha256')))
            if record['replay']!=record['final']: raise ValueError('Replay metadata/score differs')
        records[arm]=dict(training=t,streaming=s)
    if len(set(input_signatures))!=1: raise ValueError('Evaluation inputs differ')
    for key in ('training_metadata_sha256','training_pixels_sha256','initial_backbone_hash','initial_readout_hash'):
        if records['siso']['training'][key]!=records['mimo']['training'][key]:
            raise ValueError('Matched condition differs: '+key)
    final=paired(arrays['mimo']['stream_final'],arrays['siso']['stream_final'],targets)
    initial=paired(arrays['mimo']['stream_initial'],arrays['siso']['stream_initial'],targets)
    promising=(final['delta_pp']>=1 and final['bootstrap_95_pp'][0]>0 and final['delta_pp']>initial['delta_pp'])
    result=dict(plan_sha256=file_hash(output/'plan.json'),arms=records,
        primary_mimo_vs_siso=final,initial_mimo_vs_siso=initial,
        change_in_gap_pp=final['delta_pp']-initial['delta_pp'],promote_to_larger_comparison=promising,
        matched_inputs_verified=True,limitations=plan['limitations'])
    write(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('plan','train','stream','aggregate'),required=True)
    p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.mode=='plan':
        plan=prepare(a.output);print(json.dumps(dict(steps=plan['steps'],validation=len(plan['validation']),train=plan['train_count'],parameters=plan['trainable_parameters'])))
    elif a.mode=='aggregate': print(json.dumps(aggregate(a.output)['primary_mimo_vs_siso']))
    elif a.arm is None: p.error('--arm is required for train/stream')
    elif a.mode=='train': train_arm(a.output,a.arm)
    else: stream_arm(a.output,a.arm)
