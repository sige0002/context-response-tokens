"""State-only depth intervention on the archived 4000-update recognizer.

Reuses the already evaluated full-history and current-only predictions after
checking the same checkpoint and all RGB inputs. New inference erases one
adapter's prior states at a time, keeping every current-frame computation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from exploration.evaluate_video_history import model_hash,tensor_hash,summarize
from exploration.memory_layer_intervention import stream_with_memory
from train_video import configure_reproducibility
from video_checkpoint import file_hash,load_video_checkpoint
from video_data import RawVideoClips


KEPT = dict(clear_4=('7','11'),clear_8=('3','11'),clear_12=('3','7'))


def read(path): return json.loads(Path(path).read_bytes())


def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(output):
    plan=read(output/'plan.json'); summary=read(output/'summary.json')
    if (summary['plan_sha256']!=file_hash(output/'plan.json')
            or summary['logits_sha256']!=file_hash(output/'logits.npz')
            or summary['inputs_sha256']!=file_hash(output/'inputs.json')):
        raise ValueError('Diagnostic artifacts changed')
    prior=Path(plan['prior'])
    for name,digest in plan['prior_hashes'].items():
        if file_hash(prior/name)!=digest: raise ValueError('Original history diagnostic changed')
    for name,digest in plan['sources'].items():
        if file_hash(name)!=digest: raise ValueError('Diagnostic source changed')
    manifest=Path(plan['checkpoint_info']['config']['val'])
    if file_hash(manifest)!=plan['checkpoint_info']['config']['val_sha256']:
        raise ValueError('Original labels changed')
    labels={str(r['id']):r['label'] for r in (json.loads(s) for s in manifest.read_text().splitlines() if s)}
    inputs=read(output/'inputs.json')
    if inputs!=read(prior/'inputs.json'):
        raise ValueError('Current or historical RGB differs from original diagnostic')
    counts={}
    with np.load(output/'logits.npz') as saved, np.load(prior/'logits.npz') as old:
        ids=saved['ids'].tolist(); targets=saved['targets']
        if (len(ids)!=348 or len(set(ids))!=348 or ids!=[r['id'] for r in plan['subset']]
                or targets.tolist()!=[labels[i] for i in ids]
                or not np.array_equal(saved['full_stream'],old['full_stream'])
                or not np.array_equal(saved['reset_all'],old['current_only'])):
            raise ValueError('Population, targets or reused references differ')
        for name in ('full_stream','reset_all',*KEPT):
            values=saved[name]
            if values.shape!=(348,174) or not np.isfinite(values).all():
                raise ValueError('Invalid logits')
            pred=values.argmax(-1); count=int((pred==targets).sum())
            reported=summary['scores'][name]
            if (reported['correct']!=count or reported['original_label_top1']!=count/348
                    or reported['predictions']!=[dict(id=i,target=int(y),prediction=int(p)) for i,y,p in zip(ids,targets,pred)]):
                raise ValueError('Reported score differs from original labels and saved logits')
            counts[name]=count
    replays=summary['baseline_replays']
    if (len(replays)!=4 or {(r['index'],r['condition']) for r in replays} !=
            {(i,c) for i in plan['baseline_replay_indices'] for c in ('full_stream','reset_all')}
            or not summary['weights_unchanged'] or any(r['max_abs']!=0 for r in replays)):
        raise ValueError('Weight preservation or native baseline equivalence failed')
    if file_hash(plan['checkpoint'])!=plan['checkpoint_info']['checkpoint_sha256']:
        raise ValueError('Archived checkpoint changed')
    return dict(passed=True,summary_sha256=file_hash(output/'summary.json'),counts=counts,
        predictions_recounted=1740,scope='Original manifest recount and artifact checks; not a new GPU forward or a re-decode of every source video.')


@torch.inference_mode()
def run(args):
    prior,output=Path(args.prior).resolve(),Path(args.output).resolve()
    old_plan=read(prior/'plan.json'); old_summary=read(prior/'summary.json'); old_verification=read(prior/'verification.json')
    if not old_verification['passed'] or old_verification['summary_sha256']!=file_hash(prior/'summary.json'):
        raise ValueError('Original reference is not verified')
    for name,key in [('plan.json','plan_sha256'),('inputs.json','inputs_sha256'),('logits.npz','logits_sha256')]:
        if file_hash(prior/name)!=old_summary[key]: raise ValueError('Reference artifacts changed')
    model,info=load_video_checkpoint(old_plan['checkpoint'],'cpu')
    if info!=old_plan['checkpoint_info'] or info['step']!=4000:
        raise ValueError('Use the unchanged archived 4000-update checkpoint')
    model.requires_grad_(False).eval()
    if model.encoder.kind!='mamba' or tuple(model.encoder.adapters)!=('3','7','11'):
        raise ValueError('This diagnostic requires the three-layer interleaved Mamba')
    processor=SiglipImageProcessor.from_pretrained(info['config']['model'],revision=info['config']['revision'],local_files_only=True)
    dataset=RawVideoClips(info['config']['val'],processor,frames=8,classes=174)
    if file_hash(dataset.path)!=info['config']['val_sha256']: raise ValueError('Validation manifest changed')
    sources=['exploration/evaluate_memory_layers.py','exploration/memory_layer_intervention.py',
        'exploration/test_memory_layer_intervention.py','exploration/evaluate_video_history.py',
        'temporal_siglip.py','temporal_readout.py','video_checkpoint.py','video_data.py','train_video.py','pyproject.toml','uv.lock']
    plan=dict(prior=str(prior),prior_hashes={n:file_hash(prior/n) for n in ('plan.json','summary.json','verification.json','inputs.json','logits.npz')},
        checkpoint=old_plan['checkpoint'],checkpoint_info=info,subset=old_plan['subset'],
        retained_keys={k:list(v) for k,v in KEPT.items()},reused_baselines={'full_stream':'full_stream','reset_all':'current_only'},
        baseline_replay_indices=[0,347],sources={p:file_hash(p) for p in sources},
        scope='All adapters execute; only one selected layer state is erased before each observation. Absolute offsets preserved.',
        limitations=['Exploratory, previously inspected balanced 348-video validation subset, one interim checkpoint.',
            'Fixed video-trained classifier; interventions change its input feature distribution.',
            'Other layers can still carry history; this is not a standalone layer or an additive decomposition.',
            'Resetting all Mamba states equals current-only inference here; checked on the two predeclared native replay clips.',
            'No retraining, architecture superiority, long-term-memory or robotics-success conclusion.',
            'GPU may be shared with main training; no speed measurements are claimed.'])
    if args.resume:
        if read(output/'plan.json')!=plan or any((output/n).exists() for n in ('progress.jsonl','summary.json','logits.npz')):
            raise ValueError('Only an unstarted identical plan can be resumed')
    else:
        output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan)
    if args.plan_only: print('Fixed state-erasure plan saved; no GPU inference started',flush=True);return
    torch.set_num_threads(2);configure_reproducibility(42)
    before=model_hash(model)
    if before!=old_summary['model_state_sha256']: raise ValueError('Reference model weights differ')
    old_inputs=read(prior/'inputs.json')
    with np.load(prior/'logits.npz') as data:
        values={'full_stream':data['full_stream'].copy(),'reset_all':data['current_only'].copy()}
        targets=data['targets'].copy();ids=data['ids'].copy()
    if len(ids)!=348: raise ValueError('Expected original balanced subset')
    values.update({k:[] for k in KEPT});model.cuda();records=[];replays=[];state_records={}
    for number,item in enumerate(plan['subset']):
        pixels,target,metadata=dataset.read(item['index'])
        metadata.update(label=target,pixels_sha256=tensor_hash(pixels),current_pixels_sha256=tensor_hash(pixels[-1]))
        if (metadata!=old_inputs[number] or item['id']!=ids[number] or target!=targets[number]
                or file_hash(dataset.rows[item['index']]['video'])!=item['video_sha256']):
            raise ValueError('RGB or source label differs from reference')
        video=pixels[None].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if number in plan['baseline_replay_indices']:
                for name,keys in [('full_stream',tuple(model.encoder.adapters)),('reset_all',())]:
                    features,_=stream_with_memory(model.encoder,video,keys)
                    logits=model.readout(features)[0].float().cpu().numpy()
                    delta=float(np.abs(logits-values[name][number]).max())
                    replays.append(dict(index=number,id=item['id'],condition=name,max_abs=delta))
                    if delta!=0: raise ValueError('Native reference equivalence failed')
            for name,keys in KEPT.items():
                features,states=stream_with_memory(model.encoder,video,keys)
                logits=model.readout(features)[0].float().cpu().numpy()
                if not np.isfinite(logits).all(): raise FloatingPointError('Nonfinite recognition logits')
                if any(s['offset']!=i+1 or s['state_layer_keys']!=['3','7','11'] for i,s in enumerate(states)):
                    raise ValueError('Current adapter computation or absolute offset was altered')
                if number==0: state_records[name]=states
                elif states!=state_records[name]: raise ValueError('State accounting changed between clips')
                values[name].append(logits)
        records.append(metadata)
        with (output/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(completed=number+1,id=item['id'],predictions={k:int(values[k][-1].argmax()) for k in KEPT}))+'\n')
        if (number+1)%24==0 or number==347: print(json.dumps(dict(completed=number+1,total=348)),flush=True)
    for name in KEPT: values[name]=np.stack(values[name])
    if model_hash(model)!=before: raise ValueError('Model weights changed')
    for path,digest in plan['sources'].items():
        if file_hash(path)!=digest: raise ValueError('Source changed during inference')
    np.savez_compressed(output/'logits.npz',ids=ids,targets=targets,**values);write(output/'inputs.json',records)
    summary=dict(plan_sha256=file_hash(output/'plan.json'),logits_sha256=file_hash(output/'logits.npz'),
        inputs_sha256=file_hash(output/'inputs.json'),weights_unchanged=True,model_state_sha256=before,
        baseline_replays=replays,state_by_frame=state_records,scores=summarize(values,targets,ids),limitations=plan['limitations'])
    write(output/'summary.json',summary)
    print(json.dumps({k:v['original_label_top1'] for k,v in summary['scores'].items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',default='results/video_history_intervention_seed17')
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        output=Path(args.output);result=verify(output);write(output/'verification.json',result);print(json.dumps(result,indent=2))
    else: run(args)


if __name__=='__main__': main()
