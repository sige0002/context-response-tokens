"""Matched final-checkpoint state-erasure diagnostic; all current transforms stay."""
import hashlib
import json
import math
from pathlib import Path
import statistics


MODELS=('mamba_interleaved','attention_interleaved')
CONDITIONS=dict(full_stream=('3','7','11'),clear_4=('7','11'),clear_8=('3','11'),clear_12=('3','7'),reset_all=())


def read(path): return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path,value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def aggregate(records):
    expected={f'{m}_seed{s}' for m in MODELS for s in (42,43,44)}
    if set(records)!=expected: raise ValueError('All six predeclared models are required')
    for record in records.values():
        if set(record['scores'])!=set(CONDITIONS): raise ValueError('Incomplete state-erasure conditions')
        for row in record['scores'].values():
            if (row['total']!=348 or type(row['correct']) is not int or not 0<=row['correct']<=348
                    or not math.isfinite(row['original_label_top1'])
                    or row['original_label_top1']!=row['correct']/348):
                raise ValueError('Inconsistent score or cohort size')
    models={}
    for model in MODELS:
        conditions={}
        for condition in CONDITIONS:
            correct=[records[f'{model}_seed{s}']['scores'][condition]['correct'] for s in (42,43,44)]
            full=[records[f'{model}_seed{s}']['scores']['full_stream']['correct'] for s in (42,43,44)]
            if any(type(n) is not int or not 0<=n<=348 for n in correct+full):
                raise ValueError('Invalid fixed-cohort count')
            delta=[100*(a-b)/348 for a,b in zip(correct,full)]
            conditions[condition]=dict(seed_order=[42,43,44],correct=correct,total_per_seed=348,
                accuracy_percent=[100*n/348 for n in correct],delta_vs_full_pp=delta,
                mean_delta_pp=statistics.mean(delta),sample_sd_delta_pp=statistics.stdev(delta))
        models[model]=conditions
    difference={}
    for condition in CONDITIONS:
        a=models[MODELS[0]][condition];b=models[MODELS[1]][condition]
        delta=[x-y for x,y in zip(a['delta_vs_full_pp'],b['delta_vs_full_pp'])]
        difference[condition]=dict(seed_order=[42,43,44],mamba_minus_attention_state_erasure_effect_pp=delta,
            mean_pp=statistics.mean(delta),sample_sd_pp=statistics.stdev(delta))
    return dict(models=models,effect_differences=difference,
        interpretation='An effect difference compares within-model changes, not raw architecture accuracy. Larger values mean a less negative state-erasure effect for Mamba.',
        limitations=['Previously inspected 348-video validation cohort; not full benchmark or held-out test.',
            'State erasure changes the fitted readout input distribution; not a separately trained model.',
            'Other layers may retain history. Effects are not additive.',
            'Eight observations do not establish long-term memory.',
            'Three seeds provide limited uncertainty information; no speed or robotics-success claim.'])


def run_one(plan,plan_path,row,output,check_cancel):
    import numpy as np
    import torch
    from transformers import SiglipImageProcessor
    from temporal_priority import verify_video
    from temporal_siglip import StreamState
    from evaluate_video import streaming_logits
    from video_checkpoint import load_video_checkpoint
    from video_data import RawVideoClips
    from train_video import configure_reproducibility
    from exploration.evaluate_video_history import model_hash,tensor_hash,summarize
    from exploration.memory_layer_intervention import stream_with_memory

    check_cancel(); completed,reference=verify_video(row)
    model,info=load_video_checkpoint(Path(row['run'])/'last.pt','cpu')
    if (info['step']!=20000 or info['config']!=row['config']
            or info['checkpoint_sha256']!=completed['checkpoint_sha256']
            or model.encoder.placement!='interleaved' or tuple(model.encoder.adapters)!=('3','7','11')):
        raise ValueError('Final model or architecture differs')
    model.requires_grad_(False).eval();torch.set_num_threads(2);configure_reproducibility(row['seed'])
    processor=SiglipImageProcessor.from_pretrained(info['config']['model'],revision=info['config']['revision'],local_files_only=True)
    dataset=RawVideoClips(info['config']['val'],processor,frames=8,classes=174)
    if sha(dataset.path)!=info['config']['val_sha256']: raise ValueError('Validation manifest changed')
    old_inputs=read(Path(plan['cohort'])/'inputs.json'); subset=plan['subset']
    before=model_hash(model)
    provenance=dict(study_plan_sha256=sha(plan_path),row=row,checkpoint_info=info,batch_size=1,
                    conditions={k:list(v) for k,v in CONDITIONS.items()},sources=plan['sources'],
                    original_evaluation_sha256=reference['evaluation_sha256'])
    output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',provenance)
    values={k:[] for k in CONDITIONS};records=[];replays=[];states_by_condition={}
    model.cuda()
    with torch.inference_mode():
        for number,item in enumerate(subset):
            check_cancel()
            pixels,target,metadata=dataset.read(item['index'])
            metadata.update(label=target,pixels_sha256=tensor_hash(pixels),current_pixels_sha256=tensor_hash(pixels[-1]))
            if (metadata!=old_inputs[number] or target!=item['label'] or metadata['id']!=item['id']
                    or sha(dataset.rows[item['index']]['video'])!=item['video_sha256']):
                raise ValueError('Input differs from fixed cohort')
            video=pixels[None].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                for condition,keys in CONDITIONS.items():
                    features,states=stream_with_memory(model.encoder,video,keys)
                    logits=model.readout(features)[0].float().cpu().numpy()
                    if not np.isfinite(logits).all(): raise FloatingPointError('Nonfinite logits')
                    if any(r['offset']!=i+1 or r['state_layer_keys']!=['3','7','11'] for i,r in enumerate(states)):
                        raise ValueError('State offset or adapter membership changed')
                    if number==0: states_by_condition[condition]=states
                    elif states!=states_by_condition[condition]: raise ValueError('State accounting differs between clips')
                    values[condition].append(logits)
                if number in (0,347):
                    original,_=streaming_logits(model,video)
                    at_last=StreamState(id(model.encoder),1,tuple(video.shape[2:]),offset=7)
                    current,_=model.encoder.step(video[:,-1],at_last)
                    current=model.readout(current)
                    for name,logits in [('full_stream',original),('reset_all',current)]:
                        delta=float(np.abs(logits[0].float().cpu().numpy()-values[name][-1]).max())
                        replays.append(dict(index=number,condition=name,max_abs=delta))
                        if delta!=0: raise ValueError('State-erasure equivalence failed')
            records.append(metadata)
            with (output/'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(completed=number+1,id=item['id']))+'\n')
            if (number+1)%48==0 or number==347: print(json.dumps(dict(run=row['name'],completed=number+1,total=348)),flush=True)
    values={k:np.stack(v) for k,v in values.items()};ids=np.array([r['id'] for r in records]);targets=np.array([r['label'] for r in records])
    if model_hash(model)!=before: raise ValueError('Weights changed')
    if sha(Path(row['run'])/'last.pt')!=info['checkpoint_sha256']: raise ValueError('Final checkpoint changed')
    previous=dict(zip(reference['ids'],reference['predictions']['full','stream']))
    disagree=sum(int(p)!=previous[str(i)] for i,p in zip(ids,values['full_stream'].argmax(-1)))
    np.savez_compressed(output/'logits.npz',ids=ids,targets=targets,**values);write(output/'inputs.json',records)
    summary=dict(plan_sha256=sha(output/'plan.json'),logits_sha256=sha(output/'logits.npz'),inputs_sha256=sha(output/'inputs.json'),
        checkpoint_sha256=info['checkpoint_sha256'],weights_unchanged=True,model_state_sha256=before,
        baseline_replays=replays,state_by_frame=states_by_condition,scores=summarize(values,targets,ids),
        original_full_evaluation_disagreements=disagree,
        original_evaluation_scope='Original full validation may use a different batch size. All intervention contrasts here use newly recomputed batch-one baselines.')
    write(output/'summary.json',summary)
    del model
    torch.cuda.empty_cache()
    return summary


def verify_one(plan,plan_path,row,output):
    import numpy as np
    from exploration.evaluate_video_history import summarize
    metadata=read(output/'plan.json');summary=read(output/'summary.json')
    if (metadata['study_plan_sha256']!=sha(plan_path) or metadata['row']!=row or metadata['batch_size']!=1
            or metadata['conditions']!={k:list(v) for k,v in CONDITIONS.items()} or metadata['sources']!=plan['sources']
            or metadata['checkpoint_info']['config']!=row['config'] or metadata['checkpoint_info']['step']!=20000):
        raise ValueError('Diagnostic belongs to a different model or plan')
    for name,key in [('plan.json','plan_sha256'),('inputs.json','inputs_sha256'),('logits.npz','logits_sha256')]:
        if sha(output/name)!=summary[key]: raise ValueError('Diagnostic artifact changed')
    checkpoint=sha(Path(row['run'])/'last.pt')
    if checkpoint!=summary['checkpoint_sha256'] or checkpoint!=metadata['checkpoint_info']['checkpoint_sha256']:
        raise ValueError('Diagnostic checkpoint changed')
    inputs=read(output/'inputs.json')
    if inputs!=read(Path(plan['cohort'])/'inputs.json'): raise ValueError('Diagnostic input differs from cohort')
    manifest=Path(row['config']['val'])
    if sha(manifest)!=row['config']['val_sha256']: raise ValueError('Original labels changed')
    labels={r['id']:r['label'] for r in (json.loads(s) for s in manifest.read_text().splitlines() if s)}
    with np.load(output/'logits.npz') as saved:
        ids=saved['ids'];targets=saved['targets']
        if (len(ids)!=348 or len(set(ids))!=348 or ids.tolist()!=[r['id'] for r in plan['subset']]
                or targets.tolist()!=[labels[str(i)] for i in ids]): raise ValueError('Incomplete or altered cohort')
        values={k:saved[k] for k in CONDITIONS}
        if any(v.shape!=(348,174) or not np.isfinite(v).all() for v in values.values()): raise ValueError('Invalid diagnostic logits')
        if summarize(values,targets,ids)!=summary['scores']: raise ValueError('Score recomputation differs')
        original_path=Path(row['run'])/'final_clip_stream.json'
        if sha(original_path)!=metadata['original_evaluation_sha256']['final_clip_stream.json']:
            raise ValueError('Original full evaluation changed')
        original=read(original_path)['ablations']['full']['modes']['stream']['clips']
        indexed={r['id']:r['prediction'] for r in original}
        differences=sum(int(p)!=indexed[str(i)] for i,p in zip(ids,values['full_stream'].argmax(-1)))
        if differences!=summary['original_full_evaluation_disagreements']:
            raise ValueError('Original-evaluation disagreement count differs')
    replays=summary['baseline_replays']
    if (not summary['weights_unchanged'] or len(replays)!=4
            or {(r['index'],r['condition']) for r in replays}!={(i,c) for i in (0,347) for c in ('full_stream','reset_all')}
            or any(r['max_abs']!=0 for r in replays)):
        raise ValueError('Native equivalence or weight preservation failed')
    if set(summary['state_by_frame'])!=set(CONDITIONS): raise ValueError('Missing state accounting')
    for condition,keys in CONDITIONS.items():
        states=summary['state_by_frame'][condition]
        if len(states)!=8: raise ValueError('Missing per-frame state accounting')
        for i,s in enumerate(states):
            expected=79027200 if row['config']['kind']=='mamba' else 200704*(len(keys)*(i+1)+3-len(keys))
            if s['offset']!=i+1 or s['state_layer_keys']!=['3','7','11'] or s['state_bytes']!=expected:
                raise ValueError('State identity, offset or capacity differs')
    return summary
