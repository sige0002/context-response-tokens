"""Matched supervised linear readouts for frozen temporal feature conditions."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from exploration.continuous_matching import continuous_match
from exploration.evaluate_continuous_matching import score_scene,aggregate
from exploration.linear_correspondence_metric import LinearCorrespondenceMetric,metric_loss
from exploration.oracle_transport import digest,sample
from exploration.readout_diagnostic import parameter_digest,select_queries


def make_schedule(cache,config):
    rng=random.Random(config['seed']); generator=torch.Generator().manual_seed(config['seed'])
    rows=[]; visible_count=total=0
    for _ in range(config['steps']):
        scene=rng.choice(config['train']); data=cache[scene]
        candidates=torch.where(data['visible'][:,0])[0]
        selected=candidates[torch.randperm(len(candidates),generator=generator)[:config['queries']]]
        if len(selected)!=config['queries']: raise ValueError('Too few training queries')
        frames=sorted(rng.sample(range(1,data['points'].shape[1]),config['targets_per_update']))
        visible=data['visible'][selected][:,frames]
        total+=visible.numel(); visible_count+=int(visible.sum())
        rows.append(dict(scene=scene,query_frame=0,queries=selected.tolist(),frames=frames))
    return rows,dict(targets=total,visible=visible_count,invisible=total-visible_count)


@torch.no_grad()
def evaluate(model,condition,cache,config,destination,reference=None):
    destination.mkdir(parents=True)
    model.eval(); rows=[]; predictions={}
    checkpoint=destination/'metric.pt'
    torch.save(dict(config=model.config,state_dict=model.state_dict()),checkpoint)
    for scene in config['val']:
        data=cache[scene]; selected=select_queries(data,scene,config['evaluation_queries'],config['seed'])
        truth,visible=data['points'][selected],data['visible'][selected]
        features=model(data['features'][condition]).detach()
        anchor=sample(features[0],truth[:,0],config['grid'])
        coordinates={name:[] for name in config['methods'] if name!='static_query'}
        similarities={name:[] for name in coordinates}
        for frame in range(len(features)):
            output=continuous_match(anchor,features[frame],config['grid'],True,config['topk'],config['refinement_steps'])
            for name,value in output.items():
                coordinates[name].append(value['points'].numpy()); similarities[name].append(value['scores'].numpy())
        arrays={k:np.stack(v,1) for k,v in coordinates.items()}
        arrays['static_query']=np.broadcast_to(truth[:,:1].numpy(),truth.shape).copy()
        if reference is not None:
            with np.load(reference/(scene+'_'+condition+'_12.npz')) as old:
                for method,values in arrays.items(): assert np.array_equal(values,old[method]),(scene,condition,method)
        saved=dict(target_xy=truth.numpy(),visible=visible.numpy(),selected=selected.numpy(),**arrays)
        saved.update({k+'_score':np.stack(v,1) for k,v in similarities.items()})
        path=destination/(scene+'.npz'); np.savez_compressed(path,**saved); predictions[scene]=digest(path)
        rows.extend(score_scene(scene,12,truth.numpy(),visible.numpy(),arrays))
    return dict(checkpoint_sha256=digest(checkpoint),predictions_sha256=predictions,rows=rows,summary=aggregate(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache',required=True)
    parser.add_argument('--probe-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    directory,probe,output=Path(args.train_cache).resolve(),Path(args.probe_run).resolve(),Path(args.output).resolve()
    cache_plan=json.loads((directory/'plan.json').read_text()); receipt=json.loads((directory/'summary.json').read_text())
    assert receipt['plan_sha256']==digest(directory/'plan.json')
    old=json.loads((probe/'plan.json').read_text()); old_result=json.loads((probe/'summary.json').read_text())
    assert digest(probe/'plan.json')==cache_plan['probe_plan_sha256']==old_result['plan_sha256']
    assert old['checkpoint']['checkpoint_sha256']==cache_plan['checkpoint_sha256']
    config=dict(seed=17,train=cache_plan['config']['train'],val=old['config']['val'],conditions=old['config']['conditions'],
        steps=1024,checkpoints=[0,256,1024],rank=32,dim=768,lr=1e-3,weight_decay=1e-4,queries=64,targets_per_update=3,
        evaluation_queries=128,grid=14,methods=old['config']['methods'],topk=4,refinement_steps=old['config']['refinement_steps'],
        precision='CPU FP32, 2 threads',
        supervision='Visibility-masked correspondence CE with bilinear target-cell weights; no feature teacher or encoder update',
        schedule='Same scenes, frame-zero queries and three random future targets in all conditions',
        transform='x + U V x, rank32, zero-initialized U. Shared between reference/current; LN and L2 follow transformation.',
        scope='Short-budget linear readout of fixed 4000-update features; not retraining the encoder',
        limits='24 training / 8 repeatedly inspected rendered validation scenes, one seed. A limited linear probe does not settle general representation quality.')
    assert not set(config['train'])&set(config['val'])
    cache={}
    for scene in config['train']:
        path=directory/(scene+'.pt'); assert digest(path)==receipt['records'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        cache[scene]={k:data[k] for k in ('features','points','visible')}
    original=Path(old['cache']); original_receipt=json.loads((original/'summary.json').read_text())
    assert digest(original/'summary.json')==old['cache_summary_sha256']
    for scene in config['val']:
        path=original/(scene+'.pt'); assert digest(path)==original_receipt['scenes'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        feature_path=probe/(scene+'_features.pt'); assert digest(feature_path)==old_result['features_sha256'][scene]
        features=torch.load(feature_path,map_location='cpu',weights_only=True)['features']
        cache[scene]=dict(points=data['points'],visible=data['visible'],features={k:v['12'] for k,v in features.items()})
    del data,features; gc.collect()
    schedule,coverage=make_schedule(cache,config); schedule_text=json.dumps(schedule,indent=2)+'\n'
    sources=[Path(__file__),Path('exploration/linear_correspondence_metric.py'),Path('exploration/continuous_matching.py'),
        Path('exploration/evaluate_continuous_matching.py'),Path('exploration/oracle_transport.py'),
        Path('exploration/readout_diagnostic.py'),Path('tapvid_metrics.py'),Path('pyproject.toml'),Path('uv.lock')]
    plan=dict(config=config,train_cache=str(directory),train_cache_plan_sha256=digest(directory/'plan.json'),
        train_cache_summary_sha256=digest(directory/'summary.json'),probe_run=str(probe),
        probe_plan_sha256=digest(probe/'plan.json'),probe_summary_sha256=digest(probe/'summary.json'),
        schedule_sha256=hashlib.sha256(schedule_text.encode()).hexdigest(),training_coverage=coverage,
        sources={str(p):digest(p) for p in sources})
    if args.resume:
        if json.loads((output/'plan.json').read_text())!=plan or (output/'training.jsonl').exists():
            raise ValueError('Changed plan or training already started')
        assert digest(output/'schedule.json')==plan['schedule_sha256']
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n'); (output/'schedule.json').write_text(schedule_text)
    if args.plan_only:
        print(json.dumps(dict(conditions=config['conditions'],updates_per_condition=1024,coverage=coverage)),flush=True); return
    initial=None; results={}; started=time.monotonic()
    for condition in config['conditions']:
        torch.manual_seed(config['seed']); model=LinearCorrespondenceMetric(config['dim'],config['rank'])
        initial_hash=parameter_digest(model)
        if initial is not None: assert initial_hash==initial
        initial=initial_hash
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        scores={}
        for step in range(config['steps']+1):
            if step in config['checkpoints']:
                scores[str(step)]=evaluate(model,condition,cache,config,output/condition/str(step),reference=probe if step==0 else None)
                print(json.dumps(dict(event='evaluate',condition=condition,step=step,
                    summary=[r for r in scores[str(step)]['summary'] if r['method']=='field_top4' and r['task']=='motion'])),flush=True)
            if step==config['steps']: break
            model.train(); entry=schedule[step]; data=cache[entry['scene']]
            chosen=torch.tensor(entry['queries']); frames=entry['frames']; features=data['features'][condition]
            anchor=sample(features[0],data['points'][chosen,0],config['grid'])
            target=data['points'][chosen][:,frames].transpose(0,1)
            visible=data['visible'][chosen][:,frames].transpose(0,1)
            optimizer.zero_grad(set_to_none=True)
            loss=metric_loss(model,anchor[None].expand(len(frames),-1,-1),features[frames],target,visible,config['grid'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite metric loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            if (step+1)%128==0:
                row=dict(condition=condition,step=step+1,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started)
                with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        assert model.up.weight.norm()>0
        results[condition]=dict(initial_parameters_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
            initial_predictions_exactly_match_fixed_probe=True,scores=scores)
        (output/(condition+'.json')).write_text(json.dumps(results[condition],indent=2)+'\n')
        del model,optimizer,loss; gc.collect()
    report=dict(plan_sha256=digest(output/'plan.json'),elapsed_seconds=time.monotonic()-started,results=results)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
