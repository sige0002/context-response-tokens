"""Factorial CPU diagnostic: pair spacing versus visibility class weighting.

All four arms use identical scene/query schedules and initialization. Only
target-frame spacing and the visibility loss's class weights change. The
window/unweighted arm must exactly reproduce the previous trained checkpoint.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from transformers import SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from exploration.correlation_readout import CorrelationReadout, correlation_loss
from exploration.oracle_transport import digest, sample
from exploration.readout_diagnostic import parameter_digest
from exploration.train_correlation_readout import evaluate


def make_schedule(config, cache):
    rng=random.Random(config['seed'])
    span_rng=random.Random(config['seed']+10000)
    generator=torch.Generator().manual_seed(config['seed'])
    rows=[]
    stats={mode:dict(visible=0,invisible=0,moving_visible_gt16=0,gaps=Counter()) for mode in ('window','span')}
    for _ in range(config['steps']):
        scene=rng.choice(config['train']); data=cache[scene]
        frames=data['points'].shape[1]
        start=rng.randrange(frames-config['clip_frames']+1)
        candidates=torch.where(data['visible'][:,start])[0]
        chosen=candidates[torch.randperm(len(candidates),generator=generator)[:config['queries']]]
        if len(chosen)!=config['queries']: raise ValueError('Too few queries')
        future=dict(window=list(range(start+1,start+config['clip_frames'])),
                    span=sorted(span_rng.sample(range(start+1,frames),config['clip_frames']-1)))
        rows.append(dict(scene=scene,query_frame=start,queries=chosen.tolist(),**future))
        xy=data['points'][chosen,start]
        for mode,indices in future.items():
            visible=data['visible'][chosen][:,indices]
            target=data['points'][chosen][:,indices]
            displacement=torch.linalg.vector_norm((target-xy[:,None])*256,dim=-1)
            stats[mode]['visible']+=int(visible.sum())
            stats[mode]['invisible']+=int((~visible).sum())
            stats[mode]['moving_visible_gt16']+=int((visible & (displacement>16)).sum())
            stats[mode]['gaps'].update({frame-start:config['queries'] for frame in indices})
    for mode,stat in stats.items():
        total=stat['visible']+stat['invisible']
        if not stat['visible'] or not stat['invisible']: raise ValueError('Both visibility classes required')
        stat['total']=total
        stat['invisible_fraction']=stat['invisible']/total
        stat['class_weights']=dict(visible=total/(2*stat['visible']),invisible=total/(2*stat['invisible']))
        stat['mean_gap_cached_frames']=sum(g*n for g,n in stat['gaps'].items())/total
        stat['gaps']=dict(sorted(stat['gaps'].items()))
    return rows,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Execute unchanged plan, not partial-training resume')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    reference_run,output=Path(args.reference_run).resolve(),Path(args.output).resolve()
    old_plan=json.loads((reference_run/'plan.json').read_text())
    old_result=json.loads((reference_run/'summary.json').read_text())
    reference_checkpoint=reference_run/'metric_and_local.pt'
    if digest(reference_checkpoint)!=old_result['results']['metric_and_local']['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    config=dict(old_plan['config'])
    config.update(arms=['window_unweighted','span_unweighted','window_balanced','span_balanced'],
        scope='Factorial readout diagnostic only; no recurrent encoder or teacher model',
        balancing='Fixed inverse class frequencies counted on the complete planned training schedule; no validation calibration',
        spacing='Same query frame and points. Window uses next 3 cached frames; span samples 3 distinct later frames with independent seed+10000 RNG.',
        visibility_threshold=0.,limits='24 training / 8 repeatedly inspected validation rendered scenes; one seed, short budget. Not an independent test or SOTA claim.')
    directory=Path(old_plan['cache']); receipt=json.loads((directory/'summary.json').read_text())
    if digest(directory/'summary.json')!=old_plan['cache_summary_sha256']: raise ValueError('Cache summary changed')
    cache={}
    for scene in config['train']+config['val']:
        path=directory/(scene+'.pt')
        if digest(path)!=receipt['scenes'][scene]['sha256']: raise ValueError('Cache changed')
        data=torch.load(path,map_location='cpu',weights_only=True)
        cache[scene]=dict(features=data['features']['12'],points=data['points'],visible=data['visible'])
    del data
    schedule,coverage=make_schedule(config,cache)
    schedule_text=json.dumps(schedule,indent=2)+'\n'
    import hashlib
    plan=dict(config=config,cache=str(directory),cache_summary_sha256=digest(directory/'summary.json'),
        reference_run=str(reference_run),reference_plan_sha256=digest(reference_run/'plan.json'),
        reference_result_sha256=digest(reference_run/'summary.json'),training_coverage=coverage,
        schedule_sha256=hashlib.sha256(schedule_text.encode()).hexdigest(),
        sources={str(p):digest(p) for p in (Path(__file__),Path('exploration/correlation_readout.py'),
            Path('exploration/train_correlation_readout.py'),Path('exploration/readout_diagnostic.py'),
            Path('exploration/oracle_transport.py'),Path('tapvid_metrics.py'),Path('temporal_siglip.py'),Path('pyproject.toml'),Path('uv.lock'))})
    # Normalize integer histogram keys to JSON before comparing saved plans.
    plan=json.loads(json.dumps(plan))
    if args.resume:
        if json.loads((output/'plan.json').read_text())!=plan or (output/'training.jsonl').exists():
            raise ValueError('Changed plan or already-started training')
        if digest(output/'schedule.json')!=plan['schedule_sha256']: raise ValueError('Schedule changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        (output/'schedule.json').write_text(schedule_text)
    if args.plan_only:
        print(json.dumps(coverage),flush=True); return
    backbone=SiglipVisionModel.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True).eval()
    norm=backbone.vision_model.post_layernorm; norm.requires_grad_(False)
    norm_hash=parameter_digest(norm); del backbone
    if norm_hash!=old_result['post_layernorm_sha256']: raise ValueError('Normalization changed')
    with torch.no_grad():
        for data in cache.values(): data['features']=norm(data['features']).detach()
    del norm
    started,results,initial_hash=time.monotonic(),{},None
    for arm in config['arms']:
        spacing,weighting=arm.split('_')
        torch.manual_seed(config['seed'])
        model=CorrelationReadout(adapt_metric=True)
        actual=parameter_digest(model)
        if initial_hash is not None and actual!=initial_hash: raise ValueError('Initial weights differ')
        if actual!=old_result['results']['metric_and_local']['initial_parameters_sha256']:
            raise ValueError('Reference initialization differs')
        initial_hash=actual
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        scores={}
        for step in range(config['steps']+1):
            if step in config['checkpoints']:
                scores[str(step)]=evaluate(model,cache,config['val'],config,output/arm/str(step))
                print(json.dumps(dict(event='evaluate',arm=arm,step=step,summary=scores[str(step)]['summary'])),flush=True)
            if step==config['steps']: break
            model.train(); entry=schedule[step]; data=cache[entry['scene']]
            start=entry['query_frame']; chosen=torch.tensor(entry['queries']); indices=entry[spacing]
            xy=data['points'][chosen,start]; anchor=sample(data['features'][start],xy,14)
            count=len(indices)
            if spacing=='window':
                end=start+config['clip_frames']
                current=data['features'][start+1:end]
                target=data['points'][chosen,start+1:end].transpose(0,1)
                visible=data['visible'][chosen,start+1:end].transpose(0,1)
            else:
                current=data['features'][indices]
                target=data['points'][chosen][:,indices].transpose(0,1)
                visible=data['visible'][chosen][:,indices].transpose(0,1)
            anchor=anchor[None].expand(count,-1,-1); query=xy[None].expand(count,-1,-1)
            optimizer.zero_grad(set_to_none=True)
            losses=correlation_loss(model,anchor,current,query,target,visible)
            if weighting=='balanced':
                predicted=model(anchor,current,query)['visibility_logits']
                weights=coverage[spacing]['class_weights']
                class_weight=torch.where(visible,weights['visible'],weights['invisible'])
                balanced=(F.binary_cross_entropy_with_logits(predicted,visible.float(),reduction='none')*class_weight).mean()
                losses['loss']=losses['coarse']+losses['offset']+.1*balanced
                losses['visibility']=balanced
            if not torch.isfinite(losses['loss']): raise ValueError('Nonfinite loss')
            losses['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if (step+1)%128==0:
                row=dict(arm=arm,step=step+1,**{k:float(v.detach()) for k,v in losses.items()},elapsed_seconds=time.monotonic()-started)
                with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        if arm=='window_unweighted':
            reference_weights=torch.load(reference_checkpoint,map_location='cpu',weights_only=True)['state_dict']
            for key,value in model.state_dict().items():
                if not torch.equal(value,reference_weights[key]): raise ValueError(f'Reference reproduction failed: {key}')
            if any(scores[s]['rows']!=old_result['results']['metric_and_local']['scores'][s]['rows'] for s in scores):
                raise ValueError('Reference evaluation reproduction failed')
        checkpoint=output/(arm+'.pt')
        torch.save(dict(config=model.config,state_dict=model.state_dict()),checkpoint)
        results[arm]=dict(initial_parameters_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            checkpoint_sha256=digest(checkpoint),reference_exact_match=arm=='window_unweighted',scores=scores)
        (output/(arm+'.json')).write_text(json.dumps(results[arm],indent=2)+'\n')
        del model,optimizer,losses; gc.collect()
    report=dict(plan_sha256=digest(output/'plan.json'),post_layernorm_sha256=norm_hash,results=results,
                elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
