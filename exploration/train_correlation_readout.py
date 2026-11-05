"""CPU-only short-budget test of a correspondence-first point readout."""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from tapvid_metrics import compute_tapvid_metrics
from exploration.correlation_readout import CorrelationReadout, correlation_loss
from exploration.oracle_transport import digest, sample
from exploration.readout_diagnostic import parameter_digest, select_queries


@torch.no_grad()
def evaluate(model, cache, scenes, config, destination):
    model.eval()
    rows, receipts = [], {}
    destination.mkdir(parents=True)
    for scene in scenes:
        data = cache[scene]
        selected = select_queries(data, scene, config['evaluation_queries'], config['seed'])
        truth, visible = data['points'][selected], data['visible'][selected]
        features, xy = data['features'], truth[:, 0]
        reference = sample(features[0], xy, 14)
        predicted, visibility = [xy], [torch.full((len(xy),),10.)]
        for frame in range(1,len(features)):
            out = model(reference[None], features[frame:frame+1], xy[None])
            predicted.append(out['points'][0]); visibility.append(out['visibility_logits'][0])
        predicted, visibility = torch.stack(predicted,1),torch.stack(visibility,1)
        path = destination/(scene+'.npz')
        np.savez_compressed(path, prediction=predicted.numpy(), visibility_logits=visibility.numpy(),
            target_xy=truth.numpy(), visible=visible.numpy(), selected=selected.numpy())
        receipts[scene] = digest(path)
        queries = np.concatenate((np.zeros((len(xy),1),dtype=np.float32),xy[:,[1,0]].numpy()*256),-1)[None]
        displacement = torch.linalg.vector_norm((truth[:,1:]-truth[:,:1])*256,dim=-1)
        for method,value,pred_visible in (
                ('model',predicted,visibility>=0),
                ('static_query',truth[:,:1].expand_as(truth),torch.ones_like(visible))):
            metric = compute_tapvid_metrics(queries,(~visible).numpy()[None],truth.numpy()[None]*256,
                (~pred_visible).numpy()[None],value.numpy()[None]*256,'first')
            metrics = {key:float(v[0]) for key,v in metric.items()}
            distance = torch.linalg.vector_norm((value[:,1:]-truth[:,1:])*256,dim=-1)
            for stratum,mask in dict(all_visible=visible[:,1:],moving_gt2=visible[:,1:] & (displacement>2),
                                     moving_gt16=visible[:,1:] & (displacement>16)).items():
                errors = distance[mask]; count=len(errors)
                rows.append(dict(scene=scene,method=method,stratum=stratum,points=count,
                    mean_error=float(errors.mean()) if count else None,
                    **{f'pck_{q}':float((errors<q).float().mean()) if count else None for q in (4,8,16)},
                    full_sequence_metrics=metrics if stratum=='all_visible' else None))
    summary=[]
    for method in ('model','static_query'):
        for stratum in ('all_visible','moving_gt2','moving_gt16'):
            selected=[r for r in rows if r['method']==method and r['stratum']==stratum and r['points']]
            summary.append(dict(method=method,stratum=stratum,scenes=len(selected),points=sum(r['points'] for r in selected),
                **{key:sum(r[key] for r in selected)/len(selected) if selected else None for key in ('mean_error','pck_4','pck_8','pck_16')},
                full_sequence_metrics={key:sum(r['full_sequence_metrics'][key] for r in selected)/len(selected)
                    for key in selected[0]['full_sequence_metrics']} if selected and stratum=='all_visible' else None))
    return dict(rows=rows,summary=summary,predictions_sha256=receipts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Execute unchanged plan, not partial training')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    directory,output=Path(args.cache).resolve(),Path(args.output).resolve()
    cache_plan=json.loads((directory/'plan.json').read_text())
    receipt=json.loads((directory/'summary.json').read_text())
    if receipt['plan_sha256']!=digest(directory/'plan.json'):
        raise ValueError('Cache plan changed')
    config=dict(seed=17,arms=['local_only','metric_and_local'],steps=1024,checkpoints=[0,256,1024],
        batch=1,clip_frames=4,queries=32,evaluation_queries=128,lr=1e-4,weight_decay=1e-4,
        train=cache_plan['config']['train'],val=cache_plan['config']['val'],
        model=DEFAULT_MODEL,revision=DEFAULT_REVISION,precision='CPU FP32',depth=12,
        feature_transform='Frozen pretrained final post-layernorm',
        supervision='Visible target cell CE + within-cell smooth L1 + 0.1 visibility BCE; no teacher-feature loss',
        scope='Readout only, frozen images; no Mamba. 24 train and 8 already-inspected validation rendered scenes. Original frames have stride 4.',
        limits='Not a novel encoder, SOTA result, real-video benchmark, or cost-matched comparison with the 15M readout. Earlier readout training used BF16; this diagnostic uses CPU FP32.')
    plan=dict(config=config,cache=str(directory),cache_summary_sha256=digest(directory/'summary.json'),
        sources={str(p):digest(p) for p in (Path(__file__),Path('exploration/correlation_readout.py'),
            Path('exploration/readout_diagnostic.py'),Path('exploration/oracle_transport.py'),Path('tapvid_metrics.py'),
            Path('temporal_siglip.py'),Path('pyproject.toml'),Path('uv.lock'))})
    if args.resume:
        if json.loads((output/'plan.json').read_text())!=plan or (output/'training.jsonl').exists():
            raise ValueError('Changed plan or already-started training')
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.plan_only:
        print('Two fixed 1024-update CPU arms planned',flush=True); return
    backbone=SiglipVisionModel.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True).eval()
    norm=backbone.vision_model.post_layernorm; norm.requires_grad_(False)
    norm_hash=parameter_digest(norm); del backbone
    cache={}
    for scene in config['train']+config['val']:
        path=directory/(scene+'.pt')
        if digest(path)!=receipt['scenes'][scene]['sha256']: raise ValueError('Cache changed')
        data=torch.load(path,map_location='cpu',weights_only=True)
        with torch.no_grad(): feature=norm(data['features']['12']).detach()
        cache[scene]=dict(features=feature,points=data['points'],visible=data['visible'])
    del norm,data
    started,results,initial_hash=time.monotonic(),{},None
    for arm in config['arms']:
        torch.manual_seed(config['seed'])
        model=CorrelationReadout(adapt_metric=arm=='metric_and_local')
        actual=parameter_digest(model)
        if initial_hash is not None and actual!=initial_hash: raise ValueError('Initial weights differ')
        initial_hash=actual
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['lr'],weight_decay=config['weight_decay'])
        rng=random.Random(config['seed']); generator=torch.Generator().manual_seed(config['seed'])
        scores={}
        for step in range(config['steps']+1):
            if step in config['checkpoints']:
                scores[str(step)]=evaluate(model,cache,config['val'],config,output/arm/str(step))
                print(json.dumps(dict(event='evaluate',arm=arm,step=step,summary=scores[str(step)]['summary'])),flush=True)
            if step==config['steps']: break
            model.train()
            data=cache[rng.choice(config['train'])]
            start=rng.randrange(len(data['features'])-config['clip_frames']+1)
            candidates=torch.where(data['visible'][:,start])[0]
            chosen=candidates[torch.randperm(len(candidates),generator=generator)[:config['queries']]]
            if len(chosen)!=config['queries']: raise ValueError('Too few visible queries')
            xy=data['points'][chosen,start]
            reference=sample(data['features'][start],xy,14)
            end=start+config['clip_frames']; count=end-start-1
            current=data['features'][start+1:end]
            target=data['points'][chosen,start+1:end].transpose(0,1)
            visible=data['visible'][chosen,start+1:end].transpose(0,1)
            optimizer.zero_grad(set_to_none=True)
            losses=correlation_loss(model,reference[None].expand(count,-1,-1),current,
                xy[None].expand(count,-1,-1),target,visible)
            if not torch.isfinite(losses['loss']): raise ValueError('Nonfinite loss')
            losses['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if (step+1)%128==0:
                row=dict(arm=arm,step=step+1,**{k:float(v.detach()) for k,v in losses.items()},elapsed_seconds=time.monotonic()-started)
                with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        checkpoint=output/(arm+'.pt')
        torch.save(dict(config=model.config,state_dict=model.state_dict()),checkpoint)
        results[arm]=dict(initial_parameters_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            checkpoint_sha256=digest(checkpoint),scores=scores)
        (output/(arm+'.json')).write_text(json.dumps(results[arm],indent=2)+'\n')
        del model,optimizer,losses; gc.collect()
    report=dict(plan_sha256=digest(output/'plan.json'),post_layernorm_sha256=norm_hash,
                results=results,elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
