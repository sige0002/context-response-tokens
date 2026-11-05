"""CPU frozen-feature diagnostic of patch-center versus continuous localization."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL,DEFAULT_REVISION
from tapvid_metrics import compute_tapvid_metrics
from exploration.continuous_matching import continuous_match
from exploration.oracle_transport import digest,sample
from exploration.readout_diagnostic import parameter_digest,select_queries


def score_scene(scene,depth,truth,visible,arrays):
    rows=[]
    movement=np.linalg.norm((truth[:,1:]-truth[:,:1])*256,axis=-1)
    queries=np.concatenate((np.zeros((len(truth),1),dtype=np.float32),truth[:,0,[1,0]]*256),-1)[None]
    for method,coordinates in arrays.items():
        error=np.linalg.norm((coordinates-truth)*256,axis=-1)
        # Self-match is a localization diagnostic, not a supplied initial point.
        identity=error[:,0]
        rows.append(dict(scene=scene,depth=depth,method=method,task='identity',stratum='all_visible',
            points=len(identity),mean_error=float(identity.mean()),
            **{f'pck_{q}':float((identity<q).mean()) for q in (1,2,4,8,16)},full_sequence_metrics=None))
        metrics=compute_tapvid_metrics(queries,(~visible)[None],truth[None]*256,
            np.zeros_like(visible)[None],coordinates[None]*256,'first')
        for stratum,mask in dict(all_visible=visible[:,1:],moving_gt2=visible[:,1:]&(movement>2),
                                 moving_gt16=visible[:,1:]&(movement>16)).items():
            values=error[:,1:][mask]
            rows.append(dict(scene=scene,depth=depth,method=method,task='motion',stratum=stratum,
                points=len(values),mean_error=float(values.mean()) if len(values) else None,
                **{f'pck_{q}':float((values<q).mean()) if len(values) else None for q in (1,2,4,8,16)},
                full_sequence_metrics={k:float(v[0]) for k,v in metrics.items()} if stratum=='all_visible' else None))
    return rows


def aggregate(rows):
    output=[]
    for depth,method,task,stratum in sorted({(r['depth'],r['method'],r['task'],r['stratum']) for r in rows}):
        selected=[r for r in rows if (r['depth'],r['method'],r['task'],r['stratum'])==(depth,method,task,stratum) and r['points']]
        output.append(dict(depth=depth,method=method,task=task,stratum=stratum,scenes=len(selected),
            points=sum(r['points'] for r in selected),
            **{key:float(np.mean([r[key] for r in selected])) if selected else None
               for key in ('mean_error','pck_1','pck_2','pck_4','pck_8','pck_16')},
            full_sequence_metrics={key:float(np.mean([r['full_sequence_metrics'][key] for r in selected]))
                for key in selected[0]['full_sequence_metrics']} if selected and selected[0]['full_sequence_metrics'] else None))
    return output


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--control-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    reference,control,output=Path(args.reference_run).resolve(),Path(args.control_run).resolve(),Path(args.output).resolve()
    prior=json.loads((reference/'plan.json').read_text()); directory=Path(prior['cache'])
    receipt=json.loads((directory/'summary.json').read_text())
    if digest(directory/'summary.json')!=prior['cache_summary_sha256']: raise ValueError('Cache changed')
    config=dict(seed=17,val=prior['config']['val'],evaluation_queries=128,grid=14,
        depths={'4':'Raw block4 cosine','12':'Pretrained post-layernorm, then parameter-free LN at each sampled position'},
        methods=['center','field_top1','field_top4','static_query'],topk=4,
        coarse_offsets=[-.5,-.25,0.,.25,.5],refinement_steps=[.125,.0625,.03125],
        model=DEFAULT_MODEL,revision=DEFAULT_REVISION,precision='CPU FP32',visibility='Always visible; no threshold tuned',
        scope='No trained readout, encoder update or temporal state. Query appearance sampled from first image; no query coordinates in matching function.',
        limits='8 previously inspected rendered validation scenes; no held-out test. Bilinear features add no image observations. Similarity improvement is not a correctness guarantee.')
    plan=dict(config=config,cache=str(directory),cache_summary_sha256=digest(directory/'summary.json'),
        reference_run=str(reference),reference_plan_sha256=digest(reference/'plan.json'),
        control_run=str(control),control_summary_sha256=digest(control/'summary.json'),
        sources={str(p):digest(p) for p in (Path(__file__),Path('exploration/continuous_matching.py'),
            Path('exploration/oracle_transport.py'),Path('exploration/readout_diagnostic.py'),Path('tapvid_metrics.py'),
            Path('temporal_siglip.py'),Path('pyproject.toml'),Path('uv.lock'))})
    if args.resume:
        if json.loads((output/'plan.json').read_text())!=plan or (output/'progress.jsonl').exists():
            raise ValueError('Changed plan or already-started evaluation')
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.plan_only:
        print('Fixed methods, depths, query selection and search budget before results',flush=True); return
    backbone=SiglipVisionModel.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True).eval()
    norm=backbone.vision_model.post_layernorm; norm_hash=parameter_digest(norm); del backbone
    norm.requires_grad_(False)
    old_controls=json.loads((control/'summary.json').read_text())
    rows=[]; predictions={}; started=time.monotonic()
    for scene in config['val']:
        path=directory/(scene+'.pt'); assert digest(path)==receipt['scenes'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        selected=select_queries(data,scene,config['evaluation_queries'],config['seed'])
        truth,visible=data['points'][selected],data['visible'][selected]
        old_path=control/(scene+'.npz'); assert digest(old_path)==old_controls['predictions_sha256'][scene]
        with np.load(old_path) as previous:
            assert np.array_equal(selected.numpy(),previous['selected'])
            assert np.array_equal(truth.numpy(),previous['truth'])
            for depth in config['depths']:
                features=data['features'][depth] if depth=='4' else norm(data['features'][depth])
                anchor=sample(features[0],truth[:,0],config['grid'])
                coordinates={name:[] for name in config['methods'] if name!='static_query'}
                similarities={name:[] for name in coordinates}
                for frame in range(len(features)):
                    values=continuous_match(anchor,features[frame],config['grid'],depth=='12',
                        config['topk'],config['refinement_steps'])
                    for method,result in values.items():
                        coordinates[method].append(result['points'].numpy())
                        similarities[method].append(result['scores'].numpy())
                arrays={k:np.stack(v,axis=1) for k,v in coordinates.items()}
                arrays['static_query']=np.broadcast_to(truth[:,:1].numpy(),truth.shape).copy()
                previous_name='depth4_cosine' if depth=='4' else 'depth12_postnorm_inputnorm_cosine'
                assert np.array_equal(arrays['center'][:,1:],previous[previous_name]),(scene,depth)
                saved=dict(target_xy=truth.numpy(),visible=visible.numpy(),selected=selected.numpy(),**arrays)
                saved.update({name+'_score':np.stack(v,axis=1) for name,v in similarities.items()})
                destination=output/(scene+'_'+depth+'.npz'); np.savez_compressed(destination,**saved)
                predictions[destination.name]=digest(destination)
                rows.extend(score_scene(scene,int(depth),truth.numpy(),visible.numpy(),arrays))
                event=dict(scene=scene,depth=int(depth),elapsed_seconds=time.monotonic()-started)
                with (output/'progress.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
    result=dict(plan_sha256=digest(output/'plan.json'),post_layernorm_sha256=norm_hash,
        predictions_sha256=predictions,center_exactly_matches_previous_predictions=True,
        elapsed_seconds=time.monotonic()-started,rows=rows,summary=aggregate(rows))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__': main()
