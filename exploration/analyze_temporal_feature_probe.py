"""Post-result history bands and GT-correspondence descriptor diagnostics.

Ground truth enters scoring only. No predictions are changed or selected.
The first 7 future samples fit the training clip's frame count, but do not
match its original-frame spacing or its SSv2 image distribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration.continuous_matching import descriptor
from exploration.oracle_transport import digest,sample


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    torch.set_num_threads(2)
    plan=json.loads((run/'plan.json').read_text()); config=plan['config']
    result=json.loads((run/'summary.json').read_text())
    assert digest(run/'plan.json')==result['plan_sha256']
    bands={'all_1_23':(1,24),'first_1_7':(1,8),'middle_8_15':(8,16),'last_16_23':(16,24)}
    rows=[]
    for scene in config['val']:
        feature_path=run/(scene+'_features.pt'); assert digest(feature_path)==result['features_sha256'][scene]
        features=torch.load(feature_path,map_location='cpu',weights_only=True)['features']
        for condition in config['conditions']:
            for depth in ('4','12'):
                name=scene+'_'+condition+'_'+depth+'.npz'; assert digest(run/name)==result['predictions_sha256'][name]
                with np.load(run/name) as raw: data={k:raw[k] for k in raw.files}
                truth,visible=data['target_xy'],data['visible']; current=features[condition][depth]
                assert len(current)==24
                xy=torch.from_numpy(truth)
                query=descriptor(sample(current[0],xy[:,0],14),depth=='12')
                correct_similarity=torch.stack([(query*descriptor(sample(current[f],xy[:,f],14),depth=='12')).sum(-1)
                    for f in range(24)],1).numpy()
                movement=np.linalg.norm((truth-truth[:,:1])*256,axis=-1)
                for method in config['methods']:
                    squared=(((data[method]-truth)*256)**2).sum(-1)
                    for band,(start,end) in bands.items():
                        gt=visible[:,start:end]; dist=np.sqrt(squared[:,start:end])
                        for stratum,mask in dict(all_visible=gt,
                                moving_gt2=gt&(movement[:,start:end]>2),moving_gt16=gt&(movement[:,start:end]>16)).items():
                            errors=dist[mask]; count=len(errors)
                            scores=correct_similarity[:,start:end][mask]
                            aj=None
                            if stratum=='all_visible':
                                n=int(gt.sum()); total=gt.size; js=[]
                                for threshold in (1,2,4,8,16):
                                    correct=int(((squared[:,start:end]<threshold**2)&gt).sum())
                                    js.append(correct/(n+total-correct))
                                aj=float(np.mean(js))
                            row=dict(scene=scene,condition=condition,depth=int(depth),method=method,band=band,
                                stratum=stratum,points=count,mean_error=float(errors.mean()) if count else None,
                                true_correspondence_cosine=float(scores.mean()) if count else None,
                                average_jaccard=aj,
                                **{f'pck_{q}':float((errors<q).mean()) if count else None for q in (1,2,4,8,16)})
                            if band=='all_1_23':
                                old=next(r for r in result['rows'] if r['scene']==scene and r['condition']==condition
                                    and r['depth']==int(depth) and r['method']==method and r['task']=='motion' and r['stratum']==stratum)
                                for key in ('points','mean_error','pck_1','pck_2','pck_4','pck_8','pck_16'): assert row[key]==old[key]
                                if aj is not None: assert abs(aj-old['full_sequence_metrics']['average_jaccard'])<1e-12
                            rows.append(row)
    summary=[]; keys=('condition','depth','method','band','stratum')
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected=[r for r in rows if tuple(r[k] for k in keys)==group and r['points']]
        entry=dict(zip(keys,group)); entry.update(scenes=len(selected),points=sum(r['points'] for r in selected))
        for metric in ('mean_error','pck_1','pck_2','pck_4','pck_8','pck_16','true_correspondence_cosine','average_jaccard'):
            values=[r[metric] for r in selected if r[metric] is not None]
            entry[metric]=float(np.mean(values)) if values else None
        summary.append(entry)
    output=dict(summary_sha256=digest(run/'summary.json'),source_sha256=digest(__file__),bands=bands,
        post_result_analysis=True,all_band_reproduces_original_motion_rows=True,rows=rows,summary=summary,
        limits='Post-result descriptive bands; not an independent test or multiple-comparison-adjusted inference. GT coordinates used only for descriptor analysis, never for the saved predictions. Reset retains the initial query descriptor.')
    (run/'history_bands.json').write_text(json.dumps(output,indent=2)+'\n')
    for row in summary:
        if row['method']=='field_top4' and row['depth']==12 and row['stratum']=='all_visible': print(json.dumps(row),flush=True)


if __name__=='__main__': main()
