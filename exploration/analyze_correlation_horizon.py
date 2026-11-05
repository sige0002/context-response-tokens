"""Audit paired schedules and report occlusion counts without threshold tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration.correlation_horizon import make_schedule
from exploration.oracle_transport import digest


def occlusion_metrics(visible, visibility_logits):
    """Occluded is positive; AUROC counts tied positive/negative pairs as 1/2."""
    visible=np.asarray(visible,dtype=bool).ravel()
    logits=np.asarray(visibility_logits).ravel()
    if visible.shape!=logits.shape or not np.isfinite(logits).all():
        raise ValueError('Invalid visibility arrays')
    positive=~visible; predicted=logits<0
    tp=int((positive&predicted).sum()); fp=int((visible&predicted).sum())
    fn=int((positive&~predicted).sum()); tn=int((visible&~predicted).sum())
    _,group=np.unique(-logits,return_inverse=True)
    pos=np.bincount(group,weights=positive)
    neg=np.bincount(group,weights=visible)
    wins=float((pos*(np.cumsum(neg)-neg+.5*neg)).sum())
    auc=wins/((tp+fn)*(tn+fp)) if (tp+fn)*(tn+fp) else None
    return dict(targets=len(visible),tp=tp,fp=fp,fn=fn,tn=tn,
        recall=tp/(tp+fn) if tp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None,
        false_positive_rate=fp/(fp+tn) if fp+tn else None,
        accuracy=(tp+tn)/len(visible) if len(visible) else None,
        auroc=auc)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    plan=json.loads((run/'plan.json').read_text()); config=plan['config']
    report=json.loads((run/'summary.json').read_text())
    assert digest(run/'plan.json')==report['plan_sha256']
    assert digest(run/'schedule.json')==plan['schedule_sha256']
    for source,expected in plan['sources'].items(): assert digest(source)==expected,source
    cache=Path(plan['cache']); receipt=json.loads((cache/'summary.json').read_text())
    assert digest(cache/'summary.json')==plan['cache_summary_sha256']
    training={}
    for scene in config['train']:
        path=cache/(scene+'.pt'); assert digest(path)==receipt['scenes'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        training[scene]=dict(points=data['points'],visible=data['visible'])
    schedule,coverage=make_schedule(config,training)
    assert schedule==json.loads((run/'schedule.json').read_text())
    assert json.loads(json.dumps(coverage))==plan['training_coverage']
    reference=Path(plan['reference_run'])
    assert digest(reference/'summary.json')==plan['reference_result_sha256']
    assert digest(reference/'plan.json')==plan['reference_plan_sha256']
    previous=torch.load(reference/'metric_and_local.pt',map_location='cpu',weights_only=True)['state_dict']
    current=torch.load(run/'window_unweighted.pt',map_location='cpu',weights_only=True)['state_dict']
    assert previous.keys()==current.keys()
    assert all(torch.equal(current[k],v) for k,v in previous.items())
    old=json.loads((reference/'summary.json').read_text())['results']['metric_and_local']
    for step,score in report['results']['window_unweighted']['scores'].items():
        assert score['rows']==old['scores'][step]['rows']
        assert score['summary']==old['scores'][step]['summary']
    rows=[]
    for arm,result in report['results'].items():
        for step,score in result['scores'].items():
            visibles=[]; logits=[]; scenes={}
            for scene,expected in score['predictions_sha256'].items():
                path=run/arm/step/(scene+'.npz'); assert digest(path)==expected
                with np.load(path) as data:
                    visible=data['visible'][:,1:]; values=data['visibility_logits'][:,1:]
                scenes[scene]=occlusion_metrics(visible,values)
                visibles.append(visible.ravel()); logits.append(values.ravel())
            aucs=[m['auroc'] for m in scenes.values() if m['auroc'] is not None]
            rows.append(dict(arm=arm,step=int(step),
                pooled=occlusion_metrics(np.concatenate(visibles),np.concatenate(logits)),
                mean_scene_auroc=float(np.mean(aucs)) if aucs else None,
                scenes_with_both_classes=len(aucs),by_scene=scenes))
    output=dict(plan_sha256=digest(run/'plan.json'),summary_sha256=digest(run/'summary.json'),
        sources={str(p):digest(p) for p in (Path(__file__),Path('exploration/correlation_horizon.py'))},
        schedule_regenerated_exactly=True,reference_weights_and_all_scores_match=True,
        training_coverage=coverage,visibility_threshold=0,occluded_is_positive=True,
        rows=rows,scope='Saved predictions only, query frame excluded. No threshold selection or new inference. Pooled counts and AUROC are distinguished from scene-mean AJ/PCK.')
    (run/'occlusion_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    for row in rows:
        if row['step']==config['steps']:
            print(json.dumps({k:v for k,v in row.items() if k!='by_scene'}),flush=True)


if __name__=='__main__': main()
