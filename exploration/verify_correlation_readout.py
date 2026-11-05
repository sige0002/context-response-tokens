"""Recount all saved correlation-probe curves with independent NumPy formulas."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration.oracle_transport import digest
from exploration.readout_diagnostic import select_queries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    run=Path(args.run)
    report=json.loads((run/'summary.json').read_text())
    plan=json.loads((run/'plan.json').read_text()); config=plan['config']
    assert digest(run/'plan.json')==report['plan_sha256']
    for source,expected in plan['sources'].items(): assert digest(source)==expected,source
    cache=Path(plan['cache'])
    cache_summary=json.loads((cache/'summary.json').read_text())
    assert digest(cache/'summary.json')==plan['cache_summary_sha256']
    original={}
    for scene in config['val']:
        path=cache/(scene+'.pt')
        assert digest(path)==cache_summary['scenes'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        selected=select_queries(data,scene,config['evaluation_queries'],config['seed'])
        original[scene]=(data['points'][selected].numpy(),data['visible'][selected].numpy(),selected.numpy())
    counted=files=0; max_pck=max_error=max_aj=0.
    initial_hashes=set()
    for arm,result in report['results'].items():
        assert digest(run/(arm+'.pt'))==result['checkpoint_sha256']
        initial_hashes.add(result['initial_parameters_sha256'])
        weights=torch.load(run/(arm+'.pt'),map_location='cpu',weights_only=True)['state_dict']
        if arm=='local_only':
            assert not torch.count_nonzero(weights['metric.2.weight'])
            assert not torch.count_nonzero(weights['metric.2.bias'])
        else: assert weights['metric.2.weight'].norm()>0
        for step,score in result['scores'].items():
            by_scene={}
            for scene,expected in score['predictions_sha256'].items():
                path=run/arm/step/(scene+'.npz')
                assert digest(path)==expected,path
                with np.load(path) as values: data={k:values[k] for k in values.files}
                truth,visible,selected=original[scene]
                assert np.array_equal(data['target_xy'],truth)
                assert np.array_equal(data['visible'],visible)
                assert np.array_equal(data['selected'],selected)
                assert np.array_equal(data['prediction'][:,0],truth[:,0])
                by_scene[scene]=data; files+=1
            for row in score['rows']:
                data=by_scene[row['scene']]
                truth,visible=data['target_xy'],data['visible']
                model=row['method']=='model'
                prediction=data['prediction'] if model else np.broadcast_to(truth[:,:1],truth.shape)
                pred_visible=data['visibility_logits']>=0 if model else np.ones_like(visible)
                squared=((prediction-truth)*256)**2
                squared=squared.sum(-1)[:,1:]
                dist=np.sqrt(squared)
                movement=np.linalg.norm((truth[:,1:]-truth[:,:1])*256,axis=-1)
                mask=visible[:,1:].copy()
                if row['stratum']=='moving_gt2': mask &= movement>2
                if row['stratum']=='moving_gt16': mask &= movement>16
                errors=dist[mask]
                assert len(errors)==row['points']
                for q in (4,8,16):
                    difference=abs(float((errors<q).mean())-row[f'pck_{q}'])
                    assert difference<1e-7,(arm,step,row,difference)
                    max_pck=max(max_pck,difference)
                difference=abs(float(errors.mean())-row['mean_error'])
                assert difference<2e-5,(arm,step,row,difference)
                max_error=max(max_error,difference)
                if row['stratum']=='all_visible':
                    gt,pv=visible[:,1:],pred_visible[:,1:]
                    js=[]; within=[]
                    for q in (1,2,4,8,16):
                        correct=(squared<q*q)&gt
                        tp=(correct&pv).sum()
                        fp=(pv&~correct).sum()
                        js.append(float(tp/(gt.sum()+fp)))
                        within.append(float(correct.sum()/gt.sum()))
                    metrics=row['full_sequence_metrics']
                    difference=abs(float(np.mean(js))-metrics['average_jaccard'])
                    assert difference<1e-12,(arm,step,row,difference)
                    max_aj=max(max_aj,difference)
                    assert abs(float(np.mean(within))-metrics['average_pts_within_thresh'])<1e-12
                    assert abs(float((gt==pv).mean())-metrics['occlusion_accuracy'])<1e-12
                counted+=1
    assert len(initial_hashes)==1
    checks=dict(status='passed',saved_prediction_files=files,recounted_rows=counted,
        max_pck_difference=max_pck,max_mean_error_difference=max_error,max_aj_difference=max_aj,
        all_ground_truth_and_queries_match_original_cache=True,initial_parameters_identical=True,
        fixed_metric_stayed_unchanged=True,learned_metric_updated=True,
        scope='All saved predictions and metric curves rescored; no second model inference')
    (run/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks),flush=True)


if __name__=='__main__': main()
