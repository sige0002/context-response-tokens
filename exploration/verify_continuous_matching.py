"""Check raw predictions against annotations and independently recount all rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration.oracle_transport import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    plan=json.loads((run/'plan.json').read_text()); config=plan['config']
    report=json.loads((run/'summary.json').read_text())
    assert digest(run/'plan.json')==report['plan_sha256']
    for source,expected in plan['sources'].items(): assert digest(source)==expected,source
    cache=Path(plan['cache']); receipt=json.loads((cache/'summary.json').read_text())
    assert digest(cache/'summary.json')==plan['cache_summary_sha256']
    controls=Path(plan['control_run']); control=json.loads((controls/'summary.json').read_text())
    assert digest(controls/'summary.json')==plan['control_summary_sha256']
    checked=0; max_error_difference=0.; data_by_key={}
    for scene in config['val']:
        path=cache/(scene+'.pt'); assert digest(path)==receipt['scenes'][scene]['sha256']
        original=torch.load(path,map_location='cpu',weights_only=True)
        candidates=torch.where(original['visible'][:,0])[0]
        rng=torch.Generator().manual_seed(config['seed']+int(scene))
        selected=candidates[torch.randperm(len(candidates),generator=rng)[:config['evaluation_queries']]]
        truth=original['points'][selected].numpy(); visible=original['visible'][selected].numpy()
        control_path=controls/(scene+'.npz')
        assert digest(control_path)==control['predictions_sha256'][scene]
        with np.load(control_path) as old:
            for depth in config['depths']:
                name=scene+'_'+depth+'.npz'; assert digest(run/name)==report['predictions_sha256'][name]
                with np.load(run/name) as values: data={key:values[key] for key in values.files}
                assert np.array_equal(data['target_xy'],truth)
                assert np.array_equal(data['visible'],visible)
                assert np.array_equal(data['selected'],selected.numpy())
                previous_name='depth4_cosine' if depth=='4' else 'depth12_postnorm_inputnorm_cosine'
                assert np.array_equal(data['center'][:,1:],old[previous_name])
                assert np.array_equal(data['static_query'],np.broadcast_to(truth[:,:1],truth.shape))
                for method in config['methods']:
                    assert data[method].shape==truth.shape
                    assert np.isfinite(data[method]).all() and (data[method]>=0).all() and (data[method]<=1).all()
                for method in ('field_top1','field_top4'):
                    assert np.isfinite(data[method+'_score']).all()
                    assert (data[method+'_score']+2e-5>=data['center_score']).all()
                    assert (np.abs(data[method+'_score'])<=1+2e-5).all()
                data_by_key[(scene,int(depth))]=data
    for row in report['rows']:
        data=data_by_key[(row['scene'],row['depth'])]
        truth,visible=data['target_xy'],data['visible']; prediction=data[row['method']]
        squared=(((prediction-truth)*256)**2).sum(-1)
        if row['task']=='identity':
            values=np.sqrt(squared[:,0])
        else:
            movement=np.sqrt((((truth[:,1:]-truth[:,:1])*256)**2).sum(-1))
            mask=visible[:,1:].copy()
            if row['stratum']=='moving_gt2': mask &= movement>2
            if row['stratum']=='moving_gt16': mask &= movement>16
            values=np.sqrt(squared[:,1:])[mask]
        assert len(values)==row['points']
        for threshold in (1,2,4,8,16):
            assert float((values<threshold).mean())==row[f'pck_{threshold}']
        difference=abs(float(values.mean())-row['mean_error'])
        assert difference<2e-5
        max_error_difference=max(max_error_difference,difference)
        metrics=row['full_sequence_metrics']
        if metrics is not None:
            gt=visible[:,1:]; total_visible=int(gt.sum()); total=gt.size; js=[]; pcks=[]
            for threshold in (1,2,4,8,16):
                correct=int(((squared[:,1:]<threshold*threshold)&gt).sum())
                # Every prediction is visible. False positives include visible
                # points outside the positional threshold and occluded points.
                jaccard=correct/(total_visible+total-correct)
                pck=correct/total_visible; js.append(jaccard); pcks.append(pck)
                assert abs(jaccard-metrics[f'jaccard_{threshold}'])<1e-12
                assert abs(pck-metrics[f'pts_within_{threshold}'])<1e-12
            assert abs(float(np.mean(js))-metrics['average_jaccard'])<1e-12
            assert abs(float(np.mean(pcks))-metrics['average_pts_within_thresh'])<1e-12
            assert abs(total_visible/total-metrics['occlusion_accuracy'])<1e-12
        checked+=1
    for summary in report['summary']:
        keys=('depth','method','task','stratum')
        rows=[r for r in report['rows'] if all(r[k]==summary[k] for k in keys) and r['points']]
        assert len(rows)==summary['scenes'] and sum(r['points'] for r in rows)==summary['points']
        for key in ('mean_error','pck_1','pck_2','pck_4','pck_8','pck_16'):
            assert float(np.mean([r[key] for r in rows]))==summary[key]
        if summary['full_sequence_metrics'] is not None:
            for key,expected in summary['full_sequence_metrics'].items():
                assert float(np.mean([r['full_sequence_metrics'][key] for r in rows]))==expected
    result=dict(status='passed',source_sha256=digest(__file__),summary_sha256=digest(run/'summary.json'),
        prediction_files=len(data_by_key),recounted_rows=checked,max_mean_error_difference=max_error_difference,
        original_ground_truth_queries_and_old_coarse_predictions_match=True,
        coordinates_finite_and_bounded=True,refinement_scores_never_below_coarse_tolerance=2e-5,
        all_scene_averages_match=True,scope='Saved predictions rescored. Does not rerun inference or certify a tracking gain from similarity improvement.')
    (run/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
