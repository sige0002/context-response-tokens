"""Audit all curves and rerun every final learned readout from saved features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exploration.continuous_matching import continuous_match
from exploration.linear_correspondence_metric import LinearCorrespondenceMetric
from exploration.oracle_transport import digest,sample
from exploration.readout_diagnostic import parameter_digest,select_queries
from exploration.train_temporal_metric_probe import make_schedule


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    plan=json.loads((run/'plan.json').read_text()); config=plan['config']
    result=json.loads((run/'summary.json').read_text())
    assert digest(run/'plan.json')==result['plan_sha256']
    for source,expected in plan['sources'].items(): assert digest(source)==expected,source
    directory=Path(plan['train_cache']); receipt=json.loads((directory/'summary.json').read_text())
    assert digest(directory/'summary.json')==plan['train_cache_summary_sha256']
    assert digest(directory/'plan.json')==plan['train_cache_plan_sha256']
    train={}
    for scene in config['train']:
        path=directory/(scene+'.pt'); assert digest(path)==receipt['records'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        train[scene]=dict(points=data['points'],visible=data['visible'])
    schedule,coverage=make_schedule(train,config)
    assert schedule==json.loads((run/'schedule.json').read_text())
    assert digest(run/'schedule.json')==plan['schedule_sha256'] and coverage==plan['training_coverage']
    del train,data
    probe=Path(plan['probe_run']); old=json.loads((probe/'plan.json').read_text()); old_result=json.loads((probe/'summary.json').read_text())
    assert digest(probe/'plan.json')==plan['probe_plan_sha256'] and digest(probe/'summary.json')==plan['probe_summary_sha256']
    original=Path(old['cache']); original_receipt=json.loads((original/'summary.json').read_text())
    assert digest(original/'summary.json')==old['cache_summary_sha256']
    cache={}
    for scene in config['val']:
        path=original/(scene+'.pt'); assert digest(path)==original_receipt['scenes'][scene]['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        selected=select_queries(data,scene,config['evaluation_queries'],config['seed'])
        feature_path=probe/(scene+'_features.pt'); assert digest(feature_path)==old_result['features_sha256'][scene]
        features=torch.load(feature_path,map_location='cpu',weights_only=True)['features']
        cache[scene]=dict(points=data['points'][selected].numpy(),visible=data['visible'][selected].numpy(),
            selected=selected.numpy(),features={k:v['12'] for k,v in features.items()})
    torch.manual_seed(config['seed']); initial=LinearCorrespondenceMetric(config['dim'],config['rank'])
    initial_hash=parameter_digest(initial)
    rows=files=rerun_calls=0
    for condition,record in result['results'].items():
        assert record['initial_parameters_sha256']==initial_hash
        for step,score in record['scores'].items():
            folder=run/condition/step
            assert digest(folder/'metric.pt')==score['checkpoint_sha256']
            checkpoint=torch.load(folder/'metric.pt',map_location='cpu',weights_only=True)
            model=LinearCorrespondenceMetric(**checkpoint['config']); model.load_state_dict(checkpoint['state_dict'],strict=True); model.eval()
            if int(step)==0: assert parameter_digest(model)==initial_hash
            else: assert model.up.weight.norm()>0
            by_scene={}
            for scene,expected in score['predictions_sha256'].items():
                path=folder/(scene+'.npz'); assert digest(path)==expected
                with np.load(path) as raw: data={k:raw[k] for k in raw.files}
                truth,visible=cache[scene]['points'],cache[scene]['visible']
                assert np.array_equal(data['target_xy'],truth) and np.array_equal(data['visible'],visible)
                assert np.array_equal(data['selected'],cache[scene]['selected'])
                assert np.array_equal(data['static_query'],np.broadcast_to(truth[:,:1],truth.shape))
                if int(step)==0:
                    previous_path=probe/(scene+'_'+condition+'_12.npz')
                    assert digest(previous_path)==old_result['predictions_sha256'][previous_path.name]
                    with np.load(previous_path) as previous:
                        for method in config['methods']: assert np.array_equal(data[method],previous[method])
                if int(step)==config['steps']:
                    features=model(cache[scene]['features'][condition])
                    anchor=sample(features[0],torch.from_numpy(truth[:,0]),config['grid'])
                    for frame in range(len(features)):
                        repeated=continuous_match(anchor,features[frame],config['grid'],True,config['topk'],config['refinement_steps'])
                        for method,value in repeated.items():
                            assert np.array_equal(value['points'].numpy(),data[method][:,frame])
                            assert np.array_equal(value['scores'].numpy(),data[method+'_score'][:,frame])
                        rerun_calls+=1
                by_scene[scene]=data; files+=1
            for row in score['rows']:
                data=by_scene[row['scene']]; truth,visible=data['target_xy'],data['visible']
                squared=(((data[row['method']]-truth)*256)**2).sum(-1)
                if row['task']=='identity': errors=np.sqrt(squared[:,0])
                else:
                    movement=np.sqrt((((truth[:,1:]-truth[:,:1])*256)**2).sum(-1))
                    mask=visible[:,1:].copy()
                    if row['stratum']=='moving_gt2': mask &= movement>2
                    if row['stratum']=='moving_gt16': mask &= movement>16
                    errors=np.sqrt(squared[:,1:])[mask]
                assert len(errors)==row['points']
                assert abs(float(errors.mean())-row['mean_error'])<2e-5
                for threshold in (1,2,4,8,16): assert float((errors<threshold).mean())==row[f'pck_{threshold}']
                metrics=row['full_sequence_metrics']
                if metrics:
                    gt=visible[:,1:]; n=int(gt.sum()); total=gt.size; js=[]; pcks=[]
                    for threshold in (1,2,4,8,16):
                        correct=int(((squared[:,1:]<threshold**2)&gt).sum()); j=correct/(n+total-correct); pck=correct/n
                        assert abs(j-metrics[f'jaccard_{threshold}'])<1e-12
                        assert abs(pck-metrics[f'pts_within_{threshold}'])<1e-12
                        js.append(j); pcks.append(pck)
                    assert abs(float(np.mean(js))-metrics['average_jaccard'])<1e-12
                    assert abs(float(np.mean(pcks))-metrics['average_pts_within_thresh'])<1e-12
                    assert abs(n/total-metrics['occlusion_accuracy'])<1e-12
                rows+=1
            for summary in score['summary']:
                keys=('depth','method','task','stratum')
                values=[r for r in score['rows'] if all(r[k]==summary[k] for k in keys) and r['points']]
                assert len(values)==summary['scenes'] and sum(r['points'] for r in values)==summary['points']
                for key in ('mean_error','pck_1','pck_2','pck_4','pck_8','pck_16'):
                    assert float(np.mean([r[key] for r in values]))==summary[key]
                if summary['full_sequence_metrics']:
                    for key,value in summary['full_sequence_metrics'].items():
                        assert float(np.mean([r['full_sequence_metrics'][key] for r in values]))==value
            print(json.dumps(dict(condition=condition,step=int(step),prediction_files=files)),flush=True)
    verification=dict(status='passed',source_sha256=digest(__file__),summary_sha256=digest(run/'summary.json'),
        prediction_files=files,recounted_rows=rows,final_matching_rerun_calls=rerun_calls,
        schedule_regenerated_exactly=True,initial_weights_identical=True,initial_predictions_match_fixed_probe=True,
        trained_projections_updated=True,ground_truth_and_queries_match_original=True,
        scope='All saved prediction curves independently rescored. Final CPU readouts rerun on all validation features; GPU encoders and intermediate readout forwards are not rerun.')
    (run/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification),flush=True)


if __name__=='__main__': main()
