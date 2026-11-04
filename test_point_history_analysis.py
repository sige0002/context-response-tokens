import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from analyze_point_history import strata_masks,masked_metrics,aggregate,verify_reports
from point_checkpoint import file_hash
from tapvid_metrics import compute_tapvid_metrics


def test_age_bins_partition_post_query_frames_and_gap_counts_restart_at_visibility():
    occluded=np.zeros((2,40),dtype=bool)
    occluded[0,3:6]=True;occluded[0,8:18]=True;occluded[1,7:35]=True
    first=np.array([2,5]);masks=strata_masks(first,occluded)
    ages=[v for k,v in masks.items() if k.startswith('age_')]
    np.testing.assert_array_equal(sum(ages),masks['all_post_query'].astype(int))
    assert np.argwhere(masks['reappearance_gap_1_7']).tolist()==[[0,6]]
    assert np.argwhere(masks['reappearance_gap_8_23']).tolist()==[[0,18]]
    assert np.argwhere(masks['reappearance_gap_24_plus']).tolist()==[[1,35]]
    # A visible frame after reappearance is not counted repeatedly.
    assert not any(mask[0,7] for key,mask in masks.items() if key.startswith('reappearance'))
    assert all(not mask[0,:3].any() for mask in masks.values())


def test_unrestricted_counts_match_pinned_official_tap_metric():
    rng=np.random.default_rng(41)
    target=rng.uniform(.1,.9,(6,150,2)).astype('float32')
    prediction=target+rng.normal(0,.025,target.shape).astype('float32')
    occluded=rng.random((6,150))<.35;logits=rng.normal(size=(6,150)).astype('float32')
    first=np.array([0,3,11,4,8,23]);xy=target[np.arange(6),first]
    query=np.stack((first,xy[:,1]*256,xy[:,0]*256),-1)
    expected=compute_tapvid_metrics(query[None],occluded[None],target[None]*256,
        (logits<0)[None],prediction[None]*256,'first')
    actual=masked_metrics(target,occluded,prediction,logits,strata_masks(first,occluded)['all_post_query'])
    for key,value in actual['metrics'].items():assert value==expected[key][0]


def test_empty_or_all_occluded_strata_are_undefined_independent_of_predictions():
    xy=np.zeros((1,3,2));occ=np.ones((1,3),dtype=bool)
    for logits in (np.ones((1,3)),-np.ones((1,3))):
        for mask in (np.ones((1,3),dtype=bool),np.zeros((1,3),dtype=bool)):
            assert masked_metrics(xy,occ,xy,logits,mask)['metrics'] is None


def test_strict_threshold_and_visibility_false_positives_are_scored():
    target=np.zeros((1,3,2),dtype=np.float64)
    predicted=target.copy();predicted[0,0,0]=1/256;predicted[0,1,0]=20/256
    occluded=np.array([[False,False,True]]);logits=np.ones((1,3));mask=np.ones((1,3),dtype=bool)
    result=masked_metrics(target,occluded,predicted,logits,mask)
    # One GT-visible point is correct only for 2/4/8/16, the other is always
    # wrong; the occluded point is a false positive at every threshold.
    assert result['metrics']['average_pts_within_thresh']==.4
    assert result['metrics']['average_jaccard']==.2
    assert result['metrics']['occlusion_accuracy']==2/3


def test_paired_aggregation_preserves_target_cohort_and_does_not_invent_interval():
    value=dict(counts=dict(evaluated_positions=2,visible_positions=1,occluded_positions=1),
        metrics=dict(average_jaccard=.4,average_pts_within_thresh=.5,occlusion_accuracy=.6))
    row=dict(id='one',video_sha256='rgb',strata={'all':{name:copy.deepcopy(value) for name in ('full','reset_1','reset_24')}})
    row['strata']['all']['reset_1']['metrics']['average_jaccard']=.2
    report=aggregate([row])['all']
    assert report['full_minus_reset']['reset_1']['average_jaccard']==dict(difference=.2,interval=None)
    bad=copy.deepcopy(row);bad['strata']['all']['reset_1']['counts']['visible_positions']=0
    with pytest.raises(ValueError,match='support'):aggregate([bad])


def test_real_intervention_normal_branch_matches_separately_evaluated_checkpoint():
    root='results/point_history_corrected_cli_integration'
    baseline='results/point_online_integration_rgbfix_fp32/mamba/summary.json'
    with pytest.raises(ValueError,match='allow-subset'):verify_reports(root,baseline)
    _,evidence=verify_reports(root,baseline,True)
    assert evidence['normal_branch_max_xy']==0
    assert evidence['normal_branch_max_visibility_logit']==0


@pytest.mark.parametrize('fault',['partial','reset_schedule','source'])
def test_mismatched_interventions_rejected_even_with_updated_artifact_hash(tmp_path,fault):
    source=Path('results/point_history_corrected_cli_integration')
    root=tmp_path/'interventions';shutil.copytree(source,root)
    report_path=root/'reset_24/summary.json';report=json.loads(report_path.read_text())
    if fault=='partial':report['rows'].clear()
    elif fault=='reset_schedule':report['rows'][0]['reset_frames']=[]
    else:report['source_sha256']['point_history.py']='other'
    report_path.write_text(json.dumps(report))
    summary=json.loads((root/'summary.json').read_text())
    summary['conditions']['reset_24']['report_sha256']=file_hash(report_path)
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):verify_reports(root,'results/point_online_integration_rgbfix_fp32/mamba/summary.json',True)
