import torch
import numpy as np

from exploration.analyze_correlation_horizon import occlusion_metrics
from exploration.correlation_horizon import make_schedule


def test_schedule_pairs_are_causal_and_class_weights_balance_training_counts():
    visible=torch.ones(64,24,dtype=torch.bool)
    visible[::2,8:16]=False
    visible[1::3,18:]=False
    cache={'0000':dict(points=torch.rand(64,24,2),visible=visible)}
    config=dict(seed=17,steps=128,train=['0000'],clip_frames=4,queries=16)
    schedule,coverage=make_schedule(config,cache)
    again,coverage_again=make_schedule(config,cache)
    assert schedule==again and coverage==coverage_again
    assert any(max(row['span'])-row['query_frame']>3 for row in schedule)
    for row in schedule:
        q=row['query_frame']
        assert visible[row['queries'],q].all()
        assert row['window']==[q+1,q+2,q+3]
        assert len(set(row['span']))==3
        assert all(q<frame<24 for frame in row['span'])
    for stats in coverage.values():
        assert stats['total']==128*16*3
        assert sum(stats['gaps'].values())==stats['total']
        for name in ('visible','invisible'):
            assert abs(stats[name]*stats['class_weights'][name]-stats['total']/2)<1e-8


def test_occlusion_scores_agree_with_explicit_pair_ranking_and_threshold_counts():
    visible=np.array([True,False,True,False,False,True])
    logits=np.array([2.,-1.,0.,0.,3.,-2.])
    actual=occlusion_metrics(visible,logits)
    positive=-logits[~visible]; negative=-logits[visible]
    reference=((positive[:,None]>negative).astype(float)+.5*(positive[:,None]==negative)).mean()
    assert actual['auroc']==reference
    assert (actual['tp'],actual['fp'],actual['fn'],actual['tn'])==(1,1,2,2)
    assert actual['recall']==1/3 and actual['precision']==.5
    assert occlusion_metrics(visible,np.ones(6))['auroc']==.5
    assert occlusion_metrics(np.ones(6,dtype=bool),logits)['auroc'] is None
