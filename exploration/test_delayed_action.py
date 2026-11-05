import numpy as np
import pytest
import torch

from exploration.delayed_action_data import select_blocks,verify_plan
from exploration.delayed_action_probe import evaluate
from exploration.cache_delayed_action import stream_features
from test_temporal_siglip import make,activate


def fixture():
    plan=dict(classes=2,frames_per_clip=8,delays=[0,1,8],ridge_alpha=1.,
              conditions=['full_memory','suffix_only','saved_evidence'],splits={})
    arrays={}
    for split in ('train','validation'):
        rows=[dict(id=f'{split}-{i}',group=f'{split}-{i}',sha256=f'{split}-{i}',label=i%2)
              for i in range(20)]
        plan['splits'][split]=select_blocks(rows,2,2,1,71)
        labels=np.array([0,1,0,1]); base=np.eye(2)[labels]
        arrays[split+'_labels']=labels
        arrays[split+'_blocks']=np.array([0,0,1,1])
        arrays[split+'_ids']=np.array([r['id'] for b in plan['splits'][split] for r in b['evidence']])
        arrays[split+'_full_memory']=np.repeat(base[:,None],3,axis=1)
        arrays[split+'_saved_evidence']=arrays[split+'_full_memory'].copy()
        arrays[split+'_suffix_only']=np.zeros((4,2,2))
    return plan,arrays


def test_common_suffix_has_exact_chance_but_retained_evidence_decodes():
    plan,arrays=fixture()
    result,_=evaluate(arrays,plan)
    assert all(r['accuracy']==.5 for r in result['suffix_only']['validation'])
    assert all(r['accuracy']==1 for r in result['full_memory']['validation'])
    arrays['validation_full_memory'][:,-1]=0
    result,_=evaluate(arrays,plan)
    assert result['full_memory']['validation'][-1]['accuracy']==.5
    assert result['saved_evidence']['validation'][-1]['accuracy']==1


def test_validation_features_cannot_change_fitted_probe():
    plan,arrays=fixture()
    _,before=evaluate(arrays,plan)
    arrays['validation_full_memory'] *= -7
    arrays['validation_saved_evidence'] *= -7
    _,after=evaluate(arrays,plan)
    assert all(np.array_equal(before[k],after[k]) for k in before)


def test_partial_or_target_dependent_suffix_rejected():
    plan,arrays=fixture()
    arrays['validation_suffix_only'][0,0,0]=1
    with pytest.raises(ValueError,match='leak'):
        evaluate(arrays,plan)
    plan,arrays=fixture()
    arrays['validation_full_memory']=arrays['validation_full_memory'][:2]
    with pytest.raises(ValueError,match='population'):
        evaluate(arrays,plan)


def test_balanced_but_misaligned_evidence_labels_rejected():
    plan,arrays=fixture()
    arrays['validation_labels']=1-arrays['validation_labels']
    with pytest.raises(ValueError,match='aligned'):
        evaluate(arrays,plan)


def test_plan_rejects_cross_split_sources_and_unbalanced_suffixes():
    plan,_=fixture()
    verify_plan(plan)
    plan['splits']['validation'][0]['evidence'][0]['sha256']=plan['splits']['train'][0]['evidence'][0]['sha256']
    with pytest.raises(ValueError,match='leakage'):
        verify_plan(plan)
    plan,_=fixture()
    plan['splits']['validation'][0]['evidence'][1]['label']=0
    with pytest.raises(ValueError,match='every class'):
        verify_plan(plan)


def test_real_attention_stream_retains_prefix_and_reset_removes_it():
    torch.set_num_threads(1)
    encoder=make('attention'); activate(encoder)
    torch.manual_seed(32)
    a=torch.randn(8,3,32,32); b=torch.randn_like(a); suffix=torch.randn(8,3,32,32)
    first,sizes=stream_features(encoder,a,suffix,[0,1,8])
    second,_=stream_features(encoder,b,suffix,[0,1,8])
    reset_a,_=stream_features(encoder,a,suffix,[0,1,8],True)
    reset_b,_=stream_features(encoder,b,suffix,[0,1,8],True)
    assert np.array_equal(reset_a,reset_b)
    assert np.linalg.norm(first[1:]-second[1:])>1e-6
    assert sizes[0]<sizes[1]<sizes[2]
    stateless=make('none')
    first,_=stream_features(stateless,a,suffix,[0,1,8])
    second,_=stream_features(stateless,b,suffix,[0,1,8])
    assert np.array_equal(first[1:],second[1:])
