from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from evaluate_point_tracks import track_video
from point_history import clear_encoder_memory, track_history_interventions
from point_readout import PointReadout, PointTracker
from temporal_siglip import StreamState, TemporalSiglip


class PixelProcessor:
    def __call__(self, images, return_tensors):
        image = torch.from_numpy(np.asarray(images).copy()).permute(2,0,1).float()/127.5-1.
        return SimpleNamespace(pixel_values=image[None])


def tracker(kind):
    torch.set_num_threads(2)
    torch.manual_seed(29)
    config = SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,image_size=32,patch_size=16)
    encoder = TemporalSiglip(SiglipVisionModel(config),kind=kind,every=1,width=32)
    with torch.no_grad():
        for adapter in encoder.adapters.values():
            adapter.gain.fill_(.2)
    return PointTracker(encoder,PointReadout(input_dim=32,grid_size=2,dim=16,heads=2,frequencies=3,hidden=12)).eval()


def fixture():
    video = np.random.default_rng(4).integers(0,256,size=(7,32,32,3),dtype=np.uint8)
    xy = np.array([[.3,.4],[.7,.6],[.5,.2]],dtype=np.float32)
    return video,xy,np.array([2,0,4]),PixelProcessor(),torch.device('cpu')


def test_reset_clears_cache_preserves_clock_and_does_not_mutate_original_state():
    original = StreamState(owner=7,batch=1,image_shape=(3,32,32),offset=129,layers={'0':(torch.ones(2),)})
    cleared = clear_encoder_memory(None,original)
    assert cleared is not original and cleared.offset == 129 and cleared.layers == {}
    assert original.layers['0'][0].sum() == 2
    with pytest.raises(TypeError,match='Unsupported'):
        clear_encoder_memory(None,SimpleNamespace(offset=3))


def test_full_branch_matches_standard_tracking_and_reset_branches_preserve_queries_and_clock():
    model = tracker('attention')
    video,xy,first,processor,device = fixture()
    reference_xy,reference_logits,_ = track_video(model,video,xy,first,processor,device)
    result = track_history_interventions(model,video,xy,first,processor,device,(1,3,100))
    np.testing.assert_array_equal(result['full']['predicted_xy'],reference_xy)
    np.testing.assert_array_equal(result['full']['visibility_logits'],reference_logits)
    # No reset inside the video must reproduce the full path exactly.
    np.testing.assert_array_equal(result['reset_100']['predicted_xy'],reference_xy)
    assert result['reset_3']['resets'] == [3,6]
    for values in result.values():
        assert [row['absolute_frames'] for row in values['state_bytes']] == list(range(1,8))
        for q,t in enumerate(first):
            np.testing.assert_array_equal(values['predicted_xy'][q,t],xy[q])
            assert (values['predicted_xy'][q,:t] == 0).all()
            assert (values['visibility_logits'][q,:t] == -30).all()
        assert [v['queries'] for v in values['state_bytes']] == [v['queries'] for v in result['full']['state_bytes']]
    sizes = [v['encoder'] for v in result['reset_1']['state_bytes']]
    assert len(set(sizes)) == 1
    assert result['full']['state_bytes'][-1]['encoder'] == 7*sizes[0]
    assert np.max(np.abs(result['reset_1']['predicted_xy']-reference_xy)) > 1e-6
    assert result['full']['state_bytes'][2]['queries'] > result['full']['state_bytes'][1]['queries']


def test_future_pixels_cannot_change_any_branch_past_outputs_or_query_anchors():
    model = tracker('attention')
    video,xy,first,processor,device = fixture()
    a = track_history_interventions(model,video,xy,first,processor,device,(1,3))
    changed = video.copy()
    changed[5:] = 255-changed[5:]
    b = track_history_interventions(model,changed,xy,first,processor,device,(1,3))
    for name in a:
        np.testing.assert_array_equal(a[name]['predicted_xy'][:,:5],b[name]['predicted_xy'][:,:5])
        np.testing.assert_array_equal(a[name]['visibility_logits'][:,:5],b[name]['visibility_logits'][:,:5])


def test_no_temporal_encoder_is_an_exact_negative_control_for_cache_interventions():
    result = track_history_interventions(tracker('none'),*fixture(),periods=(1,3))
    for name in ('reset_1','reset_3'):
        np.testing.assert_array_equal(result['full']['predicted_xy'],result[name]['predicted_xy'])
        np.testing.assert_array_equal(result['full']['visibility_logits'],result[name]['visibility_logits'])
        assert all(row['encoder'] == 0 for row in result[name]['state_bytes'])


@pytest.mark.parametrize('config,step,limit,allowed',[
    ({'development_run':True,'steps':3},3,0,False),
    ({'development_run':False,'steps':3},2,0,False),
    ({'development_run':False,'steps':3},3,0,True),
    ({'development_run':True,'steps':3},3,1,True),
])
def test_intervention_cli_requires_finished_full_training_unless_explicit_subset(tmp_path,monkeypatch,config,step,limit,allowed):
    import evaluate_point_history as evaluation
    config = dict(config, reproducibility=evaluation.REPRODUCIBILITY)
    class ReachedData(Exception):
        pass
    def processor(*a,**kw):
        raise ReachedData
    monkeypatch.setattr('sys.argv',['evaluate_point_history.py','--checkpoint','unused.pt','--output',str(tmp_path/'run'),'--limit',str(limit)])
    monkeypatch.setattr(evaluation,'load_point_checkpoint',lambda *a,**kw:(None,{'metadata':{'config':config},'training_state':{'step':step}}))
    monkeypatch.setattr(evaluation.SiglipImageProcessor,'from_pretrained',processor)
    monkeypatch.setattr(evaluation.torch,'use_deterministic_algorithms',lambda *a:None)
    monkeypatch.setattr(evaluation.torch,'set_num_threads',lambda *a:None)
    monkeypatch.setattr(evaluation.torch.backends.cudnn,'benchmark',False)
    monkeypatch.setattr(evaluation.torch.backends.cudnn,'deterministic',True)
    with pytest.raises(ReachedData if allowed else ValueError):
        evaluation.main()
