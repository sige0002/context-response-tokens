import json

import numpy as np
import pytest
import torch

from exploration import mimo_pilot as pilot
from exploration.run_mimo_pilot import predecessor_complete
from video_checkpoint import file_hash


def test_siso_export_is_self_contained_and_exact(tmp_path):
    from transformers import SiglipVisionConfig,SiglipVisionModel
    from temporal_siglip import TemporalSiglip
    from temporal_readout import VideoRecognizer
    torch.set_num_threads(2)
    config=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=4,image_size=32,patch_size=16)
    model=VideoRecognizer(TemporalSiglip(SiglipVisionModel(config)),7,'mean',43).eval()
    path=tmp_path/'model.pt';pilot.save_siso(model,path,dict(classes=7,kind='mean',seed=43))
    restored=pilot.load_siso(path)
    assert model.state_dict().keys()==restored.state_dict().keys()
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert not restored.training
    assert not any(p.requires_grad for p in restored.encoder.backbone.parameters())


def fixture(tmp_path,monkeypatch):
    plan=dict(steps=2,validation=[dict(label=0),dict(label=1)],limitations=[],
              environments=dict(train={'runtime':'locked'},stream={'runtime':'overlay'}))
    pilot.write(tmp_path/'plan.json',plan)
    monkeypatch.setattr(pilot,'checked_plan',lambda _:plan)
    targets=np.array([0,1])
    for arm in pilot.ARMS:
        folder=tmp_path/arm;folder.mkdir();(folder/'model.pt').write_bytes(b'test export')
        for mode in ('clip','stream'):
            initial=np.zeros((2,174));initial[:,0]=1
            final=initial.copy()
            if arm=='mimo': final[1,0]=0;final[1,1]=1
            np.savez_compressed(folder/(mode+'.npz'),initial=initial,final=final,targets=targets)
            def score(x):
                correct=int((x.argmax(-1)==targets).sum())
                return dict(correct=correct,total=2,accuracy=correct/2,
                    input_metadata_sha256='same metadata',input_pixels_sha256='same pixels')
            record=dict(plan_sha256=file_hash(tmp_path/'plan.json'),model_sha256=file_hash(folder/'model.pt'),
                initial=score(initial),final=score(final),replay=score(final),
                versions=plan['environments']['train' if mode=='clip' else 'stream'])
            record[mode+'_sha256']=file_hash(folder/(mode+'.npz'))
            if mode=='clip':
                record.update(steps=2,frozen_backbone_unchanged=True,clip_export_replay_exact=True,
                    training_metadata_sha256='same metadata',training_pixels_sha256='same pixels',
                    initial_backbone_hash='same backbone',initial_readout_hash='same readout')
            else: record['stream_export_replay_exact']=True
            pilot.write(folder/('training.json' if mode=='clip' else 'streaming.json'),record)
    return tmp_path


def test_aggregate_recounts_scores_and_initial_gap(tmp_path,monkeypatch):
    output=fixture(tmp_path,monkeypatch)
    result=pilot.aggregate(output)
    assert result['primary_mimo_vs_siso']['delta_pp']==50
    assert result['initial_mimo_vs_siso']['delta_pp']==0
    assert result['change_in_gap_pp']==50


@pytest.mark.parametrize('field',['input_metadata_sha256','input_pixels_sha256'])
def test_aggregate_rejects_changed_inputs_despite_matching_scores(tmp_path,monkeypatch,field):
    output=fixture(tmp_path,monkeypatch);path=output/'mimo/streaming.json';record=pilot.read(path)
    for stage in ('initial','final','replay'):record[stage][field]='different'
    pilot.write(path,record)
    with pytest.raises(ValueError,match='Evaluation inputs differ'):pilot.aggregate(output)


def test_aggregate_rejects_reordered_targets_with_updated_file_hash(tmp_path,monkeypatch):
    output=fixture(tmp_path,monkeypatch);path=output/'mimo/stream.npz'
    with np.load(path) as saved: arrays={k:saved[k].copy() for k in saved.files}
    arrays['targets']=arrays['targets'][::-1];np.savez_compressed(path,**arrays)
    record=pilot.read(output/'mimo/streaming.json');record['stream_sha256']=file_hash(path)
    pilot.write(output/'mimo/streaming.json',record)
    with pytest.raises(ValueError,match='Targets changed'):pilot.aggregate(output)


@pytest.mark.parametrize('event,step,expected',[
    ('budget_complete',20000,True),('stopped',20000,False),('budget_complete',14000,False),('train',20000,False)])
def test_predecessor_requires_successful_full_budget(tmp_path,event,step,expected):
    (tmp_path/'progress.jsonl').write_text(json.dumps(dict(event=event,step=step))+'\n')
    assert predecessor_complete(tmp_path) is expected
