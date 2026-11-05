"""State erasure preserves modules, current transforms, and attention positions."""
import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from temporal_siglip import TemporalSiglip, StreamState
from exploration.memory_layer_intervention import stream_with_memory


@pytest.fixture
def example():
    torch.set_num_threads(2); torch.manual_seed(71)
    config = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, image_size=16, patch_size=8)
    model = TemporalSiglip(SiglipVisionModel(config),kind='attention',every=1,width=32).eval()
    for adapter in model.adapters.values(): adapter.gain.data.fill_(.2)
    return model, torch.randn(1,4,3,16,16)


def test_all_memory_matches_original_and_every_module_still_runs(example):
    encoder,video = example
    with torch.no_grad(): original = encoder(video)[:,-1]
    output,records = stream_with_memory(encoder,video,tuple(encoder.adapters))
    torch.testing.assert_close(original,output,atol=1e-5,rtol=1e-5)
    assert [r['offset'] for r in records] == [1,2,3,4]
    assert all(r['state_layer_keys']==list(encoder.adapters) for r in records)
    counts = {k:0 for k in encoder.adapters}
    handles=[]
    for key,adapter in encoder.adapters.items():
        def count(module,args,result,key=key): counts[key]+=1
        handles.append(adapter.register_forward_hook(count))
    try: stream_with_memory(encoder,video,())
    finally:
        for handle in handles: handle.remove()
    assert counts == {k:4 for k in encoder.adapters}


def test_erased_history_preserves_absolute_position_and_episode_isolation(example):
    encoder,video = example
    before={k:v.detach().clone() for k,v in encoder.state_dict().items()}
    actual,records=stream_with_memory(encoder,video,())
    at_last=StreamState(id(encoder),1,(3,16,16),offset=3)
    expected,_=encoder.step(video[:,-1],at_last)
    wrong,_=encoder.step(video[:,-1],None)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert not torch.allclose(actual,wrong,atol=1e-6,rtol=1e-6)
    assert len({r['state_bytes'] for r in records})==1
    repeated,_=stream_with_memory(encoder,video,())
    torch.testing.assert_close(actual,repeated,atol=0,rtol=0)
    assert all(torch.equal(before[k],v) for k,v in encoder.state_dict().items())


def test_one_state_erasure_keeps_other_histories_and_rejects_misuse(example):
    encoder,video=example
    one,r1=stream_with_memory(encoder,video,('1','2'))
    none,r0=stream_with_memory(encoder,video,())
    assert not torch.allclose(one,none)
    assert r1[-1]['state_bytes']>r0[-1]['state_bytes']
    assert all(r['state_layer_keys']==list(encoder.adapters) for r in r1)
    for keys in [('bad',),('1','1')]:
        with pytest.raises(ValueError): stream_with_memory(encoder,video,keys)
    encoder.train()
    with pytest.raises(ValueError): stream_with_memory(encoder,video,())
