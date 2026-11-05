import pytest
import torch

from temporal_siglip import CausalAttention
from exploration.buffered_attention import BufferedAttention,compare_streams,variants,select
from test_temporal_siglip import make,activate


def test_full_history_and_irregular_chunks_match_without_reallocating():
    torch.manual_seed(71); torch.set_num_threads(1)
    reference=CausalAttention(32).eval(); buffered=BufferedAttention(reference,19).eval()
    values=torch.randn(3,19,32)
    a=b=None; offset=0; pointers=None
    with torch.inference_mode():
        for length in (1,4,2,7,5):
            x=values[:,offset:offset+length]
            y,a=reference(x,a,offset,True)
            z,b=buffered(x,b,offset,True)
            torch.testing.assert_close(y,z,atol=1e-7,rtol=1e-6)
            current=[t.data_ptr() for t in b[:2]]
            if pointers is not None: assert pointers==current
            pointers=current; offset+=length
            # Poison unwritten capacity; it must not enter any subsequent answer.
            b[0][:,:,offset:]=float('nan'); b[1][:,:,offset:]=float('nan')
        before=[t.clone() for t in b]
        with pytest.raises(ValueError,match='capacity'):
            buffered(values[:,:1],b,offset,True)
        assert all(torch.equal(x,y) for x,y in zip(before,b))


def test_cursor_and_training_errors_are_rejected():
    reference=CausalAttention(32).eval(); buffered=BufferedAttention(reference,4)
    x=torch.randn(2,1,32)
    with pytest.raises(RuntimeError,match='no-grad'):
        buffered(x,streaming=True)
    with torch.inference_mode():
        with pytest.raises(ValueError,match='zero'):
            buffered(x,offset=1,streaming=True)
        _,cache=buffered(x,streaming=True)
        with pytest.raises(ValueError,match='cursor'):
            buffered(x,cache,offset=0,streaming=True)


def test_encoder_weights_reset_and_causal_features_preserved():
    encoder=make('attention'); activate(encoder)
    old={k:v.clone() for k,v in encoder.state_dict().items()}
    pointers={k:p.data_ptr() for k,p in encoder.named_parameters()}
    original,buffered=variants(encoder,13); select(encoder,buffered)
    assert {k:p.data_ptr() for k,p in encoder.named_parameters()}==pointers
    assert all(torch.equal(old[k],v) for k,v in encoder.state_dict().items())
    select(encoder,original)
    images=torch.randn(13,1,3,32,32)
    first=compare_streams(encoder,images,13)
    second=compare_streams(encoder,images,13)
    assert first==second
    assert max(r['max_abs'] for r in first)<1e-5
    assert len({r['states']['buffered'] for r in first})==1
    assert first[-1]['states']['buffered']==first[-1]['states']['dynamic']+16
    assert all(type(a.mixer) is CausalAttention for a in encoder.adapters.values())


def test_cost_rotation_balances_each_implementation_position():
    from exploration.benchmark_buffered_attention import traversal_order,VARIANTS
    orders=[traversal_order(i) for i in range(3)]
    assert all(len(order)==3 and set(order)==set(VARIANTS) for order in orders)
    assert all({order[position] for order in orders}==set(VARIANTS) for position in range(3))
