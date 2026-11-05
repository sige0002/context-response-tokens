import torch
import pytest

from exploration.temporal_interval import interval_weights, interval_kernel, scale_arguments


def test_intervals_preserve_original_anchor_clock():
    for factor in (1, 2, 4):
        weights = torch.tensor(interval_weights(factor))
        assert len(weights) == 7*factor+1
        assert torch.equal(weights.cumsum(0)[::factor], torch.arange(1., 9.))
    for bad in (True, 0, 3, -1, 2.):
        with pytest.raises(ValueError): interval_weights(bad)


def test_both_decay_and_integration_are_scaled_without_mutating_input():
    dt = torch.linspace(.01, .15, 15)[None,None,:]
    original = dict(DT=dt, ADT=-2*dt, Angles=torch.ones(1,15,1,16))
    scaled = scale_arguments(original, 2)
    assert torch.equal(scaled['DT'][...,0], dt[...,0])
    assert torch.equal(scaled['DT'][...,1:]*2, dt[...,1:])
    assert torch.equal(scaled['ADT'], -2*scaled['DT'])
    assert scaled['Angles'] is original['Angles']
    assert torch.equal(original['DT'], dt) and scaled['DT'].data_ptr() != dt.data_ptr()
    with pytest.raises(ValueError): scale_arguments(original, 4)


def test_native_wrapper_restores_after_failure_and_rejects_nesting(monkeypatch):
    import mamba_ssm.modules.mamba3 as native
    def original(**kwargs): return kwargs['DT']
    monkeypatch.setattr(native, 'mamba3_siso_combined', original)
    args = dict(DT=torch.ones(1,1,15), ADT=-torch.ones(1,1,15))
    with pytest.raises(RuntimeError, match='test failure'):
        with interval_kernel(2) as calls:
            result = native.mamba3_siso_combined(**args)
            assert result.sum() == 8 and len(calls) == 1
            with pytest.raises(RuntimeError, match='Nested'):
                with interval_kernel(2): pass
            raise RuntimeError('test failure')
    assert native.mamba3_siso_combined is original


def test_identity_qualification_checks_returned_states(monkeypatch):
    import mamba_ssm.modules.mamba3 as native
    def original(**kwargs):
        x = kwargs['DT']; return (x, x+1, x+2, x+3, x+4)
    monkeypatch.setattr(native, 'mamba3_siso_combined', original)
    with interval_kernel(1, qualification=True) as checks:
        result = native.mamba3_siso_combined(DT=torch.ones(1,1,8),
            ADT=-torch.ones(1,1,8), return_final_states=False)
        assert result.shape == (1,1,8)
    assert checks == [dict(output_and_four_states_max_abs=[0.]*5)]
