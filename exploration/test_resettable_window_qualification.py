import json

import pytest
import torch

from exploration import check_resettable_window_attention as check
from exploration.test_resettable_window_attention import reference
from temporal_siglip import CausalAttention


@pytest.mark.parametrize('window', [1, 8, 128])
def test_gpu_reference_matches_independent_dense_formula_on_cpu(window):
    torch.manual_seed(139)
    torch.set_num_threads(1)
    source = CausalAttention(32).double().eval()
    ref = check.AbsoluteResetReference(source, window)
    x = torch.randn(2, 145, 32, dtype=torch.float64)
    with torch.inference_mode():
        for origin in (0, 8, 137):
            cache, values = None, []
            for i in range(len(x[0])):
                value, cache = ref(x[:, i:i + 1], cache, origin + i, True)
                values.append(value)
            torch.testing.assert_close(torch.cat(values, 1), reference(source, x, window, origin),
                                       atol=1e-12, rtol=1e-12)


def test_gpu_release_requires_complete_chain_then_idle_device(tmp_path, monkeypatch):
    plan = dict(fixture_plan=dict(main_run='main'), after_run=str(tmp_path))
    calls = []
    monkeypatch.setattr(check, 'require_main_complete', lambda run: calls.append('main'))
    monkeypatch.setattr(check, 'assert_gpu_idle', lambda: calls.append('idle'))
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(dict(status='waiting_for_main_20000')))
    with pytest.raises(RuntimeError, match='not complete'):
        check.require_completed_chain(plan)
    assert calls == ['main']
    summary = tmp_path / 'summary.json'
    summary.write_text('{}')
    status.write_text(json.dumps(dict(status='complete', summary_sha256='stale')))
    with pytest.raises(ValueError, match='artifacts changed'):
        check.require_completed_chain(plan)
    assert 'idle' not in calls
    status.write_text(json.dumps(dict(status='complete', summary_sha256=check.sha(summary))))
    check.require_completed_chain(plan)
    assert calls[-2:] == ['main', 'idle']
