import copy
import json

import pytest
import torch

from exploration.benchmark_windowed_attention import (
    REPEATS, VARIANTS, compare_bounded_streams, expected_state_bytes,
    traversal_order, verify_numerics,
    require_predecessor, measure,
)
from video_checkpoint import file_hash
from test_temporal_siglip import activate, make


def test_five_rotations_balance_every_variant_at_every_position():
    orders = [traversal_order(r) for r in range(REPEATS)]
    assert REPEATS == len(VARIANTS) == 5
    assert all(set(order) == set(VARIANTS) for order in orders)
    assert all({order[position] for order in orders} == set(VARIANTS) for position in range(5))
    for value in (-1, 5, True, 1.5):
        with pytest.raises(ValueError):
            traversal_order(value)


def test_state_accounting_counts_allocated_capacity_and_reset_metadata():
    for frames in (52, 245, 1337):
        assert expected_state_bytes('mamba_interleaved', frames) == 79027200
        assert expected_state_bytes('attention_buffered', frames) == 602112 * frames + 24
        for window in (8, 32, 128):
            assert expected_state_bytes(f'attention_window{window}', frames) == 602112 * window + 48
    with pytest.raises(ValueError):
        expected_state_bytes('attention_window999', 1337)


@pytest.mark.parametrize('window', [1, 3, 8])
def test_framewise_comparison_restores_model_and_checks_all_frames(window):
    torch.manual_seed(157)
    torch.set_num_threads(1)
    encoder = make('attention')
    activate(encoder)
    original = {k: a.mixer for k, a in encoder.adapters.items()}
    weights = {k: p.clone() for k, p in encoder.state_dict().items()}
    images = [torch.randn(1, 3, 32, 32) for _ in range(19)]
    rows = compare_bounded_streams(encoder, images, window)
    config = encoder.backbone.config
    tokens = (config.image_size // config.patch_size) ** 2
    width = next(iter(original.values())).qkv.in_features
    expected = len(original) * (2 * tokens * window * width * 4 + 16)
    verify_numerics(rows, len(images), dict(max_abs=1e-5, relative_l2=1e-5), expected)
    assert all(encoder.adapters[k].mixer is mixer for k, mixer in original.items())
    assert all(torch.equal(weights[k], p) for k, p in encoder.state_dict().items())


def test_comparison_restores_original_core_after_input_failure():
    encoder = make('attention')
    activate(encoder)
    original = {k: a.mixer for k, a in encoder.adapters.items()}
    def images():
        yield torch.randn(1, 3, 32, 32)
        raise RuntimeError('Controlled reader failure')
    with pytest.raises(RuntimeError, match='reader failure'):
        compare_bounded_streams(encoder, images(), 3)
    assert all(encoder.adapters[k].mixer is mixer for k, mixer in original.items())


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'nan', 'exceeds', 'capacity'])
def test_invalid_numerical_or_memory_evidence_cannot_admit_timing(failure):
    rows = [dict(frame=i, max_abs=0., relative_l2=0., state_bytes=100) for i in range(3)]
    limits = dict(max_abs=.125, relative_l2=.005)
    verify_numerics(rows, 3, limits, 100)
    altered = copy.deepcopy(rows)
    if failure == 'missing':
        altered.pop()
    elif failure == 'duplicate':
        altered[1] = altered[0]
    elif failure == 'nan':
        altered[1]['max_abs'] = float('nan')
    elif failure == 'exceeds':
        altered[1]['relative_l2'] = .006
    else:
        altered[1]['state_bytes'] = 99
    with pytest.raises(ValueError):
        verify_numerics(altered, 3, limits, 100)


@pytest.mark.parametrize('failure', ['pending', 'summary', 'qualification', 'conditions', 'source'])
def test_cost_refuses_incomplete_or_changed_recognition_predecessor(tmp_path, failure):
    qualification = tmp_path / 'qualification'
    source = tmp_path / 'input'
    source.write_text('fixed input')
    plan = dict(qualification=str(qualification), pins={str(source): file_hash(source)})
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    summary = dict(queue_plan_sha256=file_hash(tmp_path / 'plan.json'))
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    status = dict(status='complete', summary_sha256=file_hash(tmp_path / 'summary.json'),
                  completed=[dict(seed=s, window=w) for s in (42, 43, 44) for w in (8, 32, 128)])
    (tmp_path / 'status.json').write_text(json.dumps(status))
    assert require_predecessor(tmp_path, qualification)['summary_sha256'] == status['summary_sha256']
    if failure == 'pending':
        status['status'] = 'running'
    elif failure == 'summary':
        (tmp_path / 'summary.json').write_text('{}')
    elif failure == 'qualification':
        qualification = tmp_path / 'different'
    elif failure == 'conditions':
        status['completed'][-1] = status['completed'][0]
    else:
        source.write_text('changed')
    (tmp_path / 'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError):
        require_predecessor(tmp_path, qualification)


def test_active_gpu_refusal_precedes_loading_and_output_creation(tmp_path, monkeypatch):
    import exploration.benchmark_windowed_attention as benchmark
    def occupied():
        raise RuntimeError('Controlled active GPU process')
    monkeypatch.setattr(benchmark, 'assert_exclusive', occupied)
    output = tmp_path / 'new-output'
    with pytest.raises(RuntimeError, match='active GPU'):
        measure(tmp_path / 'missing-priority', 'shortest', tmp_path / 'missing-qualification',
                output, tmp_path / 'missing-predecessor')
    assert not output.exists()
