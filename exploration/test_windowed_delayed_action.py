import json
from pathlib import Path

import numpy as np
import pytest
import torch

from exploration import cache_windowed_delayed_action as cache
from exploration.buffered_attention import select
from exploration.delayed_action_data import select_blocks
from exploration.delayed_action_probe import evaluate
from exploration.resettable_window_attention import variants
from test_temporal_siglip import activate, make


def test_checkpoint_must_be_the_original_control_with_exact_training_config(tmp_path):
    run = tmp_path / 'original'
    row = dict(model='attention_interleaved', run=str(run), config=dict(seed=42, steps=20000, lr=1e-4))
    plan = dict(runs=[row])
    info = dict(config=row['config'].copy())
    assert cache.matching_reference(plan, run / 'last.pt', info) == row
    with pytest.raises(ValueError, match='exact scheduled'):
        cache.matching_reference(plan, tmp_path / 'other/last.pt', info)
    info['config']['lr'] = 1e-3
    with pytest.raises(ValueError, match='exact scheduled'):
        cache.matching_reference(plan, run / 'last.pt', info)


def test_extracted_cohort_uses_original_probe_and_class_independent_reset():
    torch.manual_seed(149)
    torch.set_num_threads(1)
    encoder = make('attention')
    activate(encoder)
    _, bounded = variants(encoder, 3)
    select(encoder, bounded)
    plan = dict(classes=2, frames_per_clip=8, delays=[0, 1, 8], ridge_alpha=1.,
                conditions=cache.CONDITIONS, splits={})
    arrays = {}
    for split in ('train', 'validation'):
        rows = [dict(id=f'{split}-{i}', group=f'{split}-{i}', sha256=f'{split}-{i}', label=i % 2) for i in range(20)]
        plan['splits'][split] = select_blocks(rows, 2, 2, 1, 71)
        pixels = {row['id']: torch.randn(8, 3, 32, 32) for row in rows}
        pieces = {name: [] for name in cache.CONDITIONS + ['labels', 'ids', 'blocks']}
        for block in plan['splits'][split]:
            values, ids, states = cache.collect_block(encoder, block, lambda row: pixels[row['id']], plan['delays'])
            assert np.array_equal(values['suffix_only'][0], values['suffix_only'][1])
            assert np.array_equal(values['saved_evidence'], np.broadcast_to(values['full_memory'][:, :1], values['full_memory'].shape))
            assert np.array_equal(values['full_memory'][0, -1], values['full_memory'][1, -1])
            assert len({tuple(s['current_rgb_sha256'][1:]) for s in states}) == 1
            for name, value in {**values, **ids}.items():
                pieces[name].append(value)
        arrays.update({split + '_' + name: np.concatenate(values) for name, values in pieces.items()})
    result, weights = evaluate(arrays, plan)
    assert all(r['accuracy'] == .5 for r in result['suffix_only']['validation'])
    assert result['full_memory']['validation'][-1]['accuracy'] == .5
    arrays['validation_full_memory'] *= -11
    _, after = evaluate(arrays, plan)
    assert all(np.array_equal(weights[k], after[k]) for k in weights)


def test_incomplete_evidence_or_suffix_is_rejected():
    block = dict(block=0, evidence=[dict(id='a', label=0)], suffix=[dict(id='b', label=1)])
    with pytest.raises(ValueError, match='suffix'):
        cache.collect_block(None, block, lambda row: torch.zeros(7, 3, 32, 32), [0, 8])
    def reader(row):
        return torch.zeros(7 if row['id'] == 'a' else 8, 3, 32, 32)
    with pytest.raises(ValueError, match='eight'):
        cache.collect_block(None, block, reader, [0, 8])


def fake_qualification(tmp_path, monkeypatch):
    plan = dict(expected_rows=2, fixture_plan=dict(frames=2), windows=[8], origins=[0],
                tolerances=dict(fp32=dict(max_abs=.001, relative_l2=1e-5)))
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(cache, 'checked_plan', lambda _: plan)
    rows = [dict(precision='fp32', window=8, origin=0, absolute_observation=t,
                 max_abs=0., relative_l2=0., passed=True) for t in range(2)]
    (tmp_path / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary = dict(passed=True, weights_unchanged=True, plan_sha256=cache.sha(tmp_path / 'plan.json'),
                   rows=2, rows_sha256=cache.sha(tmp_path / 'rows.jsonl'))
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    return summary


def test_qualification_requires_complete_rows_and_numerical_pass(tmp_path, monkeypatch):
    summary = fake_qualification(tmp_path, monkeypatch)
    assert cache.require_qualification(tmp_path)['expected_rows'] == 2
    rows = [json.loads(s) for s in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    rows[1] = rows[0]
    (tmp_path / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summary['rows_sha256'] = cache.sha(tmp_path / 'rows.jsonl')
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='coverage'):
        cache.require_qualification(tmp_path)
    summary = fake_qualification(tmp_path, monkeypatch)
    summary['passed'] = False
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='failed'):
        cache.require_qualification(tmp_path)
    (tmp_path / 'summary.json').unlink()
    with pytest.raises(ValueError, match='required'):
        cache.require_qualification(tmp_path)
