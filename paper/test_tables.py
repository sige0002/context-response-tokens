"""Reject plausible-looking table inputs with mismatched scores or cohorts."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from paper.build_tables import ROOT, SOURCES, build, collect


@pytest.fixture
def snapshot(tmp_path):
    for relative in SOURCES.values():
        destination = tmp_path/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    return tmp_path


def update(root, key, change):
    path = root/SOURCES[key]
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data)+'\n')


def refresh_receipt(root):
    digest = hashlib.sha256((root/SOURCES['history']).read_bytes()).hexdigest()
    update(root, 'history_verification', lambda data: data.update(summary_sha256=digest))


def test_wrong_accuracy_rejected_even_with_updated_receipt(snapshot):
    update(snapshot, 'history', lambda data: data['scores']['full_stream'].update(original_label_top1=.99))
    refresh_receipt(snapshot)
    with pytest.raises(ValueError, match='accuracy'):
        collect(snapshot)


def test_different_history_checkpoint_rejected(snapshot):
    update(snapshot, 'history_plan', lambda data: data['checkpoint_info'].update(step=6000))
    digest = hashlib.sha256((snapshot/SOURCES['history_plan']).read_bytes()).hexdigest()
    update(snapshot, 'history', lambda data: data.update(plan_sha256=digest))
    refresh_receipt(snapshot)
    with pytest.raises(ValueError, match='checkpoint'):
        collect(snapshot)


def test_partial_validation_cannot_be_full_population(snapshot):
    update(snapshot, 'training', lambda data: data['validation'][-1].update(samples=348))
    with pytest.raises(ValueError, match='population'):
        collect(snapshot)


def refresh_memory_receipt(root):
    digest = hashlib.sha256((root/SOURCES['memory']).read_bytes()).hexdigest()
    update(root, 'memory_verification', lambda data: data.update(summary_sha256=digest))


def test_real_tables_separate_interim_diagnostics_and_pending_final(snapshot, tmp_path):
    output = tmp_path/'tables'
    evidence = build(snapshot, output)
    assert '9,000 & 12,551 & 50.66' in (output/'training.tex').read_text()
    assert 'After block 8 & 72 & 20.69 & -9.48' in (output/'memory_layers.tex').read_text()
    assert 'After block 12 & 50 & 14.37 & -15.80' in (output/'memory_layers.tex').read_text()
    assert (output/'comparison_pending.tex').read_text().count('& --') == 6
    assert not evidence['submission_ready']


def test_memory_counts_recounted_after_receipt_refresh(snapshot):
    update(snapshot, 'memory', lambda data: data['scores']['clear_8'].update(correct=73))
    refresh_memory_receipt(snapshot)
    with pytest.raises(ValueError, match='counts'):
        collect(snapshot)


def test_memory_cohort_checked_even_when_accuracy_unchanged(snapshot):
    update(snapshot, 'memory', lambda data: data['scores']['clear_8']['predictions'][0].update(id='different'))
    refresh_memory_receipt(snapshot)
    with pytest.raises(ValueError, match='cohort'):
        collect(snapshot)


@pytest.mark.parametrize('delta', [9.482758620689655, float('nan')])
def test_memory_delta_sign_and_finiteness(snapshot, delta):
    update(snapshot, 'memory', lambda data: data['scores']['clear_8'].update(delta_vs_stream_pp=delta))
    refresh_memory_receipt(snapshot)
    with pytest.raises(ValueError, match='delta'):
        collect(snapshot)


def test_same_step_different_memory_checkpoint_rejected(snapshot):
    update(snapshot, 'memory_plan', lambda data: data['checkpoint_info'].update(checkpoint_sha256='0'*64))
    digest = hashlib.sha256((snapshot/SOURCES['memory_plan']).read_bytes()).hexdigest()
    update(snapshot, 'memory', lambda data: data.update(plan_sha256=digest))
    refresh_memory_receipt(snapshot)
    with pytest.raises(ValueError, match='checkpoint'):
        collect(snapshot)


def test_reused_memory_predictions_must_match_not_just_accuracy(snapshot):
    def change(data):
        row = next(r for r in data['scores']['reset_all']['predictions'] if r['target'] != r['prediction'])
        row['prediction'] = next(v for v in range(174) if v not in (row['prediction'], row['target']))
    update(snapshot, 'memory', change)
    refresh_memory_receipt(snapshot)
    with pytest.raises(ValueError, match='baseline predictions'):
        collect(snapshot)


def test_memory_trace_rejects_layer_removal(snapshot):
    update(snapshot, 'memory', lambda data: data['state_by_frame']['clear_8'][0].update(state_layer_keys=['3','11']))
    refresh_memory_receipt(snapshot)
    with pytest.raises(ValueError, match='state trace'):
        collect(snapshot)
