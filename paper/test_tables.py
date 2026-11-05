"""Reject plausible-looking table inputs with mismatched scores or cohorts."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from paper.build_tables import ROOT, SOURCES, collect


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
