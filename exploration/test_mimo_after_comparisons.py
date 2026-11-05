import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from exploration import run_mimo_after_comparisons as queue


def fixture(tmp_path):
    final, pilot, main = (tmp_path / name for name in ('final', 'pilot', 'main'))
    for folder in (final, pilot, main):
        folder.mkdir()
    upstream = {}
    for name in ('priority', 'buffered'):
        folder = tmp_path / name
        folder.mkdir()
        queue.write(folder / 'status.json', dict(status='complete'))
        upstream[name] = str(folder)
    queue.write(final / 'plan.json', dict(runs=[dict(name='a'), dict(name='b')], **upstream))
    queue.write(final / 'status.json', dict(status='complete'))
    summaries = {}
    for name in ('a', 'b'):
        folder = final / name
        folder.mkdir()
        queue.write(folder / 'summary.json', dict(checked=True))
        summaries[name] = queue.sha(folder / 'summary.json')
    queue.write(final / 'summary.json', dict(plan_sha256=queue.sha(final / 'plan.json'),
                                          run_summaries=summaries))
    queue.write(pilot / 'plan.json', dict(predecessor=str(main)))
    (main / 'progress.jsonl').write_text(json.dumps(dict(event='budget_complete', step=20000)) + '\n')
    plan = dict(pilot=str(pilot), dependency=str(final), pinned_files={
        str(p): queue.sha(p) for p in (pilot / 'plan.json', final / 'plan.json')})
    return plan, final, pilot, main


def test_completed_chain_is_accepted(tmp_path):
    plan, _, _, _ = fixture(tmp_path)
    queue.verify_completion(plan)


@pytest.mark.parametrize('failure', ['running', 'failed', 'result_changed', 'missing_run',
                                   'plan_changed', 'upstream_incomplete', 'main_stopped'])
def test_exit_or_stale_success_alone_cannot_release_pilot(tmp_path, failure):
    plan, final, pilot, main = fixture(tmp_path)
    if failure in ('running', 'failed'):
        queue.write(final / 'status.json', dict(status=failure))
    elif failure == 'result_changed':
        queue.write(final / 'a/summary.json', dict(checked=False))
    elif failure == 'missing_run':
        summary = queue.read(final / 'summary.json')
        summary['run_summaries'].pop('b')
        queue.write(final / 'summary.json', summary)
    elif failure == 'plan_changed':
        queue.write(pilot / 'plan.json', dict(predecessor='somewhere_else'))
    elif failure == 'upstream_incomplete':
        queue.write(tmp_path / 'priority/status.json', dict(status='failed'))
    else:
        (main / 'progress.jsonl').write_text(json.dumps(dict(event='stopped', step=20000)) + '\n')
    with pytest.raises(ValueError):
        queue.verify_completion(plan)


def test_active_gpu_refuses_release(monkeypatch):
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: str(os.getpid()) + '\n')
    with pytest.raises(RuntimeError, match='GPU job still active'):
        queue.assert_gpu_idle()
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: '')
    queue.assert_gpu_idle()


def test_binding_rejects_unrelated_process_without_signalling_it(tmp_path):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
    try:
        with pytest.raises(ValueError, match='not the final-memory queue'):
            queue.bind_dependency(child.pid, tmp_path)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_binding_tracks_process_exit_and_validates_output(tmp_path):
    # A controlled stand-in with the expected argv; no actual queue or GPU job.
    ready = tmp_path / 'ready'
    child = subprocess.Popen([sys.executable, '-c',
                              'import sys,time; from pathlib import Path; '
                              'Path(sys.argv[1]).write_text("ready"); time.sleep(20)', str(ready),
                              'exploration.run_final_memory_layers', '--output', str(tmp_path)])
    fd = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), 'Controlled predecessor did not reach its ready marker'
        fd, bound = queue.bind_dependency(child.pid, tmp_path)
        assert bound['pid'] == child.pid and bound['start_ticks'] > 0
        poller = queue.select.poll()
        poller.register(fd, queue.select.POLLIN)
        assert not poller.poll(0)
        with pytest.raises(ValueError, match='another directory'):
            queue.bind_dependency(child.pid, tmp_path / 'wrong')
        child.terminate()
        child.wait(timeout=5)
        assert poller.poll(1000)
    finally:
        if fd is not None:
            os.close(fd)
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
