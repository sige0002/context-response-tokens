"""Serialize prepared bounded-attention qualification/probes after the MIMO pilot.

Predecessors are never signalled. A failure preserves partial artifacts;
completed stages are verified before reuse and partial stages are refused.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from exploration.run_mimo_after_comparisons import assert_gpu_idle, read, sha, write


ROOT = Path(__file__).resolve().parents[1]


def build_plan(experiment, scheduler):
    base, previous = read(experiment), read(scheduler / 'plan.json')
    qualification = Path(base['gpu_qualification'])
    qualified = read(qualification / 'plan.json')
    if Path(qualified['after_run']).resolve() != Path(previous['pilot']).resolve():
        raise ValueError('GPU qualification and scheduler depend on different pilots')
    pins = dict(base['source_sha256'])
    for p in (experiment, scheduler / 'plan.json', Path(base['comparison_plan']),
              qualification / 'plan.json', ROOT / 'exploration/run_windowed_delayed_action.py',
              ROOT / 'exploration/aggregate_windowed_delayed_action.py',
              ROOT / 'exploration/test_windowed_delayed_queue.py'):
        pins[str(p)] = sha(p)
    return dict(experiment=str(experiment), scheduler=str(scheduler), pilot=previous['pilot'],
                qualification=str(qualification), pins=pins,
                scope='Run fixed GPU qualification, all nine prepared caches/probes, then paired CPU recount; no encoder training')


def verify_plan(plan):
    for path, digest in plan['pins'].items():
        if sha(path) != digest:
            raise ValueError('Prepared queue input changed: ' + path)
    base = read(plan['experiment'])
    if sha(base['comparison_plan']) != base['comparison_plan_sha256']:
        raise ValueError('Original delayed-action comparison plan changed')
    if sha(Path(plan['qualification']) / 'plan.json') != base['gpu_qualification_plan_sha256']:
        raise ValueError('GPU qualification plan changed')


def bind_scheduler(pid, scheduler, pilot):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'Cannot bind MIMO scheduler')
    try:
        proc = Path('/proc') / str(pid)
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        if ('exploration.run_mimo_after_comparisons' not in args or '--output' not in args or '--pilot' not in args):
            raise ValueError('Dependency is not the MIMO scheduler')
        cwd = proc.joinpath('cwd').resolve()
        if ((cwd / args[args.index('--output') + 1]).resolve() != scheduler.resolve()
                or (cwd / args[args.index('--pilot') + 1]).resolve() != pilot.resolve()):
            raise ValueError('Dependency belongs to a different experiment')
        ticks = int(proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19])
        return fd, dict(pid=pid, start_ticks=ticks, scheduler=str(scheduler), pilot=str(pilot))
    except BaseException:
        os.close(fd)
        raise


def verify_dependencies(plan):
    verify_plan(plan)
    if read(Path(plan['scheduler']) / 'status.json')['status'] != 'released_to_pilot':
        raise ValueError('Preceding scheduler did not release the pilot')
    pilot = Path(plan['pilot'])
    status = read(pilot / 'status.json')
    if status.get('status') != 'complete':
        raise ValueError('MIMO pilot did not complete successfully')
    if (status.get('summary_sha256') != sha(pilot / 'summary.json')
            or read(pilot / 'summary.json')['plan_sha256'] != sha(pilot / 'plan.json')):
        raise ValueError('Completed MIMO pilot artifacts changed')
    baseline = Path(read(plan['experiment'])['comparison_plan']).parent
    if (read(baseline / 'status.json')['status'] != 'complete'
            or read(baseline / 'summary.json')['plan_sha256'] != sha(baseline / 'plan.json')):
        raise ValueError('Original delayed-action comparison is incomplete or changed')


def execute(plan, output, after_pid):
    child = fd = None
    cancelled = False
    status = dict(status='starting', completed=[], operations=[])

    def save():
        write(output / 'status.json', status)

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    def check():
        if cancelled:
            raise InterruptedError('Windowed comparison stopped; predecessor untouched')
        verify_plan(plan)

    def launch(command, name, cpu=False):
        nonlocal child
        check()
        assert_gpu_idle()
        entry = dict(name=name, command=command, status='running', started=time.time())
        status.update(status='running', current=name)
        status['operations'].append(entry)
        save()
        env = dict(os.environ, OPENBLAS_NUM_THREADS='2')
        env.pop('VIRTUAL_ENV', None)
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        with (output / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            entry['pid'] = child.pid
            save()
            while child.poll() is None:
                if cancelled:
                    raise InterruptedError('Windowed comparison cancelled')
                time.sleep(1)
            code = child.returncode
            child = None
        entry.update(status='complete' if code == 0 else 'failed', returncode=code, finished=time.time())
        save()
        if code:
            raise RuntimeError('Windowed comparison stage failed: ' + name)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        check()
        fd, bound = bind_scheduler(after_pid, Path(plan['scheduler']), Path(plan['pilot']))
        status.update(status='waiting_for_mimo_pilot', dependency=bound)
        save()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        while not poller.poll(1000):
            if cancelled:
                raise InterruptedError('Cancelled while waiting; predecessor untouched')
        os.close(fd)
        fd = None
        check()
        verify_dependencies(plan)
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        from exploration.cache_windowed_delayed_action import require_qualification
        from exploration.aggregate_windowed_delayed_action import aggregate, verify_probe
        qualification = Path(plan['qualification'])
        if not (qualification / 'summary.json').exists():
            if (qualification / 'rows.jsonl').exists():
                raise ValueError('Partial numerical qualification; inspect before reuse')
            launch(['uv', 'run', '--locked', 'python', '-m', 'exploration.check_resettable_window_attention',
                    'run', '--output', str(qualification)], 'qualification')
        require_qualification(qualification)
        base = read(plan['experiment'])
        data = read(base['data_plan'])
        for row in base['runs']:
            check()
            directory = Path(row['output'])
            ready = all((directory / p / 'summary.json').exists() for p in ('cache', 'probe'))
            if not ready:
                if directory.exists():
                    raise ValueError('Partial cache/probe must be inspected: ' + str(directory))
                name = f"window{row['window']}_seed{row['seed']}"
                launch(row['cache_command'], name + '_cache')
                launch(row['probe_command'], name + '_probe', cpu=True)
            verify_probe(directory, row['reference_row'], base['data_plan_sha256'], row['window'], data)
            status['completed'].append(dict(seed=row['seed'], window=row['window']))
            save()
        check()
        result = aggregate(Path(plan['experiment']))
        write(output / 'summary.json', dict(queue_plan_sha256=sha(output / 'plan.json'), **result))
        status.update(status='complete', summary_sha256=sha(output / 'summary.json'))
    except BaseException as exc:
        status.update(status='stopped' if cancelled else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if fd is not None:
            os.close(fd)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('results/windowed_delayed_action_plan_v1.json'))
    parser.add_argument('--scheduler', type=Path, default=Path('results/mimo_after_comparisons_v1'))
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = build_plan(args.experiment.resolve(), args.scheduler.resolve())
    verify_plan(plan)
    with args.experiment.with_suffix('.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'plan.json', plan)
        execute(plan, args.output, args.after_pid)
