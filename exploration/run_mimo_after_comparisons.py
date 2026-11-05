"""Release the unchanged MIMO pilot only after the existing GPU queue chain.

This scheduler never signals its predecessor or modifies either experiment
plan. Once released, the original pilot owns its normal lock and cancellation.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def bind_dependency(pid, directory):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'Cannot bind final-memory queue')
    try:
        proc = Path('/proc') / str(pid)
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        if ('exploration.run_final_memory_layers' not in args or '--output' not in args):
            raise ValueError('Dependency is not the final-memory queue')
        actual = proc.joinpath('cwd').resolve() / args[args.index('--output') + 1]
        if actual.resolve() != directory.resolve():
            raise ValueError('Dependency writes to another directory')
        ticks = int(proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19])
        return fd, dict(pid=pid, start_ticks=ticks, directory=str(directory))
    except BaseException:
        os.close(fd)
        raise


def check_pins(plan):
    for name, expected in plan['pinned_files'].items():
        if sha(name) != expected:
            raise ValueError('Scheduling input changed: ' + name)


def verify_completion(plan):
    check_pins(plan)
    directory = Path(plan['dependency'])
    if read(directory / 'status.json')['status'] != 'complete':
        raise ValueError('Final-memory queue has not completed successfully')
    original = read(directory / 'plan.json')
    summary = read(directory / 'summary.json')
    if summary['plan_sha256'] != sha(directory / 'plan.json'):
        raise ValueError('Final-memory summary uses another plan')
    expected = {row['name'] for row in original['runs']}
    if set(summary['run_summaries']) != expected:
        raise ValueError('Final-memory run summaries are incomplete')
    for name, digest in summary['run_summaries'].items():
        if sha(directory / name / 'summary.json') != digest:
            raise ValueError('Final-memory result changed: ' + name)
    for name in ('priority', 'buffered'):
        if read(Path(original[name]) / 'status.json')['status'] != 'complete':
            raise ValueError('Upstream comparison/cost queue is incomplete: ' + name)
    pilot_plan = read(Path(plan['pilot']) / 'plan.json')
    events = [json.loads(line) for line in
              (Path(pilot_plan['predecessor']) / 'progress.jsonl').read_text().splitlines() if line]
    if (not events or events[-1].get('event') != 'budget_complete'
            or events[-1].get('step') != 20000):
        raise ValueError('Main training has not completed 20000 updates')


def assert_gpu_idle():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(row.strip().isdigit() and Path('/proc', row.strip()).exists()
           for row in output.splitlines()):
        raise RuntimeError('GPU job still active; refusing to release the pilot')


def execute(plan, output, dependency_pid):
    state = dict(status='starting')
    fd = None
    cancelled = False

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    def check():
        if cancelled:
            raise InterruptedError('Scheduler stopped; predecessor untouched')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        check_pins(plan)
        fd, bound = bind_dependency(dependency_pid, Path(plan['dependency']))
        state.update(status='waiting_for_final_memory_layers', dependency=bound)
        write(output / 'status.json', state)
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        while not poller.poll(1000):
            check()
        check()
        os.close(fd)
        fd = None
        verify_completion(plan)
        assert_gpu_idle()
        check()
        command = ['uv', 'run', '--locked', 'python', '-m',
                   'exploration.run_mimo_pilot', '--output', plan['pilot']]
        state.update(status='released_to_pilot', command=command,
                     pilot_status=str(Path(plan['pilot']) / 'status.json'))
        write(output / 'status.json', state)
        # The pilot keeps its original plan, lock, process-group handling and
        # completion guard. No overlapping child is created by this scheduler.
        os.execvp(command[0], command)
    except BaseException as exc:
        state.update(status='stopped' if cancelled else 'failed',
                     error=f'{type(exc).__name__}: {exc}')
        write(output / 'status.json', state)
        raise
    finally:
        if fd is not None:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--dependency', type=Path, required=True)
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    pilot, dependency, output = (p.resolve() for p in (args.pilot, args.dependency, args.output))
    pilot_plan, dependency_plan = read(pilot / 'plan.json'), read(dependency / 'plan.json')
    pins = {str(ROOT / p): h for d in (pilot_plan, dependency_plan) for p, h in d['sources'].items()}
    for path in (pilot / 'plan.json', dependency / 'plan.json', Path(__file__),
                 ROOT / 'exploration/test_mimo_after_comparisons.py'):
        pins[str(path.resolve())] = sha(path)
    plan = dict(pilot=str(pilot), dependency=str(dependency), pinned_files=pins,
                scope='Scheduling only; original model/data/update plans are unchanged')
    check_pins(plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', plan)
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(plan, output, args.after_pid)


if __name__ == '__main__':
    main()
