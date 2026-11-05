"""Serialize the three prepared cost videos after bounded-recognition completion."""
import argparse
import fcntl
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from exploration.benchmark_windowed_attention import assert_exclusive, require_predecessor
from exploration.verify_windowed_cost import verify_report, verify_specification
from encoder_costs import paired_costs
from point_experiment import read_json, write_json
from run_point_comparisons import bind_process
from video_checkpoint import file_hash


ROOT = Path(__file__).resolve().parents[1]


def build_plan(specification):
    spec = verify_specification(specification)
    sources = ['exploration/run_windowed_cost.py', 'exploration/verify_windowed_cost.py',
               'exploration/test_windowed_cost_queue.py', 'run_point_comparisons.py']
    return dict(specification=str(specification), specification_sha256=file_hash(specification),
                predecessor=spec['after_queue'], sources={p: file_hash(p) for p in sources},
                scope='Three prepared full-video measurements after the existing experiment chain; no training')


def verify_plan(plan):
    if file_hash(plan['specification']) != plan['specification_sha256']:
        raise ValueError('Prepared cost specification changed')
    for path, digest in plan['sources'].items():
        if file_hash(path) != digest:
            raise ValueError('Cost queue source changed: ' + path)
    spec = verify_specification(plan['specification'])
    if spec['after_queue'] != plan['predecessor']:
        raise ValueError('Cost queue predecessor differs')
    return spec


def execute(plan, output, after_pid):
    child = fd = None
    cancelled = False
    status = dict(status='starting', completed=[], operations=[])

    def save():
        write_json(output / 'status.json', status)

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    def check():
        if cancelled:
            raise InterruptedError('Cost queue stopped; predecessor untouched')
        return verify_plan(plan)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        spec = check()
        fd = bind_process(after_pid, 'exploration.run_windowed_delayed_action', '--output', plan['predecessor'])
        ticks = int(Path(f'/proc/{after_pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        status.update(status='waiting_for_windowed_recognition', dependency=dict(pid=after_pid, start_ticks=ticks))
        save()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        while not poller.poll(1000):
            if cancelled:
                raise InterruptedError('Cancelled while waiting; predecessor untouched')
        os.close(fd)
        fd = None
        spec = check()
        require_predecessor(Path(plan['predecessor']), Path(spec['qualification']))
        measurements, artifacts = [], {}
        for row in spec['comparisons']:
            check()
            assert_exclusive()
            selection = row['video']['selection']
            directory = Path(row['output'])
            if not (directory / 'summary.json').exists():
                if directory.exists():
                    raise ValueError('Partial cost output must be inspected: ' + str(directory))
                entry = dict(selection=selection, command=row['command'], started=time.time())
                status.update(status='running', current=selection)
                status['operations'].append(entry)
                save()
                env = dict(os.environ, OPENBLAS_NUM_THREADS='2')
                env.pop('VIRTUAL_ENV', None)
                with (output / (selection + '.log')).open('x') as log:
                    child = subprocess.Popen(row['command'], cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    entry['pid'] = child.pid
                    save()
                    while child.poll() is None:
                        if cancelled:
                            raise InterruptedError('Cost measurement cancelled')
                        time.sleep(1)
                    code = child.returncode
                    child = None
                entry.update(returncode=code, finished=time.time())
                save()
                if code:
                    raise RuntimeError('Cost measurement failed: ' + selection)
            check()
            measurements += verify_report(directory, Path(plan['specification']), selection)
            artifacts[selection] = file_hash(directory / 'summary.json')
            status['completed'].append(selection)
            save()
        check()
        assert_exclusive()
        write_json(output / 'summary.json', dict(queue_plan_sha256=file_hash(output / 'plan.json'),
            measurement_summary_sha256=artifacts, measurements=measurements, **paired_costs(measurements)))
        status.update(status='complete', summary_sha256=file_hash(output / 'summary.json'))
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
    parser.add_argument('--specification', type=Path, default=Path('results/windowed_cost_plan_v1.json'))
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = build_plan(args.specification.resolve())
    with args.specification.with_suffix('.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / 'plan.json', plan)
        execute(plan, args.output, args.after_pid)
