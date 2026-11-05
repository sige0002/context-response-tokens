"""Wait for the bound main run, then execute the independent MIMO pilot.

Only this queue's own child process group is cancelled. Existing training,
comparison queues and their source/configuration files are never signalled.
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

from exploration.mimo_pilot import aggregate, checked_plan
from exploration.temporal_density import read, write
from video_checkpoint import file_hash

ROOT=Path(__file__).resolve().parents[1]


def predecessor_complete(run):
    events=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines() if line]
    return bool(events and events[-1].get('event')=='budget_complete' and events[-1].get('step')==20000)


def bind_predecessor(pid, run):
    libc=ctypes.CDLL(None,use_errno=True)
    libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint];libc.pidfd_open.restype=ctypes.c_int
    fd=libc.pidfd_open(pid,0)
    if fd<0: raise OSError(ctypes.get_errno(),'Cannot bind the main training process')
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if 'train_video.py' not in [Path(a).name for a in args] or '--output' not in args:
            raise ValueError('Dependency is not the main video trainer')
        location=Path(f'/proc/{pid}/cwd').resolve()/args[args.index('--output')+1]
        if location.resolve()!=run.resolve(): raise ValueError('Dependency writes to another run')
        ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        return fd,dict(pid=pid,start_ticks=ticks,run=str(run.resolve()))
    except BaseException:
        os.close(fd);raise


def stage_ready(output, arm, mode):
    folder=output/arm
    record=folder/('training.json' if mode=='train' else 'streaming.json')
    if not record.exists(): return False
    saved=read(record);artifact='clip' if mode=='train' else 'stream'
    if (saved['plan_sha256']!=file_hash(output/'plan.json') or
        saved['model_sha256']!=file_hash(folder/'model.pt') or
        saved[artifact+'_sha256']!=file_hash(folder/(artifact+'.npz'))):
        raise ValueError('Completed pilot stage artifacts changed')
    return True


def run(output,after_pid):
    plan=checked_plan(output);status=dict(status='starting',completed=[])
    cancelled=False;fd=None;child=None
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
    def check():
        if cancelled: raise InterruptedError('MIMO pilot cancelled; predecessor untouched')
    def save(): write(output/'status.json',status)
    def stop_child():
        nonlocal child
        if child is not None and child.poll() is None:
            # start_new_session below makes this group belong to this child.
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        predecessor=Path(plan['predecessor']).resolve()
        if after_pid is not None:
            fd,bound=bind_predecessor(after_pid,predecessor)
            status.update(status='waiting_for_main_20000',dependency=bound);save()
            poller=select.poll();poller.register(fd,select.POLLIN)
            while not poller.poll(1000): check()
            os.close(fd);fd=None
        check();checked_plan(output)
        if not predecessor_complete(predecessor):
            raise ValueError('The bound main run did not complete its 20000-update budget')
        for mode in ('train','stream'):
            for arm in plan['arms']:
                check();checked_plan(output);key=f'{arm}_{mode}'
                if stage_ready(output,arm,mode):
                    status['completed'].append(key);save();continue
                log=output/(key+'.log')
                if log.exists(): raise ValueError('Partial stage exists; inspect before restarting: '+key)
                prefix=['uv','run','--locked'] if mode=='train' else ['uv','run','--no-sync','--with','apache-tvm-ffi==0.1.10']
                command=prefix+['python','-m','exploration.mimo_pilot','--output',str(output),
                                '--mode',mode,'--arm',arm]
                status.update(status='running',current=key,command=command);save()
                with log.open('x') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    status['child_pid']=child.pid;save()
                    while child.poll() is None:
                        check();time.sleep(1)
                    if child.returncode!=0: raise RuntimeError(f'{key} failed with exit {child.returncode}')
                child=None;status.pop('child_pid',None)
                if not stage_ready(output,arm,mode): raise ValueError('Stage exited without complete artifacts')
                status['completed'].append(key);save()
        checked_plan(output);aggregate(output)
        status.update(status='complete',summary_sha256=file_hash(output/'summary.json'))
    except BaseException as exc:
        status.update(status='stopped' if cancelled else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop_child()
        if fd is not None: os.close(fd)
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--after-pid',type=int)
    a=p.parse_args();os.chdir(ROOT);output=a.output.resolve()
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(output,a.after_pid)
