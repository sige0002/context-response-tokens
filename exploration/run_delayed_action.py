"""Run matched delayed-action probes after the fixed training/cost queue.

Only this queue's children receive cancellation. Existing training is neither
stopped nor rescheduled. Partial caches are preserved and must be inspected.
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

from exploration.delayed_action_data import sha,verify_plan as verify_data


ROOT=Path(__file__).resolve().parents[1]
MODELS=('mamba_interleaved','attention_interleaved')


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path); temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def build_plan(priority,data_path):
    upstream=read(priority/'plan.json'); data=read(data_path); verify_data(data)
    candidates=[dict(r,model=r['name'],seed=42) for r in upstream['encoders']]
    candidates+=upstream['replications']
    rows=[r for r in candidates if r['model'] in MODELS]
    if len(rows)!=6 or {(r['model'],r['seed']) for r in rows}!={(m,s) for m in MODELS for s in (42,43,44)}:
        raise ValueError('Expected the fixed two-model, three-seed comparison')
    if any(r['config']['steps']!=20000 or r['config']['frames']!=8 for r in rows):
        raise ValueError('Unexpected encoder training budget')
    sources=['exploration/run_delayed_action.py','exploration/cache_delayed_action.py',
             'exploration/delayed_action_probe.py','exploration/delayed_action_data.py',
             'exploration/test_delayed_action.py','temporal_priority.py',
             'exploration/evaluate_video_history.py','temporal_siglip.py','temporal_readout.py',
             'video_data.py','video_checkpoint.py','train_video.py','pyproject.toml','uv.lock']
    return dict(priority=str(priority),priority_plan_sha256=sha(priority/'plan.json'),
        data=str(data_path),data_plan_sha256=sha(data_path),
        runs=[dict(name=r['model']+f"_seed{r['seed']}",model=r['model'],seed=r['seed'],
                   run=r['run'],config=r['config']) for r in rows],
        source_sha256={s:sha(ROOT/s) for s in sources},
        scope='Frozen final encoder, separate common-capacity probe, all two-model/three-seed results',
        limitations=data['limits'])


def verify(plan):
    if sha(Path(plan['priority'])/'plan.json')!=plan['priority_plan_sha256'] or sha(plan['data'])!=plan['data_plan_sha256']:
        raise ValueError('Pinned experimental plan changed')
    for path,digest in plan['source_sha256'].items():
        if sha(ROOT/path)!=digest: raise ValueError('Pinned probe source changed: '+path)


def bind_priority(pid,expected):
    # The uv-managed Python on this host does not expose os.pidfd_open.
    libc=ctypes.CDLL(None,use_errno=True)
    libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint]
    libc.pidfd_open.restype=ctypes.c_int
    fd=libc.pidfd_open(pid,0)
    if fd<0:
        code=ctypes.get_errno()
        raise OSError(code,os.strerror(code))
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if 'run_temporal_priority.py' not in [Path(s).name for s in args if s]:
            raise ValueError('Dependency is not the training/cost priority queue')
        location=args[args.index('--output')+1]
        cwd=Path(f'/proc/{pid}/cwd').resolve()
        if (cwd/location).resolve()!=expected.resolve():
            raise ValueError('Dependency belongs to another run')
        return fd
    except BaseException:
        os.close(fd); raise


def assert_idle():
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(s.strip().isdigit() and Path('/proc',s.strip()).exists() for s in pids.splitlines()):
        raise RuntimeError('GPU is busy; refusing to overlap training or cost measurements')


def execute(plan,output,status,pid):
    child=fd=None; cancelled=False
    def save(): write(output/'status.json',status)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    def check():
        if cancelled: raise InterruptedError('Delayed-action queue cancelled')
        verify(plan)
    def launch(command,name):
        nonlocal child
        check(); assert_idle()
        entry=dict(name=name,command=command,started=time.time(),log=f'{len(status["operations"])+1:03d}_{name}.log')
        status['status']='running'; status['operations'].append(entry); save()
        env=dict(os.environ,OPENBLAS_NUM_THREADS='2'); env.pop('VIRTUAL_ENV',None)
        with (output/entry['log']).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            entry['pid']=child.pid; save()
            if cancelled and child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
            code=child.wait(); child=None
        entry.update(returncode=code,finished=time.time()); save()
        if cancelled: raise InterruptedError('Cancelled after child exited')
        if code: raise RuntimeError('Probe stage failed: '+name)
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    try:
        if pid is not None:
            fd=bind_priority(pid,Path(plan['priority']))
            status.update(status='waiting_for_priority',dependency_pid=pid); save()
            poller=select.poll(); poller.register(fd,select.POLLIN)
            while not poller.poll(1000):
                if cancelled: raise InterruptedError('Cancelled while waiting; predecessor unchanged')
            os.close(fd); fd=None
        check()
        if read(Path(plan['priority'])/'status.json')['status']!='complete':
            raise ValueError('Training/cost/seed queue has not completed successfully')
        # Import training validators only when the predecessor actually finishes.
        from temporal_priority import verify_video
        from exploration.delayed_action_probe import evaluate
        import numpy as np
        summaries={}
        for row in plan['runs']:
            check(); completed,_=verify_video(row)
            directory=output/row['name']; cache=directory/'cache'; probe=directory/'probe'
            if not (cache/'summary.json').exists():
                if cache.exists(): raise ValueError('Partial feature cache; preserve and inspect: '+str(cache))
                launch(['uv','run','--locked','python','-m','exploration.cache_delayed_action',
                    '--plan',plan['data'],'--checkpoint',str(Path(row['run'])/'last.pt'),
                    '--output',str(cache)],row['name']+'_cache')
            cached=read(cache/'summary.json'); config=read(cache/'plan.json')
            if (cached['scope']!='complete_cache' or cached['plan_sha256']!=sha(cache/'plan.json')
                    or cached['features_sha256']!=sha(cache/'features.npz')
                    or config['checkpoint_info']['checkpoint_sha256']!=completed['checkpoint_sha256']
                    or config['data_plan_sha256']!=plan['data_plan_sha256']):
                raise ValueError('Cached features belong to a different or incomplete experiment')
            if not (probe/'summary.json').exists():
                if probe.exists(): raise ValueError('Partial probe result; preserve and inspect')
                launch(['uv','run','--locked','python','-m','exploration.delayed_action_probe',
                    '--cache',str(cache),'--output',str(probe)],row['name']+'_probe')
            measured=read(probe/'summary.json')
            if measured['cache_summary_sha256']!=sha(cache/'summary.json') or measured['readouts_sha256']!=sha(probe/'readouts.npz'):
                raise ValueError('Probe artifacts changed')
            with np.load(cache/'features.npz') as features:
                recounted,_=evaluate(features,config['data'])
            if recounted!=measured['results']: raise ValueError('Probe recomputation differs')
            summaries[row['name']]=dict(model=row['model'],seed=row['seed'],
                checkpoint_sha256=completed['checkpoint_sha256'],probe_summary_sha256=sha(probe/'summary.json'),
                results=measured['results'])
            if row['name'] not in status['completed']: status['completed'].append(row['name'])
            save()
        paired={}
        for condition in read(plan['data'])['conditions']:
            by_delay={}
            for seed in (42,43,44):
                a=summaries[f'mamba_interleaved_seed{seed}']['results'][condition]['validation']
                b=summaries[f'attention_interleaved_seed{seed}']['results'][condition]['validation']
                if [r['delay_observations'] for r in a]!=[r['delay_observations'] for r in b]:
                    raise ValueError('Model delay grids differ')
                for left,right in zip(a,b):
                    delta=100*(left['accuracy']-right['accuracy'])
                    by_delay.setdefault(str(left['delay_observations']),[]).append(delta)
            paired[condition]={d:dict(seed_order=[42,43,44],differences_pp=v,
                mean_pp=float(np.mean(v)),sample_sd_pp=float(np.std(v,ddof=1))) for d,v in by_delay.items()}
        write(output/'summary.json',dict(plan_sha256=sha(output/'plan.json'),runs=summaries,
            paired_mamba_minus_attention=paired,
            scope='All six predeclared frozen-encoder probes; no score-based exclusion',
            limitations=plan['limitations']))
        status['status']='complete'
    except BaseException as exc:
        status.update(status='stopped' if cancelled else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if fd is not None: os.close(fd)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--priority',default='results/temporal_priority_seed42_rgbfix')
    p.add_argument('--data',default='results/delayed_action_memory_v1/data_plan.json')
    p.add_argument('--output',required=True)
    p.add_argument('--after-pid',type=int)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); os.chdir(ROOT); output=Path(args.output).resolve()
    plan=build_plan(Path(args.priority).resolve(),Path(args.data).resolve())
    if args.resume:
        if read(output/'plan.json')!=plan: raise ValueError('Existing queue plan differs')
        status=read(output/'status.json'); status.pop('error',None)
    else:
        output.mkdir(parents=True,exist_ok=False); write(output/'plan.json',plan)
        status=dict(status='planned',operations=[],completed=[])
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); verify(plan)
        write(output/'status.json',status)
        if not args.plan_only: execute(plan,output,status,args.after_pid)


if __name__=='__main__': main()
