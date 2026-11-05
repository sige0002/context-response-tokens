"""Queue final state-erasure comparisons after existing training and timing.

Evaluations run inside this process. Cancellation never signals predecessors
or unrelated processes. Partially written evaluations are preserved/refused.
"""
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import select
import signal
import subprocess

from exploration.final_memory_layers import MODELS,CONDITIONS,read,write,sha,aggregate


ROOT=Path(__file__).resolve().parents[1]


def build_plan(priority,buffered,cohort):
    upstream=read(priority/'plan.json'); old=read(cohort/'plan.json'); checked=read(cohort/'verification.json')
    if not checked['passed'] or checked['summary_sha256']!=sha(cohort/'summary.json'):
        raise ValueError('Original cohort diagnostic is not verified')
    candidates=[dict(r,model=r['name'],seed=42) for r in upstream['encoders']]+upstream['replications']
    rows=[r for r in candidates if r['model'] in MODELS]
    if len(rows)!=6 or {(r['model'],r['seed']) for r in rows}!={(m,s) for m in MODELS for s in (42,43,44)}:
        raise ValueError('Expected exactly the two architectures and three seeds')
    original=old['checkpoint_info']['config']
    for r in rows:
        c=r['config']
        if (c['steps']!=20000 or c['frames']!=8 or c['classes']!=174 or c['placement']!='interleaved'
                or any(c[k]!=original[k] for k in ('model','revision','val_sha256'))):
            raise ValueError('Final model protocol differs from the original cohort')
    if len(old['subset'])!=348 or len({r['id'] for r in old['subset']})!=348:
        raise ValueError('Incomplete or duplicate cohort')
    sources=['exploration/final_memory_layers.py','exploration/run_final_memory_layers.py',
        'exploration/test_final_memory_layers.py','exploration/memory_layer_intervention.py',
        'exploration/test_memory_layer_intervention.py','exploration/evaluate_video_history.py',
        'evaluate_video.py','temporal_priority.py','temporal_siglip.py','temporal_readout.py',
        'video_checkpoint.py','video_data.py','train_video.py','pyproject.toml','uv.lock']
    return dict(priority=str(priority),priority_plan_sha256=sha(priority/'plan.json'),
        buffered=str(buffered),buffered_plan_sha256=sha(buffered/'plan.json'),cohort=str(cohort),
        cohort_hashes={n:sha(cohort/n) for n in ('plan.json','summary.json','verification.json','inputs.json')},
        subset=old['subset'],conditions={k:list(v) for k,v in CONDITIONS.items()},
        runs=[dict(name=r['model']+f"_seed{r['seed']}",model=r['model'],seed=r['seed'],run=r['run'],config=r['config']) for r in rows],
        sources={n:sha(ROOT/n) for n in sources},
        scope='Final checkpoint, matched batch-one per-frame inference. Recompute every baseline for both models and all seeds.',
        limitations=['Exploratory extension after seeing interim layer-state results; no untouched-test claim.',
            'All five conditions use identical current RGB, weights, transforms and absolute offsets.',
            'Attention reset_all uses the original observation index, not time zero.',
            'This is a dependence diagnostic, not accuracy of a separately trained reduced-memory model.'])


def verify_plan(plan):
    for key in ('priority','buffered'):
        if sha(Path(plan[key])/'plan.json')!=plan[key+'_plan_sha256']: raise ValueError('Predecessor plan changed')
    for name,digest in plan['cohort_hashes'].items():
        if sha(Path(plan['cohort'])/name)!=digest: raise ValueError('Cohort changed')
    for name,digest in plan['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Diagnostic source changed: '+name)


def bind_buffered(pid,expected):
    libc=ctypes.CDLL(None,use_errno=True);libc.pidfd_open.argtypes=[ctypes.c_int,ctypes.c_uint];libc.pidfd_open.restype=ctypes.c_int
    fd=libc.pidfd_open(pid,0)
    if fd<0: raise OSError(ctypes.get_errno(),'Cannot bind cost-control predecessor')
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if 'exploration.run_buffered_cost' not in args or '--output' not in args:
            raise ValueError('Dependency is not the buffered-cost queue')
        location=Path(f'/proc/{pid}/cwd').resolve()/args[args.index('--output')+1]
        if location.resolve()!=expected.resolve(): raise ValueError('Dependency writes to a different directory')
        return fd
    except BaseException:
        os.close(fd);raise


def assert_no_other_gpu_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(s.strip().isdigit() and int(s)!=os.getpid() and Path('/proc',s.strip()).exists() for s in text.splitlines()):
        raise RuntimeError('Another GPU job is active; refusing concurrent evaluation')


def execute(plan,output,status,pid):
    cancelled=False;fd=None
    def save(): write(output/'status.json',status)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
    def check():
        if cancelled: raise InterruptedError('Final memory diagnostic cancelled; predecessors unchanged')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if pid is not None:
            fd=bind_buffered(pid,Path(plan['buffered']));poller=select.poll();poller.register(fd,select.POLLIN)
            status.update(status='waiting_for_buffered_cost',dependency_pid=pid);save()
            while not poller.poll(1000): check()
            os.close(fd);fd=None
        check();verify_plan(plan)
        if any(read(Path(plan[k])/'status.json')['status']!='complete' for k in ('priority','buffered')):
            raise ValueError('Existing training/measurement queues did not complete')
        from exploration.final_memory_layers import run_one,verify_one
        records={}
        for row in plan['runs']:
            check();verify_plan(plan);assert_no_other_gpu_processes()
            directory=output/row['name'];status.update(status='running',current=row['name']);save()
            if not (directory/'summary.json').exists():
                if directory.exists(): raise ValueError('Partial diagnostic must be inspected before reuse')
                run_one(plan,output/'plan.json',row,directory,check)
            check();verify_plan(plan)
            records[row['name']]=verify_one(plan,output/'plan.json',row,directory)
            if row['name'] not in status['completed']: status['completed'].append(row['name'])
            save()
        write(output/'summary.json',dict(plan_sha256=sha(output/'plan.json'),
            run_summaries={n:sha(output/n/'summary.json') for n in records},**aggregate(records)))
        status['status']='complete'
    except BaseException as exc:
        status.update(status='stopped' if cancelled else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if fd is not None: os.close(fd)
        save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--priority',default='results/temporal_priority_seed42_rgbfix')
    p.add_argument('--buffered',default='results/buffered_attention_cost_control')
    p.add_argument('--cohort',default='results/video_history_intervention_seed17')
    p.add_argument('--output',required=True);p.add_argument('--after-pid',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();os.chdir(ROOT);output=Path(args.output).resolve()
    plan=build_plan(*(Path(getattr(args,k)).resolve() for k in ('priority','buffered','cohort')))
    if args.resume:
        if read(output/'plan.json')!=plan: raise ValueError('Existing final diagnostic plan differs')
        status=read(output/'status.json');status.pop('error',None)
    else:
        output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',plan)
        status=dict(status='planned',completed=[])
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_plan(plan);write(output/'status.json',status)
        if not args.plan_only: execute(plan,output,status,args.after_pid)


if __name__=='__main__': main()
