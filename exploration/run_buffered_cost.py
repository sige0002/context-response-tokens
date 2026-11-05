"""Run the attention-cache cost control after the currently scheduled GPU work."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from benchmark_real_video import summarize_rows
from encoder_costs import paired_costs
from exploration.benchmark_buffered_attention import VARIANTS,assert_exclusive,select_protocol
from point_experiment import read_json,write_json
from run_point_comparisons import bind_process
from video_checkpoint import file_hash


ROOT=Path(__file__).resolve().parents[1]


def build_plan(priority,delayed):
    for selection in ('shortest','median','longest'): select_protocol(priority,selection)
    names=['exploration/buffered_attention.py','exploration/benchmark_buffered_attention.py',
        'exploration/run_buffered_cost.py','exploration/test_buffered_attention.py',
        'exploration/evaluate_video_history.py','benchmark_real_video.py','benchmark_streaming.py',
        'encoder_costs.py','run_point_comparisons.py','temporal_priority.py','temporal_siglip.py',
        'video_checkpoint.py','robotap_data.py','train_video.py','pyproject.toml','uv.lock']
    return dict(priority=str(priority),priority_plan_sha256=file_hash(priority/'plan.json'),
        delayed=str(delayed),delayed_plan_sha256=file_hash(delayed/'plan.json'),
        videos=['shortest','median','longest'],variants=list(VARIANTS),
        sources={n:file_hash(ROOT/n) for n in names},
        scope='Same final seed42 checkpoints; implementation control after existing GPU jobs; no new model training')


def verify_plan(plan):
    for name,key in (('priority','priority_plan_sha256'),('delayed','delayed_plan_sha256')):
        if file_hash(Path(plan[name])/'plan.json')!=plan[key]: raise ValueError('Predecessor plan changed')
    for name,digest in plan['sources'].items():
        if file_hash(ROOT/name)!=digest: raise ValueError('Cost-control source changed: '+name)


def verify_report(directory,priority,selection):
    rows,cost,video=select_protocol(priority,selection)
    plan=read_json(directory/'plan.json'); report=read_json(directory/'summary.json')
    if report['plan_sha256']!=file_hash(directory/'plan.json') or report['correctness_sha256']!=file_hash(directory/'correctness.json'):
        raise ValueError('Cost report artifacts changed')
    if plan['video']!=video or plan['frames']!=video['frames'] or plan['capacity_frames']!=video['frames']:
        raise ValueError('Cost control uses a different video or capacity')
    if (plan['repeats']!=3 or plan['warmup_passes']!=1 or plan['output_tokens']!=[196,768]
            or plan['variants']!=list(VARIANTS) or plan['classifier_timed']
            or plan['priority_plan_sha256']!=file_hash(Path(priority)/'plan.json')):
        raise ValueError('Cost-control protocol differs')
    for path,digest in plan['source_sha256'].items():
        if file_hash(path)!=digest: raise ValueError('Cost measurement source changed')
    equality=read_json(directory/'correctness.json')
    checked=equality['rows']
    if equality['plan_sha256']!=report['plan_sha256'] or [r['frame'] for r in checked]!=list(range(video['frames'])):
        raise ValueError('Incomplete framewise numerical comparison')
    maximum=max(r['max_abs'] for r in checked)
    if (not 0<=maximum<=plan['max_abs_tolerance'] or maximum!=report['framewise_attention_max_abs']
            or all(r['exact'] for r in checked)!=report['framewise_attention_all_exact']
            or any(r['readout_predictions']['dynamic']!=r['readout_predictions']['buffered'] for r in checked)):
        raise ValueError('Framewise attention equivalence was not established')
    for name,row in rows.items():
        value=plan['checkpoints'][name]
        if value['config']!=row['config'] or value['step']!=20000 or value['checkpoint_sha256']!=file_hash(Path(row['run'])/'last.pt'):
            raise ValueError('Cost-control checkpoint differs')
    if len(report['measurements'])!=3 or {r['model'] for r in report['measurements']}!=set(VARIANTS):
        raise ValueError('Incomplete implementation comparison')
    for entry in report['measurements']:
        variant=entry['model']; folder=directory/variant; summary=read_json(folder/'summary.json')
        observations=[json.loads(s) for s in (folder/'timings.jsonl').read_text().splitlines() if s]
        if (entry['video']!=video or entry['measurement']!=summary
                or summary['timings_sha256']!=file_hash(folder/'timings.jsonl')):
            raise ValueError('Timing artifacts changed')
        recount=summarize_rows(observations,video['frames'],3)
        if any(summary.get(k)!=v for k,v in recount.items()): raise ValueError('Timing recount differs')
        for row in observations:
            expected=(79027200 if variant=='mamba_interleaved' else
                      602112*video['frames']+24 if variant=='attention_buffered' else 602112*row['absolute_frames_after'])
            if row['state_bytes']!=expected: raise ValueError('Allocated state accounting differs')
    if any(report.get(k)!=v for k,v in paired_costs(report['measurements']).items()):
        raise ValueError('Paired cost ratios differ from per-implementation measurements')
    return report['measurements']


def execute(plan,output,status,pid):
    child=fd=None; cancelled=False
    def save(): write_json(output/'status.json',status)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        if pid is not None:
            fd=bind_process(pid,'exploration.run_delayed_action','--output',plan['delayed'])
            poller=select.poll(); poller.register(fd,select.POLLIN)
            status.update(status='waiting_for_delayed_probe',dependency_pid=pid); save()
            while not poller.poll(1000):
                if cancelled: raise InterruptedError('Cancelled while waiting; predecessor unchanged')
            os.close(fd); fd=None
        if cancelled: raise InterruptedError('Queue cancelled')
        verify_plan(plan)
        if read_json(Path(plan['priority'])/'status.json')['status']!='complete':
            raise ValueError('Original training and cost measurements are incomplete')
        measurements=[]
        for selection in plan['videos']:
            if cancelled: raise InterruptedError('Queue cancelled')
            verify_plan(plan); assert_exclusive(); directory=output/selection
            if not (directory/'summary.json').exists():
                if directory.exists(): raise ValueError('Partial cost control; preserve and inspect it')
                command=['uv','run','--locked','python','-m','exploration.benchmark_buffered_attention',
                    '--priority',plan['priority'],'--selection',selection,'--output',str(directory)]
                entry=dict(selection=selection,command=command,started=time.time(),log=selection+'.log')
                status['operations'].append(entry); status['status']='running'; save()
                env=dict(os.environ); env.pop('VIRTUAL_ENV',None)
                with (output/entry['log']).open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    entry['pid']=child.pid; save()
                    if cancelled and child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
                    code=child.wait(); child=None
                entry.update(returncode=code,finished=time.time()); save()
                if cancelled: raise InterruptedError('Queue cancelled')
                if code: raise RuntimeError('Cost control failed: '+selection)
            measurements+=verify_report(directory,plan['priority'],selection)
        write_json(output/'summary.json',dict(plan_sha256=file_hash(output/'plan.json'),
            measurements=measurements,**paired_costs(measurements)))
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
    p.add_argument('--delayed',default='results/delayed_action_memory_v1/comparison_v2')
    p.add_argument('--output',required=True)
    p.add_argument('--after-pid',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(); os.chdir(ROOT); output=Path(args.output).resolve()
    plan=build_plan(Path(args.priority).resolve(),Path(args.delayed).resolve())
    if args.resume:
        if read_json(output/'plan.json')!=plan: raise ValueError('Cost-control plan changed')
        status=read_json(output/'status.json'); status.pop('error',None)
    else:
        output.mkdir(parents=True,exist_ok=False); write_json(output/'plan.json',plan)
        status=dict(status='planned',operations=[])
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); verify_plan(plan); write_json(output/'status.json',status)
        if not args.plan_only: execute(plan,output,status,args.after_pid)


if __name__=='__main__': main()
