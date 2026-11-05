"""Execute temporal recognition and cost before optional point-transfer work."""
import argparse
import fcntl
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from encoder_costs import verify_measurement, paired_costs
from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from run_point_comparisons import bind_process
from temporal_priority import (ROOT, UV, build_plan, verify_plan, video_verifier,
    verify_video, summarize_videos, summarize_replications)


def assert_idle():
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(s.strip().isdigit() and Path('/proc',s.strip()).exists() for s in pids.splitlines()):
        raise RuntimeError('GPU compute process present; refusing overlapping priority experiment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',default='results/ssv2_full_comparisons_seed42_rgbfix')
    parser.add_argument('--gmmix',default='results/ssv2_gmmix_comparison_seed42_rgbfix')
    parser.add_argument('--costs',default='results/encoder_costs_seed42_rgbfix_v2')
    parser.add_argument('--multidepth',default='results/ssv2_gmmix_multidepth_comparison_seed42_rgbfix')
    parser.add_argument('--replications',default='results/seed_replications_rgbfix')
    parser.add_argument('--output',required=True)
    parser.add_argument('--after-gmmix-pid',type=int)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(); os.chdir(ROOT); torch.set_num_threads(2)
    output = Path(args.output).resolve()
    plan = build_plan(*(Path(getattr(args,k)).resolve() for k in ('video','gmmix','costs','multidepth','replications')),output)
    if args.resume:
        if read_json(output/'plan.json')!=plan: raise ValueError('Priority plan changed')
        status = read_json(output/'status.json'); status.pop('error',None)
    else:
        output.mkdir(parents=True,exist_ok=False); write_json(output/'plan.json',plan)
        for name in plan['source_sha256']:
            target = output/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        status = dict(status='planned',operations=[],completed_videos={},measurements=[])
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_plan(plan)
        status['dependency_pid'] = args.after_gmmix_pid; write_json(output/'status.json',status)
        if args.plan_only:
            print('Fixed video/cost/seed protocol; no point-training execution dependency',flush=True); return
        execute(plan,output,status,args.after_gmmix_pid)


def execute(plan,output,status,predecessor):
    child = fd = None; cancelled = False
    def save(): write_json(output/'status.json',status)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    def check():
        if cancelled: raise InterruptedError('Priority queue cancelled; dependencies unchanged')
        verify_plan(plan)
    def launch(command,name):
        nonlocal child
        check(); assert_idle()
        row = dict(name=name,command=command,started=time.time(),status='running',log=f'{len(status["operations"])+1:03d}_{name}.log')
        status['operations'].append(row); status['status']='running'; save()
        env=dict(os.environ); env.pop('VIRTUAL_ENV',None)
        with (output/row['log']).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            row['pid']=child.pid; save()
            if cancelled and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
            code=child.wait(); child=None
        row.update(returncode=code,finished=time.time(),status='complete' if code==0 else 'failed'); save()
        if cancelled: raise InterruptedError('Priority queue cancelled')
        if code: raise RuntimeError(f'{name} failed; inspect {row["log"]}')
    def train_and_evaluate(row):
        check(); run=Path(row['run']); checkpoint=run/'last.pt'
        finished=False
        if checkpoint.exists():
            if read_json(run/'config.json')!=row['config']: raise ValueError('Partial training configuration changed')
            import json
            logs=[json.loads(s) for s in (run/'progress.jsonl').read_text().splitlines() if s]
            finished=bool(logs and logs[-1].get('event')=='budget_complete')
        if not finished: launch(row['command']+(['--resume'] if checkpoint.exists() else []),row['name']+'_train')
        completed=video_verifier(row['config'])(run,row['config'])
        for i,command in enumerate(row['evaluations']):
            check()
            if file_hash(checkpoint)!=completed['checkpoint_sha256']: raise ValueError('Final weights changed')
            destination=Path(command[command.index('--output')+1])
            if not destination.exists(): launch(command,row['name']+f'_eval{i}')
        checked,_=verify_video(row); status['completed_videos'][row['name']]=checked; save()
    def measure(row):
        check(); completed=video_verifier(row['config'])(row['run'],row['config'])
        config=row['config']; cost=plan['cost']; multi=config['kind']=='gmmix_multidepth'
        for video in cost['videos']:
            check(); checkpoint=Path(row['run'])/'last.pt'
            if file_hash(checkpoint)!=completed['checkpoint_sha256']: raise ValueError('Cost checkpoint changed')
            directory=output/'costs'/row['name']/video['selection']
            command=UV+['benchmark_multidepth_video.py' if multi else 'benchmark_real_video.py',
                '--checkpoint',str(checkpoint),'--encoder-type',
                'gmmix_multidepth' if multi else 'gmmix' if config['kind']=='gmmix' else 'temporal',
                '--root',cost['data_root'],'--video-id',video['id'],'--repeats',str(cost['repeats']),
                '--warmup-passes',str(cost['warmup_passes']),'--output',str(directory)]
            if not (directory/'summary.json').exists():
                if directory.exists(): raise ValueError('Partial cost measurement exists; preserve and inspect it')
                launch(command,row['name']+'_'+video['selection']+'_cost')
            verified=verify_measurement(directory,config,completed,video,cost['repeats'],cost['warmup_passes'],cost['extraction_sha256'])
            if multi and any(b['state_bytes_min']!=b['state_bytes_max'] for b in verified['bins']):
                raise ValueError('Multi-depth recurrent state grew')
            item=dict(model=row['name'],video=video,measurement=verified)
            prior=[m for m in status['measurements'] if m['model']==row['name'] and m['video']['id']==video['id']]
            if prior and prior!=[item]: raise ValueError('Previously checked cost result changed')
            if not prior: status['measurements'].append(item)
            save()
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        if predecessor:
            queue=Path(plan['execution_dependencies']['gmmix'])
            fd=bind_process(predecessor,'run_gmmix_comparison.py','--output',queue)
            poller=select.poll(); poller.register(fd,select.POLLIN)
            status['status']='waiting_for_video_comparisons'; save()
            while not poller.poll(1000):
                if cancelled: raise InterruptedError('Cancelled while waiting; dependency unchanged')
            os.close(fd); fd=None
        check()
        for path in plan['execution_dependencies'].values():
            if read_json(Path(path)/'status.json')['status']!='complete': raise ValueError('Video comparisons did not complete')
        write_json(output/'five_video_summary.json',summarize_videos(plan['encoders']))
        for row in plan['encoders']: measure(row)
        write_json(output/'five_costs.json',dict(measurements=status['measurements'],**paired_costs(status['measurements'])))
        train_and_evaluate(plan['multidepth']); measure(plan['multidepth'])
        write_json(output/'six_video_summary.json',summarize_videos(plan['encoders']+[plan['multidepth']]))
        write_json(output/'six_costs.json',dict(measurements=status['measurements'],**paired_costs(status['measurements'])))
        for row in plan['replications']: train_and_evaluate(row)
        write_json(output/'three_seed_video_summary.json',summarize_replications(plan))
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


if __name__=='__main__': main()
