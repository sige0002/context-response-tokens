"""Queue five trained encoders on three prespecified full real videos."""
import argparse
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from encoder_costs import select_videos, verify_measurement, paired_costs
from gmmix_experiment import UV, verify_completion as gmmix_completion
from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from run_point_comparisons import bind_process
from run_video_comparisons import verify_completion, verify_inputs
from summarize_full_video import verify_shared_protocol


ROOT = Path(__file__).resolve().parent
SOURCES = ('run_encoder_costs.py','encoder_costs.py','benchmark_real_video.py','benchmark_streaming.py',
    'video_checkpoint.py','gmmix_checkpoint.py','temporal_siglip.py','gmmix_siglip.py','robotap_data.py',
    'train_video.py','run_point_comparisons.py','point_experiment.py','gmmix_experiment.py',
    'run_video_comparisons.py','summarize_full_video.py','point_checkpoint.py',
    'pyproject.toml','uv.lock','baselines/gmmix/pyproject.toml','baselines/gmmix/uv.lock',
    'baselines/gmmix/source_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-comparisons', required=True)
    parser.add_argument('--after-point-pid', type=int)
    parser.add_argument('--data-audit', default='results/robotap_data_audit.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    queue, output, audit = [Path(p).resolve() for p in (args.point_comparisons,args.output,args.data_audit)]
    source_plan, source_audit = read_json(queue/'plan.json'),read_json(audit)
    if source_audit['videos'] != 265 or len(source_audit['rows']) != 265:
        raise ValueError('Expected audited full RoboTAP data before score-independent selection')
    encoders = source_plan['encoders']
    expected = {('mamba','interleaved'),('mamba','late'),('attention','interleaved'),
                ('none','interleaved'),('gmmix','after_encoder')}
    if len(encoders) != 5 or {(r['config']['kind'],r['config']['placement']) for r in encoders} != expected:
        raise ValueError('Expected all five matched encoders')
    verify_shared_protocol([row['config'] for row in encoders])
    for row in encoders:
        verify_inputs(row['config'])
    output.mkdir(parents=True,exist_ok=False)
    plan = dict(point_queue=str(queue),point_plan_sha256=file_hash(queue/'plan.json'),
        dataset_audit=str(audit),dataset_audit_sha256=file_hash(audit),
        data_root=source_plan['robotap_root'],extraction_sha256=source_plan['robotap_extraction_sha256'],
        selection='Shortest, median and longest distinct RGB videos by (frame count, id); no accuracy-based selection',
        videos=select_videos(source_audit['rows']),encoders=encoders,repeats=3,warmup_passes=1,
        source_sha256={name:file_hash(ROOT/name) for name in SOURCES},
        scope='Five final video-trained encoders, three full real RGB sequences, batch one; not point heads or recognition quality')
    write_json(output/'plan.json',plan)
    for name in SOURCES:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    if args.plan_only:
        return
    status=dict(status='starting',dependency_pid=args.after_point_pid,operations=[],measurements=[])
    descriptor,child,cancelled=None,None,False
    def save():
        write_json(output/'status.json',status)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
    def verify_plan():
        for name,digest in plan['source_sha256'].items():
            if file_hash(ROOT/name) != digest:
                raise ValueError(f'Cost source changed: {name}')
        for path,digest in [(queue/'plan.json',plan['point_plan_sha256']),
            (audit,plan['dataset_audit_sha256']),
            (Path(plan['data_root'])/'extraction.json',plan['extraction_sha256'])]:
            if file_hash(path) != digest:
                raise ValueError(f'Cost input record changed: {path}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if args.after_point_pid is not None:
            descriptor=bind_process(args.after_point_pid,'run_point_comparisons.py','--output',queue)
            status['status']='waiting_for_point_gpu_queue';save()
            poller=select.poll();poller.register(descriptor,select.POLLIN)
            while not poller.poll(1000):
                if cancelled:raise InterruptedError('Cost queue cancelled; dependencies unchanged')
            os.close(descriptor);descriptor=None
        if cancelled:raise InterruptedError('Cost queue cancelled')
        verify_plan()
        # The point queue is a GPU scheduling dependency, not an accuracy gate.
        # Video training itself must have completed in full for every model.
        completed = {}
        for row in encoders:
            verify_inputs(row['config'])
            completed[row['run']] = (gmmix_completion if row['config']['kind']=='gmmix' else verify_completion)(row['run'],row['config'])
        write_json(output/'completed_encoders.json',completed)
        status['status']='measuring';save()
        for row in encoders:
            config=row['config'];name=config['kind']+'_'+config['placement']
            for video in plan['videos']:
                if cancelled:raise InterruptedError('Cost queue cancelled')
                verify_plan()
                checkpoint=Path(row['run'])/'last.pt'
                if file_hash(checkpoint)!=completed[row['run']]['checkpoint_sha256']:
                    raise ValueError('Completed encoder checkpoint changed')
                destination=output/name/video['selection']
                command=UV+['benchmark_real_video.py','--checkpoint',str(checkpoint),
                    '--encoder-type','gmmix' if config['kind']=='gmmix' else 'temporal',
                    '--root',plan['data_root'],'--video-id',video['id'],
                    '--repeats',str(plan['repeats']),'--warmup-passes',str(plan['warmup_passes']),
                    '--output',str(destination)]
                log_path=output/(name+'_'+video['selection']+'.log')
                operation=dict(command=command,log=str(log_path),started=time.time(),status='running')
                status['operations'].append(operation);save()
                env=dict(os.environ);env.pop('VIRTUAL_ENV',None)
                with log_path.open('w') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    operation['pid']=child.pid;save()
                    code=child.wait();child=None
                operation.update(returncode=code,finished=time.time(),status='complete' if code==0 else 'failed');save()
                if cancelled:raise InterruptedError('Cost queue cancelled')
                if code:raise RuntimeError(f'Cost measurement failed: {log_path}')
                measurement=verify_measurement(destination,config,completed[row['run']],video,
                    plan['repeats'],plan['warmup_passes'],plan['extraction_sha256'])
                status['measurements'].append(dict(model=name,video=video,measurement=measurement));save()
        verify_plan()
        write_json(output/'summary.json',dict(plan_sha256=file_hash(output/'plan.json'),
            measurements=status['measurements'],**paired_costs(status['measurements'])))
        status['status']='complete'
    except BaseException as exc:
        status.update(status='stopped' if cancelled else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        if descriptor is not None:os.close(descriptor)
        save()


if __name__=='__main__':
    main()
