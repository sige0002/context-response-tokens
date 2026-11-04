"""Run prespecified state interventions on the five full frozen-encoder probes."""
import argparse
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from benchmark_streaming import idle_compute_processes
from gmmix_experiment import UV
from point_checkpoint import file_hash
from point_experiment import read_json,write_json
from run_point_comparisons import bind_process,verify_plan_inputs
from summarize_points import read_completed


ROOT=Path(__file__).resolve().parent
SOURCES=('run_point_history_comparisons.py','analyze_point_history.py','evaluate_point_history.py',
    'point_history.py','point_checkpoint.py','point_readout.py','temporal_siglip.py','gmmix_siglip.py',
    'robotap_data.py','tapvid_metrics.py','audit_robotap_predictions.py','summarize_points.py',
    'paired_metrics.py','point_experiment.py','run_point_comparisons.py','gmmix_experiment.py',
    'train_video.py','benchmark_streaming.py','pyproject.toml','uv.lock',
    'baselines/gmmix/pyproject.toml','baselines/gmmix/uv.lock','baselines/gmmix/source_manifest.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-comparisons',required=True)
    parser.add_argument('--cost-comparisons',required=True)
    parser.add_argument('--after-cost-pid',type=int)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT);torch.set_num_threads(2)
    points,costs,output=[Path(p).resolve() for p in (args.point_comparisons,args.cost_comparisons,args.output)]
    point_plan,cost_plan=read_json(points/'plan.json'),read_json(costs/'plan.json')
    if Path(cost_plan['point_queue'])!=points or cost_plan['point_plan_sha256']!=file_hash(points/'plan.json'):
        raise ValueError('Cost and point queues refer to different experiments')
    verify_plan_inputs(point_plan)
    probes=[row for row in point_plan['runs'] if row['mode']=='probe']
    expected={('mamba','interleaved'),('none','interleaved'),('attention','interleaved'),('mamba','late'),('gmmix','after_encoder')}
    if len(probes)!=5 or {(r['encoder_config']['kind'],r['encoder_config']['placement']) for r in probes}!=expected:
        raise ValueError('Expected all five frozen-encoder probes')
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(point_queue=str(points),cost_queue=str(costs),point_plan_sha256=file_hash(points/'plan.json'),
        cost_plan_sha256=file_hash(costs/'plan.json'),runs=probes,periods=[1,24],
        data_root=point_plan['robotap_root'],dataset_extraction_sha256=point_plan['robotap_extraction_sha256'],
        query_anchors='Common full-history descriptors, created at query arrival; retained under resets',
        primary='Within interleaved Mamba probe: all-post-query average Jaccard, full state minus reset every frame',
        controls='All five encoders; history-free encoder must be unchanged. Reset-24 and age/reappearance strata are exploratory.',
        scope='Conditional ongoing-feature state effect; not wholly history-free, not a guarantee of long-term retention',
        source_sha256={name:file_hash(ROOT/name) for name in SOURCES})
    write_json(output/'plan.json',plan)
    for name in SOURCES:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    if args.plan_only:return
    status=dict(status='starting',dependency_pid=args.after_cost_pid,operations=[],results={})
    descriptor,child,cancelled=None,None,False
    def save():write_json(output/'status.json',status)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    def check_plan():
        for name,digest in plan['source_sha256'].items():
            if file_hash(ROOT/name)!=digest:raise ValueError(f'History evaluation source changed: {name}')
        for path,digest in [(points/'plan.json',plan['point_plan_sha256']),(costs/'plan.json',plan['cost_plan_sha256']),
                           (Path(plan['data_root'])/'extraction.json',plan['dataset_extraction_sha256'])]:
            if file_hash(path)!=digest:raise ValueError(f'History input record changed: {path}')
    def launch(command,label,cpu=False):
        nonlocal child
        if cancelled:raise InterruptedError('History queue cancelled')
        check_plan()
        if not cpu:idle_compute_processes()
        log_path=output/(label+'.log');operation=dict(command=command,started=time.time(),status='running',log=str(log_path))
        status['operations'].append(operation);save()
        env=dict(os.environ);env.pop('VIRTUAL_ENV',None)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        with log_path.open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            operation['pid']=child.pid;save();code=child.wait();child=None
        operation.update(returncode=code,finished=time.time(),status='complete' if code==0 else 'failed');save()
        if cancelled:raise InterruptedError('History queue cancelled')
        if code:raise RuntimeError(f'History operation failed: {log_path}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if args.after_cost_pid is not None:
            descriptor=bind_process(args.after_cost_pid,'run_encoder_costs.py','--output',costs)
            status['status']='waiting_for_cost_gpu_queue';save()
            poller=select.poll();poller.register(descriptor,select.POLLIN)
            while not poller.poll(1000):
                if cancelled:raise InterruptedError('History queue cancelled; dependencies unchanged')
            os.close(descriptor);descriptor=None
        if cancelled:raise InterruptedError('History queue cancelled')
        check_plan()
        # Revalidate full training, all official ordinary predictions and their
        # raw-label audits before reusing any probe checkpoint.
        _,_,completed=read_completed(points)
        status['status']='evaluating';save()
        for row in probes:
            name=row['name'];ordinary=completed[name];checkpoint=Path(row['run'])/'last.pt'
            if file_hash(checkpoint)!=ordinary['checkpoint_sha256']:raise ValueError('Point checkpoint changed')
            destination=output/name
            launch(UV+['evaluate_point_history.py','--checkpoint',str(checkpoint),'--root',plan['data_root'],
                '--output',str(destination),'--periods','1','24'],name)
            analysis=destination/'analysis.json'
            launch(UV+['analyze_point_history.py','--interventions',str(destination),'--baseline',ordinary['report'],
                '--root',plan['data_root'],'--output',str(analysis)],name+'_analysis',cpu=True)
            report=read_json(analysis)
            if report['scope']!='Full RoboTAP paired encoder-state intervention' or report['videos']!=265:
                raise ValueError('Partial history analysis cannot be reported as the full experiment')
            status['results'][name]=dict(analysis=str(analysis),analysis_sha256=file_hash(analysis),
                checkpoint_sha256=report['checkpoint_sha256'],strata=report['strata']);save()
        check_plan()
        write_json(output/'summary.json',dict(plan_sha256=file_hash(output/'plan.json'),
            scope=plan['scope'],primary=plan['primary'],results=status['results'],
            limitations='One seed; shared full-history query anchors; state reset changes distributions. Age cohorts differ. No multiplicity adjustment or robot-success conclusion.'))
        status['status']='complete'
    except BaseException as exc:
        status.update(status='stopped' if cancelled else 'failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if descriptor is not None:os.close(descriptor)
        save()


if __name__=='__main__':main()
