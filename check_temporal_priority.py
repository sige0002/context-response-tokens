"""Check real pidfd waiting/cancellation without signalling training jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from point_checkpoint import file_hash


def identity(pid):
    path=Path('/proc')/str(pid)
    # Field22 after the parenthesized command is the process start tick.
    return dict(command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                start_ticks=(path/'stat').read_text().rsplit(')',1)[1].split()[19])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-gmmix-pid',required=True,type=int)
    parser.add_argument('--training-pid',required=True,type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output).resolve()
    before={str(pid):identity(pid) for pid in (args.after_gmmix_pid,args.training_pid)}
    assert 'run_gmmix_comparison.py' in before[str(args.after_gmmix_pid)]['command']
    assert 'train_video.py' in before[str(args.training_pid)]['command']
    base=['uv','run','--project','baselines/gmmix','--locked','python','run_temporal_priority.py','--output',str(output)]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=''); environment.pop('VIRTUAL_ENV',None)
    subprocess.run(base+['--plan-only'],check=True,env=environment)
    child=None
    try:
        with (output/'cancellation.log').open('w') as log:
            child=subprocess.Popen(base+['--resume','--after-gmmix-pid',str(args.after_gmmix_pid)],
                stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            started=time.monotonic()
            while True:
                if child.poll() is not None: raise RuntimeError('Qualification queue exited before entering its wait')
                status=json.loads((output/'status.json').read_text())
                if status['status']=='waiting_for_video_comparisons': break
                if time.monotonic()-started>60: raise TimeoutError('Qualification did not enter its wait')
                time.sleep(.1)
            assert status['operations']==[]
            os.killpg(child.pid,signal.SIGTERM)
            child.wait(timeout=15)
        status=json.loads((output/'status.json').read_text())
        assert status['status']=='stopped' and not status['operations']
        after={str(pid):identity(pid) for pid in (args.after_gmmix_pid,args.training_pid)}
        assert after==before
        plan=json.loads((output/'plan.json').read_text())
        for row in plan['replications']:
            assert not Path(row['run']).exists()
        assert not Path(plan['multidepth']['run']).exists()
        report=dict(status='passed',source_sha256=file_hash(__file__),plan_sha256=file_hash(output/'plan.json'),
            cancellation_status=status['status'],operations_started=0,dependency_identities_unchanged=after,
            order=plan['order'],replication_runs=len(plan['replications']),
            scope='Actual pidfd wait and cancellation; existing reference trainer and GMMix queue remained live with unchanged identities. Training/cost stages are not executed by this check.')
        (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()


if __name__=='__main__': main()
