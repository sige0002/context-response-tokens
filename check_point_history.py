"""Real-GPU history-intervention integration; arbitrary queries, no accuracy claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from evaluate_point_tracks import track_video
from pause_training import isolated_gpu
from point_checkpoint import file_hash,load_point_checkpoint
from point_experiment import write_json
from point_history import track_history_interventions
from point_readout import PointReadout,PointTracker
from robotap_data import iter_videos
from temporal_siglip import DEFAULT_MODEL,DEFAULT_REVISION
from video_checkpoint import load_video_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mamba-video-checkpoint',required=True)
    parser.add_argument('--gmmix-point-checkpoint',required=True)
    parser.add_argument('--root',default='data/robotap')
    parser.add_argument('--output',required=True)
    parser.add_argument('--pause-pid',type=int)
    parser.add_argument('--pause-reference')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new GPU check output')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_num_threads(4)
    torch.manual_seed(17)
    source=iter_videos(args.root)
    shard,name,example=next(source)
    video=example['video'][:12].copy()
    source.close();del source,example
    xy=np.array([[.3,.4],[.6,.5],[.5,.6]],dtype=np.float32)
    first=np.array([2,0,7])
    processor=SiglipImageProcessor.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True)
    report=dict(scope='Native GPU API/causality/reset checks on a real video prefix with arbitrary queries; not tracking accuracy',
        video_id=name,shard=shard,frames=len(video),video_prefix_sha256=hashlib.sha256(video.tobytes()).hexdigest(),
        query_xy=xy.tolist(),query_frames=first.tolist(),periods=[1,4,100],models={},
        predeclared_tolerances=dict(normalized_xy=1e-6,visibility_logit=1e-5,future_change=0),
        sources={p:file_hash(p) for p in ('check_point_history.py','point_history.py','pause_training.py','evaluate_point_tracks.py',
            'point_readout.py','point_checkpoint.py','temporal_siglip.py','gmmix_siglip.py')})
    with isolated_gpu(args.pause_pid,args.pause_reference) as paused:
        report['initially_paused_compute_pids']=paused
        for kind in ('mamba','gmmix'):
            if kind=='mamba':
                recognizer,info=load_video_checkpoint(args.mamba_video_checkpoint,'cuda')
                encoder=recognizer.encoder;del recognizer
                model=PointTracker(encoder,PointReadout().cuda()).eval()
                del encoder
            else:
                model,info=load_point_checkpoint(args.gmmix_point_checkpoint,'cuda')
            provenance=dict(checkpoint_sha256=info['checkpoint_sha256'],
                readout='Fresh deterministic random point head' if kind=='mamba' else 'Three-update engineering point checkpoint')
            del info
            ref_xy,ref_logits,_=track_video(model,video,xy,first,processor,torch.device('cuda'))
            results=track_history_interventions(model,video,xy,first,processor,torch.device('cuda'),(1,4,100))
            max_xy=float(np.max(np.abs(ref_xy-results['full']['predicted_xy'])))
            max_logit=float(np.max(np.abs(ref_logits-results['full']['visibility_logits'])))
            assert max_xy<=1e-6 and max_logit<=1e-5,(kind,max_xy,max_logit)
            np.testing.assert_allclose(results['reset_100']['predicted_xy'],ref_xy,atol=1e-6,rtol=0)
            np.testing.assert_allclose(results['reset_100']['visibility_logits'],ref_logits,atol=1e-5,rtol=0)
            changed=video.copy();changed[10:]=255-changed[10:]
            other=track_history_interventions(model,changed,xy,first,processor,torch.device('cuda'),(1,4,100))
            for branch,values in results.items():
                np.testing.assert_array_equal(values['predicted_xy'][:,:10],other[branch]['predicted_xy'][:,:10])
                np.testing.assert_array_equal(values['visibility_logits'][:,:10],other[branch]['visibility_logits'][:,:10])
                for q,t in enumerate(first):
                    np.testing.assert_array_equal(values['predicted_xy'][q,t],xy[q])
                assert [v['absolute_frames'] for v in values['state_bytes']]==list(range(1,len(video)+1))
            report['models'][kind]=dict(provenance,full_vs_standard_max_xy=max_xy,full_vs_standard_max_logit=max_logit,
                no_reset_branch_matches_standard=True,future_change_difference=0,
                reset_effect_max_xy={branch:float(np.max(np.abs(values['predicted_xy']-ref_xy))) for branch,values in results.items() if branch!='full'},
                state_bytes={branch:values['state_bytes'] for branch,values in results.items()},
                reset_frames={branch:values['resets'] for branch,values in results.items()})
            print(json.dumps(dict(model=kind,full_vs_standard_max_xy=max_xy,full_vs_standard_max_logit=max_logit,
                reset_effect_max_xy=report['models'][kind]['reset_effect_max_xy'])),flush=True)
            del model,results,other,ref_xy,ref_logits
            gc.collect();torch.cuda.empty_cache()
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,report)
    print(json.dumps(dict(output=str(output),models=list(report['models']),scope=report['scope'])),flush=True)


if __name__=='__main__':
    main()
