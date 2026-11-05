"""Fixed-weight, human-selected observation intervention on SSv2 video 3988.

Frames were selected from visual inspection before querying the changed inputs.
This is an oracle diagnostic on one known failure, not an online sampler or a
recognition benchmark. The final RGB image stays identical in all conditions.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from transformers import SiglipImageProcessor

from evaluate_video import streaming_logits
from exploration.evaluate_video_history import model_hash, tensor_hash
from exploration.temporal_density import read, write
from train_video import configure_reproducibility
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import frame_to_rgb, temporal_indices


def schedules():
    return dict(original_8=[3,11,20,28,37,45,54,59],
        throw_8=[3,9,20,28,37,45,54,59],
        fall_8=[3,11,23,28,37,45,54,59],
        both_8=[3,9,23,28,37,45,54,59],
        nearby_control_8=[3,12,21,28,37,45,54,59],
        dense_57=list(range(3,60)))


def describe(logits, target, names):
    values=np.asarray(logits,dtype=np.float64)
    if values.shape!=(174,) or not np.isfinite(values).all(): raise ValueError('Invalid logits')
    order=np.argsort(-values,kind='stable');other=values.copy();other[target]=-np.inf
    return dict(prediction=int(order[0]),predicted_name=names[int(order[0])],correct=bool(order[0]==target),
        target_rank=int(np.where(order==target)[0][0])+1,target_margin=float(values[target]-other.max()),
        top3=[dict(label=int(i),name=names[int(i)],logit=float(values[i])) for i in order[:3]])


def verify(output):
    plan,summary=read(output/'plan.json'),read(output/'summary.json')
    if summary['plan_sha256']!=file_hash(output/'plan.json') or summary['logits_sha256']!=file_hash(output/'logits.npz'):
        raise ValueError('Artifacts changed')
    for path,sha in plan['sources'].items():
        if file_hash(path)!=sha: raise ValueError('Source changed')
    if file_hash(plan['checkpoint'])!=plan['checkpoint_info']['checkpoint_sha256']: raise ValueError('Weights changed')
    if file_hash(plan['video'])!=plan['video_sha256']: raise ValueError('Video changed')
    if plan['indices']!=schedules(): raise ValueError('Observation plan changed')
    names={int(v):k for k,v in read(plan['labels']).items()}
    manifest=Path(plan['checkpoint_info']['config']['val'])
    if file_hash(manifest)!=plan['checkpoint_info']['config']['val_sha256']: raise ValueError('Manifest changed')
    row=[json.loads(s) for s in manifest.read_text().splitlines()][plan['index']]
    if row['id']!='3988' or row['label']!=plan['target']: raise ValueError('Target changed')
    with np.load(output/'logits.npz') as saved:
        for key,value in summary['results'].items():
            if describe(saved[key],plan['target'],names)!=value: raise ValueError('Score recount changed')
    if not summary['weights_unchanged'] or summary['baseline_repeat_max_abs']!=0:
        raise ValueError('Weight/repeat check failed')
    return dict(passed=True,summary_sha256=file_hash(output/'summary.json'),predictions_recounted=12,
        scope='Source/artifact/target checks and logit recount. One hand-selected failure, no independent GPU replay.')


@torch.inference_mode()
def run(output):
    if output.exists(): raise FileExistsError('Use a new output')
    torch.set_num_threads(2);configure_reproducibility(42)
    casefile=Path('results/ssv2_failure_examples_step12000/summary.json')
    case=next(x for x in read(casefile)['examples'] if x['id']=='3988')
    checkpoint=Path('results/temporal_density_step11000_v1/checkpoint.pt').resolve()
    model,info=load_video_checkpoint(checkpoint,'cpu');model.requires_grad_(False).eval()
    if info['step']!=11000: raise ValueError('Expected archived 11000-update weights')
    labels=Path('data/ssv2/labels/labels.json').resolve()
    names={int(v):k for k,v in read(labels).items()}
    manifest=Path(info['config']['val'])
    if file_hash(manifest)!=info['config']['val_sha256']: raise ValueError('Manifest changed')
    row=[json.loads(s) for s in manifest.read_text().splitlines()][case['index']]
    if row['id']!='3988' or file_hash(case['video'])!=case['video_sha256']: raise ValueError('Case changed')
    processor=SiglipImageProcessor.from_pretrained(info['config']['model'],revision=info['config']['revision'],local_files_only=True)
    with av.open(case['video']) as container:
        container.streams.video[0].thread_count=1;frames=list(container.decode(video=0))
    if len(frames)!=60 or temporal_indices(60,8)!=schedules()['original_8']: raise ValueError('Sampling changed')
    pixels=processor(images=[frame_to_rgb(f) for f in frames],return_tensors='pt').pixel_values
    indices=schedules();inputs={k:pixels[v][None] for k,v in indices.items()}
    current=inputs['original_8'][:,-1]
    if not all(torch.equal(x[:,-1],current) for x in inputs.values()): raise ValueError('Current image changed')
    sources=[__file__,'evaluate_video.py','temporal_siglip.py','temporal_readout.py','video_data.py',
        'video_checkpoint.py','train_video.py','exploration/evaluate_video_history.py',str(casefile),str(labels)]
    plan=dict(checkpoint=str(checkpoint),checkpoint_info=info,video=case['video'],video_sha256=case['video_sha256'],
        id='3988',index=case['index'],target=row['label'],correct_name=names[row['label']],labels=str(labels),
        indices=indices,timestamps={k:[frames[i].time for i in ix] for k,ix in indices.items()},
        input_sha256={k:tensor_hash(x) for k,x in inputs.items()},current_pixels_sha256=tensor_hash(current),
        sources={str(Path(p)):file_hash(p) for p in sources},
        expected_original_prediction=case['predictions']['11000'],
        limitations=['One failure selected after inspecting earlier predictions and the complete video.',
            'Human-selected event frames: not a learned sampler, online selection rule or generalization result.',
            'All conditions keep first and last RGB images fixed; five keep eight observations.',
            'Dense57 changes both information and update count for an eight-observation-trained model.',
            '11k weights, not the latest live training weights. Same classifier, no adaptation.',
            'Both clip and single-frame streaming are measured. No runtime claim while GPU is shared.'])
    output.mkdir(parents=True);write(output/'plan.json',plan)
    before=model_hash(model);model.cuda();values={};results={};memory={}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for name,images in inputs.items():
            images=images.cuda()
            for mode in ('clip','stream'):
                if mode=='clip':logits=model(images)
                else:
                    logits,sizes=streaming_logits(model,images);memory[name]=sizes
                key=f'{name}_{mode}';values[key]=logits[0].float().cpu().numpy()
                results[key]=describe(values[key],plan['target'],names)
                print(json.dumps(dict(condition=key,**results[key])),flush=True)
        repeated=model(inputs['original_8'].cuda())[0].float().cpu().numpy()
    repeat_error=float(np.abs(repeated-values['original_8_clip']).max())
    if repeat_error!=0 or model_hash(model)!=before: raise ValueError('Repeated baseline or weights changed')
    np.savez_compressed(output/'logits.npz',**values)
    summary=dict(plan_sha256=file_hash(output/'plan.json'),logits_sha256=file_hash(output/'logits.npz'),
        results=results,state_bytes=memory,weights_unchanged=True,baseline_repeat_max_abs=repeat_error,
        baseline_matches_saved_prediction=results['original_8_clip']['prediction']==plan['expected_original_prediction'],
        limitations=plan['limitations'])
    write(output/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();output=Path(args.output)
    if args.verify:
        result=verify(output);write(output/'verification.json',result);print(json.dumps(result))
    else:run(output)
