"""Audit borderline predictions across repeated calls and streaming state use.

Select whole original batches containing a previously changed prediction,
plus two fixed initial batches. This is deliberately diagnostic selection,
not an unbiased accuracy estimate. Fresh process repetitions distinguish
within-process consistency from kernel-selection differences between runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from evaluate_video import streaming_logits
from point_experiment import write_json
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def compare_logits(reference, observed):
    reference,observed=np.asarray(reference),np.asarray(observed)
    if (reference.shape!=observed.shape or reference.ndim!=2
            or reference.shape[0]<1 or reference.shape[1]<2):
        raise ValueError('Expected matching nonempty [samples,classes] logits')
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError('Nonfinite logits')
    difference=observed-reference
    changes=reference.argmax(-1)!=observed.argmax(-1)
    ordered=np.sort(reference,axis=1)
    margins=ordered[:,-1]-ordered[:,-2]
    return dict(max_abs=float(np.abs(difference).max()),rmse=float(np.sqrt(np.mean(difference**2))),
        changed_predictions=int(changes.sum()),changed_indices=np.flatnonzero(changes).tolist(),
        changed_reference_margins=margins[changes].tolist(),bitwise_equal=np.array_equal(reference,observed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--deterministic',action='store_true')
    args=parser.parse_args()
    root,output=Path(args.diagnostic),Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    prior=json.loads((root/'summary.json').read_text())
    compared=json.loads((root/'clip_stream.json').read_text())
    old=prior['results']['full']['clips']
    new=compared['ablations']['full']['modes']
    for mode in ('clip','stream'):
        if [(r['id'],r['label']) for r in new[mode]['clips']]!=[(r['id'],r['label']) for r in old]:
            raise ValueError('Different evaluation IDs or labels')
    batch=prior['checkpoint']['config']['batch_size']
    if len(old)%batch:
        raise ValueError('This diagnostic requires complete original batches')
    affected=[i for i,row in enumerate(old) if len({row['prediction'],new['clip']['clips'][i]['prediction'],
        new['stream']['clips'][i]['prediction']})>1]
    batches=sorted({0,1,*[i//batch for i in affected]})
    selected=[i for b in batches for i in range(b*batch,(b+1)*batch)]
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=args.deterministic
    model,info=load_video_checkpoint(root/'checkpoint.pt','cuda')
    if info['checkpoint_sha256']!=prior['checkpoint']['checkpoint_sha256']:
        raise ValueError('Checkpoint differs')
    config=info['config']
    processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    dataset=RawVideoClips(root/'subset.jsonl',processor,config['frames'],config['classes'])
    baseline_state=state_hash(model)
    names=('clip_first','clip_repeat','stream','clip_after_stream')
    logits={name:[] for name in names}
    inputs=[]
    with torch.no_grad():
        for b in batches:
            indices=range(b*batch,(b+1)*batch)
            values=[dataset.read(i) for i in indices]
            pixels=torch.stack([v[0] for v in values]).cuda()
            inputs.append(dict(batch=b,pixels_sha256=tensor_hash(pixels),sampling=[v[2] for v in values]))
            # Match evaluate_video: one autocast context around clip + stream.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                first=model(pixels).float()
                repeated=model(pixels).float()
                stream,_=streaming_logits(model,pixels)
                after=model(pixels).float()
            for name,value in zip(names,(first,repeated,stream,after)):
                logits[name].append(value.float().cpu().numpy())
            del pixels,first,repeated,stream,after
    final_state=state_hash(model)
    if baseline_state!=final_state:
        raise ValueError('Inference changed model parameters or persistent buffers')
    logits={name:np.concatenate(values) for name,values in logits.items()}
    np.savez_compressed(output/'logits.npz',indices=np.asarray(selected),**logits)
    report=dict(scope='Selected mismatch-containing batches and two fixed controls; no population accuracy estimate',
        deterministic=args.deterministic,checkpoint_sha256=info['checkpoint_sha256'],step=info['step'],
        original_batch_size=batch,batches=batches,selected_samples=len(selected),triggering_indices=affected,
        inputs=inputs,model_state_sha256=baseline_state,model_state_unchanged=True,
        comparisons={name:compare_logits(logits['clip_first'],values) for name,values in logits.items() if name!='clip_first'},
        predictions={name:values.argmax(-1).tolist() for name,values in logits.items()},
        original_indices=selected,logits_sha256=file_hash(output/'logits.npz'),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),triton_cache_autotuning=os.environ.get('TRITON_CACHE_AUTOTUNING'),
            mamba_deterministic=os.environ.get('MAMBA_DETERMINISTIC')),
        source_sha256={name:file_hash(name) for name in ('diagnose_video_repeatability.py','evaluate_video.py',
            'temporal_siglip.py','video_data.py','video_checkpoint.py','temporal_readout.py')})
    write_json(output/'summary.json',report)
    print(json.dumps(dict(deterministic=args.deterministic,samples=len(selected),comparisons=report['comparisons'])),flush=True)


if __name__=='__main__':
    main()
