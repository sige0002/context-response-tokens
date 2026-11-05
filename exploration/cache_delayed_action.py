"""Extract delayed-action features from a fixed encoder, never its old classifier.

The full cache uses all declared train/validation examples. Qualification uses
two prefixes from one validation block and cannot be passed to the probe fitter.
No architecture, optimizer, or active training sources are modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from exploration.delayed_action_data import sha, verify_plan
from exploration.evaluate_video_history import model_hash, tensor_hash
from temporal_siglip import StreamState
from train_video import configure_reproducibility
from video_checkpoint import load_video_checkpoint
from video_data import RawVideoClips


@torch.inference_mode()
def stream_features(encoder, prefix, suffix, delays, suffix_only=False):
    """Read only current mean tokens at declared delays; never expose cache to head."""
    state=None; vectors=[]; sizes=[]
    device=next(encoder.parameters()).device
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        if suffix_only:
            # Preserve absolute observation positions in the attention control.
            # One-frame calls avoid chunk-mask assumptions after a reset.
            state=StreamState(id(encoder),1,tuple(suffix.shape[1:]),offset=len(prefix))
        else:
            for image in prefix:
                value,state=encoder.step(image[None].to(device),state)
            vectors.append(value[0].float().mean(0).cpu().numpy())
            sizes.append(state.nbytes)
        for i,image in enumerate(suffix,1):
            value,state=encoder.step(image[None].to(device),state)
            if i in delays:
                if state.offset != len(prefix)+i:
                    raise AssertionError('Incorrect stream position')
                vectors.append(value[0].float().mean(0).cpu().numpy())
                sizes.append(state.nbytes)
            if i==max(delays): break
    result=np.stack(vectors)
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite delayed features')
    return result,sizes


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--qualify',action='store_true')
    args=p.parse_args(); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Use a fresh output; incomplete runs are not reused')
    torch.set_num_threads(2); configure_reproducibility(42)
    data=json.loads(Path(args.plan).read_text()); verify_plan(data)
    for record in data['manifests'].values():
        if sha(record['path']) != record['sha256']: raise ValueError('Dataset manifest changed')
    model,info=load_video_checkpoint(args.checkpoint,'cpu')
    if info['step'] != info['config']['steps'] and not args.qualify:
        raise ValueError('Full probe requires the final scheduled checkpoint')
    if info['config']['kind'] not in ('none','mamba','attention'):
        raise ValueError('This extractor supports TemporalSiglip encoders only')
    for split,key in (('train','train'),('validation','val')):
        if data['manifests'][split]['sha256']!=info['config'][key+'_sha256']:
            raise ValueError('Probe corpus differs from encoder corpus')
    model.requires_grad_(False).eval()
    before=model_hash(model)
    processor=SiglipImageProcessor.from_pretrained(info['config']['model'],
        revision=info['config']['revision'],local_files_only=True)
    sources=['exploration/cache_delayed_action.py','exploration/delayed_action_data.py',
        'exploration/evaluate_video_history.py','temporal_siglip.py','temporal_readout.py',
        'video_data.py','video_checkpoint.py','train_video.py','uv.lock','pyproject.toml']
    plan=dict(data=data,data_plan_sha256=sha(args.plan),checkpoint_info=info,
        checkpoint_path=str(Path(args.checkpoint).resolve()),qualification=args.qualify,
        conditions=data['conditions'],sources={f:sha(f) for f in sources},
        preprocessing='Deterministic original 8-observation sampling and SigLIP2 processor',
        representation='Mean of current 196x768 tokens; no past tokens/cache exposed to the probe',
        suffix_reset='Empty cache at evidence end, absolute observation index preserved',
        timing_scope='Concurrent main training allowed; no latency comparison')
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    model.cuda(); encoder=model.encoder
    features={}; inputs=[]; state_records=[]
    selected=('validation',) if args.qualify else ('train','validation')
    for split in selected:
        dataset=RawVideoClips(data['manifests'][split]['path'],processor,frames=8,classes=data['classes'])
        index={r['id']:i for i,r in enumerate(dataset.rows)}
        def read(row):
            i=index[row['id']]; actual=dataset.rows[i]
            if any(actual[k]!=row[k] for k in ('id','group','label','sha256','bytes')):
                raise ValueError('Source record changed')
            if sha(actual['video'])!=row['sha256']: raise ValueError('Video content changed')
            pixels,target,metadata=dataset.read(i)
            if target!=row['label']: raise ValueError('Evidence target changed')
            inputs.append(dict(split=split,video_sha256=row['sha256'],pixels_sha256=tensor_hash(pixels),
                               **metadata))
            return pixels
        values={c:[] for c in data['conditions']}; labels=[]; blocks=[]; ids=[]
        selected_blocks=data['splits'][split][:1] if args.qualify else data['splits'][split]
        for block in selected_blocks:
            suffix=torch.cat([read(row) for row in block['suffix']])[:max(data['delays'])]
            evidence=block['evidence'][:2] if args.qualify else block['evidence']
            shared=None
            for row in evidence:
                prefix=read(row)
                full,sizes=stream_features(encoder,prefix,suffix,data['delays'])
                if shared is None:
                    shared,reset_sizes=stream_features(encoder,prefix,suffix,data['delays'],True)
                elif args.qualify:
                    replay,_=stream_features(encoder,prefix,suffix,data['delays'],True)
                    if not np.array_equal(shared,replay):
                        raise ValueError('Identical suffix with fresh memory did not reproduce')
                values['full_memory'].append(full)
                values['suffix_only'].append(shared.copy())
                values['saved_evidence'].append(np.broadcast_to(full[0],full.shape).copy())
                labels.append(row['label']); blocks.append(block['block']); ids.append(row['id'])
                state_records.append(dict(split=split,block=block['block'],id=row['id'],
                    full_bytes=sizes,suffix_only_bytes=reset_sizes,
                    current_rgb_sha256=[tensor_hash(prefix[-1])]+[tensor_hash(suffix[d-1]) for d in data['delays'][1:]]))
                with (output/'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(split=split,block=block['block'],id=row['id'],completed=len(labels)))+'\n')
                if len(labels)%32==0 or args.qualify:
                    print(json.dumps(dict(split=split,completed=len(labels))),flush=True)
            del suffix
        features.update({split+'_'+c:np.stack(rows) for c,rows in values.items()})
        features.update({split+'_labels':np.array(labels),split+'_blocks':np.array(blocks),split+'_ids':np.array(ids)})
    if model_hash(model)!=before: raise ValueError('Encoder or classifier weights changed')
    for path,digest in plan['sources'].items():
        if sha(path)!=digest: raise ValueError('Source changed during extraction: '+path)
    # Current RGB is the same for every class within a block at every positive delay.
    for split in selected:
        for b in {r['block'] for r in state_records if r['split']==split}:
            rows=[r for r in state_records if r['split']==split and r['block']==b]
            if len({tuple(r['current_rgb_sha256'][1:]) for r in rows})!=1:
                raise ValueError('Current observations depend on target class')
    np.savez_compressed(output/'features.npz',**features)
    (output/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    summary=dict(scope='qualification_only' if args.qualify else 'complete_cache',
        plan_sha256=sha(output/'plan.json'),features_sha256=sha(output/'features.npz'),
        inputs_sha256=sha(output/'inputs.json'),weights_unchanged=True,model_state_sha256=before,
        states=state_records,current_rgb_independent_of_target=True,
        scores=None,limits=data['limits'])
    if args.qualify:
        x=features['validation_full_memory']
        summary['between_prefix_feature_l2_by_delay']=np.linalg.norm(x[0]-x[1],axis=-1).tolist()
        summary['suffix_replay_exact']=True
        summary['qualification_limits']='Two prefixes and one suffix block only; no readout training or recognition score'
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('states','limits')}),flush=True)


if __name__=='__main__': main()
