"""Fixed-checkpoint history intervention with an untrained continuous readout.

No production source or training queue is changed. Spatial features are read
from actual RGB in streaming mode. Every branch uses the same initial query
coordinates, and full/reset branches share their frame-zero descriptors.
"""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor,SiglipVisionModel

from temporal_siglip import TemporalSiglip
from video_checkpoint import load_video_checkpoint
from exploration.continuous_matching import continuous_match
from exploration.evaluate_continuous_matching import score_scene,aggregate
from exploration.oracle_transport import digest,sample
from exploration.readout_diagnostic import parameter_digest,select_queries


@torch.no_grad()
def extract_stream(encoder,pixels,reset=False):
    """Tap block 4 after temporal insertion, plus the final normalized output."""
    if encoder.training: raise ValueError('Evaluation mode required')
    device=next(encoder.parameters()).device
    intermediate=[]; final=[]; states=[]; state=None
    if '3' in encoder.adapters:
        module=encoder.adapters['3']
        def capture(_module,_args,output):
            intermediate.append(output[0][0,0].detach().float().cpu())
    else:
        module=encoder.backbone.vision_model.encoder.layers[3]
        def capture(_module,_args,output):
            intermediate.append(output[0].detach().float().cpu())
    handle=module.register_forward_hook(capture)
    try:
        for frame,image in enumerate(pixels):
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                output,state=encoder.step(image[None].to(device),None if reset else state)
            final.append(output[0].detach().float().cpu())
            assert state.offset==(1 if reset else frame+1)
            states.append(dict(offset=state.offset,bytes=state.nbytes,
                layer_states={k:[dict(shape=list(t.shape),dtype=str(t.dtype)) for t in tensors]
                              for k,tensors in state.layers.items()}))
    finally:
        handle.remove()
    assert len(intermediate)==len(final)==len(pixels)
    if encoder.kind!='attention':
        assert len({s['bytes'] for s in states})==1
    return {'4':torch.stack(intermediate),'12':torch.stack(final)},states


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-run',required=True)
    parser.add_argument('--matching-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    source,matching,output=Path(args.video_run).resolve(),Path(args.matching_run).resolve(),Path(args.output).resolve()
    if not args.resume:
        output.mkdir(parents=True,exist_ok=False)
        # Atomic trainer replacement cannot change the inode held open here.
        with (source/'last.pt').open('rb') as src,(output/'checkpoint.pt').open('wb') as dst:
            shutil.copyfileobj(src,dst)
    if not args.plan_only:
        # Acquire a CUDA context before staging CPU weights on unified RAM.
        torch.empty(1,device='cuda')
    recognizer,checkpoint=load_video_checkpoint(output/'checkpoint.pt','cpu')
    encoder=recognizer.encoder; del recognizer
    if checkpoint['config']!=json.loads((source/'config.json').read_text()):
        raise ValueError('Checkpoint configuration differs from its run')
    if encoder.kind!='mamba' or encoder.placement!='interleaved' or len(encoder.backbone.vision_model.encoder.layers)!=12:
        raise ValueError('This diagnostic requires the 12-layer interleaved Mamba encoder')
    frozen=SiglipVisionModel.from_pretrained(checkpoint['config']['model'],
        revision=checkpoint['config']['revision'],local_files_only=True).eval()
    backbone_hash=parameter_digest(frozen)
    if parameter_digest(encoder.backbone)!=backbone_hash:
        raise ValueError('The supposed frozen backbone differs from its pretrained weights')
    del frozen; gc.collect()
    reference=json.loads((matching/'plan.json').read_text()); config=dict(reference['config'])
    config.update(conditions=['siglip_only','mamba_full','mamba_reset1'],
        precision='Spatial GPU FP32 with TF32 off; temporal BF16; matching CPU FP32',
        depths={'4':'Block4 output, including temporal insertion where present','12':'Final post-layernorm with parameter-free LN for matching'},
        feature_taps={'4':'After block4 temporal residual, or block4 spatial output for SigLIP only','12':'Actual final post-layernorm output'},
        scope='Exploratory fixed intermediate SSv2 checkpoint, no fitting. Same readout across history interventions.',
        reset='Fresh native Mamba states for each frame. Initial query appearance stays fixed from frame zero; reset is not memoryless tracking.',
        limits='8 repeatedly inspected rendered scenes, 24 frames at original stride4. Classification-trained intermediate features; not the formal point benchmark or evidence of final superiority.')
    cache=Path(reference['cache']); receipt=json.loads((cache/'summary.json').read_text())
    if digest(cache/'summary.json')!=reference['cache_summary_sha256']: raise ValueError('Cache changed')
    cache_plan=json.loads((cache/'plan.json').read_text()); root=Path(cache_plan['root'])
    images={}
    for scene in config['val']:
        metadata=root/scene/'metadata.json'; values=json.loads(metadata.read_text())
        if digest(metadata)!=receipt['scenes'][scene]['metadata_sha256']: raise ValueError('Metadata changed')
        images[scene]=dict(metadata_sha256=digest(metadata),images_sha256=values['files']['images.zip']['sha256'])
    sources=[Path(__file__),Path('exploration/continuous_matching.py'),Path('exploration/evaluate_continuous_matching.py'),
        Path('exploration/oracle_transport.py'),Path('exploration/readout_diagnostic.py'),Path('temporal_siglip.py'),
        Path('video_checkpoint.py'),Path('temporal_readout.py'),Path('tapvid_metrics.py'),Path('pyproject.toml'),Path('uv.lock')]
    plan=dict(config=config,checkpoint=checkpoint,video_run=str(source),video_config_sha256=digest(source/'config.json'),
        matching_run=str(matching),matching_plan_sha256=digest(matching/'plan.json'),matching_result_sha256=digest(matching/'summary.json'),
        cache=str(cache),cache_summary_sha256=digest(cache/'summary.json'),rgb_root=str(root),images=images,
        frozen_backbone_sha256=backbone_hash,sources={str(p):digest(p) for p in sources})
    if args.resume:
        if json.loads((output/'plan.json').read_text())!=plan or (output/'progress.jsonl').exists():
            raise ValueError('Changed fixed plan or already-started evaluation')
    else:
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.plan_only:
        print(json.dumps(dict(step=checkpoint['step'],checkpoint_sha256=checkpoint['checkpoint_sha256'],
            conditions=config['conditions'],frozen_backbone_exact_match=True)),flush=True); return
    encoder.requires_grad_(False); encoder_hash=parameter_digest(encoder)
    foundation=TemporalSiglip(copy.deepcopy(encoder.backbone),kind='none').eval()
    encoder.cuda().eval(); foundation.cuda().eval()
    processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    started=time.monotonic(); rows=[]; predictions={}; features_receipts={}; states_receipts={}; first_frame_equal={}
    for scene in config['val']:
        cache_path=cache/(scene+'.pt'); assert digest(cache_path)==receipt['scenes'][scene]['sha256']
        data=torch.load(cache_path,map_location='cpu',weights_only=True)
        selected=select_queries(data,scene,config['evaluation_queries'],config['seed'])
        truth,visible=data['points'][selected],data['visible'][selected]
        archive_path=root/scene/'images.zip'
        assert digest(archive_path)==images[scene]['images_sha256']
        with zipfile.ZipFile(archive_path) as archive:
            rgb=[Image.open(io.BytesIO(archive.read(f'{frame:03d}.png'))).convert('RGB') for frame in data['frames']]
        pixels=processor(images=rgb,return_tensors='pt').pixel_values
        features={}; scene_states={}
        with torch.inference_mode():
            for condition in config['conditions']:
                model=foundation if condition=='siglip_only' else encoder
                features[condition],scene_states[condition]=extract_stream(model,pixels,condition=='mamba_reset1')
        for depth in ('4','12'):
            if not torch.equal(features['mamba_full'][depth][0],features['mamba_reset1'][depth][0]):
                raise ValueError('Initial full/reset query features differ')
        first_frame_equal[scene]=True
        state_path=output/(scene+'_states.json')
        state_path.write_text(json.dumps(scene_states,indent=2)+'\n'); states_receipts[scene]=digest(state_path)
        feature_path=output/(scene+'_features.pt')
        torch.save(dict(features=features,selected=selected,frames=data['frames'],
            pixels_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest()),feature_path)
        features_receipts[scene]=digest(feature_path)
        del data,pixels,rgb
        for condition in config['conditions']:
            for depth in ('4','12'):
                current=features[condition][depth]
                anchor=sample(current[0],truth[:,0],config['grid'])
                coordinates={name:[] for name in config['methods'] if name!='static_query'}
                scores={name:[] for name in coordinates}
                for frame in range(len(current)):
                    values=continuous_match(anchor,current[frame],config['grid'],depth=='12',config['topk'],config['refinement_steps'])
                    for method,result in values.items():
                        coordinates[method].append(result['points'].numpy()); scores[method].append(result['scores'].numpy())
                arrays={k:np.stack(v,1) for k,v in coordinates.items()}
                arrays['static_query']=np.broadcast_to(truth[:,:1].numpy(),truth.shape).copy()
                saved=dict(target_xy=truth.numpy(),visible=visible.numpy(),selected=selected.numpy(),**arrays)
                saved.update({k+'_score':np.stack(v,1) for k,v in scores.items()})
                name=scene+'_'+condition+'_'+depth+'.npz'; np.savez_compressed(output/name,**saved)
                predictions[name]=digest(output/name)
                rows.extend(dict(row,condition=condition) for row in score_scene(scene,int(depth),truth.numpy(),visible.numpy(),arrays))
            event=dict(scene=scene,condition=condition,elapsed_seconds=time.monotonic()-started)
            with (output/'progress.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        del features,current,anchor; gc.collect()
    if parameter_digest(encoder)!=encoder_hash: raise ValueError('Encoder weights changed during evaluation')
    if parameter_digest(foundation.backbone)!=backbone_hash: raise ValueError('Baseline weights changed')
    summary=[]
    for condition in config['conditions']:
        summary.extend(dict(row,condition=condition) for row in aggregate([r for r in rows if r['condition']==condition]))
    report=dict(plan_sha256=digest(output/'plan.json'),checkpoint_sha256=digest(output/'checkpoint.pt'),
        encoder_parameters_unchanged=True,pretrained_backbone_matches=True,full_reset_first_frame_equal=first_frame_equal,
        predictions_sha256=predictions,features_sha256=features_receipts,states_sha256=states_receipts,
        elapsed_seconds=time.monotonic()-started,rows=rows,summary=summary)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(event='complete',step=checkpoint['step'],elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__': main()
