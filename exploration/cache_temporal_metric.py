"""Cache frozen final features on training scenes for matched readout learning."""
import argparse
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

from PIL import Image
import torch
from transformers import SiglipImageProcessor

from temporal_siglip import TemporalSiglip
from video_checkpoint import load_video_checkpoint
from exploration.oracle_transport import digest
from exploration.readout_diagnostic import parameter_digest
from exploration.temporal_feature_probe import extract_stream


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    probe,output=Path(args.probe_run).resolve(),Path(args.output).resolve()
    old=json.loads((probe/'plan.json').read_text()); original=Path(old['cache'])
    cache_plan=json.loads((original/'plan.json').read_text()); receipt=json.loads((original/'summary.json').read_text())
    assert digest(original/'summary.json')==old['cache_summary_sha256']
    assert digest(probe/'checkpoint.pt')==old['checkpoint']['checkpoint_sha256']
    train=cache_plan['config']['train']; val=old['config']['val']
    assert not set(train)&set(val)
    config=dict(train=train,val=val,conditions=old['config']['conditions'],depth=12,
        frames=cache_plan['config']['frames'],scope='Training features only; same fixed intermediate checkpoint and preprocessing as validation probe')
    plan=dict(config=config,probe_run=str(probe),probe_plan_sha256=digest(probe/'plan.json'),
        checkpoint_sha256=old['checkpoint']['checkpoint_sha256'],original_cache=str(original),
        original_summary_sha256=digest(original/'summary.json'),rgb_root=old['rgb_root'],
        sources={str(p):digest(p) for p in (Path(__file__),Path('exploration/temporal_feature_probe.py'),
            Path('temporal_siglip.py'),Path('video_checkpoint.py'),Path('exploration/oracle_transport.py'),Path('pyproject.toml'),Path('uv.lock'))})
    if args.resume:
        assert json.loads((output/'plan.json').read_text())==plan
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.plan_only:
        print('24 training scenes, three matched conditions, fixed checkpoint',flush=True); return
    torch.empty(1,device='cuda')
    model,provenance=load_video_checkpoint(probe/'checkpoint.pt','cpu')
    assert provenance==old['checkpoint']
    encoder=model.encoder; del model
    assert parameter_digest(encoder.backbone)==old['frozen_backbone_sha256']
    encoder.requires_grad_(False); encoder_hash=parameter_digest(encoder)
    foundation=TemporalSiglip(copy.deepcopy(encoder.backbone),kind='none').eval()
    encoder.cuda().eval(); foundation.cuda().eval()
    processor=SiglipImageProcessor.from_pretrained(old['config']['model'],revision=old['config']['revision'],local_files_only=True)
    root=Path(old['rgb_root']); records={}; started=time.monotonic()
    for scene in train:
        destination=output/(scene+'.pt'); record_path=output/(scene+'.json')
        if destination.exists() and record_path.exists():
            record=json.loads(record_path.read_text())
            assert record['plan_sha256']==digest(output/'plan.json') and digest(destination)==record['sha256']
        else:
            path=original/(scene+'.pt'); assert digest(path)==receipt['scenes'][scene]['sha256']
            data=torch.load(path,map_location='cpu',weights_only=True); del data['features']
            metadata_path=root/scene/'metadata.json'; metadata=json.loads(metadata_path.read_text())
            assert digest(metadata_path)==receipt['scenes'][scene]['metadata_sha256']
            images=root/scene/'images.zip'; assert digest(images)==metadata['files']['images.zip']['sha256']
            with zipfile.ZipFile(images) as archive:
                rgb=[Image.open(io.BytesIO(archive.read(f'{frame:03d}.png'))).convert('RGB') for frame in data['frames']]
            pixels=processor(images=rgb,return_tensors='pt').pixel_values
            features={}; state_bytes={}
            with torch.inference_mode():
                for condition in config['conditions']:
                    current,states=extract_stream(foundation if condition=='siglip_only' else encoder,pixels,condition=='mamba_reset1')
                    features[condition]=current['12']; state_bytes[condition]=states[-1]['bytes']
            assert torch.equal(features['mamba_full'][0],features['mamba_reset1'][0])
            payload=dict(features=features,points=data['points'],visible=data['visible'],frames=data['frames'],
                pixels_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest())
            temporary=destination.with_suffix('.tmp'); torch.save(payload,temporary); temporary.replace(destination)
            record=dict(scene=scene,sha256=digest(destination),bytes=destination.stat().st_size,
                plan_sha256=digest(output/'plan.json'),input_cache_sha256=digest(path),
                metadata_sha256=digest(metadata_path),images_sha256=digest(images),
                pixels_sha256=payload['pixels_sha256'],state_bytes=state_bytes,full_reset_initial_features_match=True)
            record_path.write_text(json.dumps(record,indent=2)+'\n')
            del data,pixels,rgb,payload,features,current; gc.collect()
        records[scene]=record
        print(json.dumps(dict(scene=scene,completed=len(records),total=len(train),elapsed_seconds=time.monotonic()-started)),flush=True)
    assert parameter_digest(encoder)==encoder_hash
    result=dict(plan_sha256=digest(output/'plan.json'),records=records,total_bytes=sum(r['bytes'] for r in records.values()),
        encoder_weights_unchanged=True,elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
