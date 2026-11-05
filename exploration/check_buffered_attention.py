"""Check original vs buffered attention on real 224 RGB, without timing claims."""
import argparse
import json
from pathlib import Path

import torch
from transformers import SiglipImageProcessor,SiglipVisionModel

from exploration.buffered_attention import compare_streams
from exploration.delayed_action_data import sha
from exploration.evaluate_video_history import model_hash,tensor_hash
from temporal_siglip import TemporalSiglip,DEFAULT_MODEL,DEFAULT_REVISION
from train_video import configure_reproducibility
from video_data import RawVideoClips


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='results/delayed_action_memory_v1/data_plan.json')
    p.add_argument('--output',required=True)
    args=p.parse_args(); output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a fresh qualification output')
    torch.set_num_threads(2); configure_reproducibility(71)
    data=json.loads(Path(args.data).read_text()); record=data['manifests']['validation']
    if sha(record['path'])!=record['sha256']: raise ValueError('Input manifest changed')
    processor=SiglipImageProcessor.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True)
    dataset=RawVideoClips(record['path'],processor,frames=8,classes=174)
    index={r['id']:i for i,r in enumerate(dataset.rows)}
    block=data['splits']['validation'][0]
    chosen=block['evidence'][:1]+block['suffix']
    pixels=[]; inputs=[]
    for row in chosen:
        i=index[row['id']]
        if sha(dataset.rows[i]['video'])!=row['sha256']: raise ValueError('Source video changed')
        values,_,metadata=dataset.read(i)
        pixels.append(values)
        inputs.append(dict(**metadata,source_sha256=row['sha256'],pixels_sha256=tensor_hash(values)))
    pixels=torch.cat(pixels)
    backbone=SiglipVisionModel.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True)
    encoder=TemporalSiglip(backbone,kind='attention').eval().requires_grad_(False)
    for adapter in encoder.adapters.values(): adapter.gain.fill_(.1)
    fingerprint=model_hash(encoder)
    sources=['exploration/buffered_attention.py','exploration/check_buffered_attention.py',
        'exploration/evaluate_video_history.py','temporal_siglip.py','video_data.py','train_video.py','pyproject.toml','uv.lock']
    plan=dict(model=DEFAULT_MODEL,revision=DEFAULT_REVISION,frames=len(pixels),capacity=len(pixels),
        data_plan_sha256=sha(args.data),inputs=inputs,sources={f:sha(f) for f in sources},
        initialization='Pretrained SigLIP2, seed71 random attention, residual gates set to 0.1 so the temporal path is exercised',
        max_abs_tolerance=1e-5,scope='Numerical implementation qualification; neither trained recognition nor speed measurement')
    output.mkdir(parents=True); (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    encoder.cuda()
    rows=compare_streams(encoder,(x[None] for x in pixels),len(pixels))
    if model_hash(encoder)!=fingerprint: raise ValueError('Parameters changed')
    for path,digest in plan['sources'].items():
        if sha(path)!=digest: raise ValueError('Source changed')
    maximum=max(r['max_abs'] for r in rows)
    passed=maximum<=plan['max_abs_tolerance']
    report=dict(passed=passed,plan_sha256=sha(output/'plan.json'),rows=rows,
        all_frames_exact=all(r['exact'] for r in rows),max_abs=maximum,
        weights_unchanged=True,model_state_sha256=fingerprint,
        capacity_frames=len(pixels),reserved_kv_bytes=rows[-1]['states']['buffered']-24,
        cpu_cursor_bytes=24,scope=plan['scope'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if not passed: raise ValueError('Numerical tolerance exceeded')


if __name__=='__main__': main()
