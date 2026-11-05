"""Paired implementation-cost control: dynamic KV, buffered KV, and Mamba.

Only one encoder resides on GPU during each measured traversal. Both attention
paths are checked on every real frame before timing. No recognition claim is
made from these cost videos or from numerical output agreement.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import SiglipImageProcessor

from benchmark_real_video import replay,summarize_rows
from benchmark_streaming import idle_compute_processes
from encoder_costs import paired_costs
from exploration.buffered_attention import compare_streams,variants,select
from exploration.evaluate_video_history import model_hash
from point_experiment import write_json
from robotap_data import iter_videos,video_info
from temporal_priority import verify_video
from train_video import configure_reproducibility
from video_checkpoint import file_hash,load_video_checkpoint


VARIANTS=('attention_dynamic','attention_buffered','mamba_interleaved')


def assert_exclusive():
    if idle_compute_processes():
        raise RuntimeError('Paused GPU processes still occupy memory; require exclusive measurements')


def traversal_order(repeat):
    # Latin rotation makes each implementation appear once at each position.
    shift=repeat%len(VARIANTS)
    return VARIANTS[shift:]+VARIANTS[:shift]


def select_protocol(priority,selection):
    source=json.loads((Path(priority)/'plan.json').read_text())
    expected={r['name']:r for r in source['encoders']}
    rows={name:expected[name] for name in ('attention_interleaved','mamba_interleaved')}
    for row in rows.values():
        if row['config']['seed']!=42 or row['config']['steps']!=20000:
            raise ValueError('Expected the predeclared final seed42 encoders')
    videos=[v for v in source['cost']['videos'] if v['selection']==selection]
    if len(videos)!=1 or source['cost']['repeats']!=3 or source['cost']['warmup_passes']!=1:
        raise ValueError('Unexpected video or traversal protocol')
    return rows,source['cost'],videos[0]


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--priority',default='results/temporal_priority_seed42_rgbfix')
    p.add_argument('--selection',choices=('shortest','median','longest'),required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args(); output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a fresh measurement directory')
    assert_exclusive(); torch.set_num_threads(4); configure_reproducibility(42)
    rows,cost,expected=select_protocol(args.priority,args.selection)
    if file_hash(Path(cost['data_root'])/'extraction.json')!=cost['extraction_sha256']:
        raise ValueError('RGB dataset extraction changed')
    video=None
    for shard,name,example in iter_videos(cost['data_root']):
        if name==expected['id']:
            info=video_info(example)
            if shard!=expected['shard'] or info['video_sha256']!=expected['video_sha256'] or info['frames']!=expected['frames']:
                raise ValueError('Fixed cost video changed')
            video=example['video'].copy(); break
    if video is None: raise ValueError('Cost video missing')
    del example
    models={}; provenance={}; parameters={}; fingerprints={}; readout=None
    for name,row in rows.items():
        completed,_=verify_video(row)
        recognizer,record=load_video_checkpoint(Path(row['run'])/'last.pt','cpu')
        if record['checkpoint_sha256']!=completed['checkpoint_sha256']:
            raise ValueError('Final checkpoint changed during loading')
        encoder=recognizer.encoder
        parameters[name]=dict(count=sum(p.numel() for p in encoder.parameters()),
            bytes=sum(p.numel()*p.element_size() for p in encoder.parameters()))
        models[name]=encoder.requires_grad_(False).eval(); provenance[name]=record
        fingerprints[name]=model_hash(encoder)
        if name=='attention_interleaved': readout=recognizer.readout.requires_grad_(False).eval()
        del recognizer,encoder
    config=provenance['attention_interleaved']['config']
    processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    sources=['exploration/benchmark_buffered_attention.py','exploration/buffered_attention.py',
        'exploration/evaluate_video_history.py','benchmark_real_video.py','benchmark_streaming.py',
        'encoder_costs.py','temporal_priority.py','temporal_siglip.py','video_checkpoint.py',
        'robotap_data.py','train_video.py','pyproject.toml','uv.lock']
    plan=dict(priority_plan_sha256=file_hash(Path(args.priority)/'plan.json'),checkpoints=provenance,
        video=expected,batch_size=1,frames=len(video),repeats=3,warmup_passes=1,
        parameter_counts=parameters,source_sha256={f:file_hash(f) for f in sources},
        capacity_frames=len(video),cursor_cpu_bytes=24,variants=list(VARIANTS),
        measured_order=[list(traversal_order(r)) for r in range(3)],
        max_abs_tolerance=1e-5,output_tokens=[196,768],classifier_timed=False,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        scope='Implementation control on fixed real RGB with identical attention weights; no recognition evaluation',
        memory='Each measured pass holds only its encoder on GPU. Buffered state counts all reserved KV slots plus 24 CPU cursor bytes. Capacity is a finite bound tied to this video, not unbounded constant memory.',
        timing='Same synchronized replay as primary measurements; decoding, model loading, numerical checking and parameter transfers excluded. Preprocessing/input copy separately included in pipeline latency.',
        limitations=['No compilation, quantization or sliding window is used; not an exhaustive optimized-attention comparison',
                    'Per-video capacity is known in advance','Timing repeats are correlated, not confidence intervals'])
    output.mkdir(parents=True); write_json(output/'plan.json',plan)
    attention=models['attention_interleaved'].cuda(); readout.cuda()
    images=(processor(images=Image.fromarray(frame),return_tensors='pt').pixel_values for frame in video)
    equality=compare_streams(attention,images,len(video),readout)
    write_json(output/'correctness.json',dict(rows=equality,plan_sha256=file_hash(output/'plan.json')))
    if (max(r['max_abs'] for r in equality)>plan['max_abs_tolerance']
            or any(r['readout_predictions']['dynamic']!=r['readout_predictions']['buffered'] for r in equality)):
        raise ValueError('Trained attention paths differ; inspect correctness before measuring cost')
    attention.cpu(); readout.cpu(); del readout
    original,buffered=variants(attention,len(video))
    gc.collect(); torch.cuda.empty_cache()
    measured={v:[] for v in VARIANTS}; hashes={v:[] for v in VARIANTS}; samples={v:[] for v in VARIANTS}
    for repeat in range(-1,3):
        for variant in (VARIANTS if repeat<0 else traversal_order(repeat)):
            assert_exclusive()
            name='mamba_interleaved' if variant=='mamba_interleaved' else 'attention_interleaved'
            encoder=models[name]
            if name=='attention_interleaved': select(encoder,buffered if variant=='attention_buffered' else original)
            encoder.cuda(); gc.collect(); torch.cuda.empty_cache()
            observed,digest,memory=replay(encoder,processor,video,max(0,repeat),repeat>=0)
            if repeat>=0:
                for row in observed:
                    if name=='attention_interleaved':
                        row['logical_kv_bytes']=602112*row['absolute_frames_after']
                measured[variant]+=observed; hashes[variant].append(digest); samples[variant].append(memory)
                directory=output/variant; directory.mkdir(exist_ok=True)
                with (directory/'timings.jsonl').open('a') as stream:
                    for row in observed: stream.write(json.dumps(row)+'\n')
            encoder.cpu(); del encoder
            gc.collect(); torch.cuda.empty_cache()
            print(json.dumps(dict(variant=variant,repeat=repeat,frames=len(video))),flush=True)
    select(attention,original)
    measurements=[]
    for variant in VARIANTS:
        name='mamba_interleaved' if variant=='mamba_interleaved' else 'attention_interleaved'
        summary=summarize_rows(measured[variant],len(video),3)
        summary.update(parameter_count=parameters[name]['count'],parameter_bytes=parameters[name]['bytes'],
            hardware={k:plan[k] for k in ('gpu','torch','cuda')},repeated_final_outputs_equal=len(set(hashes[variant]))==1,
            process_memory_after_pass=samples[variant],paused_compute_pids=[],final_output_sha256_by_pass=hashes[variant],
            timings_sha256=file_hash(output/variant/'timings.jsonl'))
        write_json(output/variant/'summary.json',summary)
        measurements.append(dict(model=variant,video=expected,measurement=summary))
    for name,model in models.items():
        if model_hash(model)!=fingerprints[name]: raise ValueError('Encoder weights changed')
    for path,digest in plan['source_sha256'].items():
        if file_hash(path)!=digest: raise ValueError('Measurement source changed')
    assert_exclusive()
    report=dict(plan_sha256=file_hash(output/'plan.json'),correctness_sha256=file_hash(output/'correctness.json'),
        measurements=measurements,**paired_costs(measurements),weights_unchanged=True,
        framewise_attention_max_abs=max(r['max_abs'] for r in equality),
        framewise_attention_all_exact=all(r['exact'] for r in equality))
    write_json(output/'summary.json',report)


if __name__=='__main__': main()
