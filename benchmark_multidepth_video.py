"""Multi-depth baseline cost through the unchanged real-frame replay routine.

Protocol, timing boundaries and memory accounting match benchmark_real_video.
This standalone entry point preserves already pinned comparisons.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import SiglipImageProcessor

from benchmark_real_video import replay, summarize_rows
from benchmark_streaming import distribution, idle_compute_processes
from multidepth_checkpoint import load_multidepth_video_checkpoint
from point_checkpoint import file_hash
from point_experiment import write_json
from robotap_data import iter_videos, video_info
from train_video import configure_reproducibility, REPRODUCIBILITY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--encoder-type', choices=('gmmix_multidepth',), default='gmmix_multidepth')
    parser.add_argument('--root', default='data/robotap')
    parser.add_argument('--video-id', help='Default is the first video in the verified dataset iterator')
    parser.add_argument('--frames', type=int, default=0, help='0 uses the entire selected video')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup-passes', type=int, default=1)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    if args.frames < 0 or args.repeats < 2 or args.warmup_passes < 1:
        parser.error('Require nonnegative frames, at least two measured passes and one warmup')
    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Read and hash source RGB before loading any GPU model. Labels are never
    # passed to the encoder, and no benchmark choices depend on accuracy.
    selected=None
    for shard,name,example in iter_videos(args.root):
        if args.video_id is None or name==args.video_id:
            info=video_info(example)
            if args.frames > info['frames']:
                raise ValueError('Requested prefix exceeds the real video; frames are not repeated')
            video=example['video'][:args.frames or info['frames']].copy()
            selected=dict(id=name,shard=shard,original=info,frames_used=len(video),
                input_rgb_sha256=hashlib.sha256(video.tobytes()).hexdigest())
            break
    if selected is None:
        raise ValueError('Requested video does not exist')
    del example
    initial_paused=idle_compute_processes()
    torch.set_num_threads(4)
    configure_reproducibility(42)
    recognizer,provenance=load_multidepth_video_checkpoint(args.checkpoint,'cuda')
    encoder=recognizer.encoder
    parameter_count=sum(p.numel() for p in encoder.parameters())
    trainable_count=sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    parameter_bytes=sum(p.numel()*p.element_size() for p in encoder.parameters())
    encoder.requires_grad_(False).eval()
    del recognizer
    gc.collect()
    torch.cuda.empty_cache()
    config=provenance['config']
    if config.get('reproducibility') != REPRODUCIBILITY:
        raise ValueError('Benchmark requires the current explicit reproducibility protocol')
    processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    names=['benchmark_multidepth_video.py','benchmark_real_video.py','benchmark_streaming.py',
        'multidepth_checkpoint.py','gmmix_multidepth.py','gmmix_siglip.py','temporal_siglip.py',
        'robotap_data.py','pyproject.toml','uv.lock','train_video.py',
        'baselines/gmmix/pyproject.toml','baselines/gmmix/uv.lock','baselines/gmmix/source_manifest.json']
    plan=dict(scope='Trained encoder implementation cost on real RGB; no accuracy or retention result',
        checkpoint=provenance,encoder_type=args.encoder_type,architecture=encoder.architecture,
        video=selected,batch_size=1,repeats=args.repeats,warmup_passes=args.warmup_passes,
        output_tokens=[196,768],classifier_included=False,parameter_count=parameter_count,
        encoder_trainable_parameter_count_before_inference_freeze=trainable_count,parameter_bytes=parameter_bytes,
        initial_paused_compute_pids=initial_paused,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,precision=dict(spatial='FP32',temporal='BF16 autocast'),deterministic=True,
        reproducibility=REPRODUCIBILITY,
        source_sha256={name:file_hash(name) for name in names},
        dataset_extraction_sha256=file_hash(Path(args.root)/'extraction.json'),
        timing='GPU and model wall exclude preprocessing/input copy. Pipeline wall includes those but excludes decoding and diagnostic checks. Every frame is synchronized.',
        memory='Current process PyTorch allocated peak includes model parameters, one input, current state/output and temporaries; excludes driver/context and non-PyTorch allocations. No copied state or entire GPU video. Driver memory is only sampled after each pass.',
        limits='One real video; replayed frames are correlated. Training length may be shorter. Long-state processing does not establish useful memory or recognition. Paused other processes still occupy memory. Not an asynchronous throughput measurement.')
    write_json(output/'plan.json',plan)
    for index in range(args.warmup_passes):
        replay(encoder,processor,video,index,False)
    rows,hashes,samples=[],[],[]
    for repeat in range(args.repeats):
        idle_compute_processes()
        gc.collect()
        torch.cuda.empty_cache()
        observed,fingerprint,memory=replay(encoder,processor,video,repeat,True)
        rows+=observed;hashes.append(fingerprint);samples.append(memory)
        with (output/'timings.jsonl').open('a') as stream:
            for row in observed:
                stream.write(json.dumps(row)+'\n')
        print(json.dumps(dict(repeat=repeat,frames=len(observed),gpu_median_ms=distribution([r['gpu_ms'] for r in observed])['p50'],
            torch_peak_allocated_bytes=max(r['torch_peak_allocated_bytes'] for r in observed),process_memory_after_pass=memory)),flush=True)
    idle_compute_processes()
    report=dict(plan_sha256=file_hash(output/'plan.json'),timings_sha256=file_hash(output/'timings.jsonl'),
        **summarize_rows(rows,len(video),args.repeats),final_output_sha256_by_pass=hashes,
        repeated_final_outputs_equal=len(set(hashes))==1,process_memory_after_pass=samples)
    write_json(output/'summary.json',report)


if __name__=='__main__':
    main()
