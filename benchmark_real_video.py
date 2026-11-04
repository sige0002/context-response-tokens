"""Measure a trained dense encoder on consecutive real RGB frames.

Each pass starts with an empty state. There are no saved state copies or full
video tensors on the GPU. Model timing excludes preprocessing and input copy;
pipeline timing includes both, but excludes video decoding and metric checks.
Torch peak memory includes parameters, input, state, output and temporaries.
Driver-reported process memory is sampled between passes, not a peak measure.
This measures implementation cost, not video understanding or memory retention.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor

from benchmark_streaming import distribution, idle_compute_processes
from point_experiment import write_json
from robotap_data import iter_videos, video_info
from video_checkpoint import file_hash, load_video_checkpoint
from train_video import configure_reproducibility, REPRODUCIBILITY


HISTORY_BINS = ((0, 0), (1, 7), (8, 31), (32, 127), (128, 255), (256, None))


def summarize_rows(rows, frames, repeats):
    """Require exactly one observation for every frame of each replay."""
    expected = {(repeat, frame) for repeat in range(repeats) for frame in range(frames)}
    actual = [(row['repeat'], row['history_frames']) for row in rows]
    if not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing, duplicate or out-of-range timing observations')
    for row in rows:
        if row['absolute_frames_after'] != row['history_frames'] + 1:
            raise ValueError('State was not reset for a replay or advanced incorrectly')
        for key in ('gpu_ms', 'model_wall_ms', 'pipeline_wall_ms'):
            if not np.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('Nonpositive or nonfinite timing')
        if (row['torch_peak_allocated_bytes'] < row['torch_allocated_before_bytes']
                or row['torch_peak_allocated_bytes'] < row['torch_allocated_after_bytes']
                or row['state_bytes'] < 0):
            raise ValueError('Inconsistent memory accounting')
    bins = []
    for lower, upper in HISTORY_BINS:
        selected = [r for r in rows if r['history_frames'] >= lower
                    and (upper is None or r['history_frames'] <= upper)]
        if selected:
            bins.append(dict(history_min=lower, history_max=upper,
                samples=len(selected), unique_frames=len({r['history_frames'] for r in selected}),
                **{key:distribution([r[key] for r in selected])
                   for key in ('gpu_ms', 'model_wall_ms', 'pipeline_wall_ms')},
                state_bytes_min=min(r['state_bytes'] for r in selected),
                state_bytes_max=max(r['state_bytes'] for r in selected),
                torch_peak_allocated_bytes=max(r['torch_peak_allocated_bytes'] for r in selected)))
    return dict(bins=bins, frames_per_pass=frames, repeats=repeats,
        torch_peak_allocated_bytes=max(r['torch_peak_allocated_bytes'] for r in rows),
        max_state_bytes=max(r['state_bytes'] for r in rows),
        scope='Empirical timing distribution over fixed real frames and repeated passes; not a confidence interval')


def process_memory_sample():
    """A post-pass driver observation, explicitly separate from allocator peak."""
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(',')]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                return dict(mib=float(fields[1]), status='reported', scope='After pass, before clearing CUDA allocator cache; not peak')
            except ValueError:
                return dict(mib=None, status=fields[1], scope='Driver did not report process memory')
    return dict(mib=None, status='process absent', scope='Driver did not report process memory')


@torch.inference_mode()
def replay(encoder, processor, video, repeat, measure):
    state, rows = None, []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    final_hash = None
    for frame, image in enumerate(video):
        pipeline_started = time.perf_counter()
        pixels = processor(images=Image.fromarray(image), return_tensors='pt').pixel_values.cuda()
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model_started = time.perf_counter()
        start.record()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            features, state = encoder.step(pixels, state)
        end.record()
        torch.cuda.synchronize()
        stopped = time.perf_counter()
        peak, after = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
        if state.offset != frame + 1:
            raise ValueError('Streaming state did not advance by one frame')
        if tuple(features.shape) != (1, 196, 768) or not torch.isfinite(features).all():
            raise ValueError('Expected finite batch-one 196x768 current-frame tokens')
        if measure:
            rows.append(dict(repeat=repeat, history_frames=frame, absolute_frames_after=state.offset,
                gpu_ms=start.elapsed_time(end), model_wall_ms=(stopped-model_started)*1000,
                pipeline_wall_ms=(stopped-pipeline_started)*1000, state_bytes=state.nbytes,
                torch_allocated_before_bytes=allocated_before, torch_allocated_after_bytes=after,
                torch_peak_allocated_bytes=peak, torch_reserved_after_bytes=torch.cuda.memory_reserved()))
        if frame == len(video)-1:
            final_hash=hashlib.sha256(features.float().cpu().numpy().tobytes()).hexdigest()
        # Keep only the live streaming state between images. The last output is
        # consumed and released, as for a caller that does not store all tokens.
        del features, pixels
    torch.cuda.synchronize()
    memory_sample = process_memory_sample()
    del state
    return rows, final_hash, memory_sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--encoder-type', choices=('temporal','gmmix'), default='temporal')
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
    if args.encoder_type=='gmmix':
        from gmmix_checkpoint import load_gmmix_video_checkpoint
        recognizer,provenance=load_gmmix_video_checkpoint(args.checkpoint,'cuda')
    else:
        recognizer,provenance=load_video_checkpoint(args.checkpoint,'cuda')
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
    names=['benchmark_real_video.py','benchmark_streaming.py','video_checkpoint.py','temporal_siglip.py',
        'gmmix_checkpoint.py','gmmix_siglip.py','robotap_data.py','pyproject.toml','uv.lock','train_video.py']
    if args.encoder_type=='gmmix':
        names+=['baselines/gmmix/pyproject.toml','baselines/gmmix/uv.lock','baselines/gmmix/source_manifest.json']
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
