"""Verify paired real-video cost measurements without claiming recognition quality."""
from pathlib import Path

import json

from benchmark_real_video import summarize_rows
from point_checkpoint import file_hash
from point_experiment import read_json
from train_video import REPRODUCIBILITY


def select_videos(rows):
    """Choose shortest, median and longest distinct RGB videos, without scores."""
    unique = {}
    for row in sorted(rows, key=lambda r: (r['frames'], r['id'])):
        unique.setdefault(row['video_sha256'], row)
    ordered = list(unique.values())
    if len(ordered) < 3 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Need at least three distinct videos and unique identifiers')
    return [dict(selection=label, **{key:ordered[index][key] for key in
            ('id', 'frames', 'video_sha256', 'shard')})
            for label,index in [('shortest',0), ('median',len(ordered)//2), ('longest',len(ordered)-1)]]


def verify_measurement(directory, expected_config, checkpoint, video, repeats, warmup, extraction_hash):
    directory = Path(directory)
    plan, report = read_json(directory/'plan.json'), read_json(directory/'summary.json')
    if (report['plan_sha256'] != file_hash(directory/'plan.json')
            or report['timings_sha256'] != file_hash(directory/'timings.jsonl')):
        raise ValueError('Measurement artifacts changed')
    if (plan['checkpoint']['config'] != expected_config
            or plan['checkpoint']['checkpoint_sha256'] != checkpoint['checkpoint_sha256']
            or plan['checkpoint']['step'] != checkpoint['step']):
        raise ValueError('Measurement belongs to a different checkpoint or protocol')
    actual = plan['video']
    if (actual['id'] != video['id'] or actual['shard'] != video['shard']
            or actual['frames_used'] != video['frames']
            or actual['original']['frames'] != video['frames']
            or actual['input_rgb_sha256'] != video['video_sha256']
            or actual['original']['video_sha256'] != video['video_sha256']):
        raise ValueError('Measurement uses different RGB data or only a prefix')
    if (plan['dataset_extraction_sha256'] != extraction_hash
            or plan['batch_size'] != 1 or plan['repeats'] != repeats
            or plan['warmup_passes'] != warmup or plan['classifier_included']
            or plan['output_tokens'] != [196,768]
            or plan['reproducibility'] != REPRODUCIBILITY):
        raise ValueError('Cost protocol differs')
    for name, digest in plan['source_sha256'].items():
        if file_hash(name) != digest:
            raise ValueError(f'Measurement implementation changed: {name}')
    rows = [json.loads(line) for line in (directory/'timings.jsonl').read_text().splitlines() if line]
    recount = summarize_rows(rows,video['frames'],repeats)
    if any(report.get(key) != value for key,value in recount.items()):
        raise ValueError('Cost summary differs from raw frame measurements')
    if (len(report['final_output_sha256_by_pass']) != repeats
            or len(report['process_memory_after_pass']) != repeats):
        raise ValueError('Missing replay output or memory records')
    kind = expected_config['kind']
    state_sizes = {row['state_bytes'] for row in rows}
    if kind in ('mamba','gmmix') and (len(state_sizes) != 1 or next(iter(state_sizes)) <= 0):
        raise ValueError('Recurrent state was not fixed size')
    if kind == 'none' and state_sizes != {0}:
        raise ValueError('History-free encoder retained temporal state')
    return dict(directory=str(directory), summary_sha256=file_hash(directory/'summary.json'),
        **recount, parameter_count=plan['parameter_count'], parameter_bytes=plan['parameter_bytes'],
        hardware=dict(gpu=plan['gpu'],torch=plan['torch'],cuda=plan['cuda']),
        repeated_final_outputs_equal=report['repeated_final_outputs_equal'],
        process_memory_after_pass=report['process_memory_after_pass'],
        paused_compute_pids=plan['initial_paused_compute_pids'])


def paired_costs(measurements):
    """Ratios use identical videos/bins; no accuracy scores enter this function."""
    models = {row['model'] for row in measurements}
    videos = {row['video']['id'] for row in measurements}
    pairs = {(r['model'],r['video']['id']):r for r in measurements}
    if len(pairs) != len(measurements) or len(pairs) != len(models)*len(videos):
        raise ValueError('Incomplete or duplicated model/video measurements')
    reference = 'mamba_interleaved'
    if reference not in models:
        raise ValueError('Missing interleaved Mamba reference')
    hardware = measurements[0]['measurement']['hardware']
    comparisons = []
    for name in sorted(models-{reference}):
        for video in sorted(videos):
            a,b = pairs[reference,video],pairs[name,video]
            if a['video'] != b['video']:
                raise ValueError('Paired models use different source videos')
            am,bm = a['measurement'],b['measurement']
            if am['hardware'] != hardware or bm['hardware'] != hardware:
                raise ValueError('Measurements use different hardware/software')
            bins = []
            for x,y in zip(am['bins'],bm['bins'],strict=True):
                if any(x[k] != y[k] for k in ('history_min','history_max','samples','unique_frames')):
                    raise ValueError('Paired history bins differ')
                bins.append(dict(history_min=x['history_min'],history_max=x['history_max'],
                    mamba_over_other_gpu_median=x['gpu_ms']['p50']/y['gpu_ms']['p50'],
                    mamba_over_other_pipeline_median=x['pipeline_wall_ms']['p50']/y['pipeline_wall_ms']['p50']))
            comparisons.append(dict(other=name,video_id=video,bins=bins,
                mamba_over_other_torch_peak=am['torch_peak_allocated_bytes']/bm['torch_peak_allocated_bytes']))
    return dict(comparisons=comparisons, ratio_interpretation='Below one favors interleaved Mamba for time/memory',
        scope='Paired implementation cost on the listed fixed real videos; no accuracy or retention claim',
        limits='Conditional timing distributions, not confidence intervals. PyTorch peak excludes driver/context and non-PyTorch allocations. No throughput claim or generalized robot-success claim.')
