"""Matched real-video cost for Mamba, buffered full attention and three windows.

Five measured rotations balance traversal positions across five variants.
Correctness checks run outside timing on every frame of each cost video.
Recognition and retention are evaluated by the separate delayed-action probe.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import SiglipImageProcessor

from benchmark_real_video import replay, summarize_rows
from encoder_costs import paired_costs
from exploration.benchmark_buffered_attention import assert_exclusive, select_protocol
from exploration.buffered_attention import compare_streams, select, variants as buffered_variants
from exploration.cache_windowed_delayed_action import require_qualification
from exploration.check_resettable_window_attention import AbsoluteResetReference
from exploration.evaluate_video_history import model_hash
from exploration.resettable_window_attention import variants as windowed_variants
from point_experiment import write_json
from robotap_data import iter_videos, video_info
from temporal_priority import verify_video
from train_video import configure_reproducibility
from video_checkpoint import file_hash, load_video_checkpoint


VARIANTS = ('mamba_interleaved', 'attention_buffered', 'attention_window8',
            'attention_window32', 'attention_window128')
WINDOWS = (8, 32, 128)
REPEATS = len(VARIANTS)


def require_predecessor(queue, qualification):
    """Avoid starting in a CPU-only gap of the earlier GPU experiment queue."""
    plan = json.loads((queue / 'plan.json').read_text())
    status = json.loads((queue / 'status.json').read_text())
    if status.get('status') != 'complete':
        raise ValueError('The bounded-recognition queue must complete before cost measurements')
    if Path(plan['qualification']).resolve() != qualification.resolve():
        raise ValueError('Cost and recognition use different GPU qualifications')
    for path, digest in plan['pins'].items():
        if file_hash(path) != digest:
            raise ValueError('Preceding recognition input changed: ' + path)
    summary = queue / 'summary.json'
    if (status.get('summary_sha256') != file_hash(summary)
            or json.loads(summary.read_text()).get('queue_plan_sha256') != file_hash(queue / 'plan.json')):
        raise ValueError('Preceding recognition completion artifacts changed')
    expected = {(seed, window) for seed in (42, 43, 44) for window in WINDOWS}
    actual = [(row['seed'], row['window']) for row in status['completed']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Preceding recognition did not complete all nine conditions')
    return dict(queue=str(queue), plan_sha256=file_hash(queue / 'plan.json'),
                summary_sha256=file_hash(summary))


def traversal_order(repeat):
    if type(repeat) is not int or not 0 <= repeat < REPEATS:
        raise ValueError('Expected one of the five measured rotations')
    return VARIANTS[repeat:] + VARIANTS[:repeat]


def expected_state_bytes(variant, capacity):
    if variant == 'mamba_interleaved':
        return 79027200
    if variant == 'attention_buffered':
        return 602112 * capacity + 24
    if variant in VARIANTS and variant.startswith('attention_window'):
        return 602112 * int(variant.removeprefix('attention_window')) + 48
    raise ValueError('Unknown cost variant')


@torch.inference_mode()
def compare_bounded_streams(encoder, images, window):
    """Compare trained ring and chronological paths; always restore the core."""
    original, bounded = windowed_variants(encoder, window)
    reference = {k: AbsoluteResetReference(m, window) for k, m in original.items()}
    device = next(encoder.parameters()).device
    states, rows, pointers = {'ring': None, 'reference': None}, [], None
    try:
        for frame, image in enumerate(images):
            outputs = {}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                for name, mixers in (('ring', bounded), ('reference', reference)):
                    select(encoder, mixers)
                    value, states[name] = encoder.step(image.to(device), states[name])
                    if not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite bounded-attention features')
                    outputs[name] = value.float()
            difference = outputs['ring'] - outputs['reference']
            current = [[t.data_ptr() for t in cache] for cache in states['ring'].layers.values()]
            if pointers is not None and current != pointers:
                raise ValueError('Bounded cache was reallocated')
            pointers = current
            if any(cache[2].tolist() != [frame + 1, 0] for cache in states['ring'].layers.values()):
                raise ValueError('Unexpected absolute offset or reset origin')
            rows.append(dict(frame=frame, max_abs=float(difference.abs().max()),
                relative_l2=float(difference.norm() / outputs['reference'].norm().clamp_min(1e-12)),
                state_bytes=states['ring'].nbytes))
        if not rows:
            raise ValueError('No comparison images')
        return rows
    finally:
        select(encoder, original)


def verify_numerics(rows, frames, limits, expected_bytes):
    if [row['frame'] for row in rows] != list(range(frames)):
        raise ValueError('Numerical comparison omitted or duplicated frames')
    if any(not 0 <= row['max_abs'] <= limits['max_abs']
           or not 0 <= row['relative_l2'] <= limits['relative_l2']
           or row['state_bytes'] != expected_bytes for row in rows):
        raise ValueError('Predeclared trained-model numerical alarm or state capacity failed')


@torch.inference_mode()
def measure(priority, selection, qualification, output, after_queue):
    if output.exists():
        raise FileExistsError('Use a fresh measurement directory')
    # Reject another GPU job before loading fixtures or creating output.
    assert_exclusive()
    predecessor = require_predecessor(after_queue, qualification)
    qualified = require_qualification(qualification)
    if qualified['windows'] != list(WINDOWS):
        raise ValueError('Qualification window grid differs')
    torch.set_num_threads(4)
    configure_reproducibility(42)
    references, cost, video_row = select_protocol(priority, selection)
    if file_hash(Path(cost['data_root']) / 'extraction.json') != cost['extraction_sha256']:
        raise ValueError('RGB extraction manifest changed')
    video = None
    for shard, name, example in iter_videos(cost['data_root']):
        if name == video_row['id']:
            info = video_info(example)
            if (shard != video_row['shard'] or info['video_sha256'] != video_row['video_sha256']
                    or info['frames'] != video_row['frames']):
                raise ValueError('Fixed cost video changed')
            video = example['video'].copy()
            break
    if video is None:
        raise ValueError('Fixed cost video missing')
    del example
    models, provenance, parameters, fingerprints = {}, {}, {}, {}
    for name, row in references.items():
        completed, _ = verify_video(row)
        recognizer, record = load_video_checkpoint(Path(row['run']) / 'last.pt', 'cpu')
        if record['checkpoint_sha256'] != completed['checkpoint_sha256']:
            raise ValueError('Final checkpoint changed during loading')
        encoder = recognizer.encoder.requires_grad_(False).eval()
        models[name], provenance[name] = encoder, record
        parameters[name] = dict(count=sum(p.numel() for p in encoder.parameters()),
                                bytes=sum(p.numel() * p.element_size() for p in encoder.parameters()))
        fingerprints[name] = model_hash(encoder)
        del encoder, recognizer
    config = provenance['attention_interleaved']['config']
    processor = SiglipImageProcessor.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    sources = ['exploration/benchmark_windowed_attention.py', 'exploration/test_windowed_cost.py',
        'exploration/benchmark_buffered_attention.py', 'exploration/buffered_attention.py',
        'exploration/resettable_window_attention.py', 'exploration/check_resettable_window_attention.py',
        'exploration/check_windowed_attention.py', 'exploration/cache_windowed_delayed_action.py',
        'exploration/evaluate_video_history.py', 'benchmark_real_video.py', 'benchmark_streaming.py',
        'encoder_costs.py', 'temporal_priority.py', 'temporal_siglip.py', 'video_checkpoint.py',
        'robotap_data.py', 'train_video.py', 'pyproject.toml', 'uv.lock']
    plan = dict(priority_plan_sha256=file_hash(Path(priority) / 'plan.json'), checkpoints=provenance,
        predecessor=predecessor,
        video=video_row, frames=len(video), variants=list(VARIANTS), windows=list(WINDOWS),
        batch_size=1, repeats=REPEATS, warmup_passes=1, parameter_counts=parameters,
        measured_order=[list(traversal_order(r)) for r in range(REPEATS)], output_tokens=[196, 768],
        classifier_timed=False, qualification_summary_sha256=file_hash(qualification / 'summary.json'),
        full_attention_max_abs_tolerance=1e-5, window_tolerances=qualified['tolerances']['bf16'],
        source_sha256={p: file_hash(p) for p in sources},
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        scope='Five same-session implementation costs on identical full real RGB videos; no recognition or retention claim',
        timing='Original synchronized replay; decoding, loading, transfers between model traversals and correctness checks excluded. Pipeline latency includes preprocessing/input copy.',
        memory='One GPU encoder per measured traversal; allocated KV capacity plus CPU cursor metadata counted. PyTorch peak includes parameters, input, live state/output and temporaries, not driver allocations.',
        limitations=['Five rotations differ from the earlier three-pass experiment; all five controls are remeasured together.',
                    'Sliding windows discard evidence; useful retention must be evaluated separately.',
                    'All numerical checks precede timing and use fixed alarms, not recognition guarantees.',
                    'Synchronized per-frame distributions are correlated; not throughput or confidence intervals.'])
    output.mkdir(parents=True)
    write_json(output / 'plan.json', plan)
    attention = models['attention_interleaved'].cuda()

    def images():
        return (processor(images=Image.fromarray(frame), return_tensors='pt').pixel_values for frame in video)

    correctness = dict(full_attention=compare_streams(attention, images(), len(video)), windows={})
    write_json(output / 'correctness.json', dict(plan_sha256=file_hash(output / 'plan.json'), **correctness))
    if any(not 0 <= r['max_abs'] <= plan['full_attention_max_abs_tolerance'] for r in correctness['full_attention']):
        raise ValueError('Original/buffered full attention differs; preserve diagnostics')
    for window in WINDOWS:
        checked = compare_bounded_streams(attention, images(), window)
        correctness['windows'][str(window)] = checked
        write_json(output / 'correctness.json', dict(plan_sha256=file_hash(output / 'plan.json'), **correctness))
        verify_numerics(checked, len(video), plan['window_tolerances'],
                        expected_state_bytes(f'attention_window{window}', len(video)))
    attention.cpu()
    original, buffered = buffered_variants(attention, len(video))
    policies = {'attention_buffered': buffered}
    for window in WINDOWS:
        _, policies[f'attention_window{window}'] = windowed_variants(attention, window)
    gc.collect()
    torch.cuda.empty_cache()
    measured = {name: [] for name in VARIANTS}
    hashes, samples = {name: [] for name in VARIANTS}, {name: [] for name in VARIANTS}
    try:
        for repeat in range(-1, REPEATS):
            for variant in (VARIANTS if repeat < 0 else traversal_order(repeat)):
                assert_exclusive()
                name = 'mamba_interleaved' if variant == 'mamba_interleaved' else 'attention_interleaved'
                encoder = models[name]
                if name == 'attention_interleaved':
                    select(encoder, policies[variant])
                encoder.cuda()
                gc.collect()
                torch.cuda.empty_cache()
                observed, digest, memory = replay(encoder, processor, video, max(0, repeat), repeat >= 0)
                if repeat >= 0:
                    if any(r['state_bytes'] != expected_state_bytes(variant, len(video)) for r in observed):
                        raise ValueError('Measured cache capacity differs')
                    measured[variant] += observed
                    hashes[variant].append(digest)
                    samples[variant].append(memory)
                    directory = output / variant
                    directory.mkdir(exist_ok=True)
                    with (directory / 'timings.jsonl').open('a') as stream:
                        for row in observed:
                            stream.write(json.dumps(row) + '\n')
                encoder.cpu()
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
                print(json.dumps(dict(variant=variant, repeat=repeat, frames=len(video))), flush=True)
    finally:
        select(attention, original)
    measurements = []
    for variant in VARIANTS:
        name = 'mamba_interleaved' if variant == 'mamba_interleaved' else 'attention_interleaved'
        summary = summarize_rows(measured[variant], len(video), REPEATS)
        summary.update(parameter_count=parameters[name]['count'], parameter_bytes=parameters[name]['bytes'],
            hardware={k: plan[k] for k in ('gpu', 'torch', 'cuda')},
            repeated_final_outputs_equal=len(set(hashes[variant])) == 1,
            final_output_sha256_by_pass=hashes[variant], process_memory_after_pass=samples[variant],
            paused_compute_pids=[], timings_sha256=file_hash(output / variant / 'timings.jsonl'))
        write_json(output / variant / 'summary.json', summary)
        measurements.append(dict(model=variant, video=video_row, measurement=summary))
    for name, model in models.items():
        if model_hash(model) != fingerprints[name]:
            raise ValueError('Encoder parameters changed')
    for p, digest in plan['source_sha256'].items():
        if file_hash(p) != digest:
            raise ValueError('Cost measurement source changed: ' + p)
    assert_exclusive()
    write_json(output / 'summary.json', dict(plan_sha256=file_hash(output / 'plan.json'),
        correctness_sha256=file_hash(output / 'correctness.json'), measurements=measurements,
        weights_unchanged=True, **paired_costs(measurements)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--priority', type=Path, default=Path('results/temporal_priority_seed42_rgbfix'))
    parser.add_argument('--qualification', type=Path, default=Path('results/resettable_window_attention_qualification_v1'))
    parser.add_argument('--after-queue', type=Path, default=Path('results/windowed_delayed_action_queue_v1'))
    parser.add_argument('--selection', choices=('shortest', 'median', 'longest'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measure(args.priority, args.selection, args.qualification, args.output, args.after_queue)
