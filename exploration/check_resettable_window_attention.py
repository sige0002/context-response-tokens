"""Qualify nonzero-origin ring resets using the existing real RGB fixture.

Prepare is CPU-only and does not duplicate the large fixture. Run requires
the whole preceding experiment chain to finish and the GPU to be idle.
"""
import argparse
import json
from pathlib import Path

import torch

from exploration.buffered_attention import select
from exploration.check_windowed_attention import (
    ChronologicalWindowAttention, require_main_complete, sha, write,
)
from exploration.evaluate_video_history import model_hash, tensor_hash
from exploration.resettable_window_attention import variants
from exploration.run_mimo_after_comparisons import assert_gpu_idle
from temporal_siglip import CausalAttention, StreamState, TemporalSiglip
from train_video import configure_reproducibility


class AbsoluteResetReference(ChronologicalWindowAttention):
    def forward(self, x, cache=None, offset=0, streaming=False):
        if not streaming or x.shape[1] != 1 or self.training or torch.is_grad_enabled():
            raise ValueError('Reference requires no-grad single-observation inference')
        if type(offset) is not int or offset < 0:
            raise ValueError('Reference requires a nonnegative absolute offset')
        if cache is None and offset > 0:
            y, (k, v) = CausalAttention.forward(self, x, None, offset, True)
            return y, (k, v, torch.tensor(offset + 1, dtype=torch.int64))
        return super().forward(x, cache, offset, streaming)


def checked_plan(output):
    plan = json.loads((output / 'plan.json').read_text())
    for path, expected in plan['pinned_files'].items():
        if sha(path) != expected:
            raise ValueError('Qualification input changed: ' + path)
    return plan


def require_completed_chain(plan):
    require_main_complete(plan['fixture_plan']['main_run'])
    directory = Path(plan['after_run'])
    status = json.loads((directory / 'status.json').read_text())
    if status.get('status') != 'complete':
        raise RuntimeError('Preceding experiment chain is not complete')
    if status.get('summary_sha256') != sha(directory / 'summary.json'):
        raise ValueError('Preceding pilot completion artifacts changed')
    assert_gpu_idle()


def prepare(output, fixture, after_run):
    if output.exists():
        raise FileExistsError('Use a fresh qualification directory')
    old = json.loads((fixture / 'plan.json').read_text())
    pins = dict(old['sources'])
    pins[str(fixture / 'encoder.pt')] = old['encoder_sha256']
    pins[str(fixture / 'pixels.pt')] = old['pixels_file_sha256']
    sources = ['exploration/resettable_window_attention.py',
               'exploration/check_resettable_window_attention.py',
               'exploration/test_resettable_window_attention.py',
               'exploration/test_resettable_window_qualification.py',
               'exploration/run_mimo_after_comparisons.py',
               'exploration/cache_delayed_action.py']
    for path in [str(fixture / 'plan.json'), str(after_run / 'plan.json'), *sources]:
        pins[path] = sha(path)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Prepared fixture/source changed: ' + path)
    plan = dict(fixture=str(fixture), fixture_plan=old, after_run=str(after_run),
                pinned_files=pins, origins=[0, 8], windows=[8, 32, 128],
                tolerances=old['tolerances'], expected_rows=(136 + 128) * 3 * 2,
                scope='Real RGB, random temporal layers, absolute-time-preserving reset; no recognition or timing',
                limitation='This covers 136 observations and an origin-eight reset; not arbitrary-length GPU verification')
    output.mkdir(parents=True)
    write(output / 'plan.json', plan)
    print(json.dumps(dict(status='prepared_cpu_only', expected_rows=plan['expected_rows'],
                          plan_sha256=sha(output / 'plan.json'))))


@torch.inference_mode()
def run(output):
    plan = checked_plan(output)
    require_completed_chain(plan)
    if (output / 'summary.json').exists() or (output / 'rows.jsonl').exists():
        raise FileExistsError('Preserve previous result or inspect partial execution')
    torch.set_num_threads(2)
    configure_reproducibility(71)
    fixture = Path(plan['fixture'])
    encoder = TemporalSiglip.load_checkpoint(fixture / 'encoder.pt').requires_grad_(False)
    pixels = torch.load(fixture / 'pixels.pt', map_location='cpu', weights_only=True)
    old = plan['fixture_plan']
    if model_hash(encoder) != old['model_state_sha256'] or tensor_hash(pixels) != old['pixels_tensor_sha256']:
        raise ValueError('Decoded fixture changed')
    encoder.cuda()
    original = {key: adapter.mixer for key, adapter in encoder.adapters.items()}
    rows = []
    try:
        with (output / 'rows.jsonl').open('x') as stream:
            for precision, limits in plan['tolerances'].items():
                for window in plan['windows']:
                    for origin in plan['origins']:
                        select(encoder, original)
                        _, ring = variants(encoder, window)
                        reference = {k: AbsoluteResetReference(m, window) for k, m in original.items()}
                        states = {name: StreamState(id(encoder), 1, tuple(pixels.shape[1:]), offset=origin)
                                  for name in ('ring', 'reference')}
                        pointers = None
                        for absolute in range(origin, len(pixels)):
                            features = {}
                            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                                for name, mixers in (('ring', ring), ('reference', reference)):
                                    select(encoder, mixers)
                                    features[name], states[name] = encoder.step(
                                        pixels[absolute:absolute + 1].cuda(), states[name])
                            a, b = features['ring'].float(), features['reference'].float()
                            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                                raise FloatingPointError('Nonfinite feature')
                            difference = a - b
                            maximum = float(difference.abs().max())
                            relative = float(difference.norm() / b.norm().clamp_min(1e-12))
                            current = [[v.data_ptr() for v in c] for c in states['ring'].layers.values()]
                            if pointers is not None and current != pointers:
                                raise ValueError('Ring storage was reallocated')
                            pointers = current
                            expected_bytes = 602112 * window * (2 if precision == 'fp32' else 1) + 48
                            if states['ring'].nbytes != expected_bytes:
                                raise ValueError('Unexpected ring capacity')
                            if any(c[2].tolist() != [absolute + 1, origin]
                                   for c in states['ring'].layers.values()):
                                raise ValueError('Absolute offset or reset origin changed')
                            row = dict(precision=precision, window=window, origin=origin,
                                       absolute_observation=absolute, observations_since_reset=absolute - origin + 1,
                                       max_abs=maximum, relative_l2=relative, state_bytes=states['ring'].nbytes,
                                       passed=maximum <= limits['max_abs'] and relative <= limits['relative_l2'])
                            rows.append(row)
                            stream.write(json.dumps(row) + '\n')
                        stream.flush()
                        print(json.dumps(dict(precision=precision, window=window, origin=origin,
                                              frames=len(pixels) - origin)), flush=True)
    finally:
        select(encoder, original)
    if model_hash(encoder) != old['model_state_sha256']:
        raise ValueError('Parameters changed')
    checked_plan(output)
    if len(rows) != plan['expected_rows']:
        raise ValueError('Incomplete qualification rows')
    summary = dict(passed=all(r['passed'] for r in rows), rows=len(rows),
                   plan_sha256=sha(output / 'plan.json'), rows_sha256=sha(output / 'rows.jsonl'),
                   weights_unchanged=True, max_abs=max(r['max_abs'] for r in rows),
                   max_relative_l2=max(r['relative_l2'] for r in rows),
                   gpu=torch.cuda.get_device_name(), torch=torch.__version__, scope=plan['scope'],
                   recognition_evaluated=False, speed_measured=False)
    write(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if not summary['passed']:
        raise ValueError('Predeclared numerical alarm exceeded; preserve results for diagnosis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=Path('results/windowed_attention_qualification_v1'))
    parser.add_argument('--after-run', type=Path, default=Path('results/mimo_pilot_step11000_v2'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.fixture, args.after_run)
    else:
        run(args.output)
