"""Measure conditional SSM carry on a fixed real sequence, not recognition.

Only the homogeneous multiplier of the SSM-matrix component is measured.
It is not the sensitivity of the whole nonlinear encoder to past pixels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def carry_windows(adt, lag):
    """Log carry from an observed state to a later state, time axis first.

The multiplier at observation zero precedes the first observed state and
is excluded. For lag k, window i sums adt[i+1:i+k+1].
"""
    adt = np.asarray(adt, dtype=np.float64)
    if adt.ndim != 3 or not np.isfinite(adt).all() or np.any(adt > 0):
        raise ValueError('Expected finite nonpositive [time, coordinate, head] ADT')
    if type(lag) is not int or not 1 <= lag < len(adt):
        raise ValueError('Lag must connect two observed states')
    cumulative = np.concatenate([np.zeros_like(adt[:1]), adt[1:].cumsum(axis=0)])
    return cumulative[lag:] - cumulative[:-lag]


def summarize(adt, lags=(1, 8, 32, 128, 244)):
    result = {}
    for lag in lags:
        log_carry = carry_windows(adt, lag)
        result[str(lag)] = dict(
            windows=log_carry.shape[0], coordinates=log_carry.shape[1], heads=log_carry.shape[2],
            log_carry_quantiles=np.quantile(log_carry, [.05, .5, .95]).tolist(),
            carry_quantiles=np.quantile(np.exp(log_carry), [.05, .5, .95]).tolist(),
            fraction_carry_at_least_half=float((log_carry >= -np.log(2)).mean()),
            fraction_carry_at_least_one_percent=float((log_carry >= np.log(.01)).mean()),
            per_head_median_carry=np.median(np.exp(log_carry), axis=(0, 1)).tolist())
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(output):
    output = Path(output)
    plan = json.loads((output/'plan.json').read_text())
    result = json.loads((output/'summary.json').read_text())
    if result['plan_sha256'] != sha(output/'plan.json') or result['adt_sha256'] != sha(output/'adt.npz'):
        raise ValueError('Recorded artifacts changed')
    if not result['weights_unchanged'] or result['first_three_frames_replay_max_abs'] != 0:
        raise ValueError('Instrumentation altered the checked behavior')
    with np.load(output/'adt.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'3', '7', '11'}:
            raise ValueError('Missing adapter observations')
        recounted = {}
        for key in saved.files:
            if saved[key].shape != (245, 196, 8):
                raise ValueError('Wrong observed gate dimensions')
            recounted[key] = summarize(saved[key], plan['lags'])
    if recounted != result['conditional_ssm_carry']:
        raise ValueError('Carry statistics differ from raw gates')
    if not all(r['passed'] for r in result['native_homogeneous_checks']):
        raise ValueError('Native homogeneous propagation check failed')
    receipt = dict(passed=True, summary_sha256=sha(output/'summary.json'),
        adt_sha256=sha(output/'adt.npz'),
        scope='CPU recount of captured gate statistics; no independent GPU inference or recognition scoring')
    write_json(output/'verification.json', receipt)
    return receipt


def native_check(adt):
    """Isolate initial SSM propagation with zero future writes in native code."""
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    rows = []
    for lag in (8, 32, 128, 244):
        # One real coordinate, all heads. Frame zero has already been read.
        a = torch.tensor(adt[1:lag+1, 0], device='cuda').T[None].contiguous()
        h = a.shape[1]
        qk = torch.zeros((1, lag, 1, 64), device='cuda', dtype=torch.bfloat16)
        values = torch.zeros((1, lag, h, 64), device='cuda', dtype=torch.bfloat16)
        states = (torch.zeros(1, h, 16, device='cuda'),
                  torch.ones(1, h, 64, 64, device='cuda'),
                  torch.zeros(1, h, 64, device='cuda', dtype=torch.bfloat16),
                  torch.zeros(1, h, 64, device='cuda', dtype=torch.bfloat16))
        output = mamba3_siso_combined(Q=qk, K=qk, V=values, ADT=a,
            DT=torch.ones_like(a), Trap=torch.zeros_like(a),
            Q_bias=torch.zeros(h, 64, device='cuda'), K_bias=torch.zeros(h, 64, device='cuda'),
            Angles=torch.zeros(1, lag, h, 16, device='cuda'), Input_States=states,
            chunk_size=64, return_final_states=True)
        expected = a.double().sum(-1).exp().cpu().numpy()[..., None, None]
        actual = output[2].double().cpu().numpy()
        error = float(np.abs(actual - expected).max())
        passed = bool(np.allclose(actual, expected, rtol=2e-4, atol=2e-6))
        rows.append(dict(lag=lag, max_abs_error=error, rtol=2e-4, atol=2e-6, passed=passed))
        if not passed:
            raise ValueError(f'Native SSM carry mismatch: {rows[-1]}')
    return rows


def run(output):
    import torch
    import torch.nn.functional as F
    import mamba_ssm.modules.mamba3 as mamba3_module
    from PIL import Image
    from transformers import SiglipImageProcessor
    from exploration.evaluate_video_history import model_hash
    from robotap_data import iter_videos, video_info
    from train_video import configure_reproducibility
    from video_checkpoint import load_video_checkpoint

    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic directory')
    torch.set_num_threads(2)
    configure_reproducibility(42)
    torch.empty(1, device='cuda')
    prior = Path('results/memory_layer_intervention_seed17_v2/plan.json')
    prior_plan = json.loads(prior.read_text())
    checkpoint = Path(prior_plan['checkpoint'])
    recognizer, info = load_video_checkpoint(checkpoint, 'cpu')
    if info != prior_plan['checkpoint_info']:
        raise ValueError('Archived checkpoint changed')
    encoder = recognizer.encoder.requires_grad_(False).eval()
    del recognizer
    before = model_hash(encoder)
    if encoder.kind != 'mamba' or encoder.placement != 'interleaved':
        raise ValueError('Expected the archived interleaved Mamba model')
    priority = Path('results/temporal_priority_seed42_rgbfix/plan.json')
    cost = json.loads(priority.read_text())['cost']
    selected = next(v for v in cost['videos'] if v['selection'] == 'median')
    if selected['frames'] != 245 or sha(Path(cost['data_root'])/'extraction.json') != cost['extraction_sha256']:
        raise ValueError('Fixed real-video protocol changed')
    video = None
    for shard, name, example in iter_videos(cost['data_root']):
        if name == selected['id']:
            metadata = video_info(example)
            if shard != selected['shard'] or metadata['video_sha256'] != selected['video_sha256']:
                raise ValueError('Fixed RGB sequence changed')
            video = example['video'].copy()
            break
    if video is None or len(video) != 245:
        raise ValueError('Missing real RGB sequence')
    config = info['config']
    processor = SiglipImageProcessor.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
    sources = ['exploration/inspect_temporal_decay.py', 'temporal_siglip.py', 'video_checkpoint.py',
               'robotap_data.py', 'train_video.py', 'pyproject.toml', 'uv.lock']
    plan = dict(checkpoint=info, checkpoint_path=str(checkpoint), video=selected,
        lags=[1, 8, 32, 128, 244], prior_plan_sha256=sha(prior), priority_plan_sha256=sha(priority),
        source_sha256={p: sha(p) for p in sources}, upstream_module_sha256=sha(mamba3_module.__file__),
        scope='One real RGB sequence, one archived checkpoint, conditional SSM-matrix multiplier only',
        limitations=['Not recognition accuracy or information retention',
                    'Inputs and other state components held fixed in the homogeneous propagation interpretation',
                    'Nonlinear effects via other layers, angles, previous K/V and output gates are not included',
                    'Coordinates, heads and overlapping windows are not independent samples',
                    'GPU shared with training; no speed or memory benchmark'])
    output.mkdir(parents=True)
    write_json(output/'plan.json', plan)
    gates = {key: [] for key in encoder.adapters}
    handles = []
    for key, adapter in encoder.adapters.items():
        mixer = adapter.mixer
        def capture(_module, _inputs, projected, key=key, mixer=mixer):
            start = 2*mixer.d_inner + 2*mixer.d_state*mixer.num_bc_heads
            dt, raw_a = projected[..., start:start+2*mixer.nheads].split(mixer.nheads, -1)
            dt = F.softplus(dt + mixer.dt_bias)
            a = (-mamba3_module.heavy_tail_activation(raw_a.float())).clamp(max=-mixer.A_floor)
            gates[key].append((a*dt)[:, 0].detach().float().cpu().numpy())
        handles.append(mixer.in_proj.register_forward_hook(capture))
    encoder.cuda()
    first_outputs = []
    try:
        state = None
        with torch.inference_mode():
            for frame, rgb in enumerate(video):
                pixels = processor(images=Image.fromarray(rgb), return_tensors='pt').pixel_values.cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    features, state = encoder.step(pixels, state)
                if frame < 3:
                    first_outputs.append(features.detach().cpu())
                if (frame+1) % 25 == 0:
                    print(json.dumps({'frames': frame+1}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    arrays = {key: np.stack(values) for key, values in gates.items()}
    np.savez_compressed(output/'adt.npz', **arrays)
    state = None
    replay_error = 0.
    with torch.inference_mode():
        for frame in range(3):
            pixels = processor(images=Image.fromarray(video[frame]), return_tensors='pt').pixel_values.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features, state = encoder.step(pixels, state)
            replay_error = max(replay_error, float((features.cpu()-first_outputs[frame]).abs().max()))
        checks = [{**row, 'layer': key} for key, adt in arrays.items() for row in native_check(adt)]
    encoder.cpu()
    result = dict(plan_sha256=sha(output/'plan.json'), adt_sha256=sha(output/'adt.npz'),
        weights_unchanged=model_hash(encoder)==before, first_three_frames_replay_max_abs=replay_error,
        native_homogeneous_checks=checks,
        conditional_ssm_carry={key: summarize(adt, plan['lags']) for key, adt in arrays.items()},
        limitations=plan['limitations'])
    write_json(output/'summary.json', result)
    print(json.dumps(verify(output)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify(args.output)))
    else:
        run(args.output)
