"""Native GPU diagnostics for the exported, trained MIMO visual candidate.

One previously inspected SSv2 video is a numerical fixture, not an accuracy
benchmark. Clip/backward use the locked root environment; stream uses the
documented TVM-FFI overlay. No timing comparison while main training runs.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
from torch.nn import functional as F

from exploration.evaluate_video_history import model_hash
from exploration.mimo_fp64_reference import metric
from exploration.mimo_siglip import load_model
from train_video import configure_reproducibility
from video_checkpoint import file_hash, load_video_checkpoint


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite native visual result')
    return value


@torch.inference_mode()
def clip_values(model, pixels):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        features = model.encoder(pixels)
        logits = model.readout(features[:, -1])
    return dict(features=finite(features).float().cpu(), logits=finite(logits).float().cpu())


@torch.inference_mode()
def stream_values(model, pixels, chunks):
    state = None
    features = []
    offset = 0
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for length in chunks:
            value, state = model.encoder.encode_chunk(pixels[:, offset:offset+length], state)
            features.append(value)
            offset += length
        if offset != pixels.shape[1]:
            raise ValueError('Incomplete stream schedule')
        features = torch.cat(features, dim=1)
        logits = model.readout(features[:, -1])
    for values in state.base_state.layers.values():
        for value in values:
            finite(value)
    return dict(features=finite(features).float().cpu(), logits=finite(logits).float().cpu()), state


def compare(actual, expected):
    return {k:metric(actual[k], expected[k]) for k in ('features', 'logits')}


def run(fixture_root, mode, output):
    if output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(2)
    configure_reproducibility(42)
    prior = json.loads((fixture_root/'summary.json').read_text())
    migration_plan = json.loads((fixture_root/'plan.json').read_text())
    for name, key in [('mimo_model.pt', 'model_sha256'), ('fixture.pt', 'fixture_sha256')]:
        if file_hash(fixture_root/name) != prior[key]:
            raise ValueError('Migration artifact changed')
    if not prior['passed'] or file_hash(fixture_root/'plan.json') != prior['plan_sha256']:
        raise ValueError('Migration qualification changed or failed')
    for path, digest in migration_plan['sources'].items():
        if file_hash(path) != digest:
            raise ValueError(f'Migration source changed: {path}')
    saved = torch.load(fixture_root/'fixture.pt', map_location='cpu', weights_only=True)
    pixels = saved['pixels'].cuda()
    model = load_model(fixture_root/'mimo_model.pt', 'cuda')
    initial_hash = model_hash(model)
    versions = {p:importlib.metadata.version(p) for p in
                ('torch', 'tilelang', 'apache-tvm-ffi', 'nvidia-cutlass-dsl')}
    sources = [__file__, 'exploration/mimo_siglip.py', 'temporal_siglip.py',
               'temporal_readout.py', 'train_video.py', 'uv.lock']
    output.mkdir(parents=True)
    plan = dict(mode=mode, fixture_root=str(fixture_root),
        migration_summary_sha256=file_hash(fixture_root/'summary.json'),
        versions=versions, sources={p:file_hash(p) for p in sources},
        gpu=torch.cuda.get_device_name(), input_shape=list(pixels.shape),
        future_test='Keep first four images, flip the spatial width of last four images',
        causality_tolerance=dict(atol=1e-5, rtol=1e-5),
        stream_chunks=[1]*8, alternate_stream_chunks=[3,1,4],
        numerical_policy='Report native-vs-FP64 and native clip/stream differences; do not turn measured differences into a post-hoc accuracy guarantee.',
        scope=__doc__)
    write(output/'plan.json', plan)
    result = dict(completed=False, plan_sha256=file_hash(output/'plan.json'),
        mode=mode, recognition_accuracy_tested=False, latency_tested=False,
        full_visual_model_qualified=False, stages={})
    outputs = {}
    stage = 'native_forward'
    try:
        if mode == 'clip':
            siso, info = load_video_checkpoint(migration_plan['checkpoint'], 'cuda')
            if info['checkpoint_sha256'] != migration_plan['checkpoint_sha256']:
                raise ValueError('SISO checkpoint changed')
            outputs['siso'] = clip_values(siso, pixels)
            del siso
            outputs['mimo'] = clip_values(model, pixels)
            repeated = clip_values(model, pixels)
            if not all(torch.equal(repeated[k], outputs['mimo'][k]) for k in repeated):
                raise ValueError('Repeated native clip output changed')
            result['stages'][stage] = dict(completed=True, replay_exact=True,
                siso_vs_fp64=compare(outputs['siso'], dict(features=saved['siso_features'], logits=saved['siso_logits'])),
                mimo_vs_fp64=compare(outputs['mimo'], dict(features=saved['mimo_features'], logits=saved['mimo_logits'])),
                mimo_vs_siso=compare(outputs['mimo'], outputs['siso']))
            print(json.dumps(dict(stage=stage, completed=True)), flush=True)
            stage = 'causality'
            changed = pixels.clone()
            changed[:, 4:] = changed[:, 4:].flip(-1)
            altered = clip_values(model, changed)
            torch.testing.assert_close(altered['features'][:, :4], outputs['mimo']['features'][:, :4],
                                       **plan['causality_tolerance'])
            result['stages'][stage] = dict(completed=True,
                prefix=metric(altered['features'][:, :4], outputs['mimo']['features'][:, :4]))
            print(json.dumps(dict(stage=stage, completed=True)), flush=True)
            stage = 'backward'
            model.train()
            if model.encoder.backbone.training or any(p.requires_grad for p in model.encoder.backbone.parameters()):
                raise ValueError('Backbone is not frozen')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(pixels)
                # The fixture's true class; this is one backward pass, no update.
                loss = F.cross_entropy(logits.float(), torch.tensor([154], device='cuda'))
            loss.backward()
            invalid = [n for n,p in model.named_parameters() if p.requires_grad and
                       (p.grad is None or not torch.isfinite(p.grad).all())]
            if invalid:
                raise ValueError(f'Missing/nonfinite gradients: {invalid}')
            extra_gains = {}
            for key, adapter in model.encoder.encoder.adapters.items():
                core = adapter.mixer
                extra_gains[key] = {name:float(getattr(core, name).grad[:, 1:].abs().sum())
                                   for name in ('mimo_x', 'mimo_o')}
                if min(extra_gains[key].values()) <= 0:
                    raise ValueError('Extra MIMO gains have no learning signal')
            result['stages'][stage] = dict(completed=True, loss=float(loss.detach()),
                parameters_with_grad=sum(p.grad is not None for p in model.parameters()),
                extra_gain_gradient_l1=extra_gains, optimizer_updates=0)
            model.zero_grad(set_to_none=True)
            model.eval()
        else:
            outputs['mimo'], state = stream_values(model, pixels, plan['stream_chunks'])
            snapshot = {k:tuple(t.cpu().clone() for t in values)
                        for k,values in state.base_state.layers.items()}
            repeated, repeated_state = stream_values(model, pixels, plan['stream_chunks'])
            if not all(torch.equal(repeated[k], outputs['mimo'][k]) for k in repeated):
                raise ValueError('Repeated fresh-episode stream output changed')
            if not all(torch.equal(t.cpu(), ref) for key,values in repeated_state.base_state.layers.items()
                       for t,ref in zip(values, snapshot[key])):
                raise ValueError('Repeated fresh-episode stream states changed')
            outputs['chunks'], chunks_state = stream_values(model, pixels, plan['alternate_stream_chunks'])
            result['stages'][stage] = dict(completed=True, replay_exact=True, state_replay_exact=True,
                mimo_vs_fp64=compare(outputs['mimo'], dict(features=saved['mimo_features'], logits=saved['mimo_logits'])),
                alternate_chunks_vs_single_steps=compare(outputs['chunks'], outputs['mimo']),
                state_bytes=state.nbytes, alternate_state_bytes=chunks_state.nbytes,
                final_offset=state.base_state.offset)
        if model_hash(model) != initial_hash:
            raise ValueError('Candidate weights changed')
        if any(file_hash(p) != h for p,h in plan['sources'].items()):
            raise ValueError('Source changed during the diagnostic')
        torch.save(outputs, output/'outputs.pt')
        result.update(completed=True, weights_unchanged=True,
            outputs_sha256=file_hash(output/'outputs.pt'),
            predictions={k:int(v['logits'].argmax(-1)) for k,v in outputs.items()})
    except Exception as exc:
        result['stages'][stage] = dict(completed=False, error_type=type(exc).__name__, error=str(exc))
        (output/'error.txt').write_text(traceback.format_exc())
        write(output/'summary.json', result)
        raise
    write(output/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('clip', 'stream'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.fixture_root, args.mode, args.output)
