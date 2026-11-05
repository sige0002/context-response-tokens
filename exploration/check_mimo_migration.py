"""CPU qualification of a trained SISO-to-MIMO migration.

The image backbone/readout run in FP32 and both temporal cores are replaced
with independent FP64 reference recurrences. This tests mathematical migration
and serialization, NOT the native GPU visual model, its accuracy, or its speed.
The previously inspected SSv2 clip 3988 is a numerical fixture only.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType

import av
import torch
from transformers import SiglipImageProcessor

from exploration.evaluate_video_history import model_hash, tensor_hash
from exploration.mimo_fp64_reference import metric, reference as mimo_reference
from exploration.mimo_siglip import attach, load_model, promote_core, save_model
from exploration.test_mimo_siglip import siso_reference
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import frame_to_rgb, temporal_indices


@contextmanager
def reference_cores(encoder):
    """Temporarily replace the complete temporal core, including projections."""
    base = getattr(encoder, 'encoder', encoder)
    changed = []
    def forward(core, value):
        weights = core.state_dict()
        result = (mimo_reference(value, weights)[0] if core.is_mimo
                  else siso_reference(value, weights))
        return result.to(value.dtype)
    try:
        for adapter in base.adapters.values():
            core = adapter.mixer
            if 'forward' in core.__dict__:
                raise ValueError('An existing instance forward override would be hidden')
            core.forward = MethodType(forward, core)
            changed.append(core)
        yield
    finally:
        for core in changed:
            del core.forward


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.inference_mode()
def run(output):
    if output.exists():
        raise FileExistsError('Use a new output directory')
    torch.set_num_threads(2)
    torch.manual_seed(701)
    checkpoint = Path('results/temporal_density_step11000_v1/checkpoint.pt')
    model, info = load_video_checkpoint(checkpoint, 'cpu')
    if (info['step'] != 11000 or info['checkpoint_sha256'] !=
            '89e594f8285489f01e71fc553a685ca36bbef77cd87f98598612e788eb43a9ee'):
        raise ValueError('Expected the archived 11000-update checkpoint')
    model.eval()
    manifest = Path(info['config']['val'])
    if file_hash(manifest) != info['config']['val_sha256']:
        raise ValueError('Validation manifest changed')
    row = next(json.loads(s) for s in manifest.read_text().splitlines()
               if json.loads(s)['id'] == '3988')
    video = (manifest.parent / row['video']).resolve()
    if video.stat().st_size != row['bytes'] or file_hash(video) != row['sha256']:
        raise ValueError('Video changed')
    processor = SiglipImageProcessor.from_pretrained(info['config']['model'],
        revision=info['config']['revision'], local_files_only=True)
    with av.open(str(video)) as container:
        container.streams.video[0].thread_count = 1
        frames = list(container.decode(video=0))
    indices = temporal_indices(len(frames), 8)
    pixels = processor(images=[frame_to_rgb(frames[i]) for i in indices],
                       return_tensors='pt').pixel_values[None]
    sources = [__file__, 'exploration/mimo_siglip.py',
        'exploration/test_mimo_siglip.py', 'exploration/mimo_fp64_reference.py',
        'exploration/evaluate_video_history.py', 'temporal_siglip.py',
        'temporal_readout.py', 'video_checkpoint.py', 'video_data.py',
        'train_temporal.py', 'pyproject.toml', 'uv.lock',
        'vendor/mamba/mamba_ssm/modules/mamba3.py']
    output.mkdir(parents=True)
    plan = dict(checkpoint=str(checkpoint), checkpoint_sha256=info['checkpoint_sha256'],
        checkpoint_step=info['step'], video_id=row['id'], video_sha256=row['sha256'],
        input_sha256=tensor_hash(pixels), input_shape=list(pixels.shape), indices=indices,
        timestamps=[frames[i].time for i in indices], random_seed=701,
        core_lengths=[1, 8, 17, 65], core_batch=2,
        core_tolerances=dict(atol=1e-10, rtol=1e-10),
        image_tolerances=dict(atol=1e-5, rtol=1e-5),
        sources={str(Path(p)):file_hash(p) for p in sources},
        scope='CPU mathematical migration and weight export: FP32 SigLIP/readout plus substituted FP64 temporal recurrences. No native visual inference, accuracy benchmark, or latency claim.')
    write(output / 'plan.json', plan)
    original_hash = model_hash(model)
    core_results = []
    for key, adapter in model.encoder.adapters.items():
        core = adapter.mixer
        promoted = promote_core(core)
        for length in plan['core_lengths']:
            sample = torch.randn(2, length, 256)
            expected = siso_reference(sample, core.state_dict())
            actual, _ = mimo_reference(sample, promoted.state_dict())
            torch.testing.assert_close(actual, expected, **plan['core_tolerances'])
            core_results.append(dict(layer=int(key)+1, length=length,
                                     **metric(actual, expected)))
        print(json.dumps(dict(event='trained_core_checked', layer=int(key)+1)), flush=True)
    with reference_cores(model.encoder):
        expected_features = model.encoder(pixels)
        expected_logits = model.readout(expected_features[:, -1])
    if model_hash(model) != original_hash:
        raise ValueError('Baseline weights changed')
    print(json.dumps(dict(event='siso_reference_image_complete')), flush=True)
    attach(model)
    with reference_cores(model.encoder):
        actual_features = model.encoder(pixels)
        actual_logits = model.readout(actual_features[:, -1])
    torch.testing.assert_close(actual_features, expected_features, **plan['image_tolerances'])
    torch.testing.assert_close(actual_logits, expected_logits, **plan['image_tolerances'])
    print(json.dumps(dict(event='mimo_reference_image_complete')), flush=True)
    export = output / 'mimo_model.pt'
    save_model(model, export, dict(classes=174, kind='attention', seed=43))
    reloaded = load_model(export)
    if model_hash(reloaded) != model_hash(model):
        raise ValueError('Exported weights differ')
    with reference_cores(reloaded.encoder):
        restored_features = reloaded.encoder(pixels)
        restored_logits = reloaded.readout(restored_features[:, -1])
    if not torch.equal(restored_features, actual_features) or not torch.equal(restored_logits, actual_logits):
        raise ValueError('Restored CPU reference outputs differ')
    # Retain inputs/outputs locally for a later native-kernel qualification.
    torch.save(dict(pixels=pixels, siso_features=expected_features,
        mimo_features=actual_features, siso_logits=expected_logits,
        mimo_logits=actual_logits), output / 'fixture.pt')
    if any(file_hash(p) != digest for p,digest in plan['sources'].items()):
        raise ValueError('A qualification source changed during execution')
    result = dict(passed=True, plan_sha256=file_hash(output/'plan.json'),
        core_checks=core_results, features=metric(actual_features, expected_features),
        logits=metric(actual_logits, expected_logits), export_weights_exact=True,
        export_reference_features_exact=True, export_reference_logits_exact=True,
        model_sha256=file_hash(export), fixture_sha256=file_hash(output/'fixture.pt'),
        prediction_siso=int(expected_logits.argmax(-1)),
        prediction_mimo=int(actual_logits.argmax(-1)),
        native_visual_forward_tested=False, native_streaming_tested=False,
        recognition_accuracy_tested=False, latency_tested=False,
        full_visual_model_qualified=False, scope=plan['scope'])
    write(output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='core_checks'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
