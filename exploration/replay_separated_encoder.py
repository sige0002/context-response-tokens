"""Reload portable pilot encoders and reproduce their final RGB features.

All validation scenes and full/reset conditions are replayed independently of
the training process. This verifies serialization and RGB preprocessing, not
generalization, classification preservation, or equality of batched kernels.
"""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

from PIL import Image
import torch
from transformers import SiglipImageProcessor

from exploration.oracle_transport import digest
from exploration.readout_diagnostic import parameter_digest
from exploration.separated_encoder import SeparatedEncoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--arm', choices=['final_memory', 'separated_memory', 'separated_image'])
    args = parser.parse_args(); root = Path(args.run)
    plan = json.loads((root/'plan.json').read_text()); config = plan['config']
    arms = [args.arm] if args.arm else config['arms']
    assert all((root/(arm+'.json')).exists() for arm in arms), 'Requested arms must have completed'
    for path, expected in plan['sources'].items():
        assert digest(path) == expected, path
    probe = Path(plan['probe_run'])
    assert digest(probe/'plan.json') == plan['probe_plan_sha256']
    prior = json.loads((probe/'plan.json').read_text())
    cache = Path(plan['original_cache'])
    assert digest(cache/'summary.json') == plan['original_cache_summary_sha256']
    receipt = json.loads((cache/'summary.json').read_text())
    pixel_hashes = json.loads((root/'input_pixels.json').read_text())
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    # Obtain the CUDA context before staging weights in unified CPU/GPU RAM.
    torch.empty(1, device='cuda')
    processor = SiglipImageProcessor.from_pretrained(prior['config']['model'],
        revision=prior['config']['revision'], local_files_only=True)
    pixels = {}; inputs = {}
    for scene in config['val']:
        path = cache/(scene+'.pt')
        assert digest(path) == receipt['scenes'][scene]['sha256']
        data = torch.load(path, map_location='cpu', weights_only=True)
        frames = data['frames']; del data
        folder = Path(prior['rgb_root'])/scene
        assert digest(folder/'metadata.json') == receipt['scenes'][scene]['metadata_sha256']
        metadata = json.loads((folder/'metadata.json').read_text())
        archive_hash = digest(folder/'images.zip')
        assert archive_hash == metadata['files']['images.zip']['sha256']
        with zipfile.ZipFile(folder/'images.zip') as archive:
            rgb = [Image.open(io.BytesIO(archive.read(f'{frame:03d}.png'))).convert('RGB') for frame in frames]
        values = processor(images=rgb, return_tensors='pt').pixel_values
        actual_hash = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        assert actual_hash == pixel_hashes[scene], scene
        pixels[scene] = values
        inputs[scene] = dict(archive_sha256=archive_hash, pixels_sha256=actual_hash, frames=list(frames))
    for arm in arms:
        started = time.monotonic()
        record = json.loads((root/(arm+'.json')).read_text())
        checkpoint = root/(arm+'_encoder.pt')
        assert digest(checkpoint) == record['exported_encoder_sha256']
        model = SeparatedEncoder.load_checkpoint(checkpoint, device='cuda')
        assert model.encoder.checkpoint_revision == prior['config']['revision']
        before = parameter_digest(model); rows = []
        final = record['scores'][str(config['steps'])]
        for scene in config['val']:
            path = root/arm/str(config['steps'])/(scene+'_features.pt')
            assert digest(path) == final['features_sha256'][scene]
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['pixels_sha256'] == inputs[scene]['pixels_sha256']
            for condition, expected in saved['features'].items():
                state = None; output = []; sizes = []
                for frame, value in enumerate(pixels[scene]):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        current, state = model.step(value[None].cuda(), None if condition == 'reset1' else state)
                    assert state.offset == (1 if condition == 'reset1' else frame+1)
                    output.append(current[0].float().cpu()); sizes.append(state.nbytes)
                actual = torch.stack(output)
                assert len(set(sizes)) == 1 and sizes[-1] == final['state_bytes'][scene][condition]
                rows.append(dict(scene=scene, condition=condition, frames=len(output),
                    exactly_equal=torch.equal(actual, expected),
                    max_abs=float((actual-expected).abs().max()),
                    rmse=float((actual-expected).square().mean().sqrt()), state_bytes=sizes[-1]))
        assert parameter_digest(model) == before
        passed = all(row['exactly_equal'] for row in rows)
        report = dict(status='passed' if passed else 'failed', arm=arm, source_sha256=digest(__file__),
            plan_sha256=digest(root/'plan.json'), record_sha256=digest(root/(arm+'.json')),
            checkpoint_sha256=record['exported_encoder_sha256'], inputs=inputs, rows=rows,
            encoder_parameters_unchanged=True, elapsed_seconds=time.monotonic()-started,
            scope='Final portable checkpoint loaded in a separate process; all eight RGB validation scenes and every full/reset streaming feature replayed. No unseen-data, speed, or classification claim.')
        (root/('replay_'+arm+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(arm=arm, status=report['status'], feature_sequences=len(rows),
            frames=sum(row['frames'] for row in rows), max_abs=max(row['max_abs'] for row in rows))), flush=True)
        if not passed: raise SystemExit('Portable export replay differs; inspect the saved per-scene report')
        del model, output, actual, expected, saved, state
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
