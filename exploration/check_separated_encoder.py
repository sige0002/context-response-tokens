"""Native Mamba-3 qualification with rendered RGB and a real point loss."""
import argparse
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import torch
from transformers import SiglipImageProcessor

from video_checkpoint import load_video_checkpoint
from exploration.oracle_transport import digest
from exploration.readout_diagnostic import parameter_digest, select_queries
from exploration.separated_encoder import SeparatedEncoder
from exploration.separated_loss import correspondence_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', choices=['float32', 'bfloat16'], default='bfloat16')
    parser.add_argument('--execution', choices=['framewise', 'batched'], default='framewise')
    args = parser.parse_args(); probe = Path(args.probe_run)
    torch.set_num_threads(2); torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    prior = json.loads((probe/'plan.json').read_text())
    assert digest(probe/'checkpoint.pt') == prior['checkpoint']['checkpoint_sha256']
    torch.empty(1, device='cuda')
    recognizer, provenance = load_video_checkpoint(probe/'checkpoint.pt')
    model = SeparatedEncoder(recognizer.encoder, temporal_precision=args.precision,
                             execution=args.execution).cuda().eval(); del recognizer
    scene = prior['config']['val'][0]; frames = [0, 4, 8, 12]
    path = Path(prior['rgb_root'])/scene/'images.zip'
    assert digest(path) == prior['images'][scene]['images_sha256']
    processor = SiglipImageProcessor.from_pretrained(provenance['config']['model'],
        revision=provenance['config']['revision'], local_files_only=True)
    with zipfile.ZipFile(path) as archive:
        rgb = [Image.open(io.BytesIO(archive.read(f'{f:03d}.png'))).convert('RGB') for f in frames]
    pixels = processor(images=rgb, return_tensors='pt').pixel_values.cuda()[None]
    data = torch.load(Path(prior['cache'])/(scene+'.pt'), map_location='cpu', weights_only=True)
    selected = select_queries(data, scene, 32, 17)
    truth = data['points'][selected, :4].cuda(); visible = data['visible'][selected, :4].cuda()
    backbone_hash = parameter_digest(model.backbone)
    projection_inputs = []
    capture = model.projection.register_forward_pre_hook(
        lambda _module, inputs: projection_inputs.append(tuple(value.detach().float() for value in inputs)))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = model(pixels)
        future = pixels.clone(); future[:, 2:] = pixels[:, 2:].flip(-1)
        changed = model(future)
        torch.testing.assert_close(full[:, :2], changed[:, :2], atol=1e-5, rtol=1e-5)
        state = None; stream = []; sizes = []; reset_outputs = []
        for frame in pixels.unbind(1):
            value, state = model.step(frame, state); stream.append(value); sizes.append(state.nbytes)
            reset, _ = model.step(frame); reset_outputs.append(reset)
            assert all(len(value) == 4 for value in state.layers.values())
        streamed = torch.stack(stream, 1); reset = torch.stack(reset_outputs, 1)
        capture.remove()
        def error(a, b):
            difference = (a.float()-b.float()).abs()
            return dict(max_abs=float(difference.max()), rmse=float(difference.square().mean().sqrt()),
                fraction_outside_tolerance=float((difference > .015+.02*b.float().abs()).float().mean()))
        precision = dict(execution=args.execution, original_output=error(full, streamed))
        if args.execution == 'batched':
            base_stream = tuple(torch.cat([projection_inputs[2+2*i][j] for i in range(len(frames))], 1) for j in (0, 1))
            with torch.autocast('cuda', enabled=False):
                fp_full = model.projection(*projection_inputs[0])
                fp_stream = model.projection(*base_stream)
            precision.update(observation_inputs=error(projection_inputs[0][0], base_stream[0]),
                context_inputs=error(projection_inputs[0][1], base_stream[1]), fp32_projection=error(fp_full, fp_stream))
        precision_path = Path(args.output).with_suffix('.precision.json')
        precision_path.parent.mkdir(parents=True, exist_ok=True)
        precision_path.write_text(json.dumps(precision, indent=2)+'\n')
        torch.testing.assert_close(full, streamed, atol=.015, rtol=.02)
        torch.testing.assert_close(streamed[..., :384], reset[..., :384], atol=0, rtol=0)
        history_effect = float((streamed[..., 384:]-reset[..., 384:]).abs().max())
        assert history_effect > 1e-5 and len(set(sizes)) == 1
        first, state = model.encode_chunk(pixels[:, :2])
        last, state = model.encode_chunk(pixels[:, 2:], state)
        chunked = torch.cat((first, last), 1)
        torch.testing.assert_close(full, chunked, atol=.015, rtol=.02)
    model.train(); before = parameter_digest(model.projection)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    training_pixels = pixels.detach().requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        training_output = model(training_pixels)
        torch.testing.assert_close(training_output.detach(), full, atol=1e-5, rtol=1e-5)
        memory_objective = (training_output[:, -1, :, 384:]*torch.randn_like(training_output[:, -1, :, 384:])).mean()
        input_gradient = torch.autograd.grad(memory_objective, training_pixels, retain_graph=True)[0]
        assert torch.isfinite(input_gradient).all() and input_gradient[:, 0].abs().sum() > 0
        loss = correspondence_loss(training_output, truth, visible)
    loss.backward()
    gradients = {key: float(adapter.mixer.in_proj.weight.grad.float().norm()) for key, adapter in model.encoder.adapters.items()}
    assert all(value > 0 for value in gradients.values())
    assert all(p.grad is None for p in model.backbone.parameters())
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    assert parameter_digest(model.projection) != before
    assert parameter_digest(model.backbone) == backbone_hash
    report = dict(status='passed', source_checkpoint=provenance, scene=scene, frames=frames,
        sources={str(p): digest(p) for p in (Path(__file__), Path('exploration/separated_encoder.py'),
            Path('exploration/separated_loss.py'), Path('temporal_siglip.py'))},
        shape=list(full.shape), output_partition=[384, 384], precision=args.precision, execution=args.execution,
        stream_max_error=float((full-streamed).abs().max()), chunk_max_error=float((full-chunked).abs().max()),
        tolerance=dict(atol=.015, rtol=.02), observation_full_reset_max_error=0.,
        context_history_effect_max=history_effect, state_bytes=sizes,
        loss=float(loss.detach()), mamba_gradient_norms=gradients,
        grad_enabled_forward_max_error=float((training_output.detach()-full).abs().max()),
        first_frame_gradient_from_last_context=float(input_gradient[:, 0].norm()),
        projection_updated=True, frozen_backbone_unchanged=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        projection_parameters=sum(p.numel() for p in model.projection.parameters()),
        scope='Implementation qualification and one gradient update on rendered RGB. No recognition improvement or latency claim.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
