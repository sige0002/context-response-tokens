"""Paired history intervention on a pinned trained video recognizer.

No training or model selection. The current RGB image, weights and classifier
are fixed. Altered histories are distribution interventions, not new labelled
benchmarks or independently trained image-only baselines.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from transformers import SiglipImageProcessor

from evaluate_video import streaming_logits
from train_video import configure_reproducibility
from video_checkpoint import file_hash, load_video_checkpoint
from video_data import RawVideoClips


CONDITIONS = ('full_stream', 'current_only', 'repeat_current', 'reverse_history')


def intervene(video, condition):
    if video.ndim != 5 or video.shape[1] < 1:
        raise ValueError('Expected nonempty [B,T,C,H,W]')
    if condition == 'full_stream':
        result = video
    elif condition == 'current_only':
        result = video[:, -1:]
    elif condition == 'repeat_current':
        result = video[:, -1:].expand_as(video).contiguous()
    elif condition == 'reverse_history':
        result = torch.cat((video[:, :-1].flip(1), video[:, -1:]), dim=1)
    else:
        raise ValueError(condition)
    if not torch.equal(result[:, -1], video[:, -1]):
        raise AssertionError('Current observation changed')
    return result


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def summarize(values, targets, ids):
    reference = values['full_stream']
    ref_pred = reference.argmax(-1)
    ref_correct = ref_pred == targets
    result = {}
    for name, logits in values.items():
        pred = logits.argmax(-1)
        correct = pred == targets
        # Stratified paired bootstrap: preserve the fixed two clips per class.
        rng = np.random.default_rng(17)
        strata = [np.flatnonzero(targets == c) for c in np.unique(targets)]
        delta = (correct.astype(float) - ref_correct.astype(float)) * 100
        draws = np.concatenate([rng.choice(s, (2000, len(s))) for s in strata], axis=1)
        interval = np.quantile(delta[draws].mean(1), [.025, .975]).tolist()
        shifted = logits.astype(np.float64) - logits.max(-1, keepdims=True)
        loss = np.log(np.exp(shifted).sum(-1)) - shifted[np.arange(len(targets)), targets]
        difference = logits.astype(np.float64) - reference
        result[name] = dict(correct=int(correct.sum()), total=len(targets),
            original_label_top1=float(correct.mean()), cross_entropy=float(loss.mean()),
            gained_vs_stream=int((correct & ~ref_correct).sum()),
            lost_vs_stream=int((~correct & ref_correct).sum()),
            changed_predictions=int((pred != ref_pred).sum()),
            delta_vs_stream_pp=float(delta.mean()), bootstrap_95_pp=interval,
            logit_rmse=float(np.sqrt(np.mean(difference**2))),
            logit_max_abs=float(np.abs(difference).max()),
            predictions=[dict(id=str(i), target=int(y), prediction=int(p))
                         for i, y, p in zip(ids, targets, pred)])
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, help='Fixed temporal layer diagnostic directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prior, output = Path(args.prior).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output; no partial-result reuse')
    torch.set_num_threads(2)
    configure_reproducibility(42)
    old_plan = json.loads((prior/'plan.json').read_text())
    old_summary = json.loads((prior/'summary.json').read_text())
    for filename, key in [('video_inputs.json', 'inputs_sha256'), ('video_logits.npz', 'logits_sha256')]:
        if file_hash(prior/filename) != old_summary['videos'][key]:
            raise ValueError('Prior diagnostic artifact changed: '+filename)
    old_inputs = json.loads((prior/'video_inputs.json').read_text())
    with np.load(prior/'video_logits.npz') as data:
        old_logits, old_targets = data['111'].copy(), data['targets'].copy()
    checkpoint = Path(old_plan['probe_run'])/'checkpoint.pt'
    model, info = load_video_checkpoint(checkpoint, 'cpu')
    if info != old_plan['checkpoint'] or info['step'] != 4000:
        raise ValueError('Pinned model or training metadata differs')
    if model.encoder.kind != 'mamba' or tuple(model.encoder.adapters) != ('3', '7', '11'):
        raise ValueError('This diagnostic requires the original three-layer Mamba model')
    model.requires_grad_(False).eval()
    processor = SiglipImageProcessor.from_pretrained(info['config']['model'],
        revision=info['config']['revision'], local_files_only=True)
    dataset = RawVideoClips(info['config']['val'], processor, frames=8, classes=174)
    if file_hash(dataset.path) != info['config']['val_sha256']:
        raise ValueError('Validation manifest changed')
    chosen = old_plan['video_subset']
    rng = random.Random(17)
    regenerated = []
    for label in range(174):
        candidates = sorted([i for i, row in enumerate(dataset.rows) if row['label'] == label],
                            key=lambda i: dataset.rows[i]['id'])
        regenerated.extend(rng.sample(candidates, 2))
    if regenerated != [r['index'] for r in chosen] or len(chosen) != len(old_inputs):
        raise ValueError('Subset differs from the fixed sampling protocol')
    sources = [Path(__file__), Path('evaluate_video.py'), Path('train_video.py'),
        Path('temporal_siglip.py'), Path('temporal_readout.py'), Path('video_data.py'),
        Path('video_checkpoint.py'), Path('uv.lock'), Path('pyproject.toml')]
    plan = dict(checkpoint=str(checkpoint), checkpoint_info=info,
        prior=str(prior), prior_plan_sha256=file_hash(prior/'plan.json'),
        prior_inputs_sha256=file_hash(prior/'video_inputs.json'),
        prior_logits_sha256=file_hash(prior/'video_logits.npz'), subset=chosen,
        conditions=['full_clip', *CONDITIONS], batch_size=1, frames=8,
        sources={str(p): file_hash(p) for p in sources},
        scope='Inference intervention; same current image, model and classifier; no training',
        limitations=['Previously inspected balanced 348-clip validation subset, one 4000-update model',
            'Original labels are retained under history changes; reverse may change action semantics',
            'Current-only is an intervention on a video-trained model, not a trained image-only baseline',
            'Eight sparse observations do not demonstrate long-term memory',
            'Bootstrap is exploratory, stratified by class, conditional on fixed weights, unadjusted',
            'Concurrent training is allowed for this accuracy diagnostic; no runtime claim'])
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    before = model_hash(model)
    model.cuda()
    values = {name: [] for name in plan['conditions']}
    records, states = [], {}
    for number, item in enumerate(chosen):
        pixels, target, metadata = dataset.read(item['index'])
        metadata.update(label=target, pixels_sha256=tensor_hash(pixels))
        if (metadata != old_inputs[number] or target != old_targets[number]
                or metadata['id'] != item['id'] or target != item['label']
                or file_hash(dataset.rows[item['index']]['video']) != item['video_sha256']):
            raise ValueError(f'Input drift: {item["id"]}')
        video = pixels[None].cuda()
        row_states = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(video)
            values['full_clip'].append(logits[0].float().cpu().numpy())
            for condition in CONDITIONS:
                changed = intervene(video, condition)
                logits, sizes = streaming_logits(model, changed)
                if len(set(sizes)) != 1 or sizes[0] <= 0:
                    raise ValueError('Mamba state size is not fixed and positive')
                values[condition].append(logits[0].float().cpu().numpy())
                row_states[condition] = sizes
        if number == 0:
            states = row_states
        elif states != row_states:
            raise ValueError('State capacity changed between clips')
        if any(not np.isfinite(v[-1]).all() for v in values.values()):
            raise FloatingPointError('Nonfinite recognition logits')
        records.append(dict(**metadata, current_pixels_sha256=tensor_hash(pixels[-1])))
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(completed=number+1, id=item['id'], target=target,
                predictions={k: int(v[-1].argmax()) for k, v in values.items()}))+'\n')
        if (number+1) % 24 == 0 or number+1 == len(chosen):
            print(json.dumps(dict(completed=number+1, total=len(chosen))), flush=True)
    values = {name: np.stack(rows) for name, rows in values.items()}
    targets = np.array([r['label'] for r in records], dtype=np.int64)
    ids = np.array([r['id'] for r in records])
    np.savez_compressed(output/'logits.npz', targets=targets, ids=ids, **values)
    (output/'inputs.json').write_text(json.dumps(records, indent=2)+'\n')
    if model_hash(model) != before:
        raise ValueError('Weights changed during inference')
    for path, expected in plan['sources'].items():
        if file_hash(path) != expected:
            raise ValueError('Source changed during inference: '+path)
    difference = values['full_clip'].astype(np.float64) - old_logits
    summary = dict(plan_sha256=file_hash(output/'plan.json'),
        logits_sha256=file_hash(output/'logits.npz'), inputs_sha256=file_hash(output/'inputs.json'),
        model_state_sha256=before, weights_unchanged=True, same_original_pixels=True,
        state_bytes_by_frame=states, scores=summarize(values, targets, ids),
        prior_full_clip_replay=dict(exact=bool(np.array_equal(values['full_clip'], old_logits)),
            logit_max_abs=float(np.abs(difference).max()),
            changed_predictions=int((values['full_clip'].argmax(-1) != old_logits.argmax(-1)).sum())),
        limitations=plan['limitations'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v['original_label_top1'] for k, v in summary['scores'].items()}), flush=True)


if __name__ == '__main__':
    main()
