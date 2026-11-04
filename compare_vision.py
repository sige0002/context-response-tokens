"""Small, from-scratch Oxford-IIIT Pet trimap comparison; no teacher or pretraining."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from vision_mamba3 import VisionMamba3


class AttentionBlock(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(y)
        return x + self.ff(self.norm2(x))


class SmallViT(nn.Module):
    """Pre-LN ViT, patch4, width128, depth2, heads4, MLP ratio4, fixed2D positions."""
    def __init__(self, size=128, conv_stem=False):
        super().__init__()
        self.stem = (nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
                                  nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU())
                     if conv_stem else nn.Conv2d(3, 128, 4, 4))
        coords = torch.arange(size // 4, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        freq = 1 / (10000 ** (torch.arange(32, dtype=torch.float32) / 32))
        pos = torch.cat([f(t.flatten()[:, None] * freq) for t in (xx, yy) for f in (torch.sin, torch.cos)], -1)
        self.register_buffer('position', pos[None])
        self.blocks = nn.Sequential(AttentionBlock(), AttentionBlock())
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.stem(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) + self.position
        return self.norm(self.blocks(x)).transpose(1, 2).reshape(b, c, h, w)


class Segmenter(nn.Module):
    def __init__(self, name, size):
        super().__init__()
        if name == 'dinov3_vitb16':
            from dinov3_baseline import DINOv3Base
            self.encoder = DINOv3Base()
        else:
            self.encoder = VisionMamba3() if name == 'mamba3' else SmallViT(size, name == 'vit_conv')
        # Identical lightweight dense head for all encoders.
        self.head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 2, 1))
        if name == 'dinov3_vitb16':
            # External task adapter; official encoder width/depth/patch size are unchanged.
            self.head = nn.Sequential(nn.Conv2d(768, 128, 1), self.head)

    def forward(self, x):
        return F.interpolate(self.head(self.encoder(x)), size=x.shape[-2:], mode='bilinear', align_corners=False)


def metric(confusion):
    c = confusion.double()
    iou = c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp_min(1)
    return {'miou': iou.mean().item(), 'pet_iou': iou[0].item(), 'background_iou': iou[1].item(),
            'pixel_accuracy': (c.diag().sum()/c.sum()).item(),
            'confusion': confusion.cpu().tolist()}


@torch.inference_mode()
def evaluate(model, x, y, batch):
    model.eval()
    confusion = torch.zeros(4, dtype=torch.int64, device='cuda')
    per_image = []
    for start in range(0, len(x), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(x[start:start+batch] * 2 - 1).argmax(1)
        truth = y[start:start+batch]
        for p, t in zip(pred, truth):
            valid = t != 255
            cm = torch.bincount(t[valid] * 2 + p[valid], minlength=4)
            confusion += cm
            per_image.append(cm.cpu().tolist())
    return metric(confusion.reshape(2, 2)), per_image


@torch.inference_mode()
def timing(module, image):
    module.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(20): module(image)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(60):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); out = module(image); end.record(); end.synchronize()
            times.append(start.elapsed_time(end))
            del out
        peak = torch.cuda.max_memory_allocated() - baseline
    return {'median_ms': statistics.median(times), 'p90_ms': float(np.percentile(times, 90)),
            'incremental_peak_bytes': peak, 'batch': len(image), 'samples': 60}


def load_split(root, names, size):
    xs, ys = [], []
    for name in names:
        with Image.open(root / 'images' / f'{name}.jpg') as im:
            xs.append(np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)).copy())
        with Image.open(root / 'annotations/trimaps' / f'{name}.png') as im:
            ys.append(np.asarray(im.resize((size, size), Image.Resampling.NEAREST)).copy().astype(np.int64) - 1)
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to('cuda', torch.float32) / 255
    y = torch.from_numpy(np.stack(ys)).to('cuda')
    assert y.min() == 0 and y.max() == 2
    y[y == 2] = 255  # Official label3 is unclassified, not a semantic boundary class.
    return x, y


def run(args):
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    root, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'annotations'):
        if not (root / name).exists():
            with tarfile.open(root / f'{name}.tar.gz') as tar: tar.extractall(root, filter='data')
    rng = np.random.default_rng(20261104)
    train_names = [line.split()[0] for line in (root / 'annotations/trainval.txt').read_text().splitlines() if line and not line.startswith('#')]
    test_names = [line.split()[0] for line in (root / 'annotations/test.txt').read_text().splitlines() if line and not line.startswith('#')]
    rng.shuffle(train_names); rng.shuffle(test_names)
    splits = {'train': train_names[:args.train], 'validation': train_names[args.train:args.train+args.val], 'test': test_names[:args.test]}
    assert not (set(splits['train']) & set(splits['validation']) or set(train_names) & set(test_names))
    split_path = out / 'splits.json'
    if split_path.exists(): assert json.loads(split_path.read_text()) == splits
    split_path.write_text(json.dumps(splits, indent=2))
    data = {key: load_split(root, names, args.size) for key, names in splits.items()}
    print('loaded', {k: len(v[0]) for k, v in data.items()}, flush=True)
    for seed in args.seeds:
        for name in args.models:
            result_path = out / f'{name}_seed{seed}.json'
            if result_path.exists():
                print('already complete', result_path, flush=True); continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = Segmenter(name, args.size).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
            # Dedicated generator ensures identical batch order and flips despite architecture RNG usage.
            generator = torch.Generator(device='cuda').manual_seed(seed + 1000)
            history, best_score, best_state, best_epoch = [], -1., None, None
            torch.cuda.synchronize(); begin = time.perf_counter()
            x, y = data['train']
            for epoch in range(args.epochs):
                model.train()
                lr = args.lr * .5 * (1 + math.cos(math.pi * epoch / args.epochs))
                for group in optimizer.param_groups: group['lr'] = lr
                order = torch.randperm(len(x), device='cuda', generator=generator)
                total_loss = 0.
                for ids in order.split(args.batch):
                    a, b = x[ids], y[ids]
                    flips = torch.rand(len(ids), device='cuda', generator=generator) < .5
                    a = torch.where(flips[:, None, None, None], a.flip(-1), a)
                    b = torch.where(flips[:, None, None], b.flip(-1), b)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(model(a * 2 - 1), b, ignore_index=255)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradients')
                    optimizer.step()
                    total_loss += loss.item() * len(ids)
                val, _ = evaluate(model, *data['validation'], args.batch)
                record = {'epoch': epoch + 1, 'train_loss': total_loss / len(x), 'val': val, 'lr': lr}
                history.append(record)
                if val['miou'] > best_score:
                    best_score, best_epoch = val['miou'], epoch + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f'{name} seed={seed} epoch={epoch+1}/{args.epochs} loss={total_loss/len(x):.4f} val_mIoU={val["miou"]:.4f} elapsed={time.perf_counter()-begin:.1f}s', flush=True)
            torch.cuda.synchronize(); train_seconds = time.perf_counter() - begin
            model.load_state_dict(best_state)
            test, per_image = evaluate(model, *data['test'], args.batch)
            sample = data['test'][0][:1] * 2 - 1
            speed = timing(model.encoder, sample)
            full_speed = timing(model, sample)
            result = {'model': name, 'seed': seed, 'config': vars(args), 'encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
                      'total_parameters': sum(p.numel() for p in model.parameters()), 'selected_epoch': best_epoch,
                      'test': test, 'test_per_image_confusions': per_image, 'history': history,
                      'train_seconds_including_validation_and_initial_compile': train_seconds,
                      'encoder_timing': speed, 'segmenter_timing': full_speed,
                      'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                      'precision': 'FP32 weights, BF16 autocast; TF32 allowed',
                      'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                      'source_sha256': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['compare_vision.py', 'vision_mamba3.py']}}
            if name == 'dinov3_vitb16':
                from dinov3_baseline import COMMIT
                result['official_dinov3_commit'] = COMMIT
                result['pretrained'] = False
                result['feature_shape'] = [1, 768, args.size // 16, args.size // 16]
                result['source_sha256']['dinov3_baseline.py'] = hashlib.sha256(Path('dinov3_baseline.py').read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result, indent=2))
            torch.save(best_state, out / f'{name}_seed{seed}.pt')
            # Same first eight held-out images for every model; never select by visual quality.
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(data['test'][0][:8] * 2 - 1).argmax(1).cpu().numpy()
            palette = np.zeros((256, 3), dtype=np.uint8)
            palette[0], palette[1], palette[255] = [220, 60, 60], [40, 50, 80], [255, 220, 30]
            canvas = Image.new('RGB', (args.size * 3, args.size * 8 + 24), 'white')
            ImageDraw.Draw(canvas).text((3, 5), 'Image | Ground truth | Prediction', fill='black')
            for i in range(8):
                rgb = (data['test'][0][i].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                truth = data['test'][1][i].cpu().numpy()
                for j, arr in enumerate([rgb, palette[truth], palette[pred[i]]]): canvas.paste(Image.fromarray(arr), (j * args.size, i * args.size + 24))
            canvas.save(out / f'{name}_seed{seed}.png')
            print('RESULT', name, seed, test, speed, flush=True)
            del model, optimizer, best_state
            torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/pets')
    p.add_argument('--output', default='results/pets_compare')
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--train', type=int, default=512)
    p.add_argument('--val', type=int, default=128)
    p.add_argument('--test', type=int, default=512)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--models', nargs='+', choices=['mamba3', 'vit', 'vit_conv', 'dinov3_vitb16'], default=['mamba3', 'vit', 'vit_conv'])
    args = p.parse_args()
    if args.size % 4: p.error('size must be divisible by4')
    run(args)
