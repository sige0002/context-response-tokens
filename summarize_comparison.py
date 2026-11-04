"""Summarize completed runs; no selection using test scores."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('results/pets_compare'))
root = parser.parse_args().root
names = ['mamba3', 'vit', 'vit_conv']
runs = {name: [json.loads(p.read_text()) for p in sorted(root.glob(f'{name}_seed*.json'))] for name in names}
assert all(len(items) == 3 for items in runs.values()), 'Need all three seeds per architecture'
assert len({r['split_sha256'] for items in runs.values() for r in items}) == 1
summary = {}
for name, items in runs.items():
    scores = [r['test']['miou'] * 100 for r in items]
    summary[name] = {'seeds': [r['seed'] for r in items], 'test_miou_percent': scores,
                     'mean_miou_percent': statistics.mean(scores), 'sd_miou_percentage_points': statistics.stdev(scores),
                     'mean_pet_iou_percent': statistics.mean(r['test']['pet_iou'] * 100 for r in items),
                     'encoder_parameters': items[0]['encoder_parameters'], 'total_parameters': items[0]['total_parameters'],
                     'median_encoder_ms': statistics.median(r['encoder_timing']['median_ms'] for r in items),
                     'median_segmenter_ms': statistics.median(r['segmenter_timing']['median_ms'] for r in items),
                     'median_incremental_encoder_mib': statistics.median(r['encoder_timing']['incremental_peak_bytes'] for r in items) / 2**20,
                     'selected_epochs': [r['selected_epoch'] for r in items]}
# Paired image bootstrap, averaged over these three fixed training seeds.
# This interval describes test-image resampling, not unseen training seeds/datasets.
rng = np.random.default_rng(20261104)
differences = {}
def miou(cm):
    return np.mean(np.diagonal(cm, axis1=-2, axis2=-1) / np.maximum(cm.sum(-2) + cm.sum(-1) - np.diagonal(cm, axis1=-2, axis2=-1), 1), axis=-1)
for baseline in ['vit', 'vit_conv']:
    a = np.array([r['test_per_image_confusions'] for r in runs['mamba3']]).reshape(3, -1, 2, 2)
    b = np.array([r['test_per_image_confusions'] for r in runs[baseline]]).reshape(3, -1, 2, 2)
    values = []
    for _ in range(2000):
        ids = rng.integers(0, a.shape[1], a.shape[1])
        values.append(float((miou(a[:, ids].sum(1)) - miou(b[:, ids].sum(1))).mean() * 100))
    differences[baseline] = {'mean_difference_pp': summary['mamba3']['mean_miou_percent'] - summary[baseline]['mean_miou_percent'],
                             'paired_test_image_bootstrap_95_percent_pp': np.percentile(values, [2.5, 97.5]).tolist(),
                             'scope': 'Test-image resampling only, conditional on the three trained seeds.'}
result = {'models': summary, 'mamba_minus_baseline': differences}
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
rows = []
labels = {'mamba3': 'Mamba-3試作', 'vit': 'ViT・patch4', 'vit_conv': 'ViT・共通Conv stem'}
for name, r in summary.items():
    rows.append(f"| {labels[name]} | {r['encoder_parameters']:,} | {r['mean_miou_percent']:.2f} ± {r['sd_miou_percentage_points']:.2f} | {r['mean_pet_iou_percent']:.2f} | {r['median_encoder_ms']:.3f} | {r['median_segmenter_ms']:.3f} |")
cfg = runs['mamba3'][0]['config']
size = cfg['size']
for items in runs.values():
    for r in items:
        assert all(r['config'][key] == cfg[key] for key in ['size', 'train', 'val', 'test', 'epochs', 'batch', 'lr'])
assert all([r['seed'] for r in items] == [r['seed'] for r in runs['mamba3']] for items in runs.values())
m = summary['mamba3']
v = summary['vit_conv']
report = f"""# Mamba-3と小型ViT：{size}×{size}での比較

Mamba-3と通常ViTのmIoU差は{differences['vit']['mean_difference_pp']:+.2f}ポイント、共通Conv stem版ViTとの差は{differences['vit_conv']['mean_difference_pp']:+.2f}ポイント。共通Conv stem版ViTに対するエンコーダ推論時間の比は{m['median_encoder_ms']/v['median_encoder_ms']:.2f}倍。

## 実測結果

mIoUは前景・背景の2クラス平均。平均 ± 標本標準偏差はseed42/43/44の3回。推論時間は各回60サンプルの中央値を取り、さらに3回の中央値を表示する。

| モデル | エンコーダパラメータ | テストmIoU (%) | 前景IoU (%) | エンコーダ (ms) | head込み (ms) |
|---|---:|---:|---:|---:|---:|
""" + '\n'.join(rows) + f"""

## 比較条件と限界

- [Oxford-IIIT Pet公式配布](https://www.robots.ox.ac.uk/~vgg/data/pets/)。公式trainvalから学習{cfg['train']}枚・検証{cfg['val']}枚、公式testからテスト{cfg['test']}枚を固定抽出。[画像一覧](splits.json)。全データ評価ではない。
- RGB{size}×{size}、bilinear resize。正解はnearest resize。label1=前景、2=背景、3=未分類。未分類は損失・採点から除外。境界精度・幾何は未評価。解像度間のmIoUはそれぞれのサイズに変換した正解で測り、同一画素上の比較ではない。
- 全モデル未学習から訓練。事前学習・蒸留なし。{cfg['epochs']}epochs、batch{cfg['batch']}、AdamW、初期lr{cfg['lr']}、cosine decay、weight decay0.01、gradient clip1。左右反転のみ。同一seedではミニバッチ順序と反転を共通化。
- 検証mIoU最高のepochを採用後、テストで評価。モデル別のハイパーパラメータ調整なし。同じ更新回数の比較で、同一FLOPs・同一訓練時間ではない。
- 幅128・深さ2・出力{size//4}×{size//4}位置。同一のConv3×3(128→64)、GELU、Conv1×1(64→2)、bilinear upsampleのhead。
- ViTはpre-LN、4head全域attention、MLP比4、固定2D sin/cos位置埋め込み。通常版はpatch4、Conv版はMambaと同構成の畳み込みstem。既製のViT-BやDINOの学習済みモデルではない。
- Mamba-3は4方向共有SISO、局所depthwise Conv、MLP比2、LayerScale。Conv版ViTと比較しても局所Conv・MLP・残差構成の違いが残る。パラメータ数も完全一致ではなく、Mamba演算単体の因果効果を分離していない。
- GB10、FP32重み、BF16 autocast、TF32許可。推論batch1、20回warm-up後60回CUDA event計測。画像読込・初回コンパイルを除外。
- JSONのincremental peak memoryはwarm-up後のエンコーダ推論で増えたPyTorch割当メモリ。モデル重み・データ・CUDAコンテキスト等を含む総使用量ではない。訓練時メモリでもない。

3seedは訓練のばらつきを部分的に測るが、データ分割は1種類。小規模・短期学習の結果を、十分な学習後の上限やDINOv3 ViT-Bへの優劣に一般化しない。高解像度で細部を認識できるかは別途検証が必要。

[集計JSON](summary.json)のbootstrap区間は3学習済みseedを固定したテスト画像の再標本化。未知seed・別データへの信頼区間ではない。

各PNGは同じ先頭8テスト画像。左から画像、正解、予測。赤=前景、紺=背景、黄=未分類（採点対象外）。結果の良い画像だけを選んでいない。

## 再実行

ルートディレクトリから、[README](../../README.md)のuv環境で実行する。公式配布tar.gzをdata/pets/に置く。

```sh
uv run --locked compare_vision.py --size {size} --output {root.as_posix()}
uv run --locked summarize_comparison.py --root {root.as_posix()}
```

完了済みモデル・seedはskipする。条件変更時は別outputを指定する。各JSONには学習曲線・テスト混同行列・設定・ソースSHA256を保存。重みは.ptに保存する。
"""
(root / 'REPORT.md').write_text(report)
