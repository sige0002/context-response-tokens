"""Compare official random DINOv3-B/16 with the completed small Mamba prototype."""
import json
from pathlib import Path
import statistics

root=Path('results/pets_dinov3_224')
paths=sorted(root.glob('dinov3_vitb16_seed*.json'))
assert len(paths)==3
runs=[json.loads(p.read_text()) for p in paths]
old=Path('results/pets_compare_224')
assert (root/'splits.json').read_bytes()==(old/'splits.json').read_bytes()
for r in runs:
    assert r['pretrained'] is False
    assert r['encoder_parameters']==85669632
    for key in ['size','train','val','test','epochs','batch','lr']:
        assert r['config'][key]==json.loads((old/'mamba3_seed42.json').read_text())['config'][key]
scores=[r['test']['miou']*100 for r in runs]
s={'model':'DINOv3 ViT-B/16 official random initialization','pretrained':False,
   'official_commit':runs[0]['official_dinov3_commit'],
   'encoder_parameters':runs[0]['encoder_parameters'], 'total_parameters':runs[0]['total_parameters'],
   'test_miou_percent':scores,'mean_miou_percent':statistics.mean(scores),'sd_miou_pp':statistics.stdev(scores),
   'encoder_median_ms':statistics.median(r['encoder_timing']['median_ms'] for r in runs),
   'encoder_incremental_mib':statistics.median(r['encoder_timing']['incremental_peak_bytes'] for r in runs)/2**20,
   'selected_epochs':[r['selected_epoch'] for r in runs], 'same_split_verified':True}
(root/'summary.json').write_text(json.dumps(s,indent=2))
m=json.loads((old/'summary.json').read_text())['models']['mamba3']
report=f'''# DINOv3 ViT-B/16：ランダム初期化からの小規模比較

DINOv3の公式ViT-B/16を縮小せず使用した。DINOv2ではない。事前学習済み重みは取得も使用もせず、公式の`pretrained=False`による初期化で訓練した。

| モデル | エンコーダパラメータ | 空間出力 | mIoU平均 ± 標本SD (%) | エンコーダ推論 (ms) | 推論中の増分メモリ (MiB) |
|---|---:|---|---:|---:|---:|
| DINOv3 ViT-B/16・ランダム初期化 | {s['encoder_parameters']:,} | 14×14×768 | {s['mean_miou_percent']:.2f} ± {s['sd_miou_pp']:.2f} | {s['encoder_median_ms']:.3f} | {s['encoder_incremental_mib']:.2f} |
| Mamba-3試作・ランダム初期化 | {m['encoder_parameters']:,} | 56×56×128 | {m['mean_miou_percent']:.2f} ± {m['sd_miou_percentage_points']:.2f} | {m['median_encoder_ms']:.3f} | {m['median_incremental_encoder_mib']:.2f} |

## 条件と解釈

- Oxford-IIIT Pet、224×224、学習512枚・検証128枚・テスト512枚。3seed(42/43/44)、8epochs・batch16・256更新。AdamW lr0.001、cosine decay、weight decay0.01、左右反転。画像分割・順序・augmentationは既存比較と共通。
- DINOv3本体は公式の幅768・12層・12head・patch16・4 storage tokens・RoPE・LayerScaleなどを維持。入口はkernel16/stride16のConv2dによるパッチ線形写像。公式ソースcommitは`{s['official_commit']}`。vendorの公式実装は変更していない。
- DINOv3の768次元出力を外付け1×1 Convで128次元へ変換し、その後は既存比較と同じ分割head。追加変換はエンコーダのパラメータ・単体推論時間に含めず、モデル全体には含める。分割headが作用する空間解像度は異なる。
- メモリはwarm-up後、推論中に増えたPyTorch割当量。重みや常駐データを含む総使用量ではない。DINOv3の約85.7Mの重みが不要という意味ではない。
- **これはDINOv3の自己教師あり事前学習の再現ではなく、DINOv3で使うViTアーキテクチャを小規模な教師あり分割で訓練した比較。学習済みDINOv3の能力を評価した結果ではない。**
- パラメータ数・出力位置数・head入力幅が異なる。DINOv3はMamba試作の約190倍のパラメータ数であり、同容量比較ではない。512枚・8epochsは十分な学習を保証せず、同一lrが各モデルの最適値でもない。結果からアーキテクチャの上限・一般的な優劣を結論しない。
- 検証mIoUでepochを選び、選択後にtestで評価。未分類ラベルは損失・採点から除外。境界・幾何の評価ではない。前回と同じ224正解マスクで比較。

## 再現

ルートディレクトリのuv環境を使う。データ取得は[既存224レポート](../pets_compare_224/REPORT.md)参照。

```sh
bash setup_dinov3.sh
uv run --locked compare_vision.py --size 224 --models dinov3_vitb16 --output results/pets_dinov3_224
uv run --locked summarize_dinov3.py
```

[公式モデル定義](https://github.com/facebookresearch/dinov3/blob/{s['official_commit']}/dinov3/hub/backbones.py)・[ラッパー](../../dinov3_baseline.py)・[集計JSON](summary.json)。各seedのJSONに学習曲線、ソースSHA256、テスト混同行列を保存。PNGは同じ先頭8テスト画像で、左からRGB・正解・予測。
'''
(root/'REPORT.md').write_text(report)
print(json.dumps(s,indent=2))
