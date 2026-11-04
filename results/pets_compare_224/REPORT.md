# Mamba-3と小型ViT：224×224での比較

Mamba-3と通常ViTのmIoU差は+0.96ポイント、共通Conv stem版ViTとの差は-0.66ポイント。共通Conv stem版ViTに対するエンコーダ推論時間の比は3.24倍。

## 実測結果

mIoUは前景・背景の2クラス平均。平均 ± 標本標準偏差はseed42/43/44の3回。推論時間は各回60サンプルの中央値を取り、さらに3回の中央値を表示する。

| モデル | エンコーダパラメータ | テストmIoU (%) | 前景IoU (%) | エンコーダ (ms) | head込み (ms) |
|---|---:|---:|---:|---:|---:|
| Mamba-3試作 | 450,064 | 71.71 ± 1.55 | 63.43 | 2.942 | 3.031 |
| ViT・patch4 | 403,072 | 70.75 ± 0.10 | 61.77 | 0.577 | 0.704 |
| ViT・共通Conv stem | 472,832 | 72.37 ± 1.80 | 64.23 | 0.908 | 1.030 |

## 比較条件と限界

- [Oxford-IIIT Pet公式配布](https://www.robots.ox.ac.uk/~vgg/data/pets/)。公式trainvalから学習512枚・検証128枚、公式testからテスト512枚を固定抽出。[画像一覧](splits.json)。全データ評価ではない。
- RGB224×224、bilinear resize。正解はnearest resize。label1=前景、2=背景、3=未分類。未分類は損失・採点から除外。境界精度・幾何は未評価。解像度間のmIoUはそれぞれのサイズに変換した正解で測り、同一画素上の比較ではない。
- 全モデル未学習から訓練。事前学習・蒸留なし。8epochs、batch16、AdamW、初期lr0.001、cosine decay、weight decay0.01、gradient clip1。左右反転のみ。同一seedではミニバッチ順序と反転を共通化。
- 検証mIoU最高のepochを採用後、テストで評価。モデル別のハイパーパラメータ調整なし。同じ更新回数の比較で、同一FLOPs・同一訓練時間ではない。
- 幅128・深さ2・出力56×56位置。同一のConv3×3(128→64)、GELU、Conv1×1(64→2)、bilinear upsampleのhead。
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
uv run --locked compare_vision.py --size 224 --output results/pets_compare_224
uv run --locked summarize_comparison.py --root results/pets_compare_224
```

完了済みモデル・seedはskipする。条件変更時は別outputを指定する。各JSONには学習曲線・テスト混同行列・設定・ソースSHA256を保存。重みは.ptに保存する。
