# Mamba-3画像試作と小型ViTの比較

今回の条件ではMamba-3の優位性は確認できなかった。通常のpatch4 ViTより平均mIoUは1.85ポイント高いが、同構成のConv stemを使うViTより1.03ポイント低く、エンコーダ推論は約3.4倍遅かった。Conv stem版ViTのパラメータはMambaより約5%多い。入力部だけでなく局所畳み込み・MLP・残差構成も異なるため、差をMamba単体の性質とは断定しない。

同じ先頭8画像の予測を目視しても、脚・尾などの細い部分の欠落が両方式に残る。これは定量的な境界評価ではないが、粗い二値分割の改善を細部認識の達成と扱えないことは確認できる。

## 実測結果

mIoUは前景・背景の2クラス平均。平均 ± 標本標準偏差はseed42/43/44の3回。推論時間は各回60サンプルの中央値を取り、さらに3回の中央値を表示する。

| モデル | エンコーダパラメータ | テストmIoU (%) | 前景IoU (%) | エンコーダ (ms) | head込み (ms) |
|---|---:|---:|---:|---:|---:|
| Mamba-3試作 | 450,064 | 72.86 ± 1.22 | 65.07 | 1.323 | 1.374 |
| ViT・patch4 | 403,072 | 71.01 ± 0.29 | 62.14 | 0.289 | 0.366 |
| ViT・共通Conv stem | 472,832 | 73.89 ± 0.71 | 65.75 | 0.393 | 0.459 |

## 比較条件

- データ：[Oxford-IIIT Pet公式配布](https://www.robots.ox.ac.uk/~vgg/data/pets/)。公式trainvalから学習512枚・検証128枚、公式testからテスト512枚を固定抽出。各集合の画像名は[splits.json](splits.json)。標準の全データ評価ではない。
- RGB128×128、bilinear resize。正解はnearest resize。公式label1=前景、2=背景、3=未分類。未分類は損失・混同行列から除外。境界の厳密な精度や幾何推定は評価していない。
- 全モデル未学習から訓練。事前学習・蒸留なし。8epochs、batch16、256更新、AdamW、初期lr0.001、cosine decay、weight decay0.01、gradient clip1。augmentationは左右反転のみ。モデル間で同一seedの順序と反転を共通化。
- 検証mIoUが最高のepochを採用し、テストは採用後にのみ評価。ハイパーパラメータ探索なし。テスト結果を使った選択なし。
- 幅128・深さ2・出力32×32位置を共通化。同じConv3×3(128→64)、GELU、Conv1×1(64→2)、bilinear upsampleのhead。
- ViTはpre-LayerNorm、4headの全域self-attention、MLP比4、固定2D sin/cos位置埋め込み。patch4の通常入力と、Mambaと同構成のConv stem入力を比較。既製のViT-Tiny/ViT-BやDINOの学習済み重みではない。
- Mambaは既存試作をそのまま使用。4方向共有SISO、局所depthwise Conv、MLP比2、LayerScale。Conv stem比較でもこれらの違いは残り、Mamba演算単体の因果効果を分離した実験ではない。パラメータ数も完全一致ではない。
- NVIDIA GB10、FP32重み、BF16 autocast、TF32許可。推論batch1、20回warm-up後60回CUDA event計測。画像読込・初回コンパイルを除外。モデル全体の訓練時間はJSONに記録したが、validationと初回コンパイルを含む。

## 解釈の範囲

これは同じ小規模データ・更新回数での比較。全データで十分に訓練した性能、計算予算を揃えた性能、最適化後の上限、DINOv3 ViT-Bへの優位性を示すものではない。3seedは訓練のばらつきを部分的に測るが、データ分割は1種類のみ。粗い二値分割のmIoUから、小物体・微細境界・3D幾何の能力は結論できない。

各モデルのPNGは同じ先頭8テスト画像で、良い例を選んでいない。左から画像、正解、予測。赤=前景、紺=背景、黄=未分類（採点対象外）。

[集計JSON](summary.json)にはペア画像bootstrapの差の区間も保存する。この区間は3つの学習済みseedを固定したテスト画像の再標本化であり、訓練seed全体・別データへの信頼区間ではない。

## 再実行

リポジトリのルートから実行する。uv環境の作成方法は[README](../../README.md)。データtar.gzは公式ページから`data/pets/images.tar.gz`と`data/pets/annotations.tar.gz`に保存する。

```sh
uv run --locked pytest -q test_compare_vision.py test_vision_layout.py
uv run --locked compare_vision.py
uv run --locked summarize_comparison.py
```

完了済みのモデル・seedは再実行時にskipする。条件を変更するときは別の`--output`ディレクトリを使う。集計スクリプトは既定の`results/pets_compare`にある3seed×3モデルを対象とする。モデル重みは`.pt`、各回の設定・学習曲線・テスト混同行列・ソースSHA256はJSONに保存する。
