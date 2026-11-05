# 現在画像の特徴と時間文脈を分けるエンコーダ

2026-11-05。**実装して予備学習を実行中であり、性能改善を達成したという報告ではない。**
前の[時間層の診断](TEMPORAL_LAYER_DIAGNOSTIC.md)では、早い時間層による位置対応の悪化が大きかった。
そこで、時間状態が書き換えない出力経路を設け、その効果を直接学習で調べる。

## 構造

元のSigLIP2＋第4・8・12層後のMambaを意味・時間処理の経路として残す。
第4層の出力を最初の時間層の**直前**で取り出し、独立した経路から最終出力へ渡す。

```text
現在画像 → 空間1–4層 ─────────────→ 観測射影384次元 ─┐
                    ↓                              ├→ 196×768
            時間層＋残りの空間層 → 文脈射影384次元 ─┘
```

各射影はLayerNorm・線形射影・LayerNormで、二つの出力を連結する。
連結後に両経路を一緒に正規化しない。画像エンコーダを二度実行せず、出力トークン数と総幅も増やさない。
追加した射影は594,432パラメータ。時間Mambaを含む学習対象は3,096,240パラメータとなる。

固定した重みで、観測側384次元は現在画像だけの関数となる。
これは履歴に依存しないという性質であり、384次元への射影で全ての細部を保つ保証ではない。
後段が観測経路を使えること、時間文脈が認識を改善することも別途学習・評価が必要。

[実装](exploration/separated_encoder.py)は、一括・現在フレーム・逐次・chunk入力と、
自己完結したcheckpointの保存・読み込みを持つ。既存の主学習コードは変更しない。

## 既存研究との関係

浅い層の特徴を使うことには[DPT](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html)などの先行例がある。
凍結画像モデルの複数深度特徴と時間Mambaの組合せも、
[Frozen Image＋GMMix §3.2–3.3](https://arxiv.org/html/2605.19137v1#S3)にある。
今回の分離や結合だけを新しい原理とは主張しない。

本試作で問うのは、**同じ出力幅の中で現在の観測を保持することが、時間情報の利用と位置対応を両立させるか**である。
浅い特徴を追加しただけの改善を、記憶の改善と呼ばないために次の対照を置く。

## 3条件の比較学習

| 条件 | 前半384次元 | 後半384次元 | 学習する時間層 |
| --- | --- | --- | --- |
| final_memory | 時間統合後の最終層 | 同じ最終層 | あり |
| separated_memory | 時間層前の第4層 | 時間統合後の最終層 | あり |
| separated_image | 現在画像の第4層 | 現在画像の最終層 | なし |

3条件の射影の初期重み・パラメータ数は同じ。前2条件の時間層も同じ初期重みを使う。
画像だけの対照は時間層のパラメータ・状態を持たず、総学習パラメータ数は小さい。
同じ総パラメータ数の比較であるとは主張しない。

- 前2条件は主学習の固定4,000更新checkpointから開始する。
- 元のSigLIP2は固定し、Mambaと出力射影をRGBと対応点ラベルから更新する。
- 24学習シーン・8検証シーン。各条件1,024更新、途中256更新も保存し、結果で停止時点を選ばない。
- 各更新は同じシーン・64点・初期画像と後続7観測。同じ学習列を使う。
- 教師特徴への蒸留はしない。可視対応点の正解位置に対する交差エントロピーを使う。
- 評価では24時刻を逐次入力し、同じ連続位置cosine照合を使う。
  時間層ありでは状態を毎回消した対照も評価する。
- query外観は初期画像から保持するため、状態消去は追跡器全体の完全な記憶消去ではない。

[学習コード](exploration/train_separated_encoder.py)と[損失](exploration/separated_loss.py)を参照。
これは少数の既観察レンダリングシーンでの予備学習であり、実写の最終評価やSOTA比較ではない。
可視性分類器はなく、常に可視と予測する。分類器は新しい出力に合わせて再学習する必要があり、
この予備実験では分類能力の維持をまだ検証しない。

## 一括処理と逐次処理の数値差

最初のGPU検査では、元の時間エンコーダの一括処理と逐次処理で、
射影へ入る最終特徴に最大0.342、RMSE 0.00927の差があった。
新しい出力にも最大0.0994の差が現れ、既存の要素別許容誤差を超えた。
射影前の第4層特徴の差は最大5.30e-6だった。

FP32へ切り替えた試行でも完全には解消しなかった。
固定した公式SISO実装は入口でQ・K・VをBF16へ変換し、
内部では一括のchunk内積と、過去状態からの読み出しが異なる順序で丸められる。
これは差が発生し得る根拠であり、観測した差の全てを一つの丸め操作に帰属させる解析ではない。
[BF16試行](results/separated_encoder_qualification.precision.json)と
[FP32射影・演算試行](results/separated_encoder_fp32_qualification.precision.json)を保存した。

候補の既定実装では、学習・一括呼び出し・逐次呼び出しを、同じ1フレーム単位のnative状態更新へ揃えた。
学習時も4種類の状態を通して勾配を伝える。状態を途中でdetachしない。
これにより訓練時の時間方向の並列scanは使えなくなるため、費用の増加は評価対象に含める。
元の20,000更新の主学習と固定済みの比較には、この変更を適用しない。

## 実装検査と実行状態

[CPU検査](results/separated_encoder_tests.log)は、観測経路の履歴非依存性、未来からの漏洩がないこと、
一括・逐次・chunkの一致、checkpoint再読込、勾配が流れる経路と凍結された経路を確認する。
[native Mambaの検査コード](exploration/check_separated_encoder.py)では、実RGB入力、状態管理、
対応点損失の逆伝播、optimizer更新を確認する。

新しいCUDA contextの作成がメモリ確保で一度失敗したため、完了済み診断の`.pt`に対してのみ
`POSIX_FADV_DONTNEED`を行った。ファイルは削除せず、主学習を停止せず、再試行でGPU起動を確認した。
この問題を認識精度の問題と混同しない。

[native検査の結果](results/separated_encoder_framewise_qualification.json)では、一括・逐次・chunkの最大差が0、
履歴あり・消去での観測経路の最大差も0だった。勾配を有効にしたforwardも同じ出力になった。
最後の文脈だけを読む損失から最初の画像へ届いた勾配ノルムは0.01220で、
3箇所のMambaの勾配、対応点損失による更新、SigLIP2の重みが不変であることを確認した。
状態量は各時刻79,027,200 bytesだった。4フレームでの実装検査であり、長期保持や速度の証明ではない。

[比較学習の計画](results/separated_encoder_pilot_seed17_framewise/plan.json)と
[全条件共通の学習列](results/separated_encoder_pilot_seed17_framewise/schedule.json)を固定して実行を開始した。
各条件1,024更新で、結果は未確定。主学習とGPUを共用するので、学習の経過時間から推論速度を比較しない。

## 再実行

前段の[4,000更新checkpointとRGB](TEMPORAL_FEATURE_PROBE.md)が必要。新しい出力ディレクトリを使う。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_separated_encoder.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_CACHE_AUTOTUNING=1 uv run --locked python -m exploration.check_separated_encoder --probe-run results/temporal_feature_probe_seed17_v1 --execution framewise --output results/new_separated_qualification.json
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.train_separated_encoder --probe-run results/temporal_feature_probe_seed17_v1 --output results/new_separated_pilot --plan-only
CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_CACHE_AUTOTUNING=1 uv run --locked python -m exploration.train_separated_encoder --probe-run results/temporal_feature_probe_seed17_v1 --output results/new_separated_pilot --resume
```

`--resume`は未実行の固定計画を開始する指定であり、途中学習の再開ではない。
元の一括経路の数値差を調べるには、検査の`--execution batched --precision bfloat16`を使う。
これは既知の許容誤差違反を再現する診断で、検査成功を期待するコマンドではない。
