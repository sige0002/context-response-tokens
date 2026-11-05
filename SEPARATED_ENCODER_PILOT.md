# 現在画像の特徴と時間文脈を分けるエンコーダ

2026-11-05。**3条件の予備学習は完了した。現在の時間認識研究の主構成としては採用しない。**
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
各条件1,024更新。主学習とGPUを共用するので、学習の経過時間から推論速度を比較しない。

対照`final_memory`の学習・評価・書き出しは完了した。
固定した1,024更新での結果は次のとおり。PCKは256画素基準の閾値未満の割合を8シーンで等重み平均する。

| 対照の状態条件 | 全可視点PCK4 | 全可視点PCK16 | 16画素超移動点PCK16 |
| --- | ---: | ---: | ---: |
| 履歴あり | 18.65% | 56.47% | 19.57% |
| Mamba状態を毎観測消去 | 28.29% | 61.39% | 22.72% |

[全保存時点の結果](results/separated_encoder_pilot_seed17_framewise/final_memory.json)、
[CPU独立検証](results/separated_encoder_pilot_seed17_framewise/verification_final_memory.json)、
[元RGBからの再実行](results/separated_encoder_pilot_seed17_framewise/replay_final_memory.json)を保存した。
CPUでは48予測ファイル・768指標行を再採点し、最終特徴から384回の位置探索を再実行した。
別プロセスで書き出したエンコーダを読み込み、8シーンの履歴あり・消去、計384フレームの特徴が完全一致した。
この対照では履歴を残すと対応精度が低い。学習したMambaの一般的な能力を否定する結果ではない。

残る2条件も固定した1,024更新で完了した。

| 分離構成の条件 | 全可視点PCK4 | 全可視点PCK16 | 16画素超移動点PCK16 |
| --- | ---: | ---: | ---: |
| Mamba・履歴あり | 33.73% | 67.62% | 34.78% |
| Mamba・毎観測状態消去 | 33.72% | 67.24% | 33.46% |
| 画像のみ | 36.08% | 68.72% | 37.26% |

[全保存時点の曲線・層別指標・検証範囲](results/separated_encoder_pilot_seed17_framewise/outcome.json)を保存した。
3条件すべてでCPUによる予測再採点・最終位置探索・書き出し重みの照合が通過した。
元RGBからのGPU再実行は2つのMamba条件を対象にし、それぞれ384フレームで完全一致した。
画像のみの書き出しについては、この別プロセスGPU再実行を追加実施していない。

分離構成は最終層だけの対照より高いが、この診断では画像のみの対照がさらに高い。
浅い観測特徴の効果を、Mambaの記憶による改善とは呼べない。
時間認識の主目的をこの対応点指標へ置き換えず、補助診断として終了する。
評価の出力が一時的に滞ったが、元の2プロセスを再起動せずに処理が進み、pilotは正常完了した。

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

## 保存結果とエンコーダ書き出しの独立再検証

[CPU検証](exploration/verify_separated_encoder.py)は、学習プロセスと独立に、
学習列・初期重み・凍結重み・最終書き出し重みを照合する。
全保存時点の予測を正解座標から再採点し、最終時点では保存特徴から位置探索も再実行する。
これだけではGPUエンコーダの実行やRGB前処理を再検証したことにはならない。

別の[RGB再実行](exploration/replay_separated_encoder.py)では、自己完結したcheckpointを
別プロセスで読み込み、元画像アーカイブから前処理をやり直す。
全8検証シーン・全24観測・履歴あり／状態消去の各条件で、保存特徴との一致と状態量を確認する。
この再実行は学習後のエンコーダ書き出しを検証するもので、未知画像への汎化や速度を示すものではない。

[結果の集計](exploration/analyze_separated_encoder.py)はCPU検証を通過した記録を使い、
全可視点・移動点・遮蔽経験後・再出現直後・経過観測数を分ける。
遮蔽と再出現は4フレーム間隔で選んだ観測列上の定義であり、間の全フレームを検査していない。
シーンを単位とする対応付きbootstrap区間は探索的なもので、多重比較を補正した有意差の主張ではない。

完了した実行に対して、次をリポジトリのルートで実行する。
最初の2コマンドは`--arm final_memory`などで完了済みの1条件だけにも適用できる。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_separated_encoder --run results/new_separated_pilot
CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_CACHE_AUTOTUNING=1 uv run --locked python -m exploration.replay_separated_encoder --run results/new_separated_pilot
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.analyze_separated_encoder --run results/new_separated_pilot
```

## 追加学習しない新規シーン評価の準備

**研究目的の確認後、この追加評価は開始しない。**
細部対応を主目的とする探索から、[時間認識・履歴利用・Attentionとの費用比較](RESEARCH_PROTOCOL.md#研究目的の確認2026-11-05)へ優先順位を戻す。
以下は準備済みのコード・固定計画の記録であり、次に実行する主実験を意味しない。

8シーンは既に複数の探索で観察したため、それとは別に
[32シーンを固定した計画](results/separated_encoder_holdout_seed17/plan.json)を作成した。
公式ではなく、このリポジトリで共通に使うseed 42の300シーン検証分割から、
対応特徴キャッシュの学習・検証シーンとoracle診断のシーンを除外した。
準備済みの候補をseed 17で選び、評価前にIDと元データのハッシュを固定する。
後からデータ準備が進んでも、固定した32シーンを変えない。

[評価コード](exploration/evaluate_separated_holdout.py)は、3条件の1,024更新の書き出しをそのまま使用し、
学習やcheckpoint選択を行わない。前処理・可視点の選択・連続位置照合は予備実験と揃える。
[32シーン・768観測の元RGBと注釈の検査記録](results/separated_encoder_holdout_seed17/input_qualification_summary.json)を保存したが、エンコーダの評価結果はまだない。
同じKubricデータ内の追加検証であり、実写への汎化や独立した最終testを意味しない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.evaluate_separated_holdout --pilot-run results/new_separated_pilot --output results/new_separated_holdout --plan-only
CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_CACHE_AUTOTUNING=1 uv run --locked python -m exploration.evaluate_separated_holdout --pilot-run results/new_separated_pilot --output results/new_separated_holdout --resume
```

実行には完了した3条件と各CPU検証記録、および先のoracle診断の計画が必要。
`--resume`は固定した未実行計画の開始を意味し、途中評価の再開には対応しない。
