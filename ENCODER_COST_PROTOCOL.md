# 実動画でのエンコーダ計算費用

分類精度と計算費用を同じ最終checkpointで比較するための測定である。
現在の実測は3更新の動作検査用checkpointに限られ、本学習済み5方式の性能比較は未完了である。

## 確認できた実測

RoboTAPの同じ296フレームの動画を、batch 1、224入力で先頭から逐次処理した。
1周をwarmupとし、状態を初期化して3周を測定した。空間層はTF32を無効化したFP32、時間層はBF16。
学習プロセスを一時停止し、測定終了後に再開した。

| 指標 | 中間層Mamba-3 | 公開GatedMambaMix |
| --- | ---: | ---: |
| エンコーダのGPU時間、中央値 | 11.58 ms | 12.19 ms |
| 前処理・転送込み、中央値 | 13.18 ms | 13.98 ms |
| エンコーダのパラメータ数 | 95,386,032 | 141,048,576 |
| 状態テンソル | 75.37 MiB | 91.88 MiB |
| PyTorch割当のピーク | 599.86 MiB | 708.06 MiB |
| 3周の最終出力 | 完全一致 | 完全一致 |

上記の時間は全888件のフレーム処理を集計した条件付きの中央値であり、統計的な優位性や認識精度の保証ではない。
GMMixは公開コアを変更せず、この環境のPyTorch畳み込みfallbackを使用する。最適化カーネルのある別環境の速度とは区別する。
両者とも3更新のcheckpointであり、これらの値を本学習済みモデルの最終比較へ流用しない。

[全実測と元データの照合](results/real_video_cost_integration_rgbfix/verification.json)、
[Mamba-3の各フレーム記録](results/real_video_cost_integration_rgbfix/mamba/timings.jsonl)、
[GMMixの各フレーム記録](results/real_video_cost_integration_rgbfix/gmmix/timings.jsonl)、
[15件の検査](results/encoder_costs_cpu_tests_final.log)を保存している。

## 本学習後の比較条件

[実行計画](results/encoder_costs_seed42_rgbfix_v2/plan.json)に、同じ最終20,000更新の5方式を固定した。
履歴なしSigLIP 2、中間層Attention、中間層Mamba-3、出力側Mamba-3、GMMixを比較する。
分類用の読み出しと点追跡用の読み出しは費用に含めず、現在画像の196×768トークンまでを測る。

動画は精度を見て選ばない。監査済みRoboTAPの重複RGBを除き、フレーム数とIDで並べた最短・中央・最長を使う。

| 選択 | フレーム数 | 動画ID |
| --- | ---: | --- |
| 最短 | 52 | `gear_v2_12-basket_front_left_rgb_img-0` |
| 中央 | 245 | `gear_v2_4-wrist_back_rgb_img-1` |
| 最長 | 1,337 | `15402003961862946817` |

各方式・各動画を独立したプロセスで読み、1周warmup、3周測定する。異なる方式の同時GPU実行は拒否する。
計算費用のキューは点追跡のGPUキューが終了してから動く。これはGPU利用順の制御であり、点追跡の成功を計算費用の前提にはしない。
動画分類の5モデルについては、全更新と予定された検証の完了を確認してから測定する。

```sh
# POINT_QUEUE_PIDは、同じpoint-comparisons出力を使う実際のPythonプロセス
uv run --locked python run_encoder_costs.py \
  --point-comparisons results/point_comparisons_with_gmmix_seed42_rgbfix \
  --after-point-pid "$POINT_QUEUE_PID" \
  --output results/new_encoder_costs
```

## 測定値の意味と限界

現在の時間Attentionは3層、196座標、幅256、BF16のK/Vを保持するため、batch 1の状態は
`3 × 2 × 196 × 256 × 2 × T = 602,112 × T` bytesとなる。
8フレームなら約4.59 MiBであり、Mamba-3の約75.37 MiBより小さい。
状態テンソルだけの逆転は132フレームからである。これはコード上の容量計算であり、
総メモリや実時間の逆転点を示す値ではない。長い履歴での容量上の利点と、実際の認識効果を分けて評価する。

- GPU時間はエンコーダ演算のみ。前処理・転送込み時間にも、動画ファイルの復号時間は含めない。
- 各フレームで同期する遅延測定であり、非同期処理やbatch増加による最大throughputの測定ではない。
- 状態を消してから測る先頭フレームと、履歴1–7、8–31、32–127、128–255、256以上を分ける。warmup後なので、プロセス起動・JITコンパイルのcold startではない。
- メモリのピークは、当該プロセスでPyTorchが確保した重み・現在入力・状態・出力・一時領域を含む。状態の複製や動画全体のGPU保持は行わない。
- CUDAのcontextやPyTorch外の割当はこのピークに含まれない。ドライバが返すプロセスメモリは各周の後に別途記録し、ピークとは呼ばない。
- 固定サイズで長い動画を処理できても、必要な情報を覚えている証拠にはならない。認識・点追跡・履歴介入の結果が別途必要である。
- 3本の動画・1つの機械・1つの学習seedの測定であり、一般的な高速性やSOTAを示す実験ではない。
