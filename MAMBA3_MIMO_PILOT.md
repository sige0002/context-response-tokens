# SISOとMIMOの同条件追加学習

目的は、SigLIP2内の時間アダプタをnative Mamba-3のrank 4 MIMOへ変えると、
現在のSISOを同じデータで継続学習するより時間認識が改善するかを測ること。
[数値・実行検査](MAMBA3_MIMO_FEASIBILITY.md)とは分けた、認識精度の開発実験である。
MIMO自体は既存の方式であり、置換したことだけを新規性やSOTAとは主張しない。

## 固定した条件

[実行前の計画](results/mimo_pilot_step11000_v2/plan.json)と
[比較コード](exploration/mimo_pilot.py)に、次の条件を固定した。

| 項目 | 条件 |
| --- | --- |
| 開始重み | 保存済みの11,000更新時点。同じSISOから数学的な初期動作を保つMIMOを作成 |
| 空間エンコーダ | SigLIP2-B/16、224×224、重み固定 |
| 時間アダプタ | 第4・8・12層後。幅256、状態次元64。SISOとrank 4 MIMO |
| 学習対象 | 時間アダプタと共通の分類器。蒸留はしない |
| 学習量 | 各512更新、microbatch 4、勾配蓄積4。同じ8,192本を同じ順序・変換で1回ずつ使用 |
| 最適化 | 両方で新しいAdamW、最大学習率`3e-5`、warmup 32、weight decay `1e-4`、同じcosine schedule |
| 入力 | 現在画像を含む8観測。分類器は最後の画像の特徴だけを読む |
| 評価 | 174クラス各4本、計696本。初期値と512更新後だけを評価 |
| 主指標 | 1観測ずつ状態を更新した最後の特徴によるtop-1。4つの独立動画をバッチ化 |
| 副指標 | 同じ動画の一括clip評価、初期値との差、保存・再読込による全696本の予測一致 |

学習可能パラメータはSISOが9,133,662、MIMOが9,456,222で、差は322,560。
データと更新数は揃えるが、パラメータ数・FLOPsを揃えた比較ではない。
初期化の数学的な同等性は確認した一方、BF16経路の丸め差により初期予測が一致するとは仮定しない。
初期時点の精度差も保存する。

評価対象は、以前の348本の履歴診断と696本の条件付け実験から除外して選んだ。
ただし、主学習では検証集合全体の集計値をすでに見ているため、未使用の最終テスト集合とは呼ばない。
クラスごとの本数を揃えた開発用の正解率であり、全24,777本の通常分布での正解率と直接同列に置かない。

## 環境と検査

学習・clip評価は固定したルートuv環境、逐次評価はTVM-FFIだけを0.1.10にしたuvの一時環境を使う。
両アームの同じ評価モードは同じ環境で実行し、実行時のバージョンが計画と異なれば拒否する。
ルート環境のTVM-FFIは0.1.9のまま。PyTorch 2.14.1+cu130、TileLang 0.1.8、CUTLASS DSL 4.8.1は共通。

学習で読んだメタデータ・画素のハッシュ、および初期・最終・再読込評価の入力ハッシュを両アーム間で照合する。
保存したlogitから正解数を再計数し、ラベル順序、モデルハッシュ、学習予算、凍結した重みを確認する。
失敗時は部分的な結果を残し、黙って再学習しない。

[CPU検査9件のログ](results/mimo_pilot_cpu_tests_v2.log)と
[実データを使った計画の照合](results/mimo_pilot_step11000_v2/cpu_verification.json)を保存した。
これは比較学習が終了したという結果ではない。精度差はまだ得られていない。

## 実行順と判断

[追加の待機処理](exploration/run_mimo_after_comparisons.py)は、既存の比較・費用測定・履歴診断の最後にある
`run_final_memory_layers`の実プロセスをPIDファイルディスクリプタで束縛して待つ。
終了後に正常完了、計画と各結果のハッシュ、主学習の20,000更新完了、GPUが空いていることを確認し、
[元の実験コード](exploration/run_mimo_pilot.py)を起動する。
SISO学習、MIMO学習、SISO逐次評価、MIMO逐次評価、比較集計の順序と、入力・重み・学習条件は維持する。

既存の比較キューは、各処理の開始時に他のGPUプロセスがあれば停止する設計である。
そのため、MIMOも主学習の終了直後に開始する設定では競合する。
MIMOの元の待機プロセスだけを学習開始前に停止し、既存の比較列の後へ予約し直した。
主学習と既存の比較キューのプロセス・ソース・設定は変更していない。
[待機の実確認](results/mimo_after_comparisons_v1/launch_verification.json)と
[変更前後の記録](results/mimo_queue_reschedule_v1.json)を保存した。
旧pilotの`status.json`の`stopped`は、この待機処理を止めた記録である。
再予約の状態は`results/mimo_after_comparisons_v1/status.json`を確認する。
この試験は認識精度の比較であり、速度評価は別に行う。

待機処理の[CPU検査11件](results/mimo_after_comparisons_cpu_tests.log)が通過した。
先行処理の未完了・失敗、結果や計画の変更、GPU使用中を拒否し、無関係なプロセスを操作しないことを検査した。
初回の検査では模擬プロセスの起動直後の読み取りが失敗したため、テストに起動完了の同期を追加した。
[初回ログ](results/mimo_after_comparisons_cpu_tests_initial_failure.log)も残している。

大きい比較へ進める条件は、あらかじめ次の3点にした。

1. 最終の逐次精度がSISOより1ポイント以上高い。
2. 同じ動画を使うクラス層化bootstrapの差の95%区間の下端が0を超える。
3. 最終時点の差が初期時点の差より大きい。

満たしても、1 seed・短い追加学習での開発結果である。複数seed、同計算量の対照、
全件評価、自然な長い履歴、ロボティクスでの評価、実測コストは別途必要になる。

再実行はリポジトリのルートで行う。未使用の出力先を選び、`<final-memory-queue-pid>`を
実際に確認した`results/final_memory_layers_v1`の待機プロセスのPIDへ置き換える。
この例は既存の比較・費用測定・履歴診断キューを準備済みの場合に使う。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.mimo_pilot \
  --output results/new_mimo_pilot --mode plan
uv run --locked python -m exploration.run_mimo_after_comparisons \
  --pilot results/new_mimo_pilot \
  --dependency results/final_memory_layers_v1 \
  --after-pid "<final-memory-queue-pid>" --output results/new_mimo_schedule
```

計画はソースと入力をハッシュで固定するため、待機開始後に対象ファイルを変更すると実行を拒否する。
