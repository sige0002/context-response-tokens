# 学習する読み出しで履歴介入を再評価

2026-11-05。**小さな線形読み出しを学習しても、履歴ありMambaの外観照合の差は解消しなかった。**
全可視点のPCK4は、SigLIP2単独24.68%、Mamba履歴あり17.52%、毎フレーム状態消去24.69%だった。
全体の指標は学習前より改善したが、大きく移動した点の精度は3条件とも悪化した。
この読み出しを精密認識の改良として主実験へ採用しない。

## 前の診断から変えたもの

[固定したcosine照合による診断](TEMPORAL_FEATURE_PROBE.md)では、時間文脈で変わった特徴を、
未学習の照合規則で比較していた。その不利を学習で補えるかを検査した。
エンコーダの重みは前と同じ4,000更新時点で固定し、変更していない。

読み出しの前に、共有する線形変換

\[
T(x)=x+UVx,\qquad U\in\mathbb R^{768\times32},\quad V\in\mathbb R^{32\times768}
\]

を置く。Uをゼロで初期化し、学習前は元の特徴と完全に同じにする。
queryと現在画像へ同じ変換を適用し、その後のLayerNorm・L2正規化・連続位置探索は前と揃えた。
線形変換なので特徴補間と順序を交換できる。学習時と細かな位置探索時で、別の特徴変換にはならない。

U・Vと学習用照合温度の計49,153パラメータを学習した。温度は座標探索のargmaxを変えず、学習損失に使う。
これは表現を測るための限定した線形プローブであり、新しいエンコーダ原理の提案ではない。

## 揃えた条件

- 3条件：SigLIP2単独、Mamba履歴あり、Mamba毎フレーム状態消去。
- 検証8シーンと重ならない24シーンを学習に使う。シーン集合は以前の対応実験と共通。
- 各更新で同じシーン、初期フレームの64点、後続23時刻から選んだ同じ3時刻を使う。
- 読み出しの初期重み、1,024更新、AdamW、学習率1e-3、weight decay 1e-4、勾配clip 1を共通にする。
- 損失は可視な正解対応に対する交差エントロピー。正解位置を近傍4パッチへの双線形重みで表す。
  不可視点の座標は損失から除き、教師特徴への蒸留は行わない。
- 学習前・256・1,024更新の全結果を保存し、検証結果による停止や最良時点の選択はしない。

学習列は196,608点時刻で、可視163,841、不可視32,767だった。
queryの外観は初期画像から取得し、現在位置の正解を推論に渡さない。
全条件で可視性分類器は追加せず、常に可視と予測する。

学習用の最終層特徴は同じcheckpointから実RGBを1フレームずつ入力して取得した。
空間FP32・TF32無効、時間BF16で取得し、読み出しの学習と探索はCPU FP32・2スレッドで行った。
Mamba・SigLIP2の更新や、主学習の中断はしていない。

根拠：[特徴取得](exploration/cache_temporal_metric.py)、[変換と損失](exploration/linear_correspondence_metric.py)、
[学習コード](exploration/train_temporal_metric_probe.py)、[実行前の計画](results/temporal_metric_probe_seed17/plan.json)、
[共通の学習列](results/temporal_metric_probe_seed17/schedule.json)。

## 最終1,024更新の結果

以下は最終層・上位4パッチからの連続位置探索。256画素基準のPCKと0–100表記のAJを、8検証シーンで等重み平均した。
この8シーンは既に探索で観察したKubricのレンダリング画像であり、未使用testや実写ベンチマークではない。

| 条件 | 学習前PCK4 | 学習後PCK4 | 学習後PCK16 | 学習前AJ | 学習後AJ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SigLIP2単独 | 22.56% | 24.68% | 57.81% | 16.69 | 19.18 |
| Mamba・履歴あり | 14.53% | 17.52% | 54.13% | 11.19 | 14.56 |
| Mamba・毎フレーム消去 | 22.67% | 24.69% | 58.29% | 16.46 | 19.13 |

初期位置を答え続ける共通基準は、PCK4 39.11%、PCK16 57.46%、AJ 30.67だった。
学習した3条件ともAJではこの基準を下回る。

16画素を超えて動いた可視点だけでは、次のようになった。

| 条件 | 学習前PCK16 | 学習後PCK16 | 学習前の平均誤差 | 学習後の平均誤差 |
| --- | ---: | ---: | ---: | ---: |
| SigLIP2単独 | 26.73% | 20.48% | 49.99画素 | 57.61画素 |
| Mamba・履歴あり | 22.63% | 19.40% | 55.27画素 | 59.14画素 |
| Mamba・毎フレーム消去 | 25.53% | 21.47% | 50.11画素 | 56.19画素 |

全体指標と大きな移動の指標が逆方向になったため、全体のPCKやAJの改善だけを採用理由にしない。
256更新時点を含む[全曲線・全シーン](results/temporal_metric_probe_seed17/summary.json)と
[損失](results/temporal_metric_probe_seed17/training.jsonl)も公開した。

## 確認した範囲と判断

[2件のCPU検査](results/linear_correspondence_metric_tests.log)で、初期の恒等変換、
補間との交換、不可視座標の無視、学習経路の有限な勾配を確認した。
3条件とも学習前の予測は、前の固定照合の保存予測と全件完全一致した。

[検証](results/temporal_metric_probe_seed17/verification.json)では学習列を再生成し、
3条件の初期重み、元注釈、query、checkpointと保存予測のハッシュを照合した。
72予測ファイルの1,152行とシーン平均を独立に再採点し、最終重みから576回の照合を再実行して一致した。
GPUエンコーダと中間256更新のforwardを再実行した検証ではない。

今回分かったのは、**rank32の線形読み出しをこの条件で学習するだけでは、履歴ありの照合低下を補えなかった**こと。
非線形の読み出し、より多いデータ、長い学習、正式な点追跡学習でも同じとは断定しない。
同時に、分類精度の改善を、細部や動きの対応の改善とみなす前提も置かない。
読み出しの小変更を性能上の貢献として採用せず、正式な比較を継続しながら、
時間更新と空間的な対応付けを両立する設計へ検討を戻す。

## 再実行

リポジトリのルートで、前段の[4,000更新の履歴介入](TEMPORAL_FEATURE_PROBE.md)と元RGBシーンを用意する。
以下はその固定checkpointを利用する診断であり、実行中の最新checkpointへ置き換えない。
出力先は未使用のものを指定する。大きな重み・画像・特徴・予測配列はGitHubに含めない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_linear_correspondence_metric.py
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.cache_temporal_metric --probe-run results/temporal_feature_probe_seed17_v1 --output results/new_temporal_metric_cache --plan-only
uv run --locked python -m exploration.cache_temporal_metric --probe-run results/temporal_feature_probe_seed17_v1 --output results/new_temporal_metric_cache --resume
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.train_temporal_metric_probe --train-cache results/new_temporal_metric_cache --probe-run results/temporal_feature_probe_seed17_v1 --output results/new_temporal_metric_probe --plan-only
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.train_temporal_metric_probe --train-cache results/new_temporal_metric_cache --probe-run results/temporal_feature_probe_seed17_v1 --output results/new_temporal_metric_probe --resume
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_temporal_metric_probe --run results/new_temporal_metric_probe
```

特徴取得は一致する完成シーンを再利用できる。学習側の`--resume`は未実行の固定計画を開始する指定で、途中学習の再開ではない。
