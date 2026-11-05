# 学習途中のMamba特徴に対する履歴介入

2026-11-05。**4,000更新時点のMambaは、この固定した外観照合では、履歴を使うと最終層の位置精度が下がった。**
上位4パッチから連続位置を探す読み出しのPCK4は、SigLIP2単独22.56%、履歴あり14.53%、毎フレーム状態消去22.67%だった。
履歴統合による細部認識の改善を主張できる結果ではない。
SSv2分類で学習中の特徴を、追加学習なしのcosine照合で測る探索的な診断であり、
学習済みの点読み出しや最終エンコーダの優劣とは区別する。

## 何を固定して比較したか

[連続位置照合の対照](CONTINUOUS_MATCHING_DIAGNOSTIC.md)を、実際の時間エンコーダの出力へ適用した。
稼働中の主学習の`last.pt`を一度開いて別ファイルへコピーし、4,000更新の全モデル状態を固定した。
これはbestの選択ではない。実行中に主学習が新しいcheckpointを書いても、この評価の重みは変わらない。

- 画像側：SigLIP 2-B/16、224入力。保存した画像側の全重みが、指定revisionの事前学習済み重みと完全一致することを確認。
- 時間側：4・8・12層の後のMamba-3。主学習のSSv2分類損失で4,000／20,000更新した途中の重み。
- 画像：既に探索で使用したKubricの8シーン、0・4・…・92フレームの24時刻、各128点。
- 推論：全条件を1フレームずつGPUで再計算。空間FP32・TF32無効、時間BF16。照合はCPU FP32。
- 読み出し：パッチ中心・上位1・上位4の3方式をすべて保存。以下の主表は上位4方式。
- 特徴：第4層は時間残差を加えた直後、最終層は実際のpost-layernorm後の出力。
  最終層の照合には、前の対照と同じパラメータなしLayerNormも適用。

| 条件 | エンコーダの履歴 | 初期点の外観特徴 |
| --- | --- | --- |
| SigLIP2単独 | 時間層なし | 保持 |
| Mamba・履歴あり | 同じエピソードの状態を継続 | 保持 |
| Mamba・毎フレーム消去 | 各フレームを新しい状態から計算 | 保持 |

履歴ありと消去は同じMamba重みを使い、初期フレームの特徴も完全一致した。
消去側でもqueryの外観特徴には初期画像の情報があるため、**追跡器全体から記憶をなくす比較ではない**。
変えるのはエンコーダ内の履歴である。現在位置の正解、可視性、未来画像は推論へ渡さない。

全条件の画像特徴を今回再計算し、以前の8フレーム一括計算のキャッシュ値を基準スコアとして流用しない。
主学習・比較キュー・正式な点読み出しは変更していない。

根拠：[実装](exploration/temporal_feature_probe.py)、[実行前の計画とcheckpointのSHA](results/temporal_feature_probe_seed17_v1/plan.json)。

## 結果

位置指標は256画素基準。PCKは正解で可視の点の誤差が閾値未満の割合で、8シーンの等重み平均。
AJは0–100表記。全条件が常に可視と予測し、最初のquery時刻はmotionの採点から除外する。

| 特徴 | 条件 | 全可視点PCK4 | PCK16 | 16画素超移動点PCK4 | 同PCK16 | AJ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 第4層 | SigLIP2単独 | 37.08% | 68.24% | 5.17% | 36.03% | 28.62 |
| 第4層 | Mamba・履歴あり | 35.12% | 67.62% | 5.20% | 35.55% | 26.04 |
| 第4層 | Mamba・毎フレーム消去 | 36.54% | 67.64% | 5.08% | 35.32% | 28.06 |
| 最終層 | SigLIP2単独 | 22.56% | 51.66% | 3.39% | 26.73% | 16.69 |
| 最終層 | Mamba・履歴あり | 14.53% | 43.91% | 1.82% | 22.63% | 11.19 |
| 最終層 | Mamba・毎フレーム消去 | 22.67% | 50.42% | 2.91% | 25.53% | 16.46 |

初期位置を答え続ける共通基準は、全可視点PCK4 39.11%、PCK16 57.46%、AJ 30.67、16画素超移動点PCK16 0%だった。
すべての点で一律に記憶が悪いという結果ではないが、全体指標や最終層の大変位点では改善を確認できなかった。

## 長い区間だけの問題か

結果を見た後の追加分析として、query後の1–7、8–15、16–23サンプルに分けた。
これは事前に固定した主要検定ではなく、記述的な診断である。
1–7はqueryを含めて8フレームなので学習時のフレーム数の範囲内だが、
SSv2の画像分布や元フレームのサンプリング間隔とは異なる。

| query後のサンプル | SigLIP2単独PCK4 | Mamba履歴ありPCK4 | 毎フレーム消去PCK4 |
| --- | ---: | ---: | ---: |
| 1–7 | 32.77% | 25.50% | 33.19% |
| 8–15 | 21.99% | 12.19% | 21.86% |
| 16–23 | 12.78% | 5.99% | 12.73% |

表は最終層・上位4方式・全可視点。後半ほど対象の変位や可視性も変わるため、区間の絶対値の低下を忘却量とは呼ばない。
ただし同じ区間内での履歴あり対消去の差は、1–7でも残る。
「24フレームへの延長だけが問題」という説明では、この差を扱えない。

正解対応位置で特徴を採取して調べる追加分析でも、最終層の初期queryとのcosineは、
履歴あり0.582、状態消去0.698だった。正解座標を用いるのはこの診断の採点だけで、保存した予測は変更していない。
この差は履歴に伴う外観特徴の変化と整合するが、学習した別の読み出しでも情報が取り出せない証明ではない。

全区間、全方式、全層の値を[追加分析](results/temporal_feature_probe_seed17_v1/history_bands.json)へ保存した。
全区間の再集計が元の予測スコアと一致することも検査した。

## 検証と限界

- [CPUテスト2件](results/temporal_feature_probe_tests.log)で、特徴取得の位置、初期フレームの一致、
  状態消去した出力と独立した1フレーム推論の一致を確認した。小さなAttentionモデルと画像のみのモデルを用いた検査である。
- 実際のMamba推論では、3挿入層の各4個のnative状態を保持し、履歴ありの時刻が1–24、消去側が毎回1となることを確認した。
  状態量は両Mamba条件で79,027,200 bytesで一定だった。SigLIP2単独の時間状態は0。
  query特徴や一時メモリはこの状態量に含めない。
- [検証コード](exploration/verify_temporal_feature_probe.py)でRGBの前処理を再実行し、元注釈・query選択・入力ハッシュと照合した。
  保存特徴から1,152回のCPU照合を再実行し、48個の保存予測と一致した。
  768行の位置指標・AJとシーン平均も独立した式で再採点した。
- [検証記録](results/temporal_feature_probe_seed17_v1/verification.json)は、GPUエンコーダの再推論を含まない。
  評価中にエンコーダの重みが変わらなかったことは実行側でも検査した。

この探索は学習途中の1 checkpointと観察済み8シーンに限定される。
固定cosine照合は特徴の座標系の変化にも影響されるため、低い値からエンコーダ一般の表現力を否定しない。
一方、分類の改善を、そのまま細かな対応や記憶の改善へ読み替える根拠も得られなかった。
GPUは主学習と共有しており、所要時間は速度ベンチマークとして使用しない。

今回の設計判断は、**履歴を入れるだけで精密な対応が良くなるという前提を置かない**こと。
外観の対応付けに使える情報と、動作分類に役立つ時間文脈を同時に保てるかが次の検証点となる。
正式な長期学習・学習する点読み出し・Attentionとの比較を継続し、この診断だけで主構成を採否しない。
IROSに向けた性能上の貢献は未立証である。

## 再実行

リポジトリのルートで、特徴キャッシュ・RGBシーン・主学習checkpoint・前段の照合診断を用意する。
次のコマンドは実行時の`last.pt`を固定するので、4,000更新と異なる重みなら別の実験になる。
同じ結果の再現には同じcheckpointが必要で、GitHubへ大きな重み・画像・特徴・予測配列は含めない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_temporal_feature_probe.py
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.temporal_feature_probe --video-run results/ssv2_full_mamba_interleaved_seed42_rgbfix --matching-run results/continuous_matching_seed17 --output results/new_temporal_feature_probe --plan-only
uv run --locked python -m exploration.temporal_feature_probe --video-run results/ssv2_full_mamba_interleaved_seed42_rgbfix --matching-run results/continuous_matching_seed17 --output results/new_temporal_feature_probe --resume
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_temporal_feature_probe --run results/new_temporal_feature_probe
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.analyze_temporal_feature_probe --run results/new_temporal_feature_probe
```

`--resume`は未実行の固定計画を開始する指定で、部分実行からの再開ではない。
[全結果](results/temporal_feature_probe_seed17_v1/summary.json)には、中心・上位1・上位4・初期位置維持の全値を残した。
