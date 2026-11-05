# 時間間隔と遮蔽例の重み付けを分けた診断

2026-11-05。**短期学習の時間間隔を広げても、遮蔽例を重くしても、今回の点追跡AJは改善しなかった。**
遮蔽例の重み付けで見逃しは減ったが、可視点を不可視とする誤りが増えた。
この変更は主実験へ採用しない。長い履歴を使うMambaの性能を否定する結果ではなく、
[凍結画像特徴からの読み出し対照](CORRELATION_READOUT_PILOT.md)に限った診断である。

## 切り分けた仮説

前の対照では、queryの次の3時刻だけで学習し、検証では23時刻先まで照合していた。
学習で不可視点が少ないことと、時間間隔の不一致を区別するため、2×2条件を固定した。

| 要因 | 条件1 | 条件2 |
| --- | --- | --- |
| queryからの間隔 | 次の3キャッシュ時刻 | 後続から異なる3時刻を抽出 |
| 可視性BCE | 通常の平均 | 学習列のクラス頻度の逆数で重み付け |

queryとなるシーン・時刻・32点、初期重み、1024更新、損失の他の項は共通である。
広い間隔を選ぶ乱数列は別にし、queryの抽出列が変わらないようにした。
可視性の重みは予定した学習列だけから計算し、検証データで調整していない。
出力の可視性logitの判定閾値は全条件で0。検証後の閾値選択はしていない。

学習24・検証8シーン、SigLIP 2-B/16の凍結した最終特徴、264132パラメータの読み出し、
CPU FP32・2スレッドを使用した。8検証シーンは既に観察したKubricレンダリング画像であり、未使用testでも実写評価でもない。
Mamba・再帰状態・エンコーダの更新・蒸留はこの診断に含めない。

元フレームの4フレーム間隔で作ったキャッシュを使う。次の3キャッシュ時刻は元画像の4・8・12フレーム先であり、
連続する元フレームではない。広い条件は最大92フレーム先を含む。

| 学習列 | 点時刻数 | 不可視率 | 平均間隔（キャッシュ時刻） | 16画素超移動した可視例 |
| --- | ---: | ---: | ---: | ---: |
| 短い間隔 | 98304 | 6.85% | 2.00 | 10617 |
| 広い間隔 | 98304 | 15.39% | 7.09 | 26158 |
| 検証 | 23552 | 15.33% | 12.00 | 8021 |

不可視率は近づいたが、検証の時間間隔分布と同一になったわけではない。
記録：[学習前の計画](results/correlation_horizon_seed17/plan.json)、
[全query・時刻の列](results/correlation_horizon_seed17/schedule.json)。

## 最終1024更新の結果

PCKは256画素基準の位置閾値未満の割合で、正解で可視の点を対象とする。
PCK・AJは8シーンの等重み平均。遮蔽のrecall・precisionは全点時刻をまとめた値で、**不可視を陽性**とする。

| 条件 | 全可視点PCK4 | 全可視点PCK16 | 16画素超移動点PCK16 | AJ | 遮蔽recall | 遮蔽precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 短い間隔・通常 | 8.60% | 48.31% | 24.59% | 10.82 | 0.00% | 判定なし |
| 広い間隔・通常 | 7.80% | 47.63% | 24.18% | 10.25 | 0.06% | 16.67% |
| 短い間隔・重み付け | 8.12% | 48.24% | 24.52% | 9.99 | 47.05% | 26.77% |
| 広い間隔・重み付け | 7.88% | 47.60% | 24.18% | 10.11 | 32.98% | 32.34% |

AJは0–100表記。初期位置を答え続け、常に可視とする基準はAJ 30.67、全可視点PCK4 39.11%、
16画素超移動点PCK16 0%だった。全体値と移動点を分けて読む必要がある。

正解の不可視点時刻は3611、可視点時刻は19941だった。

| 条件 | 正しく不可視と判定 | 可視なのに不可視と判定 | 不可視を見逃し |
| --- | ---: | ---: | ---: |
| 短い間隔・通常 | 0 | 0 | 3611 |
| 広い間隔・通常 | 2 | 10 | 3609 |
| 短い間隔・重み付け | 1699 | 4648 | 1912 |
| 広い間隔・重み付け | 1191 | 2492 | 2420 |

「全件を可視と答えた」ことと「特徴に可視性の情報が全くない」ことは区別する。
閾値を選ばずスコアの順位を測った遮蔽AUROCは、通常・短い間隔で全件集計0.732、シーン平均0.652だった。
重み付け・短い間隔では0.695、0.614で、順位付けの改善も観察されなかった。
全件集計とシーン平均の差があるため、全件AUROCだけから各シーンでの判別能力を主張しない。

[0・256・1024更新の全位置指標](results/correlation_horizon_seed17/summary.json)、
[全時点・シーンの可視性集計](results/correlation_horizon_seed17/occlusion_analysis.json)、
[学習損失](results/correlation_horizon_seed17/training.jsonl)を公開した。

## 検証と設計判断

- 短い間隔・通常の条件は、前の対照の最終重み全テンソルと、全評価時点のスコアが完全一致した。
- 学習キャッシュからquery列、間隔、可視性の件数と重みを再生成し、保存した計画との一致を確認した。
- [再採点](results/correlation_horizon_seed17/verification.json)で96個の予測ファイルを元注釈と照合し、
  576行を独立したNumPyの式で再計算した。PCK差は最大2.98e-8、平均誤差差1.15e-5、AJ差0。
  シーン平均の集計も確認した。保存予測の再採点であり、モデル推論の再実行ではない。
- [2件の検査](results/correlation_horizon_tests.log)で、同一query・因果的な時刻抽出・重みの釣り合いと、
  同点を含むAUROCの全組合せ計数との一致を確認した。

**今回の予算では、時間間隔とクラス頻度の変更だけでは弱い読み出しを解消できなかった。**
長く学習した場合まで無効とは結論しない。可視性の校正と、細かな座標を特徴から取り出す能力が残っている。
次は、学習済み読み出しとは別の局所特徴照合で細かな位置を復元できるかを切り分ける。
それができなければ、時間記憶だけを変えた結果を精密認識の改善として解釈しにくい。
この診断の完了を、新しいエンコーダ原理やIROS投稿に足る性能改善の達成とは扱わない。

## 再実行

リポジトリのルートで実行する。先に[前の対照](CORRELATION_READOUT_PILOT.md)の特徴キャッシュと参照実験を再現し、
未使用の出力先を指定する。保存した計画の絶対パスは実施履歴であり、別環境ではコマンドから作り直す。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_correlation_horizon.py
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.correlation_horizon --reference-run results/new_correlation_readout --output results/new_correlation_horizon --plan-only
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.correlation_horizon --reference-run results/new_correlation_readout --output results/new_correlation_horizon --resume
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_correlation_readout --run results/new_correlation_horizon
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.analyze_correlation_horizon --run results/new_correlation_horizon
```

`--resume`は未実行の固定計画を開始する指定で、途中の学習からの再開ではない。
GitHubにはコード・計画・query列・集計・検証記録を公開し、特徴・重み・予測配列は含めない。
