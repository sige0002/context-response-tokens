# 時間認識とAttention比較を先に実行する

2026-11-05。研究対象は、SigLIP 2の画像表現に履歴を取り込み、時間Attentionに対して
認識精度・実測時間・メモリの面で有利な視覚エンコーダを作ること。
細部の位置対応を改善する探索を主目的にしていた優先順位を修正した。
この変更は研究成果や性能優位の報告ではない。

## 実行順の変更

以前の列では、点追跡9本とその履歴評価が、費用測定、複数深度GMMix、追加seedの動画比較の前にあった。
新しい[実行コード](run_temporal_priority.py)と[固定計画](results/temporal_priority_seed42_rgbfix/plan.json)では、
既存の動画比較が完了した後、次の順で実行する。

| 順序 | 実行内容 | 分かること |
| --- | --- | --- |
| 1 | 5方式の全SSv2認識・履歴介入を再集計 | 通常履歴の逐次精度、過去の置換・順序変更の影響 |
| 2 | 5方式の実動画での費用測定 | 同じ52・245・1,337フレームでの速度・状態量・割当メモリ |
| 3 | 複数深度GMMixの動画学習・認識・費用測定 | 近接研究の複数層特徴も含む6方式比較 |
| 4 | 主要4方式をseed 43・44で追加学習・認識評価 | 元のseed 42と合わせた3 seedの変動と対応付き差 |

主比較は中間層Mamba-3と中間層時間Attention。出力側Mambaと複数深度GMMixは追加比較である。
履歴なしと最終層GMMixは元の計画どおりseed 42のみ。
学習データ、画像モデル、入力、読み出し、20,000更新、最終checkpoint選択、seedの集合は変更しない。
追加の動画学習は新しい出力先を使う。旧計画・結果は削除しない。

点追跡の学習・履歴評価・旧統合集計など、未開始の待機キュー9プロセスを停止した。
[変更記録](results/temporal_priority_seed42_rgbfix/rescheduling.json)で、対象の実プロセス、未開始状態、
停止後の状態、本学習と既存動画比較3プロセスが維持されたことを確認できる。
元の点追跡計画は補助評価の記録として残すが、自動実行しない。
新しいキューが待つのは既存の動画比較であり、点追跡の完了ではない。

## 検査と未検証の範囲

[8件のCPU検査](results/temporal_priority_tests.log)では、実行先の変更、同名prefixを含む入力パス、
更新数・seed・評価checkpointの取り違え、点追跡コマンドへの置換、ソース変更の拒否を確認した。
[実プロセスを使った検査](results/temporal_priority_qualification_v2/verification.json)では、
実際のGMMix待機プロセスへpidfdで結び付き、検査用キューだけを停止し、
本学習と依存キューのプロセスID・開始時刻が変わらず生存していることを確認した。
検査は動画学習や費用測定を起動していない。

新しい列はソースと元計画を照合し、全予定更新・全検証の完了を確認してから評価を受け入れる。
既存のGPU計算が残っていれば新しいGPUジョブを起動しない。
一部だけ書かれた費用測定を、正常完了として読み飛ばしたり、自動削除したりしない。
旧計画に含まれる点追跡関連のリンクは由来の照合に残るが、点追跡のデータ準備・学習完了は実行条件ではない。

SSv2は短い動画から8観測を選ぶ評価であり、長期記憶の能力を実証する課題とは区別する。
長い実動画で状態量と時間を測っても、遠い過去の情報を認識に利用できた証明にはならない。
長期の認識能力に関する評価は未完了。現在は対照モデルへの性能優位も未確定である。

## 再実行

リポジトリのルートで、既存の5方式・費用・複数深度・追加seedの固定計画がある状態を前提とする。
引数の既定値は今回の計画の場所であり、別の実行先では`--video`、`--gmmix`、`--costs`、
`--multidepth`、`--replications`で対象を指定する。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q test_temporal_priority.py
CUDA_VISIBLE_DEVICES='' uv run --project baselines/gmmix --locked python run_temporal_priority.py --output results/new_temporal_priority --plan-only
# GMMIX_QUEUE_PIDは対応するrun_gmmix_comparison.pyの実プロセスを確認して設定する。
uv run --project baselines/gmmix --locked python run_temporal_priority.py --output results/new_temporal_priority --resume --after-gmmix-pid "$GMMIX_QUEUE_PID"
```

既存の動画比較が正常完了済みの場合だけ、`--after-gmmix-pid`を省略できる。
実行中の依存プロセスが見つからないことだけでは、学習完了と判断しない。
計画だけを作成するCPU指定を、GPUジョブを実行するコマンドへ付けない。
