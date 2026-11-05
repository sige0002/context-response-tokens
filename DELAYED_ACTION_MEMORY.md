# 共通の後続映像を使う遅延保持診断

2026-11-05。SigLIP 2＋時間Mambaと時間Attentionについて、必要な過去の情報が
現在の特徴から読み出せるかを調べる補助評価を実装した。
自然な連続動画での認識評価を代替するものではなく、独自アーキテクチャの貢献とも呼ばない。

## 課題と対照

各例は、SSv2の動作を表す8観測と、別動画から作った共通の後続映像を連結する。
答えは「最初の8観測で起きた動作」の174分類。後続映像の動作ラベルへ切り替えない。
同じブロックでは174クラスの各1例に、完全に同じ後続映像を与える。
0・1・8・32・128観測の遅延で、現在の196トークンを平均した768次元特徴を記録する。

正の遅延では、現在画像も後続映像全体もブロック内の全クラスで同一である。
したがって、これらだけに依存する決定的な174分類器は、各ブロックで必ず1/174正解となる。
これはデータ構成による性質であり、学習済みモデルの測定精度ではない。
ブロックID・例の順番・ラベル・過去の出力は読み出しへの入力にしない。

| 条件 | 現在の読み出しへ渡すもの | 役割 |
| --- | --- | --- |
| full_memory | 証拠から後続映像まで状態を維持した現在特徴 | 遅延後の情報の読み出し |
| suffix_only | 証拠終了時に状態を消去し、同じ後続映像だけで作った特徴 | 証拠なしの対照。絶対観測番号は維持 |
| saved_evidence | 証拠終了時の特徴を別に保存して渡す | 証拠自体を読めるかの参照。提案モデルには含めない |

既存の動画分類器は最新の動作を答えるよう学習されているため、そのまま使わない。
エンコーダは固定し、各条件に同じ容量のridge線形読み出しを学習する。
各条件で遅延をまとめて学習し、遅延値は読み出しへ入力しない。
中心化は学習集合のみ、正則化係数は1に固定し、検証精度による選択はしない。
suffix_onlyは証拠終了時の0遅延に対応する入力がないため、正の遅延だけを扱う。

## 固定したデータと実行順

[データ計画](results/delayed_action_memory_v1/data_plan.json)は、元の公式train/validationから
スコアを見ずにseed 71で選んだ。学習4ブロック・696証拠動画、検証2ブロック・348証拠動画。
後続映像用にはそれぞれ64・32動画を使い、証拠と重複させない。
元データ全体と選んだ集合で、ID・元動画group・SHAによる分割間の重複を検査する。
各クラスの証拠動画もブロックごとに異なる。

[データ生成](exploration/delayed_action_data.py)、
[特徴抽出](exploration/cache_delayed_action.py)、
[読み出し学習](exploration/delayed_action_probe.py)、
[比較の実行コード](exploration/run_delayed_action.py)を分けた。
正式な特徴抽出は、予定した20,000更新を完了したcheckpointだけを受け付ける。

[比較計画](results/delayed_action_memory_v1/comparison_v2/plan.json)は、
中間層Mambaと時間Attentionのseed 42・43・44、計6本をすべて実行する。
既存の[時間認識・費用・追加seedの実行列](TEMPORAL_PRIORITY.md)の完了を待ち、
速度測定や学習へ重ならない順序にした。現時点では6本とも未実行。
結果の良し悪しによる採否選択はせず、各遅延の生値と同じseed間の方式差を保存する。

## 実施した検査

[CPUの6検査](results/delayed_action_tests.log)で、共通後続映像の正解率制約、
検証特徴が学習済み読み出しを変えないこと、ラベルと特徴の対応、部分集合の混入拒否、
分割漏洩、実際の小型Attentionによる状態維持・消去を確認した。

[実画像の動作確認](results/delayed_action_memory_v1/qualification/summary.json)では、
保存済み4,000更新モデル、検証の二つの証拠動画と16本の共通後続動画を使った。
128観測後まで動作し、正の各遅延で現在RGBが二つの例間で一致した。
証拠を消去した後続映像の再実行は特徴が完全一致し、全重みは変化しなかった。
Mambaの状態は各遅延とも79,027,200 bytesだった。
異なる証拠からの特徴差は残ったが、意味情報を読み出せる証拠ではない。
読み出し学習・認識スコア・Attentionとの性能比較は、この動作確認では実施していない。
部分的な動作確認データは、正式なprobeの学習入力として拒否する。
[18本の元動画をCPUで再デコードした検証](results/delayed_action_memory_v1/qualification/verification.json)では、
144枚の入力、元ラベル、各遅延の現在画像hash、保存した特徴の対照を再照合した。
GPU特徴の独立再実行ではない。
[動作確認データの拒否検査](results/delayed_action_rejection_verification.json)では、
読み出し学習が開始されず、出力も作られないことを確認した。

待機処理の初回実行は、このuv Pythonに`os.pidfd_open`がないため失敗した。
libc経由の同じ機構へ修正し、旧失敗記録を保持して`comparison_v2`へ移行した。
[実プロセスでの停止検査](results/delayed_action_queue_cancel_verification.json)では、
待機中の自分のキューだけを停止し、既存の学習・比較4プロセスが同一のまま生存することを確認した。
[本キューの起動記録](results/delayed_action_queue_activation.json)には待機先と計画hashを保存した。

## 解釈の限界

- 独立した動画を連結している。128観測は自然動画の128秒を意味しない。
- 課題は観測前から既知であり、任意の後発質問に答える記憶の検証ではない。
- 最初の証拠特徴だけを保存すれば解ける課題なので、この診断単独でMambaの優位性や新規性を主張しない。
- エンコーダの学習時は8観測だった。長い入力では長さの外挿も含み、低いスコアを方式の本質的限界と断定しない。
- 線形読み出しの制約と少ない学習例により、保持情報があっても読めない可能性がある。
- 検証集合は過去に観測済みのSSv2 validationで、未使用の外部testではない。2ブロックから精密な信頼区間を作らない。

自然動画での証拠区間・質問時刻を持つ評価は、引き続き別に必要である。
[EgoProbeの一次資料](https://arxiv.org/html/2610.17128v1)は、証拠終了から質問までの遅延と、
途中の類似イベントによる混同を分けている。ただし同版はコード・データを公開予定と記しており、
実行済み評価として扱わない。この診断の完了によって自然な長期記憶評価を完了扱いにはしない。

## 再実行

ルートから実行する。元動画・SigLIP 2のキャッシュ・学習済みcheckpointが必要。
保存済み計画には実行時の絶対パスが含まれるため、別の配置ではデータ計画を新しく生成する。
`<final-checkpoint>`は同じデータ分割で20,000更新を完了したcheckpointへ置き換える。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_delayed_action.py
uv run --locked python -m exploration.delayed_action_data --output results/new_delayed_data.json
uv run --locked python -m exploration.cache_delayed_action \
  --plan results/new_delayed_data.json --checkpoint "<final-checkpoint>" \
  --output results/new_delayed_cache
OPENBLAS_NUM_THREADS=2 uv run --locked python -m exploration.delayed_action_probe \
  --cache results/new_delayed_cache --output results/new_delayed_probe
```

本番の特徴抽出は他のGPU実験がない時間に行う。`--qualify`は二例だけの動作確認で、
未完了checkpointを許すが、その出力を正式な読み出しへ渡すことはできない。
特徴配列・重み・元動画はGitへ含めず、コード・計画・小さな検査記録を公開する。
