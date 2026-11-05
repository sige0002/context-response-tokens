# 近接研究との比較で揃っている点と不足

2026-11-05時点。IROS向けの性能主張には、同条件の対照実験と、近接研究の有力な構成との比較が必要である。
現在の5方式の実験は前者の一部であり、それだけで既存手法への優位性を結論しない。

## SSv2の公表値と比較条件

[原論文v1の表1・§4.2・図2](https://arxiv.org/html/2605.19137v1#S4.SS2)には、
次のstreaming評価がある。画像エンコーダを凍結し、時間モジュールと読み出しを学習する。
174クラスの動画分類で、最後のフレームの出力からTop-1正解率を測る。

| 凍結画像エンコーダ | 時間モジュール | 学習更新数 | SSv2 Top-1 |
| --- | --- | ---: | ---: |
| DINOv3-L | Mamba | 20,000 | 63.3% |
| DINOv3-L | MambaMix | 20,000 | 66.4% |
| DINOv3-L | GatedMambaMix | 20,000 | 66.9% |
| DINOv3-L | RVMの再帰コア | 20,000 | 67.1% |
| SigLIP2（規模・重みの一致は未確認） | GatedMambaMix | 20,000 | 65.0% |

[学習設定の表5・7](https://arxiv.org/html/2605.19137v1#S7)はstreamingを20,000更新、
warmupを1,000更新とする。これは画像モデルの事前学習を含まない。
同論文の100,000更新はofflineのRNN fine-tuning、40,000更新はofflineの読み出しのみの学習であり、
上表の学習予算と混同しない。

SigLIP2＋GMMixの65.0%も参照値として扱う。DINOv3とSigLIP2の違いだけで、こちらとの精度差を説明しない。
ただし、原論文と固定した公開コードから、SigLIP2の正確なモデルID・重み・学習batchを確認できていない。
この行を、こちらの`google/siglip2-base-patch16-224`と条件を揃えた対照として扱わない。
下記の同じ画像重み・8フレーム・更新予算を使う追加対照によって、実装内での差を測る。

## GatedMambaMixの複数層特徴

[原論文§3.2・Fig. 3](https://arxiv.org/html/2605.19137v1#S3.SS2)は、画像エンコーダの4深度から特徴を取り出し、
深度ごとの残差MLPを通して平均する。現在予約しているGMMix対照は最終層の特徴だけを使うため、この部分が異なる。
公表値の再現としても、同論文の最良構成に勝った証拠としても使えない。

| 項目 | 現在の予約済み対照 | 追加する複数層対照 |
| --- | --- | --- |
| 画像重み | 同じSigLIP 2-B/16 224 | 同じSigLIP 2-B/16 224 |
| 取り出す特徴 | 最終層 | 第3・6・9・12層 |
| 深度ごとの正規化 | 最終LayerNorm | 凍結した最終LayerNormを各深度へ適用 |
| 深度ごとの適応 | なし | 公開実装のBatchNorm＋残差MLP |
| 深度の統合 | なし | 4深度の算術平均 |
| 時間コア | 公開GMMix、4層 | 同じ公開GMMix、4層 |
| 空間出力数 | 196 | 196 |
| 実行状態 | 全学習キューで待機 | 数値検証を分離して完了、非決定性を明記して全学習キューへ登録 |

公開コードで照合した箇所は、固定commit `59249bf83311bc34bae277e2e8adec287ffe5d0f` の
[DINO wrapper](https://github.com/tue-mps/towards-video-image-frozen/blob/59249bf83311bc34bae277e2e8adec287ffe5d0f/models/image_encoder/dino_wrappers.py)と
[推論設定](https://github.com/tue-mps/towards-video-image-frozen/blob/59249bf83311bc34bae277e2e8adec287ffe5d0f/scripts_inference/utils.py)である。
公開推論経路はDINOv3用で、CLS・registerも利用する。追加実装は、その特徴統合処理をSigLIP 2へ適用した対照である。
SigLIP 2には同じCLS・register列がないため、公開DINOv3モデルと同一とは呼ばない。
公開SSv2推論例は16フレーム・6 FPSを指定しており、現在の8フレーム設定とも異なる。
これは公開推論例の条件である。コードには6 FPSを学習時のサンプリング率に合わせるという注記があるが、
それだけで論文の全学習・評価におけるフレーム数、batch、増強方法を確認済みとはしない。

**観測する時間範囲も一致していない。** 固定した公開コードの
[`load_video_frames`](https://github.com/tue-mps/towards-video-image-frozen/blob/59249bf83311bc34bae277e2e8adec287ffe5d0f/scripts_inference/utils.py#L103)は、
元FPSと6 FPSの比を整数に丸めた間隔で動画中央から16枚を取る。短い動画では最後に取得したフレームを反復して埋める。
一方、こちらの[`temporal_indices`](video_data.py)は、最終フレーム以前を7区間に分け、
評価時は各区間の中央1枚と動画そのものの最終フレームを取る。
例えば60フレーム・12 FPSの動画なら、公開推論例は`[14, 16, …, 44]`、
こちらは`[3, 11, 20, 28, 37, 45, 54, 59]`となる。
これはサンプリング規則を照合した例であり、認識精度の実験ではない。
両方が「最後の入力フレームから分類」していても、その入力が同じ動画時刻とは限らない。
したがって公表値との違いをフレーム数だけで説明せず、同条件の対照では取得位置と最終観測も固定する。

[gmmix_multidepth.py](gmmix_multidepth.py)は、公開の`_IntermFeatMLP`を変更せず、ソースハッシュを確認して読み込む。
空間特徴の抽出はFP32・勾配なし、4個の特徴MLPとGMMixコアは学習対象とする。
現在の追加構成では特徴MLPもFP32、GMMixコアはBF16で計算する。公開論文の全処理BF16と同じ精度条件とは呼ばない。
特徴MLPを誤って`no_grad`内へ入れない。独立したcheckpoint形式で、学習済みBatchNormの統計も保存する。
既存5方式のコード・実行計画は変更していない。

**BatchNormの解釈：** 公開処理は学習中にbatch・time・patchをまとめて統計を計算する。
この実装もそれに従うため、学習中の過去フレームの特徴は、同じクリップの後続フレームから影響を受け得る。
評価モードでは固定した統計を使い、未来フレームを参照しない。
「学習・推論ともに同一の因果的処理」とは記述しない。
また、勾配蓄積による実効batch 32は、BatchNormも一度に32動画を見るという意味ではない。microbatch 4も比較条件として固定する。

[CPU検査](test_gmmix_multidepth.py)では、実際の小型SigLIPの中間出力とTransformersのhidden statesを照合し、
特徴統合を公開DINO wrapperと比較した。さらに4個のMLPへの勾配、凍結重みの保持、評価時の未来非依存性、
学習時のBatchNorm依存性、保存・復元を検査した。FP32の設定後は、autocast中のMLP演算精度も含めて
[7件が通過した](results/gmmix_multidepth_cpu_tests_fp32.log)。
この検査の時間コアは明示した代用品であり、native GPUの検証を代替しない。
[実サイズでの構築とパラメータ数](results/gmmix_multidepth_architecture_fp32.json)もCPU上で記録する。

[学習](train_multidepth_video.py)、[checkpoint復元](multidepth_checkpoint.py)、
[評価](evaluate_multidepth_video.py)の入口を追加した。データ、最適化、読み出し、評価関数は既存実験と共通である。
実行中のソース固定を保つため、学習ループは既存の最終層GMMix trainerから独立した入口へ移した。

最初の[GPU結合検査](results/gmmix_multidepth_training_integration/progress.json)は、
3更新後の中断再開との差が最大`1.584e-4`で、事前の重み差`5e-6`基準に届かなかった。
失敗記録を保持し、許容値を緩めて通過扱いにはしていない。
[最初のバッチの切り分け](results/multidepth_first_step_comparison.json)では入力、初期重み、記録した全順伝播出力が一致し、
差はMambaコアの`A_log`、`D`、`dt_proj.bias`の勾配12個に限られ、最大約`1.31e-10`だった。
この小さい差だけで3更新後の差を説明したとは結論しない。
公開selective-scan CUDA逆伝播には該当パラメータへのatomic加算があるため、PyTorchの決定的設定だけで全拡張のビット一致を保証しない。

特徴統合のFP32化後の[再検査](results/gmmix_multidepth_training_integration_fp32/progress.json)も、
再開との差が最大`1.564e-4`となり、同じ重み差基準に届かなかった。
[独立再学習との差](results/gmmix_multidepth_training_integration_fp32/repeatability_diagnosis.json)も最大`1.495e-4`である。
特徴MLPのFP32化だけでは解消せず、再開処理だけの問題とも判断できない。

[全3更新の追跡コード](trace_multidepth_training.py)で本来のtrainerを変更せず観測し、
独立した2プロセスを[比較した](results/multidepth_trace_comparison.json)。
全9回の入力とCUDA乱数状態は一致し、最初の2回の順伝播でも記録した14箇所の出力が一致した。
一方、最初のoptimizer更新前に既に62個の勾配テンソルが異なり、更新後は41個のパラメータが異なった。
保存した第1ゲート重みの差は、勾配で最大`2.573e-8`、更新後の重みで`2.366e-6`だった。
次の順伝播ではlogit差が最大`0.001953125`になった。
保存・復元を行う前から差が発生することを確認した。
これらは観測用hook付きの短い実行であり、速度の測定や低い認識精度の原因確定には使わない。

### 数値の正しさと再現性を分けた検証

続く[同一入力のCUDA逆伝播反復](results/multidepth_scan_diagnostic/report.json)では、
実動画2バッチで使った8回のscanについて、同じ入力を各8回ずつ演算した。
`A`・`D`・`delta_bias`に加え、一部の`B`・`C`勾配にも差を確認した。
公開CUDAソースはこれらをatomic加算し、`B`・`C`の蓄積後に入力のBF16へ変換する。
PyTorchの決定的設定を指定していても、この公開拡張のビット一致を保証しない。
同じ拡張を使う最終層特徴のGMMix対照にも、この実装上の制約はある。
そちらの短い再現性検査が許容値内だったことを、全学習のビット一致保証へ拡張しない。

以下の検証を追加した。

| 検証 | 結果 | 範囲 |
| --- | --- | --- |
| [独立したFP64漸化式との照合](results/multidepth_reference_resume/kernel_reference.json) | native FP32の順伝播相対L2差`1.16e-7`、8入力の勾配は最大`9.18e-7` | 実動画から取得した2系列・8時刻。全入力域の保証ではない |
| [参照scanでの連続・再開・独立再学習](results/multidepth_reference_resume/verification.json) | 重み、optimizer、検証履歴、CPU/CUDA RNG、整数bufferが完全一致 | 同じtrainerでscanのみ公開PyTorch参照演算へ置換。3更新・epoch端数を含む |
| [native保存モデルの再評価](results/gmmix_multidepth_qualification/verification.json) | trainerと評価CLIの保存予測・損失が一致 | 3動画。認識精度の証拠にはしない |
| 同モデルの一括・逐次推論 | 特徴の相対L2差`0.001960`、未来変更による過去出力差`0` | 実動画1本・8時刻。状態96,337,920 bytesで一定 |

この証拠に基づき、**全学習へ進む判定を、nativeの重み差単独から、演算の正しさ・復元処理・非決定性を分ける判定へ変更した。**
以前の`5e-6`基準は失敗のまま保存し、閾値を書き換えて通過したとは記録しない。
nativeでの検証損失差の既存基準`0.002`、同じ予測、因果性、一括・逐次の特徴差基準は維持した。
この判断は、短い実装検査で公開CUDAを比較実験に使うためのものに限る。
長期学習の安定性やseed間分散、認識精度の向上を証明したものではない。

[qualify_multidepth.py](qualify_multidepth.py)が別の検証記録を作り、
[実験計画](results/ssv2_gmmix_multidepth_comparison_seed42_rgbfix/plan.json)には
`native_bitwise_training=false`と元の重み検査失敗を明記した。
`torch.use_deterministic_algorithms(True)`は設定値として保持するが、全カーネルの決定性を意味しない。
参照演算への置換は診断専用で、予約した本学習・速度比較は公開native CUDAを使う。

[全件比較キュー](run_multidepth_comparison.py)の旧計画には点追跡への依存があった。
現在は[時間認識を優先するキュー](run_temporal_priority.py)へ移行し、点追跡を待たずに、
同じ20,000更新、8フレーム、microbatch 4・蓄積8で追加対照を学習・評価する。
[計画・完了・集計の検査5件](results/multidepth_experiment_protocol_tests_complete.log)も通過した。
全データ学習が済む前に、この追加対照の精度や速度を研究結果として報告しない。

### 共通点追跡への接続

以下の点追跡・点履歴診断は過去の実装検証記録である。現在の主評価や実行待ち項目ではない。
研究範囲は[時間認識・履歴利用の評価方針](TEMPORAL_RECOGNITION_EVIDENCE.md)に従う。

[点学習](train_multidepth_points.py)と[逐次評価](evaluate_multidepth_points.py)は、
既存対照と同じ点読み出し・教師損失・データ読み込み・公式指標を使う。
[専用checkpoint](multidepth_point_checkpoint.py)には、4段の特徴MLPとBatchNormの統計値を含む
全モデル状態を保存する。既存の実行中キューと形式・ソース記録を混同しない。

| 検査 | 実施範囲・結果 |
| --- | --- |
| [CPU 8検査](results/multidepth_point_cpu_tests.log) | 凍結と更新、BNを含む復元、異なるソース・カーネル記録の拒否。時間coreは明示的なstub |
| [native GPU結合検査](results/multidepth_point_integration/verification.json) | 実合成3シーン、probe 3更新。再開・独立再学習の重み、optimizer、損失、予測の差は0 |
| 同検査のadapt | 1更新で4段すべての特徴MLP・時間core・空間core・gateが更新。SigLIP本体は不変。BN更新回数は各2 |
| 実動画の評価 | RoboTAP 1動画・296フレーム・38点。保存予測を元注釈から再採点して一致。精度の主張には使わない |
| [計画・完了検査9件](results/multidepth_point_protocol_tests.log) | 更新予算・入力条件の共通化、未完了・誤った形式・不正な集計の拒否 |

[2実験の追加計画](results/multidepth_point_comparisons_seed42_rgbfix/plan.json)では、
24フレーム・64点・実効batch 8・20,000更新を固定した。
probeの一致はencoder backwardを実行しない条件の結果であり、native adaptの再現性へ拡張しない。
本学習したモデルでの転移精度、履歴利用、費用の結果はまだない。

### 費用・履歴の同条件評価

[追加診断キュー](run_multidepth_diagnostics.py)は、追加point学習の終了後に実行する。
費用は既存計画と同じ3動画・3回測定、履歴は同じquery記述子を共有する通常・reset-1・reset-24の全265動画である。
既存5方式の費用記録を再検証して6方式の対応付き比率を作る。
履歴評価の通常枝が元のpoint評価と一致することを検査し、各条件を元注釈から再採点する。

[実GPUの接続記録](results/multidepth_diagnostics_integration/verification.json)では、
52フレーム・2回の費用測定で最終出力が一致し、状態サイズは96,337,920 bytesだった。
296フレームの履歴評価では通常枝の座標・可視性logitの差が0だった。
いずれも3更新モデルの工学的検査である。
[計画と状態検査4件](results/multidepth_diagnostic_protocol_tests.log)も通過し、
[全件実行計画](results/multidepth_diagnostics_seed42_rgbfix_v2/plan.json)を保存した。

保存した履歴区間のJSON配列と、再集計したPython tupleを直接比較する不具合を修正した。
[実動画296フレームの再集計](results/history_json_roundtrip_regression.json)では、
数値の変更なしに保存形式へ揃えると一致した。
[キュー移行](results/multidepth_diagnostic_queue_migration_v2.json)はGPU操作開始前に行い、
同じデータ・条件・前段依存を保った。旧計画・停止記録は保持し、学習はやり直していない。

## 構造だけで主張できないこと

| 近接研究 | 既に存在する要素 | 今回必要な実証 |
| --- | --- | --- |
| [MEM](https://arxiv.org/html/2603.03596v1#S3.SS3) | ViT内のパッチ別時間Attentionと現在フレームの出力 | 同じタスク条件での精度・履歴費用。現在のAttention対照はMEM全体の再現ではない |
| [TRecViT](https://arxiv.org/abs/2412.14294) | 時間再帰と空間Attentionの交互適用 | 事前学習画像層を固定して使う場合の有効性と費用 |
| [SDQES](https://proceedings.neurips.cc/paper_files/paper/2024/file/b689b90ddf2b47d3103decabe6d47446-Paper-Datasets_and_Benchmarks_Track.pdf) | 凍結画像モデル内の状態を持つ時間アダプタ | アダプタ挿入そのものを新規性にせず、認識・転移での改善を示す |
| [Frozen Image＋GMMix](https://arxiv.org/abs/2605.19137) | 凍結画像基盤モデルと時間Mamba系の統合 | 複数層特徴を含む有力な対照に対する性能と計算費用 |

現状の研究仮説は、学習した時間更新を凍結空間層の途中へ戻すことにより、出力後の同規模Mambaより有用な表現を作れるか、である。
中間層Mambaと出力側Mambaの比較は挿入位置を分離する。一方、GMMixとの比較はパラメータ数も追加空間処理も異なるため、機構だけの比較にはならない。

複数seedの時間認識性能、履歴介入、長い履歴を必要とする認識、速度・メモリの結果が揃い、
ロボティクスでの履歴利用への有効性と近接研究との差を説明できるまでは、
IROS提出可能な成果や新しい一般原理を確立したとは判断しない。
