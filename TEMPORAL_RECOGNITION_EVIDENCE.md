# 時間認識の証拠と長期記憶評価の選び方

2026-11-05。目的はSigLIP 2の画像表現へ履歴を取り込み、時間Attentionに対して
認識精度と実用的な費用で優れるエンコーダを作ること。
細部の点対応や幾何の改善を、この研究の主な成功条件にはしない。

## 同じ現在画像に対する履歴介入

[評価実装](exploration/evaluate_video_history.py)は、保存済み4,000更新checkpointと、
各174クラスからseed 17で2本ずつ選んだ348動画を使う。
この集合は以前の層介入で観測済みであり、未使用testとは呼ばない。
全24,777本の通常検証とは母集団の重み付けも異なるため、両者の正解率を直接比較しない。

全条件でSigLIP 2、時間層、分類器、最後のRGB画像は同じ。
層を除去せず、入力履歴だけを変える。新しい学習は行わない。

| 条件 | 入力 | 調べること |
| --- | --- | --- |
| full_clip | 元の8観測を一括処理 | 以前の保存logitとの再現性 |
| full_stream | 元の8観測を順に処理 | 実運用の逐次経路と一括経路の差 |
| current_only | 最後の画像だけ、初期状態から処理 | 同じ学習済みモデルから過去情報を取り除く影響 |
| repeat_current | 最後の画像を8回処理 | 8回更新するだけの効果と、異なる過去観測の効果 |
| reverse_history | 過去7観測を逆順にし、最後の画像は維持 | 過去観測の順序への依存 |

`current_only`は画像だけで独立に学習した対照ではない。
また逆順化は動作の意味自体を変える場合がある。
元ラベルへの正解率低下を、そのまま新しい動画への誤認識や時間理解の証明とは解釈しない。
予測の変化件数・正解から不正解への変化・逆方向の変化も保存する。
区間はクラス内で対応を保つ2,000回bootstrap、探索的・多重比較未補正である。

元動画のhashと、以前の全348入力のRGB前処理hashを照合する。
同じcheckpointの重みが変化していないこと、一括経路の以前のlogitとの差、
逐次状態量を記録する。GPUを本学習と共有する診断なので実行時間の性能比較には使わない。
[独立したCPU再計数](exploration/verify_video_history.py)は元manifestのラベルと全予測を照合する。
これは各条件のGPU推論を独立に再実行する検査ではない。

## 実行結果

[固定計画](results/video_history_intervention_seed17/plan.json)、
[全条件の集計と全予測クラス](results/video_history_intervention_seed17/summary.json)、
[元ラベルからの独立再計数](results/video_history_intervention_seed17/verification.json)を保存した。
結果は次のとおり。通常履歴の30.17%は、この348本での4,000更新モデルの値である。
6,000更新時の全24,777本の46.20%とはcheckpointも評価集合も異なる。

| 条件 | 元ラベルへの正解数 / 348 | 正解率 | 通常逐次推論から予測が変わった本数 |
| --- | ---: | ---: | ---: |
| 通常履歴・一括 | 105 | 30.17% | 0 |
| 通常履歴・逐次 | 105 | 30.17% | 0 |
| 現在画像だけ | 26 | 7.47% | 306 |
| 現在画像を8回反復 | 27 | 7.76% | 288 |
| 過去7観測だけ逆順 | 93 | 26.72% | 110 |

この学習済みモデルでは、異なる過去観測を与えることで、現在画像だけ・現在画像の反復より
元の動作ラベルを多く正解した。現在画像だけへの置換では、通常履歴の正解87本を失い、8本を得た。
単に同じ画像を8回処理することだけでは通常履歴の成績に届かなかった。
逆順化で110本の予測が変わったため、順序にも依存している。
ただし逆順の意味・入力分布が変わるので、差をそのまま一般的な時間理解能力とは呼ばない。

以前保存した一括logitは全要素が完全一致した。
一括と逐次のlogitには最大絶対差0.0625があるが、この集合での予測クラスの相違は0本だった。
状態量は全条件・全観測で79,027,200 bytes（約75.37 MiB）。
これは短い動画での履歴利用と推論経路の診断であり、Attentionへの優位性や長期保持の結果ではない。

[3件のCPU検査](results/video_history_tests.log)では、最後の画像と入力順序の維持、
呼び出し間の状態初期化、予測変更件数と正解変化件数の区別を確認した。
独立再計数は全1,740件の予測クラスについて一致した。
重みの変更や新しいモデルの採用はしていない。

## 各深さの「過去状態」だけを消す診断

以前の[層を外す診断](TEMPORAL_LAYER_DIAGNOSTIC.md)は、履歴だけでなく、その層が現在画像へ適用する変換も除去していた。
そこで[状態だけを消す処理](exploration/memory_layer_intervention.py)を追加した。
第4・8・12層の全アダプタを毎フレーム実行し、選んだ1層の過去状態だけを次のフレームの直前に破棄する。
他の層の状態と、絶対観測番号は維持する。Mambaの選択層では、SSM行列だけでなく位相・前回K/Vを含む4状態すべてを初期化する。

対象は上記と同じ4,000更新checkpoint・同じ348動画・同じRGB・同じ分類器である。
通常履歴と全状態消去の基準値には、検証済みの`full_stream`と`current_only`を再利用する。
このMamba実装では全層の状態を毎回消すと、最後のフレームの出力は現在画像だけの逐次処理と同じになる。
ただしAttentionでは絶対位置の扱いが異なるため、現在画像だけを時刻0へ戻す対照と一般に同一視しない。

[実行計画](results/memory_layer_intervention_seed17_v2/plan.json)では、以下の3介入と、先頭・末尾の2動画での基準値のnative再実行を固定した。
全348動画で入力と元ファイルhashを再確認し、主学習の途中checkpointへ差し替えない。

| 介入 | 維持する過去状態 | その時刻のアダプタ計算 |
| --- | --- | --- |
| `clear_4` | 第8・12層 | 全3層を実行 |
| `clear_8` | 第4・12層 | 全3層を実行 |
| `clear_12` | 第4・8層 | 全3層を実行 |

[CPU検査](exploration/test_memory_layer_intervention.py)では、小型の実Attentionエンコーダで、全アダプタの実行回数、
絶対位置の維持、呼び出し間の状態初期化、重み不変、選んだ層以外の履歴保持を確認した。
最初のGPU実行は、状態を消して再登録したときの辞書の挿入順を、層の実行順と誤って比較する検査で止まった。
[失敗記録](results/memory_layer_intervention_seed17/failure.json)を保持し、状態の所属層を正規順序で記録するよう修正した。
修正後に3件のCPU検査を再実行し、新しい出力先でGPU評価を開始した。

全348動画の評価を完了した。[全予測と集計](results/memory_layer_intervention_seed17_v2/summary.json)、
[元ラベルからの再計数](results/memory_layer_intervention_seed17_v2/verification.json)を保存した。
新たに計算した3介入の1,044予測と、再利用した2基準の696予測、計1,740件の再計数が一致した。
事前に指定した2動画では、通常履歴・全状態消去の両方で元の基準logitと完全一致した。

| 条件 | 正解数 / 348 | 正解率 | 通常履歴との差 | 通常から正解を得た / 失った本数 |
| --- | ---: | ---: | ---: | ---: |
| 通常履歴（基準を再利用） | 105 | 30.17% | — | — |
| 第4層の過去状態だけ消去 | 106 | 30.46% | +0.29 pp | 7 / 6 |
| 第8層の過去状態だけ消去 | 72 | 20.69% | −9.48 pp | 12 / 45 |
| 第12層の過去状態だけ消去 | 50 | 14.37% | −15.80 pp | 13 / 68 |
| 全層の過去状態を消去（現在画像のみの基準を再利用） | 26 | 7.47% | −22.70 pp | 8 / 87 |

**このcheckpoint・集合では、第8・12層の状態を失う影響が大きい。**
第4層の状態消去は正解数で1本の増加だったが、50動画で予測が変わった。
精度差の探索的な区間は−1.15〜+1.72 ppであり、「第4層の記憶は不要」「同等性を証明した」とは言わない。
第8層は−12.07〜−6.90 pp、第12層は−18.97〜−12.64 ppだった。
これらは各クラス内で対応を保つ2,000回bootstrapで、重み固定・観測済み小集合・多重比較未補正である。

今回の実装は捨てる状態も毎回作るため、全介入で状態計上は79,027,200 bytesのままだった。
これは記憶の使われ方の診断であり、省メモリ化や速度改善の結果ではない。
最終checkpointでも同じ傾向が続くか、状態保持を減らして再学習したモデルが精度を保てるかは未検証である。

### 最終モデルとAttentionでの再検証

[最終比較計画](results/final_memory_layers_v1/plan.json)を固定し、
[評価・集計](exploration/final_memory_layers.py)と[実行キュー](exploration/run_final_memory_layers.py)を追加した。
中間層Mamba・時間Attentionの各seed 42・43・44、計6モデルで、通常履歴・3層の個別状態消去・全状態消去の5条件をすべて実行する。
各モデルは元の20,000更新を完了したcheckpointに限定する。
この比較では基準値も含めて同じbatch 1の逐次処理で新しく計算し、4,000更新時点の予測は流用しない。
通常の全件評価とbatch sizeが異なる可能性があるため、その予測不一致数も別に記録する。

Attentionの全状態消去でも、最後の画像には元の絶対観測番号7を使う。
先頭・末尾の動画で、通常の逐次処理との一致と、同じ絶対位置で現在画像だけを処理した結果との一致を検査する。
Mambaも同じ手順を使う。全予測を元manifestへ照合し、5条件×348動画×6モデルを集計する。

集計ではモデルごとの通常履歴からの変化と、その変化のMamba–Attention差を分ける。
**状態を消しても精度が落ちにくいことは、認識精度が高いことや記憶が優れていることを直接意味しない。**
各条件の正解数・精度も併記し、変化量だけで方式の優劣を決めない。
3 seedの差の平均と標本標準偏差は、4,000更新診断のbootstrap区間とは別の量である。

CPU検査は既存の状態消去検査を含めて6件通過した。
[実プロセスでの待機・停止検査](results/final_memory_layers_cancel_verification.json)では、既存6プロセスが同じ個体のまま生存した。
[未完了学習の拒否](results/final_memory_layers_incomplete_verification.json)も確認した。
[本キューの起動記録](results/final_memory_layers_activation.json)の通り、既存の学習・保持診断・KVバッファ費用測定が終わるまで待機する。
起動記録作成時の最初のPID検査は、uvの主スレッドの子だけを探して失敗したが、
全スレッドの子と実行コマンドを照合して、既に稼働していたワーカーを確認した。キューは再起動していない。
**最終6モデルの推論と比較結果はまだ未取得である。** 新しいモデルの学習は追加していない。

この診断は、学習済みモデルが各深さの状態へどれだけ依存するかを測る。
他層経由の履歴は残るので、ある層を消して精度が保たれても、その層単独に記憶能力がないとは結論しない。
層間の効果も単純には足し合わせられない。再学習した軽量モデルの性能、長期保持、Attentionへの優位性は別に検証する。
主学習とGPUを共有する診断であり、経過時間を性能比較には使わない。

## 固定容量と、過去状態に掛かる減衰を分ける

[計測コード](exploration/inspect_temporal_decay.py)を追加し、同じ4,000更新checkpointを、
事前に長さだけで選定済みの245フレームの実動画へ適用した。
使用したのは`gear_v2_4-wrist_back_rgb_img-1`のRGBだけで、点対応ラベルや追跡精度は使わない。
これは1動画・1つの途中モデルの内部更新の診断であり、長期認識の成績ではない。

[公式Mamba-3実装](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py#L393-L444)では、
SSM行列に掛かる過去からの持ち越し係数は`exp(ADT)`である。
実際のモジュールの投影出力から、各観測・196座標・8ヘッドの`ADT`を記録した。
同じ今後の入力、角度状態、前回K/Vを条件として、SSM行列だけの初期差を考えると、

$$
\Delta H_{t+k}=\exp\!\left(\sum_{j=t+1}^{t+k}\mathrm{ADT}_j\right)\Delta H_t
$$

となる。この量は、固定した更新列におけるSSM行列の同次項の倍率である。
過去画像を変えたときのエンコーダ全体の差は、後続層の入力、角度、K/V、書き込み、読み出しも変えるので、この式だけでは決まらない。
倍率が大きくても、必要な内容が書き込まれていることや、認識器が取り出せることは保証しない。

[計画](results/temporal_decay_4000_median/plan.json)と[結果](results/temporal_decay_4000_median/summary.json)を保存した。
下表は各間隔に対する全有効開始位置・座標・ヘッドの持ち越し倍率の中央値である。
百分率の認識精度ではない。重複する時間窓や座標を独立標本として統計的有意性を主張しない。

| 時間層の位置 | 8観測後 | 128観測後 | 244観測後 |
| --- | ---: | ---: | ---: |
| 第4層の後 | 0.9431 | 0.3243 | 0.0901 |
| 第8層の後 | 0.8835 | 0.1140 | 0.0148 |
| 第12層の後 | 0.9580 | 0.3930 | 0.1081 |

244観測後でも、第4層の一部ヘッドの座標間中央値は0.69、第12層では0.60だった。
一方、第4層には約`3.3e-114`まで小さくなるヘッドもあった。
したがって、このモデルの全状態を一律に「8観測程度で消える」とみなすことはできない。
浅い層の状態消去で分類精度がほとんど変わらなかった先の結果も、
「浅い層は必ず短い時定数だから」と決めつけない。二つの診断は同じcheckpointだが異なる入力集合であり、因果的な説明には追加の比較が要る。
現時点では、単に減衰を遅くする変更の有効性も実証していない。

保存した実測`ADT`を公式GPUカーネルへ入れ、今後の書き込みをゼロにして初期SSM行列だけを伝播させた。
3層×4間隔の12条件で上式との最大絶対差は`1.23e-7`だった。
これは係数の計算を照合する人工的な分離検査であり、実動画の認識検査ではない。
計測hookの有無で最初の3フレームの出力は完全一致し、モデル重みも変わらなかった。
[CPU再集計](results/temporal_decay_4000_median/verification.json)に加え、
prefix sumを使わず各窓を直接足し合わせた[別集計](results/temporal_decay_4000_median/direct_recount.json)でも分位点が一致した。
インストール済みのモジュール・カーネル3ファイルが固定した公式ソースと一致することも確認した。
[6件のCPU検査](results/temporal_decay_tests.log)は、最初の状態を作る更新と後続の減衰の区別、座標・ヘッド軸、未観測区間の拒否を確認する。

再実行はリポジトリのルートで、保存済み4,000更新checkpointと固定動画を用いる。
出力先には新しいディレクトリを指定する。`--verify`はローカルの`adt.npz`も必要で、公開JSONだけから新たなGPU推論を再現する処理ではない。

```sh
uv run --locked python -m exploration.inspect_temporal_decay --output results/temporal_decay_rerun
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.inspect_temporal_decay \
  --output results/temporal_decay_rerun --verify
```

実行は主学習とGPUを共有したため、速度・メモリの比較値を報告しない。
主学習と比較キューのソース・重み・条件は変更していない。
この内部診断で、予定している認識精度による保持評価やAttentionとの比較を代替しない。

## 観測画像の追加と更新回数を分ける診断

[実装](exploration/temporal_density.py)では、11,000更新のモデルを[固定保存](results/temporal_density_step11000_v1/snapshot.json)し、
同じ既観測348動画で、8観測を15・29観測へ増やした。
元の8画像を捨てず、隣接する観測の間へ1枚または3枚を追加する。
対照では、その追加位置を直前の元画像で埋める。両者で更新回数・元画像の位置・最後の画像・分類器・重みを揃える。
最初の元画像より前の履歴は追加しない。全条件をbatch 1の因果的clip経路で新たに推論する。

[固定計画](results/temporal_density_step11000_v1/evaluation_v2/plan.json)、
[結果](results/temporal_density_step11000_v1/evaluation_v2/summary.json)、
[元ラベルからの再計数](results/temporal_density_step11000_v1/evaluation_v2/verification.json)を保存した。
この348動画は各クラス2本ずつの集合であり、全24,777本の53.35%とは集計の重みが異なる。

| 条件 | 正解数 / 348 | 正解率 | 8観測との差 |
| --- | ---: | ---: | ---: |
| 元の8観測 | 163 | 46.84% | — |
| 新しい途中画像を追加・15観測 | 165 | 47.41% | +0.57 pp |
| 直前の元画像を保持・15観測 | 161 | 46.26% | −0.57 pp |
| 新しい途中画像を追加・29観測 | 150 | 43.10% | −3.74 pp |
| 直前の元画像を保持・29観測 | 135 | 38.79% | −8.05 pp |

29観測では、新しい画像を使う方が保持対照より15本多く正解した（+4.31 pp、探索的な対応付き95%区間[2.59, 6.32]）。
一方、元の8観測には13本及ばなかった。15観測の8観測に対する差の区間は[−1.72, 2.88] ppで、明確な改善とは扱わない。
区間はクラス内2本の対応を保つ2,000回bootstrapであり、重みとこの集合に条件付けた、多重比較未補正の値である。

**このモデルでは、同じ動画を細かく読むだけでは改善しなかった。**
新しい画像のない保持対照でも、更新回数を増やすと成績が下がった。
これは系列長・観測間隔の変化への感度を示すが、原因をSSMの忘却だけに限定しない。
空間特徴の変化、読み出しの入力分布、8観測だけの学習も影響し得る。
長い系列で再学習したモデルの上限、他方式への優位性、長期記憶の性能を示す実験ではない。
次の構造検討では、追加画像の情報と、観測の細分化に伴う状態更新量を分けて扱う。

全条件1,740予測の再計数が一致し、元の8観測の入力hashも以前の記録と一致した。
モデル重みは不変で、先頭・末尾の2動画では8観測の再推論logitが完全一致した。
[同じ動画の全件検証記録との照合](results/temporal_density_step11000_v1/evaluation_v2/global_validation_comparison.json)では、
batchサイズの異なる以前の推論と予測が1本異なったが、両方とも不正解で、正解数は163本で一致した。
新たな異なる画像は15観測で平均6.98枚、29観測で平均20.26枚だった。短い動画では同じ元フレームを再利用する。

[CPU検査3件](results/temporal_density_tests.log)で、元画像の保持、細分化の包含関係、
保持対照が新しい画像を使わないこと、短い動画、正解増減と予測変更数の区別を確認した。
最初の実行は[CUDA context確保時のメモリ不足](results/temporal_density_step11000_v1/evaluation.log)で、推論前に停止した。
使用していないRoboTAP元ファイル5本の読み込みキャッシュを`POSIX_FADV_DONTNEED`で解放し、
[物理空きメモリが約5.8 GBから28.5 GBへ増えた](results/temporal_density_step11000_v1/cache_reclaim.json)後、
同じ実装・重み・条件を新しい出力先で実行した。ファイルの削除や主学習の再起動はしていない。
この挙動は[NVIDIAの共有メモリに関する既知の問題](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)と整合するが、
ドライバ内部の原因を特定したとは主張しない。主学習とGPUを共有したため速度比較にも使わない。

再実行には11,000更新の固定checkpointと、既存の履歴診断の入力記録が必要である。
生logit・全入力記録・重みはローカル成果物であり、Gitには含めない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_temporal_density.py
uv run --locked python -m exploration.temporal_density \
  --checkpoint results/temporal_density_step11000_v1/checkpoint.pt \
  --output results/new_temporal_density
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.temporal_density \
  --output results/new_temporal_density --verify
```

## 関連研究から分かる評価の範囲

| 一次資料 | 実際に測るもの | 今回への含意 |
| --- | --- | --- |
| [Frozen Image + GMMix, §6.1](https://arxiv.org/html/2605.19137v1) | SSv2で最後のフレームのトークンから174分類 | 現在の短時間認識比較に直接対応する。長期保持の立証には別の証拠が必要 |
| [MEM, §III-C/D](https://arxiv.org/html/2603.03596v1) | 視覚履歴と長期の言語記憶を組み合わせたロボット評価 | 長いロボットタスクの成功を、視覚エンコーダ単体の保持能力へ転記できない |
| [SDQES, §6・付録D](https://arxiv.org/html/2412.03567v1) | 自然言語で指定したイベント開始を逐次検出。60・300秒窓と未切断動画で評価 | 長いストリームでの認識・誤報・遅延を測る候補。ただし動画が長いだけで遠い過去への依存は保証されない |

MEMでは事前学習時6観測を1秒間隔で用い、post-trainingで最大18観測・54秒へ拡張する。
15分のタスクには別の言語記憶も含むため、Mambaの視覚状態との比較対象を混同しない。
これは論文の構成の確認であり、当実装で同じ成績を再現したという主張ではない。

SDQESは追加の言語読み出しとイベント開始の教師を必要とする。
現行SSv2分類器をそのまま適用できるベンチマークではない。
現段階では採用候補の調査にとどめ、データ取得や別タスクの学習は開始しない。

## 公開注釈まで確認した追加候補

[EgoStream v2](https://arxiv.org/html/2605.31557v2)は、証拠の観測完了から質問までの時間と、
答えが引き続き有効な期間を区別する。質問は読み取り時だけ渡し、観測中は質問を知らずに記憶を作る。
この評価原理は今回の目的に合う。ただし2026-11-05に確認した[公式サイト](https://saroo25.github.io/Egostream/)は
GitHubを公開予定と表示し、[取得できたリポジトリの固定版](https://github.com/saroo25/Egostream/tree/8cb0dab06659a56a7de75b88770f31eecec17944)は
サイトのソースのみだった。現時点で実行できる評価として採用したとは言わない。

代わりに公開済み[OVO-Benchの注釈](https://github.com/JoeLeelyf/OVO-Bench/blob/c34093f9a072d93f9e1c6b923446dfba71f4528e/data/ovo_bench_new.json)を
918,443 bytes取得し、Git blob hashを照合して全件を監査した。
[監査コード](exploration/audit_ovo_memory.py)と[実行結果](results/long_memory_benchmark_audit/ovo_with_baselines.json)を保存した。
元動画や注釈本文はこのリポジトリへ再配布しない。

| 固定版の内容 | 実際に数えた値 |
| --- | ---: |
| 元動画のパス | 644 |
| 注釈レコード | 1,640 |
| Forwardの各質問時刻まで展開した評価回数 | 3,035 |
| Backward Tracing | 631件・275動画 |
| Real-Time Visual Perception | 837件 |
| Forward Active Responding | 1,567回 |

[README](https://github.com/JoeLeelyf/OVO-Bench/blob/c34093f9a072d93f9e1c6b923446dfba71f4528e/README.md)の3,100件と
固定版の展開数は一致しないため、論文・評価では実際に使った版と数を明記する。
Backwardの専用フィールドには質問時刻はあるが、証拠区間や答えの有効期限はない。
質問時刻が193秒でも、193秒前の情報を覚えたという意味にはならない。
現行注釈だけから証拠距離ごとの忘却曲線を作らない。

Backwardの選択肢数は2〜5個で、ランダム選択の期待値も一律25%ではない。
3タスクを同じ重みで平均すると、一様ランダム選択の期待値は26.84%、
常に最初の選択肢を選ぶ参照値は34.86%だった。これは注釈からの計算で、モデルの成績ではない。
公式のBackward平均はタスク平均なので、全質問を一括平均した36.29%とは区別する。

採点にも注意が必要である。[固定版の採点実装](https://github.com/JoeLeelyf/OVO-Bench/blob/c34093f9a072d93f9e1c6b923446dfba71f4528e/utils/OVOBenchScore.py)は
正解文字が応答文字列に含まれるかを見る。複数候補を列挙した応答を正解に数えないよう、
接続時には一つの選択肢IDから一文字だけを出力し、厳密な一致も別に集計する必要がある。
自由記述の`answer`と`options[gt]`が完全一致しない420件も記録した。
言い換え・句読点の差を含むため、420件の誤ラベルとは扱わず、元の`gt`を変更していない。
[根拠と最初の監査失敗の記録](results/long_memory_benchmark_audit/evidence.json)を参照。

**採否：** OVOのBackwardは過去に関する認識の応用評価候補に残す。
証拠距離を制御した長期保持評価としては、現行注釈だけでは採用しない。
いずれのQA課題も、現在の174分類器では解けず、別データで学習する共通の言語読み出しが必要になる。
視覚エンコーダを比較する際は読み出し・教師・学習量を揃え、評価用QAを学習へ混ぜない。
この監査によって新しいQA学習や大容量の動画取得を自動的に追加せず、既存の時間Attention比較を継続する。

## 遅延を対応付けられるRIVERの評価準備

[RIVER原論文のRetro-Memory](https://arxiv.org/html/2603.03985v1#S3.SS2)は、同じ過去の出来事への質問を異なる時刻で尋ねる。
公式リポジトリからリンクされた[公開注釈の固定版](https://huggingface.co/datasets/nanamma/RIVER/blob/0338ea79419e375b5760c11a1f9a3e4194af6add/annotations/Retro-Memory.json)を取得し、
[GitHub版](https://github.com/OpenGVLab/RIVER/blob/8aab82325d0986824f8504e86669fa52025593d2/annotations/Retro-Memory.json)とGit blob hashが一致することを確認した。
画像を別動画へ継ぎ足す遅延診断とは異なり、元動画の時間軸上で評価する候補である。
動画そのものはまだ取得・検査していない。

[監査・採点実装](exploration/river_memory.py)、[集計](results/river_memory_audit/summary.json)、
[実注釈からの再計数](results/river_memory_audit/verification.json)を保存した。
1,488レコードは独立した1,488問ではなく、**91動画の372問を4時刻で繰り返したもの**だった。
質問文・選択肢・正解・証拠区間は4条件で一致し、質問時刻だけが変わる。
識別子には`question_id`と`question_type`の両方が必要で、前者だけで保存すると4条件が上書きされる。
行単位のランダム分割も同じ問題を学習・評価へ混ぜるため禁止する。

2問の証拠区間が逆転していた（`moALQl818ZY@1`: [515, 495]、`YlQugR7KSKg@6`: [3800, 215]）。
元注釈を推測で修正せず、この2問の4条件すべてを除いた**370問×4時刻=1,480レコード**を準備した。
公式全件のスコアとは区別する。別の3レコードでは質問時刻が動画長を1e-12秒未満だけ超えており、
数値表現の差として1e-6秒の許容差で記録する。元時刻は変更していない。
これは未来フレームの入力を認める許容差ではなく、実際のデコードはPTSが質問時刻以下のフレームに限定する必要がある。

| 条件 | 同じ質問の数 | 証拠区間の終了から質問までの中央値 |
| --- | ---: | ---: |
| short | 370 | 22.94秒 |
| medium | 370 | 45.70秒 |
| long | 370 | 600.21秒 |
| very_long | 370 | 2,343.13秒 |

各時刻の363問が4択、7問が5択であり、一様ランダム選択の期待値は24.905%となる。
常にAを選ぶ参照値は各時刻89/370=24.054%。これはモデルの認識精度ではない。
初回の監査では全問4択と仮定して停止したが、実注釈に従って修正し、全件を再実行した。
[CPUテスト](exploration/test_river_memory.py)の5件では、4時刻の対応、欠損・重複・複数回答の拒否、
逆転区間の扱い、数値境界、5択の採点を検査した。
採点は選択肢番号の厳密一致のみとし、正解文字を含む自由文や複数候補は受け付けない。

**採否：** 証拠距離を伴う長期認識の追加候補としてRIVERを残す。
ただし時間経過とともに現在画像・途中の出来事も変わるので、精度変化を忘却だけの効果とは断定しない。
元動画の証拠を再び見せていないか、現在画像だけでも答えられないかは実画像での検証が必要である。
出典はLVBench・LongVideoBench・Vript-RRであり、ロボット作業の成功率を測るデータではない。
Hugging Faceのカードに`train`とあっても、論文で評価に使用した問題を読み出し学習へ流用しない。
エンコーダに質問を事前提示せず、質問時点の特徴と質問から答える共通読み出しを別データで学習する必要がある。
この接続と学習は未実装で、現在の174分類器によるQA評価や新たな学習キューは追加していない。

併せて確認した[S-EMBER公式スキーマ](https://github.com/facebookresearch/S-EMBER/blob/main/data/README.md)も証拠区間・質問時刻を持つ。
ただし[配布元](https://huggingface.co/datasets/facebook/S-EMBER)の固定版`897886cc967b2d0e6e45f1200605f5d3b0ffbb23`はアクセス制限があり、
現在の認証では注釈取得がHTTP 403で拒否された。内容を監査したデータとして扱わず、採用は保留する。

## 長期記憶の主張に必要な追加条件

長い動画で固定容量の状態を更新できることは、遠い過去を覚えられることと異なる。
評価例には、答えを決める過去の証拠と、その証拠から判断時点までの距離が必要になる。
同じモデルで保持履歴を制限し、必要な証拠を含む場合と含まない場合を比較する。
学習済みの履歴なし対照も必要であり、状態消去だけで方式間の優位性を結論しない。

履歴長は観測数と実時間を併記する。間引きを増やして長時間を覆う条件と、
逐次更新の回数を増やす条件を区別する。8観測で学習したモデルへ数百観測を入力する場合は、
学習時との長さ・観測間隔の違いを明記する。

まず同条件の動画認識比較を完成させる。
[共通の後続映像を使う遅延保持診断](DELAYED_ACTION_MEMORY.md)では、同じ現在画像でも
過去の証拠に応じて答えが異なる人工的な連結列を作り、固定エンコーダの読み出しを比較する。
実画像での機構の動作確認まで完了したが、性能比較と自然な長期記憶評価は未完了である。
長期評価を追加するときは、公開注釈が上の証拠距離を指定できるか、履歴なしで解けないか、
言語読み出しの能力をエンコーダ性能と混同しないかを確認してから選ぶ。
この文書は評価範囲の整理であり、新しいアーキテクチャの貢献やSOTAを示したものではない。

## 再実行

リポジトリのルートで実行する。元動画、保存済みcheckpoint、以前の層介入の出力が必要。
既存の出力ディレクトリへ上書き・部分再開はしない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_video_history.py
uv run --locked python -m exploration.evaluate_video_history \
  --prior results/temporal_layer_intervention_seed17 \
  --output results/new_video_history_intervention
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_video_history \
  --run results/new_video_history_intervention
```

OVO注釈の監査は、上記の固定版JSONを提供元から取得してから実行する。
`--annotations`はそのファイルへのパス。内容のblob hashが違う版は拒否する。

```sh
uv run --locked python -m exploration.audit_ovo_memory \
  --annotations results/long_memory_benchmark_audit/raw/ovo_bench_new.json \
  --output results/new_ovo_annotation_audit.json
```

層ごとの状態消去は、上の検証済み履歴診断と保存済み4,000更新checkpointを使う。
`--resume`は未開始の計画を実行する指定であり、途中で失敗した評価の部分再開はしない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_memory_layer_intervention.py
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.evaluate_memory_layers \
  --output results/new_memory_layer_intervention --plan-only
uv run --locked python -m exploration.evaluate_memory_layers \
  --output results/new_memory_layer_intervention --resume
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.evaluate_memory_layers \
  --output results/new_memory_layer_intervention --verify
```

最終6モデルの状態消去計画は次で作成できる。既存の本キューが待機中なので、同じ環境で二重に起動しない。
`--plan-only`を外す実行は、先行する主学習・KVバッファ費用比較の完了を要求する。
実プロセスを待つ場合は`--after-pid`へ、その出力先に対応するKVバッファ費用キューのPython PIDを指定する。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q \
  exploration/test_memory_layer_intervention.py exploration/test_final_memory_layers.py
uv run --locked python -m exploration.run_final_memory_layers \
  --output results/new_final_memory_layers --plan-only
```

RIVERの注釈を取得・監査する場合は次を実行する。ダウンロード対象は注釈JSONだけである。
`RIVER_ANNOTATIONS`はダウンロードされた実ファイルのパスとなる。

```sh
RIVER_ANNOTATIONS=$(uv run --locked python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("nanamma/RIVER", "annotations/Retro-Memory.json", repo_type="dataset", revision="0338ea79419e375b5760c11a1f9a3e4194af6add"))')
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_river_memory.py
uv run --locked python -m exploration.river_memory audit \
  --annotations "$RIVER_ANNOTATIONS" --output results/new_river_audit.json
```

予測ファイルには`annotation_sha256`と`predictions`配列が必要である。
各予測は`question_id`・`question_type`・ゼロ始まりの整数`choice_index`を持ち、除外済み2問以外の全1,480件を含める。
`python -m exploration.river_memory score`へ同じ`--annotations`、予測JSONへの`--predictions`、新規の`--output`を渡す。
この採点だけでは入力の因果性・モデルの由来・学習データの分離を確認できず、それらは実行時に別途検証する。
