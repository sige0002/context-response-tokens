# Mamba-3をGB10で試す

公式Mamba-3のSISO版を、uvプロジェクトとして動かす最小構成。依存は[pyproject.toml](pyproject.toml)と[uv.lock](uv.lock)で管理する。`.venv`はuvが管理する保存先で、システムPythonへインストールしない。

## 画像エンコーダの試作

### SigLIP 2＋時間Mamba-3（履歴エンコーダ）

[研究仮説・比較条件・途中結果・未完了事項](RESEARCH_PROTOCOL.md)を別文書にまとめている。
全方式の性能比較は未完了であり、動作検査の通過を性能優位性の証拠にはしない。
[アーキテクチャ探索](ARCHITECTURE_SEARCH.md)では、既存研究との差分候補・演算試作・棄却実験を記録する。
[正解対応による記憶移送の診断](ORACLE_TRANSPORT_DIAGNOSTIC.md)では、直交移送と座標更新が改善しなかった結果を公開している。
[画像対応を使うMambaの試作](ROUTED_MEMORY_PILOT.md)では、学習した対応と点追跡を実装し、単純基準にも負けた短期学習の結果を記録している。
[ベンチマークの適用範囲](BENCHMARK_FIT.md)では、SSv2の秒数とRoboTAP全件の遮蔽・再出現を調べ、長期記憶について未検証の範囲を区別している。
[点読み出しの診断](READOUT_DIAGNOSTIC.md)では、短期学習で位置情報を利用できていない問題と、初期位置を保つだけで全体スコアが上がる問題を切り分けている。
[対応照合を使う読み出し](CORRELATION_READOUT_PILOT.md)も実装した。移動点のPCKは小幅改善したが、不可視点を検出できずAJは単純基準を下回った。
[時間間隔と遮蔽例の重み付けの比較](CORRELATION_HORIZON_DIAGNOSTIC.md)でもAJは改善せず、
見逃し減少と誤検出増加を分けて記録した。この変更は主実験へ採用しない。
[公開内容と再実行の範囲](PUBLICATION.md)も参照。学習重み・元動画・環境はGitへ含めない。

[実動画での速度・メモリ測定と最終比較の条件](ENCODER_COST_PROTOCOL.md)も記録している。
[点追跡での履歴利用の評価条件](POINT_HISTORY_PROTOCOL.md)では、状態消去と経過フレーム・再出現の集計を定義した。
[近接研究との比較の不足](BASELINE_FIDELITY.md)では、GMMixの複数層特徴を含む追加対照と、その実装・検証状況を記録している。
[6方式の統合集計](RESEARCH_PROTOCOL.md#6方式を結ぶ最終集計)は、動画分類・点追跡・費用・履歴介入を、
同じcheckpointとデータに結び付けて比較する。全件実験が完了するまで最終比較表は生成しない。
[主要4方式の3 seed反復](RESEARCH_PROTOCOL.md#4方式を3-seedで反復する)も計画・実行コードを固定した。
追加seed 43・44の動画学習とpoint probeは既存実験列の終了後に実行する。結果はまだ出ていない。

**入力不具合の訂正：** 元動画の既定RGB変換で、奇数幅の右端が不定になることが分かった。
共有する変換を修正し、実動画64本で反復・ワーカー間一致を確認した。
旧全データ学習は4,370更新で停止し、以下の旧pilot・旧全件検証の数値は診断記録として残している。
修正後の正式結果には使わない。[原因・修正・影響範囲](RGB_INPUT_REPAIR.md)を参照。

修正後は[新しい全データ学習](results/ssv2_full_mamba_interleaved_seed42_rgbfix/config.json)を開始した。
空間演算のTF32を無効化し、学習・評価の再現性設定を比較方式間で揃えた。時間層は引き続きBF16を使用する。
修正後の全24,777本の正解率は、1,000更新で16.10%、2,000更新で29.07%（7,203本正解）だった。
[公式ラベルと保存予測の再集計](results/ssv2_implementation_audit/rgbfix_full_run_metadata_audit_step2000.json)でも一致した。
予定20,000更新の途中値であり、同条件の対照モデルとの比較は未完了である。

[temporal_siglip.py](temporal_siglip.py)は、事前学習済み
`google/siglip2-base-patch16-224`の第4・8・12層の直後に時間アダプタを挿入する。
既定の重みrevisionは`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`に固定する。
このSigLIP 2固定解像度モデルは公式設定の`model_type`が`siglip`なので、
Transformersの`SiglipVisionModel`と`SiglipImageProcessor`で読み込む。SigLIP 1の重みは使わない。
前処理はSigLIP 2の設定を使用する（リサイズ方式もSigLIP 1と異なる）。
空間Attentionは元のSigLIPを使い、時間処理は同じパッチ座標ごとに行う。
時間層は768→256次元への射影、Mamba-3 SISO（状態次元64、expand 2）、
256→768次元の射影、ゼロ初期化した残差ゲートで構成する。
カメラは別バッチとして扱い、カメラ間統合は行わない。

`kind="none" / "attention" / "mamba"`で履歴なし・因果的時間Attention・
時間Mambaを比較できる。時間Attentionは4ヘッド、フレーム番号のsin/cos位置符号を使う。
共通の挿入位置・幅だが追加パラメータ数は一致しない。MEMの重み共有Attentionの完全再現ではない。
ゼロゲートで初期出力は元のSigLIPと一致するが、学習後の画像認識性能維持は別途評価が必要。

`placement="interleaved"`が中間層へ挿入する既定方式。
`placement="late"`では同じ3個の時間アダプタを全空間層の後・最終LayerNormの前にまとめる。
同じkind・seedなら時間層の数、幅、初期重み、追加パラメータ数を一致させられる。
これは挿入位置だけの対照実験であり、空間混合も追加する既存GatedMambaMixの再現ではない。
中間層方式が遅い場合も含めて、精度と実測計算量の両方を比較する。

```python
import torch
from temporal_siglip import TemporalSiglip

encoder = TemporalSiglip.from_pretrained(kind="mamba").cuda().eval()
# video: checkpointのSiglipImageProcessorで前処理済みの[B,T,3,224,224]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    features = encoder(video)       # [B,T,196,768]
    current = features[:, -1]        # 後段へ渡す196トークン
    state = None                    # エピソードごとにリセット
    for image in video.unbind(1):
        current, state = encoder.step(image, state)
```

利用APIは次の4つ。状態を使うAPIは推論専用で、同じ入力列を分割して処理できる。

| API | 入力 | 出力 |
| --- | --- | --- |
| `encoder(video)` | `[B,T,3,224,224]` | 全時刻の`[B,T,196,768]`、学習可能 |
| `encoder.encode_current(video)` | 同上 | 最後の時刻の`[B,196,768]`、学習可能 |
| `encoder.step(image, state)` | 新しい1枚`[B,3,224,224]` | 現在特徴と更新した状態 |
| `encoder.encode_chunk(video, state)` | 新しい複数フレーム | 各時刻の特徴と更新した状態 |

アルゴリズムの処理順：各画像をパッチ化 → SigLIP 2の空間層を4層実行 →
同じ座標の特徴を`[B×196,T,256]`へ並べ替えて因果的Mamba-3で更新 →
元の画像座標へ戻して残差加算 → 残りの空間層でも同じ処理を繰り返す。
Mambaの重みは座標間で共有するが、状態は各座標・各挿入層で独立する。
特殊な記憶トークン、物体追跡、VLM、Action Expertはこのアルゴリズムに含めない。
空間Attentionが位置間で情報を交換するが、物体の対応付けは保証しない。

学習後のエンコーダは`encoder.save_checkpoint("results/encoder.pt")`で保存し、
`TemporalSiglip.load_checkpoint("results/encoder.pt", device="cuda")`でネットワークアクセスなしに復元できる。
構造設定・SigLIP 2の重み・時間層の重みをまとめて保存する。エピソード状態とoptimizerは含めない。
復元後はevalモード・履歴なし。学習を再開する場合は`train()`を呼び、optimizerを別途復元する。
部分保存による破損を避けるため同じディレクトリの一時ファイルから置き換える。

推論状態は呼出し側が管理する。Mambaは4種類の状態テンソルだけを引き継ぎ、過去画像を再実行しない。
Attentionは過去KVを保持するため履歴に比例して状態が増える。
`step`は推論専用、学習は`forward`を使う。初版は等間隔フレームを想定し、実時間の間隔は入力しない。
バッチの並び・カメラの対応はエピソード内で維持する。個別サンプルだけのリセットには未対応。

SigLIPパラメータは初期設定で凍結するが、中間アダプタへの勾配経路は維持する。
初回更新ではゼロゲートが学習され、その後に時間層内部が学習される。
GB10でBF16のSigLIPにバッチサイズ依存の数値差を確認したため、空間層はFP32、時間層はBF16 autocastとする。
モデル全体を`.bfloat16()`に変換しない。ストリーミングは公式Triton SISOの
`Input_States`を使う。CuteDSLのstep経路は現在のtvm-ffiとのAPI不整合があるため使わない。

```sh
uv sync --locked
uv run --locked pytest -q test_temporal_siglip.py
uv run --locked try_temporal_siglip.py
# 時間順のフレーム画像がある場合
uv run --locked try_temporal_siglip.py --images frame01.jpg frame02.jpg frame03.jpg --frames 3
```

初回はHugging FaceからSigLIP重みを取得する。スモーク検査は既定で`data/demo_dog.jpg`を6回入力する。
[検査結果](results/temporal_siglip2/smoke.json)には初期出力一致、因果性、一括・逐次の差、
状態メモリ、GPU時間、逆伝播・optimizer更新の成否を保存する。
時間層は未学習で、検査ではゲートを0.1にして非自明な経路を通す。
認識精度の比較・実動画の時間理解評価ではない。時間は少数回の動作確認値で、正式な速度ベンチマークではない。
検査中に別のテストプロセスがGPUを使用した可能性もあるため、この時間値で方式の速度順位を主張しない。
初版のパッチ別Mamba状態はbatch 1で約75.4 MiB。6フレームの時間Attention KVは約3.45 MiBで、
短い履歴ではMambaの方が状態メモリを多く使う。一定サイズという性質と、小さいという性質は別。
この値は状態テンソルのみで、重み・活性値・一時領域を含まない。

設計の参考：

- [MEM §III-C](https://arxiv.org/html/2603.03596v1)：視覚エンコーダ内部で空間と時間を分離し、現在フレームのトークンを出力する方針。
- [π0.7](https://www.pi.website/download/pi07.pdf)：MEM由来の履歴エンコーダを採用。今回の実装はその再現ではない。
- [XR-0 §2.2](https://arxiv.org/html/2602.12684v1)：Qwen3-VLが現在画像と言語を処理し、別のDiTが動作を生成する。
  視覚言語能力を保つ学習設計を参考にするが、XR-0を時間Mambaの先行例とは扱わない。
- [Streaming Detection of Queried Event Start §5.1](https://proceedings.neurips.cc/paper_files/paper/2024/file/b689b90ddf2b47d3103decabe6d47446-Paper-Datasets_and_Benchmarks_Track.pdf)：
  凍結ViTの中間層へ、パッチ別の状態を保持する時間アダプタを挿入する先行例。
- [TRecViT](https://arxiv.org/abs/2412.14294)：時間方向の再帰と空間Attentionを交互に適用する先行例。
- [Towards Data-Efficient Video Pre-training with Frozen Image Foundation Models](https://arxiv.org/abs/2605.19137)：
  凍結画像基盤モデルと時間Mamba/MambaMix/GatedMambaMixを比較した近接研究。
  中間層への挿入、固定状態、SigLIP 2とMambaの組合せだけで新規性を主張しない。

次の精度検証では動画の教師ラベルを使い、同じ空間重み・データ分割・学習予算で3方式を比較する。
履歴除去・順序反転、履歴長変更を併せて評価する。蒸留はこの実装の要件ではない。

#### 教師ラベルで時間層を学習する

[train_temporal.py](train_temporal.py)は動画分類の学習入口。SigLIP 2を固定し、時間層と
現在時刻の空間平均特徴を読む線形分類器を学習する。ラベルはクロスエントロピーで直接学習し、
教師モデルは使わない。アクション生成やロボット成功率の評価は含まない。

学習用・検証用に次の形式のJSONLをそれぞれ用意する。フレームは時間順で列挙し、
相対パスはJSONLのディレクトリが基準。`group`は切り出す前の動画・エピソードを示す。
同じ元動画からの複数クリップを異なる分割へ入れない。

```json
{"id":"video001_clip01","group":"video001","frames":["frames/001.jpg","frames/002.jpg","frames/003.jpg"],"label":0}
```

```sh
# train.jsonl、val.jsonlとクラス数は使用データに合わせる
uv run --locked train_temporal.py --train train.jsonl --val val.jsonl --classes 10 --kind mamba --frames 6 --epochs 10 --output results/video_mamba
# 同じ設定で中断したepochの次から再開。完了epoch数を延ばすことも可能
uv run --locked train_temporal.py --train train.jsonl --val val.jsonl --classes 10 --kind mamba --frames 6 --epochs 20 --output results/video_mamba --resume
```

`--kind attention`と`--kind none`にも対応する。出力先は別にする。
`--placement late`で出力後の時間統合を比較できる。既定は`--placement interleaved`。
各クリップ全体から固定等間隔で指定枚数を取り、最後の観測を必ず含める。
短いクリップではフレームが重複するので、評価データの長さとサンプリング条件を報告する。
現行版は固定間隔の系列を前提とし、動画ごとに実時間間隔が違う場合の補正は行わない。

検証では通常履歴・現在フレームの繰り返し・過去だけの順序反転を記録する。
`none`でも同じフレーム数を入力するため、これは精度比較の入口であり最適化済み速度比較ではない。
groupとフレームパスの重複は拒否するが、別名コピーされた同一内容やラベル由来の漏洩の監査は別途必要。

出力は`config.json`（重みrevision・manifestハッシュ・条件）、`history.json`（各epochの指標・予測）、
`best_encoder.pt`と`best_head.pt`、再開用の`last.pt`（モデル・optimizer・乱数状態）。
ベストepochは通常履歴の検証精度だけで決める。テスト集合はまだこのCLIに含めない。
再開時はmanifestと主要条件の一致を確認する。フレーム画像自体は実行中に変更しない。
新規実行では`source/`へ学習コード・エンコーダコード・依存設定・uv.lockを保存し、
再開時は保存済み学習コードとの一致も確認する。分類ヘッドの初期化乱数は時間層から分離し、
同じseedなら方式によらず同じ初期ヘッドを使う。`progress.jsonl`に50batchごとの進捗を残す。

保存済みモデルの評価は[evaluate_temporal.py](evaluate_temporal.py)で行う。

```sh
uv run --locked evaluate_temporal.py --run results/video_mamba --manifest val.jsonl --role validation --output results/video_mamba/evaluation.json
```

学習は行わず、選択済みのencoder/headを読み込む。通常履歴・現在画像の繰り返し・過去順序反転の
精度、存在するクラスのmacro recall、混同行列、動画ごとの予測を保存する。
`--frames`で学習時と違う履歴長も検査できるが、元のフレームリストが短ければ重複サンプリングになる。
評価ファイルにはモデル・ヘッド・manifestのSHA256を記録する。
`--role test`では学習だけでなくモデル選択に使ったvalidationとのgroup/path重複も拒否する。
元の分割manifestが移動した場合は`--train-manifest`と`--val-manifest`で指定する。
分割manifestの一致とファイルパスの独立性を確認するが、別名コピーの内容一致までは検出しない。

`data/temporal_training_fixture`と`results/temporal_training_integration`は学習入口の動作検査専用。
静止画の繰り返しと仮ラベルを使うため、これらの損失・精度は研究結果として扱わない。

#### Something-Something V2の準備

[公式配布ページ](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads)
から取得する[prepare_ssv2.py](prepare_ssv2.py)と、公式分割を維持してフレームを抽出する
[extract_ssv2.py](extract_ssv2.py)を用意した。用途は視覚エンコーダの時間理解の評価であり、
このデータからロボットの行動成功率は測れない。

```sh
uv run --locked prepare_ssv2.py
uv run --locked extract_ssv2.py --train-per-class 16 --val-per-class 4 --frames 6
uv run --locked train_temporal.py --train data/ssv2/pilot/train.jsonl --val data/ssv2/pilot/val.jsonl --classes 174 --kind mamba --frames 6 --epochs 10 --output results/ssv2_pilot_mamba
```

動画アーカイブの合計は約19.4 GB。途中ファイルを保持し、再実行時はサーバーがRangeに対応すれば再開する。
SHA256はローカル取得物の記録であり、配布元のチェックサム照合ではない。
抽出は分割gzipをストリームとして読み、選択済みの動画だけをデコードする。
ラベル・分割・seedから対象IDを事前に決め、selection.jsonへ保存する。
デコード不能や不足した動画を黙って除外せずエラーにする。初回抽出中の再実行はしない。

既定は各クラス学習16本・検証4本のpilot。全データの結果やSOTA比較として扱わない。
`--train-per-class 0 --val-per-class 0`で全件を対象にできるが、別の出力先を指定する。
フレームは動画全体から等間隔で抽出し、元の時刻・フレーム番号も記録する。
元動画IDをgroupとし、人物単位の独立性はこの注釈からは保証しない。
公式validationは検証用であり、未知のtest成績とは呼ばない。

#### 低精度の診断と全動画を使う学習入口

3 epoch・seed 42の[pilot比較](results/ssv2_pilot_comparisons/summary.json)は次の結果だった。

| 方式 | 通常履歴 | 現在画像の繰り返し | 過去の順序反転 |
| --- | ---: | ---: | ---: |
| 履歴なし | 5.89% | 5.89% | 5.89% |
| 中間層時間Attention | 5.75% | 5.46% | 5.60% |
| 中間層Mamba | 6.32% | 6.18% | 6.18% |
| 出力後Mamba | 7.04% | 5.17% | 7.04% |

中間層Mambaは44/696、出力後Mambaは49/696。中間挿入の優位性は確認できていない。
同じ中間層Mambaへ現在画像を繰り返すと43/696で、履歴による改善も確認できていない。
出力後Mambaには履歴除去による低下が見られるが、順序反転では精度が下がらず、
順序理解の改善とは断定できない。1 seed・少数標本・検証集合でのモデル選択という制限がある。
[診断](results/ssv2_implementation_audit/audit.json)では公式ラベル・時間順・分割、
独立した前処理計算、事前学習重みの一致、時間層への勾配、予測の再集計を確認した。
人工的な順序ラベルを付けた8例の暗記は成功したが、意味認識や汎化の評価ではない。
[固定特徴の分類器の追加学習](results/ssv2_implementation_audit/readout_optimization.json)では
学習100%・検証6.75%であり、履歴なしの低精度を単なる反復回数不足と説明できない。

[学習済みモデルの数値診断](results/ssv2_implementation_audit/streaming_numerics.json)では、
2動画・6フレームで、一括入力時の独自ラッパーと公式Mamba-3実装の差は0。
同じ投影済み入力でも公式BF16カーネルの一括・逐次計算には相対L2で約0.23〜0.28%の差があり、
FP32の公式参照計算では約1e-7。エンコーダ全体の差は相対L2で約0.21%、最大絶対差0.679だった。
この2動画での分類予測は一致したが、全検証集合の逐次精度や長い履歴の誤差を保証しない。
カーネルは内部で主要な入力をBF16に変換するため、外側のautocastを無効にするだけでは解消しない。

元動画を使う全分割は[prepare_ssv2_videos.py](prepare_ssv2_videos.py)で準備する。
学習168,913本・検証24,777本を抽出し、動画ごとのサイズとSHA256をJSONLに記録する。
元動画の合計は約17.2 GB。フレームJPEGを全件保存する必要はない。

```sh
uv run --locked python prepare_ssv2_videos.py
uv run --locked pytest -q test_video_data.py test_video_training.py
```

[video_data.py](video_data.py)は元動画をデコードし、過去の区間からフレームを取得する。
最後の観測は履歴長にかかわらず元動画の最終フレームに固定する。
学習時は過去の取得位置と画像全体の80〜100%を残すcropを変え、cropは全フレームで共有する。
左右の動作ラベルを壊す反転はしない。検証は各区間の中央を選び、cropなしで224へ変換する。
取得結果はseed・epoch・動画IDで決まり、ワーカー数に依存しない。
デコーダのfork後の停止を再現したため、読み込みワーカーにはspawnを使う。
旧pilotはJPEG読み込み・workers 0で実行しており、この停止は旧pilotの低精度の原因ではない。

[train_video.py](train_video.py)は全動画用の別の学習入口。旧pilotのコード・結果は変更しない。
既定は8フレーム、microbatch 4・勾配蓄積8、20,000更新、1,000更新のwarmup、
cosine学習率、AdamW（peak 1e-4、weight decay 1e-4）。Mambaの`_no_weight_decay`指定と
1次元パラメータを減衰から除外する。epoch末の小さいbatchも実際の標本数で勾配を平均する。
20,000更新・実効batch 32は約64万例、全訓練集合の約3.8周に相当する。
この予算だけで収束を仮定せず、検証曲線が上昇中なら追加予算も全方式で揃えて評価する。

読み出しには、近接研究の公開実装にある現在時刻だけを読むRVM/4DS型Attention readout
（1 query、768次元、12 heads）を使用できる。これは評価用の既存手法で、新規性には数えない。
ソースは[公開実装](https://github.com/tue-mps/towards-video-image-frozen/tree/59249bf83311bc34bae277e2e8adec287ffe5d0f)
のcommitとファイルハッシュに固定し、MITライセンスを保存する。`setup.sh`が取得を行う。
`--readout mean`で従来の空間平均＋線形読み出しも選べる。どちらも過去トークンを直接読まない。

```sh
uv run --locked python train_video.py --output results/ssv2_full_mamba
# 設定・コード・manifestを変えずにoptimizer更新境界から再開
uv run --locked python train_video.py --output results/ssv2_full_mamba --resume
```

検証は1,000更新ごとに公式validation全件で行い、bestは通常履歴の精度で選ぶ。
`last.pt`にはoptimizer、乱数、epoch内の位置を保存する。`--stop-after-step`は再開検査用の
実行制限で、総更新数や学習率スケジュールを変えない。中断・再開で条件を黙って変更しない。
AdamWのCPU更新カウンタをGPUへ移さないよう、checkpointはCPU経由で読み込む。
MambaのTriton逆伝播には原子的加算があるため、再現性をビット単位の一致とは区別する。
旧pilotの本実験は全方式とも再開なしで完了しており、この読み込み修正によるスコア変更はない。
フレーム数・batch・モデル規模などが近接研究と異なるため、その公表スコアの再現とは呼ばない。

[実動画での学習入口検査](results/video_training_integration_v2/verification.json)では、
5本の学習動画・3本の検証動画で3更新を実行した。epoch末の1標本だけのbatchも含め、
連続実行・独立再実行・中断再開を比較した。再開による重みの最大絶対差は約3.7e-9、
検証予測は一致した。損失には最大約9.6e-4の差があり、ビット単位の一致を主張しない。
この検査は精度・汎化の評価ではない。

[evaluate_video.py](evaluate_video.py)で、全動画用の`last.pt`を一括・逐次の両方で評価する。
encoderとreadoutを同じatomic checkpointから読み、読み込んだファイルのSHA256を保存する。
通常履歴・現在画像の繰り返し・過去だけの逆順で、同じ最終観測を維持する。
逐次入力の状態は各動画でリセットし、最終batchの標本数が違っても状態を使い回さない。
`--role test`では訓練集合とモデル選択用validationの両方とのgroup/path重複を拒否する。
別名コピーされた動画内容の一致はこの入口では検査しない。

```sh
uv run --locked python evaluate_video.py --checkpoint results/ssv2_full_mamba_interleaved_seed42_rgbfix/last.pt --manifest data/ssv2/videos_trainval/val.jsonl --role validation --output results/ssv2_full_mamba_interleaved_seed42_rgbfix/clip_stream_evaluation.json
uv run --locked pytest -q test_video_evaluation.py
```

[学習入口検査のcheckpointでの再評価](results/video_training_integration_v2/evaluation_clip_stream.json)では、
3本×3種の履歴条件で一括と逐次の予測クラスが一致した。logitの最大絶対差は0.00390625。
3更新だけのcheckpointなので、ここでの分類精度をモデル品質の結果には使わない。
学習時に保存した検証予測との一致も[照合結果](results/video_training_integration_v2/evaluation_verification.json)に記録する。

[benchmark_streaming.py](benchmark_streaming.py)では履歴長を固定して、新しい1枚の推論を測る。
各測定は同じ事前状態の深いコピーから始め、コピー・過去の入力・画像読み込みを時間に含めない。
他のGPU計算プロセスが稼働していれば測定を拒否する。学習を一時停止して測定し、終了後に再開した。
batch 1、224入力、空間FP32・時間BF16、各条件5回のwarmupと50回の測定による結果：

| 過去フレーム数 | 時間Attention GPU中央値 | Mamba-3 GPU中央値 | Attention状態（今回入力後） | Mamba-3状態（今回入力後） |
| --- | ---: | ---: | ---: | ---: |
| 6 | 10.67 ms | 10.76 ms | 4.02 MiB | 75.37 MiB |
| 128 | 11.28 ms | 10.80 ms | 74.07 MiB | 75.37 MiB |
| 256 | 12.93 ms | 10.91 ms | 147.57 MiB | 75.37 MiB |

これは未学習の時間層へ同じ画像を繰り返した**演算コストの測定**で、長期記憶の正確さを示さない。
両方式ともゼロ初期化の残差ゲートを使用するが、時間演算は省略せず実行する。
状態の数値はテンソル要素の合計で、重み・一時領域を含む総メモリではない。
Mambaは短い履歴では状態メモリが大きい。速度はこの実装・GPU・精度設定の測定範囲に限定する。
履歴なしの空間エンコーダは約7.9〜8.0 ms。
JSONにはGPU時間と同期込みwall time、状態サイズ、一時的なPyTorch割当増分を別々に保存する。
割当増分の基準は元の事前状態とそのコピーを保持した時点であり、総GPU使用量ではない。

```sh
# GPUで他の学習等を実行していないときに測定する
uv run --locked python benchmark_streaming.py --kind mamba --output results/new_streaming_cost.json
```

数値の原本：[履歴なし](results/streaming_cost/none_seed42.json)、
[時間Attention](results/streaming_cost/attention_seed42.json)、[Mamba-3](results/streaming_cost/mamba_seed42.json)。

全分割の実行条件・再現性設定は[config.json](results/ssv2_full_mamba_interleaved_seed42_rgbfix/config.json)、
経過は[progress.jsonl](results/ssv2_full_mamba_interleaved_seed42_rgbfix/progress.jsonl)に保存する。
学習の開始や更新回数は性能向上の証拠として扱わず、検証結果と対照実験で判断する。

以下の途中精度はRGB修正前の履歴であり、新しい学習の成績ではない。

1,000更新時の最初の全件検証は3,978/24,777、top-1 16.06%だった。
この時点で処理した学習動画は32,000例分で、学習集合の1周に達していない。
pilotからデータ量・読み出し・入力フレーム数・検証集合が変わっており、
6.32%との差をMambaの効果とは解釈しない。
同じ全データ設定の2,000更新時には7,201/24,777、top-1 29.06%だった。
[保存予測の独立再集計](results/ssv2_implementation_audit/full_run_metadata_audit_step2000.json)でも一致した。
この時点の学習処理数は64,000例分であり、まだ学習集合の1周に達していない。
同条件の対照モデルは未完了なので、精度上昇をMambaの優位性とは扱わない。
3,000更新時の全件検証は8,754/24,777、top-1 35.33%となり、
[保存予測の再集計](results/ssv2_implementation_audit/full_run_metadata_audit_step3000.json)でも一致した。
この時点の学習処理数は96,000例分であり、引き続き1周未満の学習途中の結果である。
4,000更新時の全件検証は9,735/24,777、top-1 39.29%だった。
[再集計](results/ssv2_implementation_audit/full_run_metadata_audit_step4000.json)では公式ラベル、
全分割、実行ソースと保存予測を再照合した。この値は8フレーム一括推論での途中評価であり、
最終結果や全件逐次推論の結果ではない。処理した学習例は128,000例分である。
[独立した記録監査](results/ssv2_implementation_audit/full_run_metadata_audit.json)で、
全193,690本の公式ラベル・分割・ファイルサイズ、実行中ソースとmanifestのハッシュ、
保存予測からの正解数を再照合した。動画内容のSHA256は抽出時の記録を使い、
この監査では全動画の再ハッシュやモデルの再推論は行っていない。

[保存重みの監査](results/ssv2_implementation_audit/full_run_weights_audit.json)も2,000更新の
checkpointで実施した。空間エンコーダの208テンソルは固定した公式SigLIP2重みと完全一致し、
全270モデルテンソルとoptimizer状態は有限だった。62個のoptimizer状態の更新数はすべて2,000で一致した。
3箇所の時間ゲートは各768成分が非ゼロになっており、初期のゼロゲートのままではなかった。
これは学習状態の確認で、履歴が認識に役立つことの証明ではない。履歴除去と対照モデルの評価が必要になる。

```sh
uv run --locked python audit_video_run.py --run results/ssv2_full_mamba_interleaved_seed42_rgbfix --output results/ssv2_implementation_audit/full_run_metadata_audit.json
uv run --locked python audit_video_weights.py --checkpoint results/ssv2_full_mamba_interleaved_seed42_rgbfix/last.pt --output results/ssv2_implementation_audit/full_run_weights_audit.json
```

[run_video_comparisons.py](run_video_comparisons.py)は全データ用の対照実験を逐次実行する。
中間層Mambaを基準に、履歴なし・時間Attention・出力後Mambaを、同じデータ・seed・
更新回数・読み出し・optimizer条件で学習する。パラメータ数と実時間の予算は一致しない。
実行中の基準モデルにはLinux pidfdで結び付け、終了後にcheckpoint、予定された全検証、
保存予測からの精度再集計、正常完了の記録を確認する。部分終了した基準から比較を始めない。
各実行前にソース・依存設定・manifest・公開readoutのハッシュを再確認し、差があれば停止する。
独立したGPU実行が残っている場合も同時学習を始めない。

```sh
# VIDEO_TRAIN_PIDにはps等で確認したtrain_video.pyのPIDを設定する
uv run --locked python run_video_comparisons.py --reference results/ssv2_full_mamba_interleaved_seed42_rgbfix --after-pid "$VIDEO_TRAIN_PID" --output results/new_full_comparisons
# 基準学習が正常に完了済みなら--after-pidを省略する
# 実行計画だけを保存する場合は、新しい出力先と--plan-onlyを指定する
uv run --locked pytest -q test_video_comparisons.py test_video_summary.py
```

進行中の比較計画は[plan.json](results/ssv2_full_comparisons_seed42_rgbfix/plan.json)に保存した。
全学習の完了後、最終更新の同じcheckpointで通常履歴の一括・逐次推論と、
一括推論での履歴除去・逆順を評価する。履歴なしモデルは通常履歴の評価だけを行う。
SIGINT/SIGTERMでキューを止めると、実行中の子ジョブへ停止を通知し、後続実験を開始しない。
基準モデルの終了を待っている段階では、キュー停止は基準モデルを停止しない。

[summarize_video.py](summarize_video.py)は、最終更新の逐次推論精度を主指標に集計する。
同じ動画ID・ラベル・元動画groupで予測を照合し、記録された精度・混同行列も再計算する。
データ・読み出し・更新予算・ソース版が異なる実験の比較は拒否する。
履歴除去の効果は一括推論どうしで比較し、逐次推論との差を混ぜない。
差の区間は元動画単位の対応付きbootstrapで、学習seed間のばらつきは含まない。
最良epochの選択は行わないが、公式validation上の比較であり、独立testの性能とは呼ばない。

```sh
# 全学習と評価が完了してから実行する
uv run --locked python summarize_video.py --runs results/ssv2_full_mamba_interleaved_seed42_rgbfix results/ssv2_full_comparisons_seed42_rgbfix/none_interleaved_seed42 results/ssv2_full_comparisons_seed42_rgbfix/attention_interleaved_seed42 results/ssv2_full_comparisons_seed42_rgbfix/mamba_late_seed42 --output results/ssv2_full_comparisons_seed42_rgbfix/summary.json
```

#### ロボット動画での密な対応の評価準備

動画分類と、物体上の位置を保てることは別の評価になる。
[RoboTAP](https://robotap.github.io/)の実ロボット動画265本と点軌跡を、追加の評価用データとして準備する。
データは評価専用とし、SSv2学習へ追加しない。
[prepare_robotap.py](prepare_robotap.py)は公式配布の約13.6 GBのZIPを再開可能な形で取得し、
約38.2 GBの5つのpickle shard等を展開する。配布元ETag・サイズとローカルSHA256を記録し、
ZIPの各メンバーのCRC32も検証する。上流の独立したSHA256との照合はしていない。
動画・注釈はCC BY 4.0で、RoboTAPの著者に帰属する。

```sh
uv run --locked python prepare_robotap.py
uv run --locked python robotap_data.py --output results/robotap_data_audit.json
uv run --locked pytest -q test_robotap_data.py test_tapvid_metrics.py
```

[robotap_data.py](robotap_data.py)は全5 shardのハッシュ、動画・点・遮蔽ラベルの形状、
最初に可視になるquery以降の評価対象を確認する。元のフレーム列を間引かず保持し、
全265本の動画長、注釈数、動画内容の重複を集計する。全動画を一度にメモリへ載せない。
取得・展開と[全件検査](results/robotap_data_audit.json)は完了した。
72,050フレーム・11,592軌跡で、動画長は52〜1,337フレーム。可視点の範囲外座標は0件だった。
名前の異なる8組で映像のSHA256が一致し、固有の映像内容は257件だった。
公式スコアは265本を維持し、独自の信頼区間では同一映像を同じgroupとして扱う。
同一映像でも注釈が同一とは限らず、複数カメラなどによる依存関係も残る。
[注釈を重ねた確認画像](results/robotap_annotation_audit.png)も保存した。
これは可視の正解点を描いたもので、モデルの追跡結果ではない。
配布ファイル内の`mediapy._VideoArray`は画素を保持した通常のNumPy配列として読み、
学習環境にmediapy等を追加していない。コンテナのFPSをこの配列から推定しない。

[tapvid_metrics.py](tapvid_metrics.py)には、
[公式実装の固定commit](https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapvid/evaluation_datasets.py)
からNumPyだけで動く評価関数とfirst-visible query関数を残した。
関数本体は変更せず、元の[Apache-2.0ライセンス](licenses/tapnet_APACHE2.txt)を同梱する。
指標の距離は256×256を基準にし、モデルの224入力とは区別する。公式の集計は動画単位の平均である。
この段階ではデータ・評価関数・未学習の読み出しの準備であり、点追跡精度や
ロボット制御の成功率を実証していない。
RoboTAPでは点ごとにqueryが現れる時刻が異なるため、共通読み出しはquery画像上の特徴と
座標を、その時刻に取得して保持する必要がある。座標だけを全時刻へ渡す方式では、
同じ座標を異なる時刻に通る別の物体を指定し分けられない。
点同士のqueryを混ぜず、query以後の正解座標や遮蔽ラベルを推論へ渡さない。
SSv2で学習した時間層を固定して読み出しだけ学習する転移評価と、
別タスクで時間層も学習する適応評価は、分けて報告する。

[point_readout.py](point_readout.py)は共通の点追跡読み出しを実装する。
query時刻の特徴を座標で双線形補間し、座標のFourier特徴と一緒に固定したquery表現へ変換する。
各queryは現在の画像特徴へcross-attentionを行い、座標と可視性を出力する。
query間のself-attentionはなく、既定ではエンコーダを凍結する。
これは評価用の共通部品であり、エンコーダの独自性や先行手法の完全再現とは主張しない。
損失はqueryより後だけに適用し、位置のHuber損失は可視点、可視性BCEは全評価点を対象にする。
各項は動画内で平均し、動画間には同じ重みを与える。可視点数が違っても、
microbatch分割によって損失の重みが変わらないようにする。
[CPU検査](test_point_readout.py)の7件で、座標の補間、query時刻の参照、未来・別queryからの
情報遮断、一括と逐次の一致、リセット、損失の対象範囲と勾配を確認した。
BF16 autocast下でも、指定されたquery座標を逐次推論時に丸めずFP32で保持する。
この問題は実GPU結合検査で見つけて修正した。出力座標と可視性logitをFP32へ戻し、
CPUでもautocastを使う回帰検査を加えた。

[実GPU結合検査](results/point_readout_gpu_integration.json)では、学習途中のSigLIP2＋Mamba-3へ
未学習の共通読み出しを接続し、実SSv2動画1本・8フレームを入力した。
一括と逐次の座標差は256画素基準で最大0.250画素、可視性logit差は最大0.001953だった。
未来画像を変更しても、それ以前の出力差は0だった。人工的な座標目標で読み出しへの勾配と
optimizer更新、エンコーダ凍結も確認した。この数値は正解軌跡に対する誤差ではない。
この初期検査後、補間を同じ座標規約のgather演算へ変更し、grid_sampleとの値・特徴勾配の一致を
CPUで確認した。CUDAの決定的な逆伝播に対応するためで、
[PyTorch 2.14の制約](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)に従う。
変更後のGPU学習・再開検査は以下のv2の結果を参照する。
全データでの読み出し学習とRoboTAP全件の精度比較は未実施。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q test_point_readout.py
```

[point_checkpoint.py](point_checkpoint.py)はエンコーダ・読み出し・構造・ソースのハッシュを
一つのcheckpointへ保存し、外部から別の重みを取得せず復元する。
[evaluate_point_tracks.py](evaluate_point_tracks.py)はRoboTAPの全フレームを因果的に処理する。
指定点は最初に可視になった時刻にだけモデルへ渡し、到着順で管理した出力を元の点順へ戻す。
正解軌跡や未来の遮蔽ラベルは指標計算にのみ使用する。
動画ごとに状態をリセットし、全265本の動画平均、予測ファイル、checkpointとデータのハッシュを保存する。
`--limit`を指定した実行は部分的な動作検査と明記し、全件の結果とは扱わない。
[評価入口の検査](test_point_evaluation.py)では、既知の等速運動に対する公式指標の一致、
queryの順序復元、未来の正解を変更しても予測が変わらないこと、checkpointの出力一致を確認した。
この検査も追跡モデルの認識精度の評価ではない。
実CLIの[接続検査](results/robotap_cli_integration/summary.json)も完了した。
RoboTAPの1動画・296フレーム・38点を処理し、予測と指標を保存した。
使用したモデルは3更新のみの動作確認用checkpointで、AJは0だった。
この実行から追跡精度を評価しない。全件評価では、development扱いまたは学習途中のcheckpointを拒否する。
この拒否条件と、明示した部分検査を許す条件もCPUで確認した。
エンコーダ状態はこの動画で79,027,200 bytes、query状態は最大78,128 bytesだった。
query状態は点の数で増えるため、点追跡器全体のメモリまで無条件に固定とは呼ばない。

[audit_robotap_predictions.py](audit_robotap_predictions.py)は、保存した予測を元のRoboTAP注釈から
再採点する。元動画のハッシュ、可視になった最初の時刻、queryの元順序、座標の保持を照合する。
指標関数は固定した上流実装とASTで一致することも確認する。
上記1動画の[独立再集計](results/robotap_cli_integration/raw_label_metric_audit.json)では、全指標が一致した。
モデルの再推論や正式な全件評価を行った結果ではない。部分検査には明示的な`--allow-subset`が必要となる。

```sh
# point_checkpoint.py形式の学習済みcheckpointを作成してから実行する
uv run --locked python evaluate_point_tracks.py --checkpoint results/tracker.pt --output results/robotap_tracking
# 全件評価の保存予測を元注釈で再集計する
uv run --locked python audit_robotap_predictions.py --report results/robotap_tracking/summary.json --output results/robotap_tracking/raw_label_metric_audit.json
```

点追跡の学習データには、公開された[CoTracker3 Kubric](https://huggingface.co/datasets/facebook/CoTracker3_Kubric)
を使う。固定版`a23b6e6e332595c9d92944680a10415e93bfd1e1`の全ページを列挙すると、
配布archiveは5,869件だった。丸めた6,000という規模を、取得できた実数として報告しない。
seed 42でシーンを分け、5,569件を学習、300件を検証に使う計画を先に固定する。
これは独自に分離した合成データの検証集合であり、RoboTAPは外部評価専用のままである。

[prepare_kubric_points.py](prepare_kubric_points.py)は、公開LFS SHA256と照合したarchiveから
全120フレーム、全32,768軌跡を保存する。元の512 RGBはPIL bilinearで224へ変換し、PNGとして保存する。
座標は元のfloat32バイトを可逆に並べ替えて圧縮し、量子化せず往復一致を確認する。
[`visibility`の扱いは公式readerの固定版](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/datasets/kubric_movif_dataset.py)
に合わせる。配布配列のTrueは遮蔽を意味し、可視性としてそのまま使わない。
取得archiveは保存結果を検証した後に削除し、空き容量40 GiBを残す。
全件完了するまで正式な学習・検証manifestを出力しない。
準備に使うソースは`data/cotracker3_kubric/preparation_source/`へ保存し、再開時の変更を検出する。

```sh
uv run --locked python prepare_kubric_points.py --workers 2
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q test_kubric_data.py test_point_readout.py test_point_evaluation.py test_tapvid_metrics.py
```

[kubric_data.py](kubric_data.py)は、シーンとepochから再現可能な時間区間・query点を選び、
元画像の幅・高さで正規化したxyと遮蔽を返す。検証時の選択はepochに依存しない。
queryは抽出区間内で最初に可視となる時刻に置き、最後のフレームにしか現れない点は学習queryにしない。
最初の公開2シーンで[8通りの読み込み](results/kubric_data_integration.json)と
[正解座標の可視化](results/kubric_annotation_audit.png)を確認した。
画像上の点は正解注釈であり、モデルの予測ではない。
全データの取得・変換は進行中で、経過は[準備ログ](results/prepare_kubric_points_full_v2.log)に保存する。
配布シーン0488には、遮蔽中の1点・1時刻だけ無限大座標があり、当初の「全座標が有限」という
読み込み条件で準備が停止した。可視点には有限値を必須とし、遮蔽点の元バイトはそのまま保存するよう修正した。
既存490シーンを再ハッシュし、失敗したシーンも全120フレーム・全32,768軌跡を保持して変換できた。
[元データ診断](results/kubric_scene_0488_source_diagnosis.json)と[修復検証](results/kubric_0488_repair_verification.json)を保存した。
遮蔽点は位置損失から除外する。非有限の可視座標を拒否すること、遮蔽座標で損失・勾配・評価指標が
変わらないことをCPUで検査した。この停止はSSv2の動画分類結果には影響しない。

#### 共通の点追跡読み出しを学習する

[train_points.py](train_points.py)は、`--mode probe`でエンコーダ全体を固定して読み出しだけを学習し、
`--mode adapt`で読み出しと時間層を更新する。どちらもSigLIP2本体は固定する。
probeを主な転移評価とし、点追跡の教師で時間層を学習するadaptとは結果を分ける。
既定は24フレーム・64点、microbatch 2・蓄積4、20,000更新、warmup 1,000、
AdamW・最大学習率1e-4・weight decay 1e-4。これだけで収束を仮定しない。
読み出しの初期化は、エンコーダの種類に依存しない同じseedから行う。
学習対象はprobeで14,932,995パラメータ、中間層Mambaのadaptで17,434,803パラメータとなる。

本実験では、全シーンの準備完了・分割ハッシュ・シーンと元archiveの非重複を検査し、
動画学習が指定更新数まで完了したcheckpointだけを初期値に使う。
`--development-run`は少数例での動作検査専用で、configにもその範囲を記録する。
学習設定・ソース・manifestを固定し、optimizer境界で保存する。
停止要求も予定された勾配蓄積を終えてから扱い、予定された検証を省略しない。
再開時には自己完結checkpointから全重みを復元し、元の動画学習の重みで置き換えない。
合成検証集合では、全シーンの予測・query・正解・遮蔽を保存し、公式TAP-Vid指標を動画平均で集計する。

```sh
# 全データ準備と初期値の動画学習が完了してから実行する
uv run --project baselines/gmmix --locked python train_points.py --encoder-checkpoint results/ssv2_full_mamba_interleaved_seed42_rgbfix/last.pt --output results/kubric_mamba_probe_seed42 --mode probe
# 同じ引数に--resumeを加えて再開する
uv run --project baselines/gmmix --locked python train_points.py --encoder-checkpoint results/ssv2_full_mamba_interleaved_seed42_rgbfix/last.pt --output results/kubric_mamba_probe_seed42 --mode probe --resume
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q test_point_training.py test_point_readout.py test_point_evaluation.py test_kubric_data.py test_tapvid_metrics.py
```

最初の[実GPU検査](results/point_training_integration/repeat_diagnosis.json)では、保存からの再開は一致したが、
同じseedでの最初からの再実行では重み差が最大1.36e-4となり、事前に決めた検査基準を超えた。
検査基準を広げず、PyTorchの決定的実行とcuDNN/CUBLAS設定を明示し、補間も対応させた。
[再検査](results/point_training_integration_v2/verification.json)では、公開3シーンの学習・別1シーンの検証で
3更新を実行し、連続・再開・再実行の重み・損失・予測差はすべて0だった。
probeではエンコーダがビット単位で不変、adaptの1更新ではSigLIP2本体を保って3時間層の各15テンソルが変わった。
これはGB10上の指定条件での小規模検査であり、任意の実行環境や全学習のビット一致を保証しない。
Mamba側は現在の環境で決定的モード用の固定したautotune候補を選ぶ。cached autotuningを勧める警告は記録に残す。

[実注釈の学習診断](results/point_learning_diagnostic/diagnosis.json)では、1つの固定した合成動画の
8フレーム・8点を、固定したエンコーダ特徴から学習した。事前に決めた400更新後、
損失は0.76975から7.03e-5へ下がり、学習対象の可視点はすべて1画素以内、可視性も一致した。
これは少数例の暗記であり、検証精度や汎化性能の結果には使用しない。

点追跡・学習・評価・データ・実験キューとGMMix比較に対するCPU検査は合計30件が通過した。
[検査記録](results/gmmix_point_comparison_cpu_verification.json)には対象コードのハッシュと実行範囲、
[進捗記録](results/point_pipeline_status.json)には完了事項と未実施事項を分けて保存する。
その後の遮蔽座標の修正と履歴介入を含む[CPU回帰検査](results/point_history_cpu_verification.json)は35件が通過した。
この2回の対象集合には重複があり、件数を合算しない。
全データでの点追跡学習と正式なRoboTAP比較はまだ実行していない。

[run_point_comparisons.py](run_point_comparisons.py)は、全動画の比較学習と合成データ準備の完了を待ち、
同条件の点追跡学習とRoboTAP評価を順番に実行する。履歴なし・時間Attention・中間層Mamba・
出力後Mamba・公開GMMixの5方式について、まずエンコーダ固定のprobeを行う。続いて時間層を持つ4方式を
adaptで学習する。履歴なしでは更新できる時間層がなく、固定空間エンコーダと同じ読み出しの
学習になるため、adaptの対照にも同じprobe結果を使用する。probeとadaptの結果は混ぜない。
同じseed・分割・query選択・読み出し初期値・20,000更新の条件を揃えるが、パラメータ数と実時間は揃わない。
全方式の点追跡学習には同じ`baselines/gmmix`のuv環境を使用する。
GMMixの読み込みには`--encoder-type gmmix`を指定し、専用の動画checkpointを復元する。

準備中のプロセスにはpidfdで結び付き、終了後にも全件データ・予定更新数・検証結果の完了を確認する。
中断した依存ジョブや、部分取得したデータから正式実験を始めない。
[point_experiment.py](point_experiment.py)で、合成検証の全予定回について保存予測からTAP-Vid指標を
再集計し、最終checkpointをRoboTAP全265動画へ適用する。RoboTAPのスコアでcheckpointを選ばない。
各RoboTAP評価の後には、元の注釈を読み直す独立再採点も実行する。
対応付き信頼区間・複数seed・長い履歴の利用能力の検証は別途必要である。

```sh
# 各*_PIDにはpsで確認した対応するPythonプロセスのPIDを設定する
uv run --locked python run_point_comparisons.py --video-comparisons results/ssv2_full_comparisons_seed42_rgbfix --gmmix-comparison results/ssv2_gmmix_comparison_seed42_rgbfix --output results/new_point_comparisons --after-video-pid "$VIDEO_QUEUE_PID" --after-data-pid "$KUBRIC_PREP_PID" --after-gmmix-pid "$GMMIX_QUEUE_PID"
# 依存ジョブが正常完了済みなら--after-*-pidを省略できる
# キュー再開は同じ出力先と条件に--resumeを追加する
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q test_gmmix_experiments.py test_point_experiments.py test_point_training.py test_point_readout.py test_point_evaluation.py test_kubric_data.py test_tapvid_metrics.py
```

現在の[実行計画](results/point_comparisons_with_gmmix_seed42_rgbfix/plan.json)は5 probe＋4 adaptの9実験である。
[状態記録](results/point_comparisons_with_gmmix_seed42_rgbfix/status.json)に加え、実プロセスの生存も確認している。
以前の7実験のキューは、点追跡学習を1つも始める前に停止し、この計画へ置き換えた。
さらにデータ準備の修正に合わせ、未開始の9実験を新しいソースハッシュと準備プロセスへ結び直した。
GPUジョブは同時に1つだけ実行する。中断時は子の学習へ停止を通知し、キューを再開すると
保存済みのoptimizer状態から学習を続ける。未完了のRoboTAP評価は保存して別ディレクトリで全件をやり直す。

#### 点追跡で時間状態が役立つかを検査する

[evaluate_point_history.py](evaluate_point_history.py)は、同じ動画に対し通常履歴と1・24フレームごとの
エンコーダ状態消去を比較する。絶対時刻を維持し、queryは到着時に作った通常履歴の記述子を全条件で共有する。
したがってquery記述子に含まれる過去は残る。これは継続するエンコーダ状態の寄与を調べる条件付きの介入であり、
完全な履歴なしモデルや速度比較とは呼ばない。全件評価には完了した本学習checkpointが必要となる。

[CPU検査](test_point_history.py)では通常推論との一致、未来画像の非影響、絶対時刻の維持、
履歴なしエンコーダでの条件間一致を確認した。
[実GPU検査](results/point_history_gpu_integration.json)でも、実動画の12フレームとMamba-3・GMMixで
通常推論と通常履歴枝の座標・logitが完全一致し、未来2フレームの変更はそれ以前の予測へ影響しなかった。
この検査はランダム初期化または3更新だけの点読み出しを使っており、追跡精度の成果ではない。
[評価CLIの結合検査](results/point_history_cli_integration/verification.json)も、実動画1本・296フレーム・38点で完了した。
通常履歴と2つの状態消去条件について、保存予測を元注釈から再採点して指標とquery座標の一致を確認した。
3更新だけのGMMix読み出しでAJは全条件0であり、この値から履歴の有効性を判断しない。

```sh
# 完了した本学習の点checkpointを使う。別GPUジョブが動いていないときに実行する
uv run --project baselines/gmmix --locked python evaluate_point_history.py --checkpoint results/your_point_run/last.pt --output results/new_point_history
```

#### 近接研究GatedMambaMixの比較準備

近接研究の[公開GatedMambaMix実装](https://github.com/tue-mps/towards-video-image-frozen/blob/59249bf83311bc34bae277e2e8adec287ffe5d0f/models/ssm_modules/mamba_modules.py)
は`mamba_ssm.Mamba`（Mamba-1）を使う。現在のMamba-3環境には、そのselective-scan CUDA拡張がない。
[prepare_gmmix_environment.py](prepare_gmmix_environment.py)で別のuvプロジェクトを生成し、
依存バージョンを既存のlockから引き継ぐ。Mambaのソースも固定commitからコピーし、
変更はビルド対象をGB10のSM121へ絞る設定と、PyTorch 2.14のヘッダが要求するC++20指定である。
モデル・カーネルの数式は変更しない。
元のプロジェクトの依存設定や環境は更新しない。

```sh
# 新しいcheckoutで一度だけ実行する。既存の出力先は上書きしない
uv run --locked python prepare_gmmix_environment.py
CUDA_HOME=/usr/local/cuda MAMBA_KEEP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 uv sync --project baselines/gmmix --locked
```

[ソース記録](baselines/gmmix/source_manifest.json)と[ビルド対象の差分](baselines/gmmix/sm121_build.patch)を保存する。
最初のビルドは上流のC++17指定がPyTorchの要求に合わず失敗し、C++20へ修正した。
修正後のビルドとCUDA拡張の読み込みは完了した。
追加の`causal-conv1d`拡張は含めておらず、畳み込みにはMamba-1のPyTorch経路を使っている。
出力後Mambaと公開GatedMambaMixは別の比較対象として扱う。

[gmmix_siglip.py](gmmix_siglip.py)は、同じSigLIP2の最終パッチ特徴を公開GMMixへ渡す。
GMMixは4層、幅768、状態次元16、空間Attention 16 heads、学習時dropout 0.1である。
各層が空間Attention・MLP、パッチ別Mamba-1の残差更新、学習ゲートを持つ。
公開コアは変更せず読み込み、ソースのハッシュを確認する。
この比較では空間エンコーダをSigLIP2-Bに揃えており、DINOv3を使う公表スコアの再現とは呼ばない。

| 学習する部分 | 中間層Mamba-3 | 公開GMMixを接続したSigLIP2 |
| --- | ---: | ---: |
| 時間処理・追加空間処理 | 2,501,808 | 48,164,352 |
| 共通の分類読み出しを含む合計 | 9,133,662 | 54,796,206 |

パラメータ数は一致しない。表は構造の規模であり、精度や速度の優劣を示さない。

[check_gmmix.py](check_gmmix.py)の[GPU検査](results/gmmix_integration/verification.json)では、
長さ7と257の入力でnative selective scanの出力と全入力の勾配を公式PyTorch参照計算と比較した。
FP32の最大相対L2差は約4.53e-7、BF16は約0.00355で、事前に決めた許容範囲内だった。
実SSv2動画1本・4フレームでも接続を検査し、一括・逐次の特徴差はFP32で約4.85e-6、
時間処理BF16で約0.00198の相対L2差だった。未来フレームの変更による過去出力の差は0。
状態は96,337,920 bytesで、5枚目を追加しても増えなかった。総GPUメモリや長期記憶精度の測定ではない。
逆伝播・optimizer更新後もSigLIP2は完全一致し、自己完結checkpointの保存・復元後の出力も一致した。

[train_gmmix_video.py](train_gmmix_video.py)は、この基準モデルの動画学習入口である。
実行中のMamba-3 trainerを変更せず、同じデータ処理・読み出し・学習率・正則化対象の選択を使う。
停止要求は予定された勾配蓄積の境界で扱い、予定された検証も省略しない。
完全な再開用checkpointは専用のformat 3で保存し、
[gmmix_checkpoint.py](gmmix_checkpoint.py)が外部重みを取得せず復元する。
カーネルのビルドハッシュと畳み込み経路も記録し、異なる実行経路での評価を拒否する。
[evaluate_gmmix_video.py](evaluate_gmmix_video.py)は共通の動画評価関数を使い、
一括・逐次、履歴除去・逆順の各条件を計算する。

```sh
# 他のGPU学習が動いていないときに実行する
uv run --project baselines/gmmix --locked python train_gmmix_video.py --output results/ssv2_full_gmmix_seed42
uv run --project baselines/gmmix --locked python evaluate_gmmix_video.py --checkpoint results/ssv2_full_gmmix_seed42/last.pt --manifest data/ssv2/videos_trainval/val.jsonl --role validation --output results/ssv2_full_gmmix_seed42/evaluation.json
```

[修正後の学習・再開のGPU検査](results/gmmix_training_integration_rgbfix_fp32/verification.json)は、実動画5本の学習と
別3本の検証で3更新を実行した。中断再開と連続実行の重み差は最大1.19e-7、
独立再実行との差は最大2.74e-6で、事前の許容値5e-6以内だった。
検証予測は一致し、評価CLIの再推論も保存済み予測と一致した。
3動画×3履歴条件では一括・逐次の予測クラスが一致したが、任意の動画での一致を保証しない。
これは実行経路の確認である。GMMixの全データ学習と正式な性能比較はまだ開始していない。

[run_gmmix_comparison.py](run_gmmix_comparison.py)で、既存4方式の動画比較が正常完了した後に、
同じ全データ・8フレーム・実効batch 32・20,000更新・読み出しでGMMixを学習する。
[実行計画](results/ssv2_gmmix_comparison_seed42_rgbfix/plan.json)と
[待機状態](results/ssv2_gmmix_comparison_seed42_rgbfix/status.json)を保存した。
このキューは、GPUで検査済みのbuildと実装を固定し、全予定回の検証予測を再集計する。
最終checkpointで一括・逐次推論と履歴の除去・逆順を評価する。

```sh
# VIDEO_QUEUE_PIDは既存4方式の比較キューのPythonプロセス
uv run --project baselines/gmmix --locked python run_gmmix_comparison.py --video-comparisons results/ssv2_full_comparisons_seed42_rgbfix --integration results/gmmix_training_integration_rgbfix_fp32 --output results/new_gmmix_comparison --after-video-pid "$VIDEO_QUEUE_PID"
```

GMMixの[点追跡学習検査](results/gmmix_point_training_integration/verification.json)でも、
3学習シーン・1検証シーン・3更新で、連続・中断再開・再実行の重み・損失・予測差は0だった。
probeではエンコーダが不変で、adaptではSigLIP2を保ち、4層すべての空間演算・Mamba・ゲートが更新された。
GMMixのadaptで学習する部分は共通の点読み出しを含めて63,097,347パラメータである。
保存したモデルで[実RoboTAP動画1本・296フレーム](results/gmmix_robotap_cli_integration/summary.json)の
オンライン処理も完了し、[元注釈からの独立再採点](results/gmmix_robotap_cli_integration/raw_label_metric_audit.json)と一致した。
この3更新モデルのAJは0であり、検査の成功を認識性能の成功とは扱わない。

[研究用途の利用条件](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf)
に従い、動画・画像・注釈は`data/`内で扱う。`.gitignore`でデータとモデル重みを除外している。
再現用コードの公開とデータ・学習重みの公開は分けて扱う。

実行中の中間層Mambaを基準にした同条件のpilot比較は
[run_temporal_comparisons.py](run_temporal_comparisons.py)で逐次実行する。
`--after-pid`には実際の`train_temporal.py`のPIDを、`--reference`にはその出力先を指定する。
Linux pidfdで当該プロセスの終了を待ち、指定epochまでの完了とコード・依存設定の一致を確認してから、
履歴なし・中間層時間Attention・出力後Mambaの順に学習する。並列GPU学習はしない。
ログと実行状態は`results/ssv2_pilot_comparisons/`へ保存する。
`status.json`は記録であり、稼働確認にはプロセスまたは実行セッションも確認する。
これは1 seedのpilotで、統計的な性能比較や論文の最終結果の代替ではない。

[summarize_temporal.py](summarize_temporal.py)は保存された予測を元動画group単位で対応づけ、
通常履歴と履歴変更の正誤差・条件付きbootstrap区間を集計する。
複数runを指定した場合はデータ・seed・学習条件の一致を要求し、指定epoch数が全runに記録された場合のみ方式間の差を出す。
区間は既存の学習済み重みと検証集合に条件付けられており、seed間変動、人物の重複、
validationによるepoch選択の楽観バイアスは含まない。pilotの最終性能の証明には使わない。

```sh
uv run --locked summarize_temporal.py --runs results/ssv2_pilot_mamba_interleaved_seed42 --epochs 3 --output results/ssv2_pilot_mamba_interleaved_seed42/paired_summary.json
```

[vision_mamba3.py](vision_mamba3.py)は、公式SISOを画像向けに包んだ小型エンコーダ。RGB → stride 4の畳み込みstem → 4方向Mamba-3ブロック × 2 → 位置ごとの特徴、という構成。

行優先・列優先の走査と、それぞれの逆順を同じ重みで処理する。状態は方向ごとに独立し、出力を元の座標へ戻して平均する。局所depthwise畳み込みとFFNも使う。既存の複数方向走査を使った試作であり、新規原理や既存論文の完全再現を主張しない。

```sh
uv run --locked try_vision.py --train-steps 20
uv run --locked try_vision.py --image /path/to/image.jpg --train-steps 20
uv run --locked pytest -q test_vision_layout.py
```

画像を指定しなければPyTorch Hubの公開サンプル画像を`data/`へ取得する。画像を128×128へリサイズし、16×16のブロックを25%隠した固定入力から、同じ画像を20回更新で復元する。教師モデルや蒸留は使わない。1枚への当てはめによる動作確認であり、未知画像の認識精度やmIoUの評価ではない。

GB10での確認結果（既定設定）：エンコーダ450,064パラメータ、入力`[1,3,128,128]`、出力`[1,128,32,32]`。全勾配は有限でMamba部分にも非ゼロの勾配が流れた。隠した領域のMSEは0.146885から0.012091へ低下。warm-up後のエンコーダ単体のGPU実行時間中央値は1.319 ms（batch 1、30回）。初回コンパイル・画像読み込み・復元headはこの推論時間に含まない。

FP32重み・状態パラメータとBF16 autocastを使う。エンコーダを直接利用する場合もCUDA上で同じautocast設定を使う。入力正規化は呼び出し側で行い、このデモではRGBの値域を`[-1,1]`にする。

[数値結果](results/vision/result.json)と[画像プレビュー](results/vision/preview.png)を保存する。プレビューは左から原画像、隠した入力、復元予測、隠した部分だけの補完。`results/vision/demo.pt`にはデモの重みを保存する。復元はまだ粗く、細部・境界・幾何の性能向上を示したものではない。

## 実行

### ViTとの実画像比較

Oxford-IIIT Petで、Mamba-3試作・小型ViT・共通Conv stem版ViTを、未学習から同条件で3seedずつ訓練した。[比較レポート](results/pets_compare/REPORT.md)に条件・結果・限界を記載している。

テストmIoUは順に72.86 ± 1.22%、71.01 ± 0.29%、73.89 ± 0.71%。この小規模条件では、Conv stem版ViTがMamba-3より高精度かつ高速だった。事前学習済みViT-Bとの比較ではない。

評価コードは[compare_vision.py](compare_vision.py)、集計は[summarize_comparison.py](summarize_comparison.py)。データの取得先・再実行コマンドはレポートを参照。

224×224でも同じ画像分割・学習条件・3seedで再評価した。[224比較レポート](results/pets_compare_224/REPORT.md)。出力は全モデル56×56位置。mIoUはMamba-3が71.71 ± 1.55%、通常ViTが70.75 ± 0.10%、Conv stem版ViTが72.37 ± 1.80%。Conv stem版との差は小さく、3seedだけで精度の優劣は断定しない。

224でのエンコーダ推論は順に2.942 / 0.577 / 0.908 ms、推論中に増えたPyTorch割当メモリは81.53 / 16.05 / 16.08 MiB。メモリ値はモデル重み・常駐データ等を除いた増分で、総使用量ではない。224でも今回のMamba試作の速度・メモリ優位は確認できなかった。入力解像度以外の学習条件は共通だが、正解マスクも各解像度に変換しているため、解像度間のスコア差は同一画素での差ではない。

```sh
uv run --locked compare_vision.py --size 224 --output results/pets_compare_224
uv run --locked summarize_comparison.py --root results/pets_compare_224
```

Mambaの期待される利点は、幅・状態サイズを固定したときに位置数に対して線形に全体の文脈を計算できること。ただし、全情報の保存やAttention以上の精度を保証しない。[Mamba原論文](https://arxiv.org/abs/2312.00752)。4方向走査は[VMamba](https://arxiv.org/abs/2401.10166)にもある既存の原理。今回の実装は方向ごとの全位置出力を保存・統合するため、逐次decodeでの一定サイズ状態の利点を、そのまま画像推論全体の一定メモリと解釈しない。

[Mamba-3論文](https://arxiv.org/abs/2603.15569)は離散化・複素数としての状態更新・MIMOを改善しているが、そこでの言語・状態追跡性能は、この画像試作での細部や幾何の改善を証明しない。今回使用したのはSISOで、MIMOは未評価。

### DINOv3 ViT-B/16との比較

DINOv3の公式ViT-B/16を、幅768・12層・patch16・85,669,632パラメータのまま、`pretrained=False`で追加した。DINOv2や縮小版への置換はしていない。公式ソースを固定commitで使い、必要な出力変換は本体の外に置く。

同じ224入力・データ分割・8epochs・3seedの小規模教師あり分割では、mIoU 69.07 ± 3.74%、エンコーダ推論5.773 msだった。[詳細レポート](results/pets_dinov3_224/REPORT.md)。これはDINOv3自己教師あり学習の再現でも、学習済みDINOv3の性能でもない。Mamba試作とは約190倍のパラメータ差があり、出力位置数も14×14対56×56で異なるため、同容量・同構造比較とは扱わない。

```sh
bash setup_dinov3.sh
uv run --locked compare_vision.py --size 224 --models dinov3_vitb16 --output results/pets_dinov3_224
uv run --locked summarize_dinov3.py
```

### Mamba単体の確認

このディレクトリで実行する。

```sh
uv run --locked try_mamba3.py
uv run --locked try_mamba3.py --train-steps 20
uv run --locked check_reference.py
```

- `try_mamba3.py`：GPUのforward/backward、有限値、warm-up後の推論時間を確認。
- `--train-steps 20`：固定した一つのbatchで、一時刻前の入力を予測する最適化デモ。汎化精度を測る実験ではない。
- `check_reference.py`：公式のFP32参照計算と、BF16 Tritonカーネルの出力・勾配を比較。長さ64/197、state64、head幅64。相対L2誤差の閾値は出力5%、各勾配10%。

デフォルト入力は`[batch=2, length=256, width=128]`。ランダム初期化のMamba-3ブロックであり、学習済みの言語モデル・画像認識器ではない。確認範囲はSISOの一括入力。MIMOや一tokenずつのdecodeを検証したとは扱わない。

結果は`results/`にJSONで保存する。初回はTritonのコンパイル・autotuneで時間がかかる。初回の壁時計時間とwarm-up後のGPU時間を分けて出力する。

## 環境を再作成

Linux aarch64、NVIDIA GB10、CUDA13.0対応driver、uv、Gitを使用する。GPU依存のため他のOS向け構成ではない。

```sh
bash setup.sh
uv sync --locked
uv pip check
```

[setup.sh](setup.sh)が公式ソースを固定commitで取得し、`uv sync --locked`で依存を同期する。手動の`pip install`や`python -m venv`は不要。

公式ソース：[state-spaces/mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)。固定commitは`e9594ce1c732d97440f0332fdc43170a2294dbfa`。Python package表示は`2.3.2.post1`だが、利用するclassは`Mamba3`。vendorの公式コードは変更していない。

GB10で実際に遭遇した依存問題への対処：

- TileLang0.1.8とapache-tvm-ffi0.1.12の組合せでimportが失敗したため、ffiを0.1.9に固定。
- NVIDIA cuSPARSELt0.8.1のARM64 wheel内部tagが`sbsa`になっており、uvが再インストールを繰り返す。setup内の[修正スクリプト](fix_arm64_wheel_tag.py)はライブラリのELFがAArch64であることを検査し、WHEELのtagとRECORDのみ修正する。ライブラリ本体は変更しない。この配布tag問題は[既存プロジェクトの導入資料](https://rosettacommons.github.io/tmol/latest/installation.html)にも記載されている。

Mamba-3本体の説明は[公式論文](https://arxiv.org/abs/2603.15569)を参照。
