# 保存長を制限した時間Attentionの対照

2026-11-05。長い履歴でのMambaの費用を評価する際には、全履歴を保存するAttentionとの比較に加え、
古いK/Vを破棄するAttentionとの比較も必要になる。
[全履歴のバッファ対照](ATTENTION_CACHE_CONTROL.md)は保存方法だけの比較だが、今回の対照は保持できる入力履歴も変える。
Sliding-window attentionとリングバッファは既存の方法である。
[Mistral 7B §2](https://arxiv.org/html/2310.06825v1#S2)も、固定容量のキャッシュと、層を重ねることで窓より広い履歴が届く点を説明している。
これは新しいアーキテクチャ案ではなく、性能主張を検証するための対照である。

## 実装した範囲

[WindowedAttention](exploration/windowed_attention.py)は、既存の時間AttentionのQ/K/V・出力射影を
同じパラメータオブジェクトのまま共有する。位置符号にはエピソード先頭からの観測番号を使い続ける。
窓長Wは**現在の観測を含む**保持数とする。各層でK/VをW個分確保し、観測tを`t % W`の位置へ書く。
満杯になると最古のK/Vを上書きする。未使用領域はAttentionに渡さない。
KとVを同じ順序で並べ替えても、厳密な算術ではAttentionの結果が変わらないため、
リング全体を時刻順へコピーし直す処理は不要である。有限精度での完全一致を保証する性質ではない。

評価モードかつ勾配なしで使用する。`variants(encoder, W)`で元の時間Attentionと固定窓版を取得し、
既存の`select`で切り替える。各方式は独立したエピソード状態で開始する。
一括入力も検査用に使えるが、T×Tのマスクを作るため、一定容量の逐次推論の費用として測らない。
費用の評価対象は1観測ずつの`encoder.step`とする。

重みだけの保存では窓長・破棄規則を表せない。通常のcheckpointは元の全履歴Attentionとして復元されるので、
固定窓を使う評価では、元checkpointのhashとともに窓長・この実装のhashを別に記録し、復元後に対照を適用する。
現在の8観測学習ではW≥8なら入力履歴を削らないため、同じ学習済み重みを使って推論時の制約を比較できる。
長い動画での認識維持や高速化は、まだ測定していない。

## 状態容量と入力履歴の範囲

196位置・3時間層・幅256・K/VがBF16の場合、確保量は`602,112 × W + 24 bytes`。
24 bytesは3層のCPUカーソルで、モデル重み・一時テンソル・画像処理のメモリは別である。
以下は構造からの計算値で、GPU上の実測値ではない。

| 各層の窓長W | K/V＋カーソル | MiB | 3時間層を通じて影響し得る入力観測数の上限 |
| ---: | ---: | ---: | ---: |
| 8 | 4,816,920 bytes | 4.59 | 22 |
| 32 | 19,267,608 bytes | 18.38 | 94 |
| 128 | 77,070,360 bytes | 73.50 | 382 |

主Mambaの実測キャッシュは79,027,200 bytes（75.37 MiB）なので、W=128は近い状態容量になる。
総GPUメモリや速度が同じになるという意味ではない。

時間層lの出力は、その直前の層の時刻tからt−W+1までを参照する。
画像内の空間演算は時間範囲を増やさないため、L個の時間層を通じた過去方向の最大距離は`L(W−1)`、
現在を含む観測数は`1 + L(W−1)`となる。これは到達できる入力範囲の上限であり、その範囲内の情報を有効に保持する保証ではない。

したがって、W=128の3層モデルを「128観測より前の画像の影響がない」と解釈してはいけない。
既存の[遅延保持診断](DELAYED_ACTION_MEMORY.md)の最大128観測の遅延だけでは、
この対照の入力範囲を超えた保持を立証できない。W=32の最大範囲は94観測であり、同じ診断でも条件が異なる。
自然動画での長期保持については、実際の証拠時刻と質問時刻を使って別途評価する必要がある。

## 実施した検査と残る作業

[CPU検査6件](results/windowed_attention_cpu_tests.log)が通過した。
[検査コード](exploration/test_windowed_attention.py)は次を確認する。

- W=1・3・8、29観測の複数回のリング周回と、長さの異なるchunk。FP64で独立した時刻順の行列積・softmax参照と、絶対・相対許容値を各1e-12として一致。
- 確保したバッファのアドレスと容量を維持し、未使用領域をNaNにしても結果へ混ざらないこと。
- カーソルやshapeの不一致、勾配あり・学習モードを拒否し、拒否時に既存キャッシュを変更しないこと。
- 未学習・合成画像の小型SigLIPで、破棄が起こる前の全履歴版との一致、パラメータ共有、状態初期化。
- 2時間層・W=3では、最初の画像の影響が1層分の窓より遠くへ伝わり、最大範囲を超えると消えること。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_windowed_attention.py
```

[確認記録](results/windowed_attention_cpu_verification.json)はCPU上の数値・状態検査に限る。
224入力の実画像・BF16 GPU経路、最終学習済み重みでの認識、速度測定は未実施である。
主学習と既存の予約キューは維持する。次はGPU数値検査を既存のGPU実験列が完了し、GPUが空いてから行い、
その結果を確認してから同じ最終Attention重みで窓長8・32・128の認識と費用を比較する。
固定窓を含む結果がない段階で、Attention全般に対する長履歴の優位性を主張しない。

## 実画像GPU検査の準備

[検査用コード](exploration/check_windowed_attention.py)を追加し、
[計画](results/windowed_attention_qualification_v1/plan.json)をCPU上で保存した。
既存のバッファ対照と同じSSv2の17動画・136観測を使う。事前学習済みSigLIP2と未学習の時間Attentionを使い、
時間経路が出力へ影響するよう残差ゲートを0.1に固定する。入力画像と重みはローカルのfixtureへ保存し、hashで固定する。
この検査は分類器を評価せず、認識精度の実験とは区別する。

参照側は時刻順のK/Vを切り詰め、変更していない元のAttentionへ1観測ずつ渡す。
窓長8・32・128をFP32とBF16の両方で比較し、各フレームの特徴差、状態量、バッファのアドレス維持を記録する。
全特徴に対する事前の警戒値は、FP32では最大絶対差0.001・相対L2差1e-5、
BF16では最大絶対差0.125・相対L2差0.005とした。数理的な誤差保証や精度維持の保証ではない。
超過した場合は結果を保存して失敗とし、その結果に合わせて値を広げて通過扱いにはしない。

元のCPU検査と合わせて[14件が通過した](results/windowed_attention_qualification_cpu_tests.log)。
追加検査ではW=128を含む145観測で参照側も独立した行列積・softmax実装と照合し、
主学習が未完了の場合と、完了ログがあっても対応する学習プロセスが生きている場合の起動拒否を確認した。
[実際の主学習中の拒否確認](results/windowed_attention_preflight_refusal.json)でも、GPU検査結果は生成されず、準備済みの入力は変わらなかった。
[準備記録](results/windowed_attention_qualification_preparation.json)を保存した。初版の計画自体は未実行・未登録で、
後述のリセットを扱う版を、既存実験列の後に検査する。

再生成はリポジトリのルートで実行する。出力先は新しいディレクトリを使う。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.check_windowed_attention prepare \
  --output results/new_windowed_qualification
uv run --locked python -m exploration.check_windowed_attention run \
  --output results/new_windowed_qualification
```

後半は主学習の20,000更新完了とプロセス終了を要求する。進行中の主学習を停止する処理はない。
この起動ガードが確認するのは主学習だけなので、実行時には既存の比較列も完了し、GPUが空いていることを別に確認する。
比較キューは各処理の開始時に他のGPUプロセスがあれば停止するため、主学習が終わっただけでこの検査を同時起動しない。
この検査で時間を測らず、正式な費用測定には別途GPUの占有を確認した条件が必要である。

## 履歴を消して絶対時刻を維持する対照

遅延保持テストの`suffix_only`は、最初の8観測の状態を消してから、観測番号8で後続画像を読み始める。
上記の初版は空のキャッシュを時刻0からだけ開始できるため、この対照へそのまま接続すると拒否される。
絶対時刻を0へ戻すと位置符号の条件まで変わるため、
[リセットを扱う別実装](exploration/resettable_window_attention.py)を追加した。初版とその固定済み計画は維持する。

新しいキャッシュには、次の絶対時刻tとリセット時刻t0を保存する。
書き込み位置は`(t - t0) % W`、読み出す有効な個数は`min(t - t0 + 1, W)`とし、位置符号にはtを使う。
これにより、時刻が進んでいても未使用スロットを読むことなく、履歴だけを消せる。
この版は`encoder.step`による1観測ずつの推論専用で、一括clipや複数観測をまとめたchunkは受け付けない。
パラメータは元のAttentionと共有する。BF16の3層・196位置の状態量は`602,112 × W + 48 bytes`で、
48 bytesは各層のCPU上の2つのint64値である。初版の24 bytesとは区別する。
入力の最大到達範囲`1 + 3(W - 1)`は変わらない。

[CPU検査22件](results/resettable_window_attention_qualification_cpu_tests.log)が通過した。
時刻0・8・137から開始し、W=1・3・8・32・128で145観測を処理した結果を、独立した時刻順の行列積・softmax計算とFP64で照合した。
時刻0では初版の1観測処理と完全一致し、確保容量・アドレスの維持、未使用スロットのNaN非混入も確認した。
小型の未学習SigLIPと合成画像では、既存の`stream_features`を変更せずに履歴消去の対照まで実行できた。
異なる証拠画像から状態を消した後の出力は一致し、窓の到達範囲を超えると証拠画像の差は消えた。
これは実画像の認識改善や、学習済み最終モデルによる比較ではない。

[実画像GPU検査のコード](exploration/check_resettable_window_attention.py)と
[計画](results/resettable_window_attention_qualification_v1/plan.json)も準備した。
初版と同じ保存済み重み・136観測を再利用し、時刻0からの136観測と、時刻8からの128観測を、
W=8・32・128、FP32・BF16で比較する。特徴の比較数は計1,584件で、数値の警戒値は初版から変更しない。
参照側は元のAttentionへ時刻順のK/Vを渡し、過去分をW−1個へ切り詰める。

主学習の完了に加え、既存の比較列の後に予約したMIMO試験の正常完了と、GPUが空いていることを要求する。
[現在の学習中に起動を拒否する検査](results/resettable_window_attention_preflight.json)も通過し、GPU結果は作成されなかった。
GPU実行は後述の比較キューへ予約したが、まだ実施していない。数値検査の通過後に、最終Attention重みで遅延保持の特徴を取得し、
同じデータ・容量・正則化の読み出しを学習して比較する。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q \
  exploration/test_resettable_window_attention.py exploration/test_resettable_window_qualification.py
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.check_resettable_window_attention prepare \
  --output results/new_resettable_window_check
uv run --locked python -m exploration.check_resettable_window_attention run \
  --output results/new_resettable_window_check
```

再実行時は初版の保存済みfixtureと先行実験を用意し、必要に応じて`prepare`の`--fixture`と`--after-run`で対応する出力先を指定する。

## 同じ遅延保持課題への接続

[特徴取得用コード](exploration/cache_windowed_delayed_action.py)を追加し、
[W=8・32・128、seed=42・43・44の9条件](results/windowed_delayed_action_plan_v1.json)を準備した。
元の比較計画にある最終Attention checkpointのパスと学習設定を照合し、20,000更新の重みだけを使う。
元のデータ計画と同じ696本の学習用証拠・348本の検証用証拠、共通の後続画像、遅延0・1・8・32・128観測を用いる。
`stream_features`とCPUのridge読み出しは、既存のMamba・全履歴Attention比較と同じコードを呼ぶ。
エンコーダは再学習せず、現在トークンの平均特徴だけを読み出しへ渡す。
各窓長・各条件について、学習集合だけで読み出しを学習し、正則化を検証スコアで選び直さない。

[CPU検査4件](results/windowed_delayed_action_cpu_tests.log)では、小型SigLIPと合成画像から
元の読み出し形式へ特徴を渡し、クラス非依存の後続画像の対照が正確にchanceになること、
検証特徴を変更しても学習した読み出しが変わらないこと、別のcheckpointや学習設定を拒否することを確認した。
未完了・失敗・比較行が欠けたGPU数値検査も拒否する。
[現時点の起動確認](results/windowed_delayed_action_preparation.json)はGPU検査未完了で停止し、出力やGPUジョブを作成しなかった。

実データの特徴取得・読み出し学習・方式間集計は未実施で、下記のキューで先行実験の完了を待機している。
数値検査の通過後、各条件について次の形式で実行する。`<final-attention-checkpoint>`には
元の比較計画に記載された同じseedの最終checkpointを指定する。

```sh
uv run --locked python -m exploration.cache_windowed_delayed_action \
  --checkpoint "<final-attention-checkpoint>" --window 128 \
  --output results/new_window128_cache
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.delayed_action_probe \
  --cache results/new_window128_cache --output results/new_window128_probe
```

この対照は人工的に遅延を入れた課題の評価であり、自然動画の長期記憶やSOTAを直接測るものではない。
最大128観測の遅延ではW=128の3層モデルの最大到達範囲を超えない点も維持する。

## 予約した実行順と比較結果の再計数

[比較キュー](exploration/run_windowed_delayed_action.py)を追加し、
[固定した実行計画](results/windowed_delayed_action_queue_v1/plan.json)で待機を開始した。
MIMOの待機プロセスをPIDファイルディスクリプタで束縛する。MIMO側が実行プログラムへ切り替わっても同じプロセスを追跡し、
終了後にMIMOと元の遅延保持比較の正常完了を確認してから、次の順で進める。

1. リセットを扱う固定窓Attentionの実画像GPU数値検査。
2. 固定した3窓長×3 seedの特徴取得とCPU読み出し学習。
3. 元のMamba・全履歴Attentionの6条件と、新しい9条件の保存結果を再検証して比較。

[集計コード](exploration/aggregate_windowed_delayed_action.py)は、動画の入力ハッシュ、証拠IDとラベル順序、
Attentionの同一checkpoint、同じデータ計画を照合する。
元のCPU処理で読み出しを再学習した結果に加え、保存した読み出し重みからも予測を再現する。
窓長・遅延・seedごとに、Mambaと固定窓Attentionの差、全履歴Attentionと固定窓Attentionの差を別々に示す。
対応する動画の正誤の増減、3 seedの平均差と標本標準偏差を保存する。統計的な有意差やSOTAの判定とは呼ばない。

[CPU検査13件](results/windowed_delayed_queue_cpu_tests.log)では、集計の符号と百分率、seed/窓長の欠落・重複、
入力や重みの取り違え、保存読み出しの不一致、先行処理の失敗を検査した。
模擬プロセスで、待機をキャンセルしても先行プロセスを停止しないことも確認した。
[実際の待機接続](results/windowed_delayed_action_queue_v1/launch_verification.json)では、
先行プロセスへのPIDファイルディスクリプタの接続、未完了時の拒否、主学習と既存キューの維持を確認した。
これらは実行管理と集計処理の検査であり、認識性能の結果ではない。

キューを再作成する場合は、先に実験計画とMIMO待機処理を用意し、未使用の出力先を指定する。
`<mimo-scheduler-pid>`は`run_mimo_after_comparisons`の実プロセスを確認して置き換える。

```sh
uv run --locked python -m exploration.run_windowed_delayed_action \
  --after-pid "<mimo-scheduler-pid>" --output results/new_windowed_queue
```

同じ実験計画のキューを重複起動できないようロックする。途中まで作成された結果は自動で上書きせず、原因を確認する。
このキューには費用測定を含めておらず、固定窓Attentionとの速度・総メモリ比較は別途必要である。

## 固定窓を含む速度・メモリ測定の準備

[測定コード](exploration/benchmark_windowed_attention.py)と
[3実動画の計画](results/windowed_cost_plan_v1.json)を追加した。
既存の費用比較と同じ52・245・1,337フレームのRGB動画を全フレーム処理する。
Mamba、全履歴のバッファ版Attention、W=8・32・128の固定窓Attentionを比較する。
MambaとAttentionには、それぞれseed=42の20,000更新時の最終重みを使う。
4つのAttention方式は同じAttention重みを共有する。

各方式は1回のウォームアップ後に5回測定する。実行順を巡回させ、5方式が各順番に1回ずつ来るようにする。
既存の3回測定の値をこの比較へ流用せず、5方式を同じ測定セッションで取り直す。
入力は224、batch=1、出力は現在画像の196×768特徴で、分類器は計時しない。
測定中にGPUへ置くエンコーダは1個とし、方式間の重み転送・動画デコード・数値照合は計時から除く。
前処理と入力転送を含むパイプライン時間も、モデル時間と分けて保存する。

計時前には、各動画の全フレームで全履歴の元実装とバッファ版を照合する。
各固定窓についても、時刻順にK/Vを切り詰める参照と特徴を照合する。
全履歴対照の最大絶対差の許容値は1e-5、固定窓のBF16警戒値はGPU数値検査と同じ最大絶対差0.125・相対L2差0.005とする。
超過した結果を保存して停止する。固定窓と全履歴の出力を一致させる検査ではなく、各方式が意図した計算を行うかの検査である。
確保したK/V容量とCPU時刻情報、PyTorchが追跡するピーク使用量を区別して記録する。

[CPU検査17件](results/windowed_cost_cpu_tests.log)が通過した。
小型SigLIPによる19観測の参照照合、処理失敗時の元の演算への復帰、行や容量の取り違え、
比較キューの未完了・結果改変・条件不足の拒否を検査した。
[実際の起動前確認](results/windowed_cost_preparation.json)でも、主学習のGPU使用を検出して停止し、
CUDAの初期化や測定出力の作成は起こらなかった。
GPUが一時的に空いていても先行する認識比較キューが未完了なら開始しない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_windowed_cost.py
uv run --locked python -m exploration.benchmark_windowed_attention \
  --selection shortest --output results/new_windowed_cost_shortest
```

後半の実行には、既存の学習・比較列、GPU数値検査、固定窓の認識比較キューの正常完了が必要である。
`--selection`には`shortest`、`median`、`longest`を指定し、毎回新しい出力先を使う。
この準備記録を保存した時点では費用測定は未予約で、その後、以下の待機処理と再計数へ接続した。
実測値はまだ得られておらず、速度や総メモリの優位性は未確認である。認識保持は別途の遅延保持結果と合わせて判断する。

## 費用測定の待機接続と結果検証

[費用測定キュー](exploration/run_windowed_cost.py)を起動し、
[固定した計画](results/windowed_attention_cost_queue_v1/plan.json)に従って、先行する固定窓の認識比較の終了を待機している。
先行プロセスをPIDファイルディスクリプタで追跡する。終了しただけでは測定を開始せず、
認識比較の正常完了、全9条件、対応するGPU数値検査と保存結果を照合してから、3動画を順番に測定する。
各動画の測定前にはGPUの占有も確認する。既存の主学習・比較キューを停止する処理はない。

[結果検証](exploration/verify_windowed_cost.py)は、全フレーム・全5巡の測定行を読み直し、
履歴長別の時間分布、最大状態容量、PyTorchピーク使用量、Mambaとの費用比を再計数する。
予定された実動画・最終checkpoint・方式・窓長・実行順・数値の警戒値と一致しない記録を拒否する。
数値照合の全フレームが揃っていること、記録されたGPU使用量が生存中のパラメータとGPU状態を下回らないことも確認する。
これは保存された観測値の検証であり、GPU推論を独立に再実行したことや、数学的な精度保証を意味しない。

[追加のCPU検査17件](results/windowed_cost_queue_cpu_tests.log)が通過した。
合成した時間記録による費用比の向き、hashを更新しても検出される測定条件・数値・容量の取り違え、
模擬子プロセスによる3動画の順次実行、途中の結果の保持、待機キャンセル時に先行プロセスを維持することを確認した。
[実際の待機接続の記録](results/windowed_attention_cost_queue_v1/launch_verification.json)では、
先行プロセスへの接続と、既存9プロセス・固定したソースの維持を確認した。GPUを使用しているのは主学習だけだった。
これらのCPU検査から速度・認識性能の改善は判断しない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_windowed_cost_queue.py
uv run --locked python -m exploration.run_windowed_cost \
  --after-pid "<windowed-recognition-queue-pid>" --output results/new_windowed_cost_queue
```

再作成時には、先行する`run_windowed_delayed_action`が実際に稼働中であることを確認してPIDを置き換える。
同じ費用計画を複数のキューから実行しないようロックする。既に完了した測定は検証後に再利用し、途中の結果は上書きしない。
3動画×5方式の実測、認識保持と費用を合わせた評価、論文への最終結果の反映は残っている。
