# Mamba-3 MIMOをGB10で使うための実行確認

現在の主学習はSISOのまま固定する。MIMOへの置換は、認識精度を改善できるかを調べる次候補であり、
新しい原理や性能向上を立証したものではない。
native GPU実装は、ランダム重みの単体ブロックに加え、学習済み重みと実動画1本で順伝播・逆伝播・逐次処理を確認した。
CPU参照式による移行も照合した。MIMOの認識精度比較は未実施。

## 変わる計算

[Mamba-3原論文§3.3・付録C](https://arxiv.org/html/2603.15569v1#S3.SS3)のMIMOは、
一つの状態行列に対する書き込み・読み出しを複数の経路にする。
書き込みは外積一つから、複数の外積の和へ変わる。4枚の画像を追加するという意味ではない。
主なSSM行列の形を保てる一方、入力投影のパラメータと直前のキーを保持する状態は増える。
論文の言語タスクの改善を、動画認識やGB10での速度改善へそのまま移さない。

同じ幅256・状態次元64・head幅64・expand 2で構築した単体ブロックは次の通り。

| 構成 | パラメータ | 1パッチ用に確保する4状態の合計 |
| --- | ---: | ---: |
| SISO | 437,392 | 133,632 bytes |
| MIMO rank 4 | 544,912 | 136,704 bytes |

状態は公式`allocate_inference_cache`にBF16のK/Vを指定した値であり、画像モデル全体のGPUメモリではない。
主学習のSISOは別のTriton状態受け渡し経路を使うため、この割り当て値をその実測状態量と混同しない。
3層すべてのコアを同じ幅のMIMOへ置き換える場合、コアの追加パラメータは合計322,560となる。

## 実機で確認した範囲

[実行probe](exploration/check_mimo_native.py)は、公式のMIMO順伝播・逆伝播・逐次経路を試す。
失敗もJSONへ記録する診断用CLIであり、プロセスが終了したことを全検査合格と扱わない。

| 実行条件 | 結果 | 記録 |
| --- | --- | --- |
| 元環境、chunk 16 | 順伝播は完了。逆伝播が共有メモリ確保で失敗 | [結果](results/mimo_native_probe_v1/summary.json) |
| 元環境、chunk 8 | 順伝播と11パラメータへの有限な勾配を確認。逐次実行はAPI不一致で失敗 | [結果](results/mimo_native_probe_chunk8_v1/summary.json) |
| TileLang 0.1.15＋TVM-FFI 0.1.12の一時環境、chunk 8 | 順伝播は完了。逆伝播のコンパイルで失敗 | [結果](results/mimo_native_probe_overlay_v1/summary.json) |
| TileLang 0.1.8＋TVM-FFI 0.1.10の一時環境 | 一括経路のコンパイルは失敗。保存済みの同じ重み・入力による逐次実行は完了 | [一括](results/mimo_native_probe_ffi010_v1/summary.json)・[逐次](results/mimo_decode_ffi010_v1/summary.json) |

chunk 16の逆伝播は101,408 bytesを要求し、GB10のブロック当たりの上限101,376 bytesを超えた。
公式実装にも共有メモリ不足時のchunk縮小が案内されており、chunk 8では逆伝播が完了した。
これはGPU全体の空きメモリ不足を示すものではない。

元環境はTileLang 0.1.8・TVM-FFI 0.1.9・CUTLASS DSL 4.8.1。
逐次実行は`map_dataclass_to_tuple`という引数を古いTVM-FFIが受け取れずに止まった。
TVM-FFIだけを0.1.10へ切り替えた別プロセスでは、公式の逐次カーネルを実行できた。
両環境でPyTorchは同じ2.14.1+cu130を使う。
**一つの環境で学習・一括・逐次のすべてを確認したわけではない。**
学習用環境を更新せず、uvの`--no-sync --with`で呼び出し単位に依存を切り替えた。
[元環境のバージョン・lockの再照合](results/mimo_root_environment_verification.json)も一致した。
FFI 0.1.12だけの変更ではTileLang 0.1.8のimportが失敗したため、単に最新版へ上げる方針は採らない。
[importの失敗記録](results/mimo_ffi012_import.log)も保存する。

元環境で[保存したfixture](results/mimo_native_fixture_v1/summary.json)には、入力、全重み、一括出力を含める。
[逐次probe](exploration/check_mimo_decode.py)は同じfixtureを読み、2系列×8観測を1枚ずつ実行した。
一括との最大絶対差は0.015625、相対L2差は0.005074だった。bitwise一致とは報告しない。

[独立したCPU/FP64漸化式](exploration/mimo_fp64_reference.py)でも、同じ入力・重みから全ブロックの出力と4状態を計算した。
投影・RMS正規化・回転・台形則・MIMOの読み書きをPyTorchの通常演算で計算し、native SSMカーネルは呼ばない。
[照合結果](results/mimo_fp64_reference_v1.json)は次の通り。

| native出力とFP64参照の比較 | 相対L2差 |
| --- | ---: |
| 一括出力 | 0.006267 |
| 逐次出力 | 0.005327 |
| 最後のSSM行列 | 0.003741 |

ここまでの表は2系列の単体ブロックの数値診断である。実動画での追加検査は下記に分けて示す。
実行中の学習とGPUを共有したため、速度比較にも使わない。
[importされたMambaソースとvendorの照合](results/mimo_native_provenance.json)も記録した。

## 視覚アダプタへの接続とCPU検証

[MIMOアダプタ](exploration/mimo_siglip.py)を、実行中の学習とは独立した探索用モジュールとして実装した。
既存の正規化・幅変換・残差ゲートを引き継ぎ、内部のSISOコアをMIMOへ移す。
元のコアと同じ働きをする経路を一つ残し、追加経路の書き込み・読み出しの係数をゼロから学習させる。
学習済みSISOの重みを引き継げる設計であり、下記のCPU参照式で移行を確認した。

SISOとMIMOでは回転する座標の並べ方が異なるため、B/C投影、バイアス、正規化の重みを同じ置換で並べ替える。
状態次元64の場合は、回転対象の隣接対をMIMOの前半・後半の対へ移す。
この座標変更と追加経路のゼロ係数によって、実数演算では元の漸化式を保つ。
**BF16のnativeカーネル間で出力が一致するという保証ではない。**

[CPUテスト](exploration/test_mimo_siglip.py)は5件すべて通った。
[ログ](results/mimo_siglip_cpu_tests_v1.log)と[対象ソース・検証範囲](results/mimo_siglip_cpu_verification_v1.json)を保存した。

| 検査 | 条件と確認した範囲 |
| --- | --- |
| 移行前後の漸化式 | ランダム重み3条件、各2系列、長さ1・7・17。非一様なB/C正規化とバイアスを含め、独立したSISO/MIMOのFP64計算を絶対・相対許容誤差`1e-10`で照合 |
| 追加経路の学習可能性 | CPU参照式の自動微分で、追加の書き込み・読み出し係数に非ゼロ勾配。その係数を1回更新すると、追加B/C投影にも非ゼロ勾配 |
| 保存・復元と状態の分離 | 小型のランダムSigLIPモデルを用い、全重みの保存・復元が完全一致。別モデルの状態、旧SISO状態、学習モードの逐次呼び出しを拒否 |

最後の検査はファイル形式と状態の受け渡しの検査であり、画像の順伝播は行っていない。
初版のCPUテストとは別に、下記でnative画像経路を検査した。
主学習・比較待ちのジョブ・ルートuv環境には適用していない。

CPU検査だけを再実行する場合は、リポジトリのルートで次を実行する。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_mimo_siglip.py
```

## 学習済み重みと実動画によるCPU移行検査

[追加の検査コード](exploration/check_mimo_migration.py)では、保存済みの11,000ステップ時点の重みを使う。
3層それぞれでランダム入力2系列×長さ1・8・17・65を照合し、全12条件が絶対・相対許容誤差`1e-10`を満たした。
さらに、以前から失敗解析に使っているSSv2の動画3988を、通常の8観測・224×224で入力した。
SigLIP2と分類器はFP32で計算し、時間コア全体だけをSISO/MIMOそれぞれの独立したFP64参照式へ置き換える。

[結果](results/mimo_trained_migration_cpu_v1/summary.json)では、移行前後の全フレームの特徴と分類出力は完全一致した。
MIMOモデルを保存・再読込した後も、全重み・CPU参照式の特徴・分類出力が完全一致した。
[計画とソースハッシュ](results/mimo_trained_migration_cpu_v1/plan.json)、
[実行ログ](results/mimo_trained_migration_cpu_v1.log)も残した。
入力・特徴のfixtureと全重みのexportはローカルに保存し、公開リポジトリには含めない。

これは**学習済みモデルの数学的な初期動作と保存形式の検査**である。
GPUカーネルを呼ばず、候補の再学習も行っていない。動画は既知の1本であり、精度・汎化・速度の評価には使わない。
native画像経路の1動画での動作確認は下記の通りであり、同条件SISOとの認識精度比較は未完了である。

同じローカルの保存済みチェックポイント・動画がある場合に、未使用の出力先で再実行できる。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.check_mimo_migration \
  --output results/new_mimo_trained_migration_cpu
```

## 学習済み視覚モデルのnative GPU経路

[GPU検査コード](exploration/check_mimo_visual.py)で、同じ動画3988の8観測を入力した。
一括処理はルート環境、逐次処理はTVM-FFI 0.1.10のuv一時環境を使う。
最新の[CPU移行検査](results/mimo_trained_migration_cpu_v2/summary.json)が出力した全重みと入力は、初版とファイルハッシュまで同一である。
逐次処理の修正後も[CPUテスト5件](results/mimo_siglip_cpu_tests_v2.log)は通った。

| 検査 | 実測結果 |
| --- | --- |
| 一括順伝播 | 全特徴・分類出力が有限。同じ入力の再実行で完全一致 |
| 因果性 | 後半4枚を左右反転しても、前半4枚の特徴は完全一致 |
| 逆伝播 | 学習対象の71パラメータテンソルへ有限な勾配。3層すべての追加書き込み・読み出し係数に非ゼロ勾配 |
| 逐次推論 | 8枚を1枚ずつ処理でき、エピソードをリセットして再実行した出力・4種類の最終状態も完全一致 |
| 分割方法の変更 | 修正後は`1×8`と`3/1/4`で、特徴・分類出力・最終状態が完全一致 |
| 保持する状態 | バッチ1、196パッチ、3層で80,381,952 bytes（約76.66 MiB）。モデル全体のGPU使用量ではない |

[一括・逆伝播の記録](results/mimo_visual_clip_v2/summary.json)、
[逐次処理の記録](results/mimo_visual_stream_v2/summary.json)、
[保存出力の照合](results/mimo_visual_comparison_v2.json)を残した。
勾配の検査ではoptimizerを更新しておらず、重みは変えていない。
再実行では一部の勾配ノルムに小さな差があったため、native逆伝播のbitwise再現性まで確認済みとはしない。
有限な勾配があることだけでは、独立した参照式との勾配一致を意味しない。追加の照合は下記に分けて示す。

### 逐次入力の分割による数値差と修正

[初版の逐次経路](results/mimo_visual_stream_v1/summary.json)では、`1×8`と`3/1/4`の分割で特徴の最大絶対差0.934761、相対L2差0.004989が出た。
[記録した初版コード](results/mimo_visual_stream_v1/mimo_siglip_source.py)と同じ重み・入力で、
[差の発生箇所を調べるコード](exploration/check_mimo_chunk_numerics.py)も実行した。
[結果](results/mimo_chunk_origin_reproduced_v1.json)では、最初の時間アダプタへ入る正規化済み特徴にすでに小さな差があり、BF16投影後には差が大きくなった。
したがって、Mambaの状態更新だけを同じ順に実行しても、入力の数値は揃わない。

候補の`encode_chunk`を、受け取った画像を1枚ずつ同じ空間処理・投影・状態更新へ通す実装に変えた。
これにより上記の分割比較は出力・状態とも完全一致した。主学習のSISO実装には変更を加えていない。
この方法では、逐次APIに複数枚を渡しても画像側の処理を時間方向にバッチ化しない。速度上の影響は未測定である。

### 一括と逐次の数値差は別に残る

| 全フレームの画像特徴の比較 | 相対L2差 |
| --- | ---: |
| native SISO一括とCPU参照 | 0.005068 |
| native MIMO一括とCPU参照 | 0.005912 |
| native MIMO逐次とCPU参照 | 0.005280 |
| native MIMO一括とnative SISO一括 | 0.005302 |
| native MIMO一括とnative MIMO逐次 | 0.006307 |

MIMOへ移行した時のnative一括特徴の最大絶対差は0.619733、分類出力では0.015625だった。
MIMOの一括と逐次の特徴には最大絶対差1.109680があり、分類出力の最大絶対差は0.0625だった。
CPU参照式での完全一致を、BF16のnative経路の完全一致へ言い換えない。
この1動画では全経路の予測クラスは同じだったが、分類精度の不変性や改善の証拠にはしない。
GPUを主学習と共有しているため、速度も比較していない。
[主学習のソース・ルート環境が変わっていないこと](results/mimo_visual_environment_verification.json)も再確認した。

同じローカルfixtureがある場合は、未使用の出力先で再実行する。

```sh
uv run --locked python -m exploration.check_mimo_visual \
  --fixture-root results/mimo_trained_migration_cpu_v2 --mode clip \
  --output results/new_mimo_visual_clip
uv run --no-sync --with 'apache-tvm-ffi==0.1.10' python -m exploration.check_mimo_visual \
  --fixture-root results/mimo_trained_migration_cpu_v2 --mode stream \
  --output results/new_mimo_visual_stream
```

今後必要なのは、複数動画・長系列での検査、勾配照合の条件の拡張、同条件SISOとの認識精度比較、専有時の実測コストである。
`full_visual_model_qualified`は引き続き`false`とし、1動画の動作確認をモデル全体の性能評価として扱わない。

## native逆伝播と独立したCPU参照式の照合

[勾配検査](exploration/check_mimo_gradients.py)で、11,000ステップ時点の第4層のコアを調べた。
入力はランダムな隠れ特徴2系列×8観測であり、画像ではない。出力と固定したランダム目標の要素積の平均を目的関数にする。
同じ入力・重み・目的関数を用い、native BF16コアと通常のPyTorch演算によるCPU/FP64漸化式から、
11パラメータテンソルおよび入力への勾配をそれぞれ計算した。
追加経路をゼロにした移行直後と、追加の書き込み・読み出し係数を非ゼロにした場合を分ける。

| 条件 | パラメータ・入力ごとの勾配の最大相対L2差 | 追加で分解した投影・rank別の最大相対L2差 |
| --- | ---: | ---: |
| 移行直後 | 0.014849 | 0.008921 |
| 追加経路を有効化 | 0.010296 | 0.011103 |

[結果](results/mimo_gradient_reference_v1/summary.json)と[保存配列の再計数・投影/rank別の内訳](results/mimo_gradient_reference_v1/verification.json)を公開した。
大きな投影行列全体のノルムだけでは小さい追加経路の不具合が隠れ得るため、B/C投影を各rankに分けて照合した。
この2条件では、参照勾配が厳密にゼロの各ブロックはnativeでもゼロだった。
nativeを2回実行した勾配も、この小規模条件では完全一致した。上記の視覚モデル全体の逆伝播の再現性を保証する結果ではない。

実行前の[計画](results/mimo_gradient_reference_v1/plan.json)では、参照ノルムが`1e-8`を超え、相対L2差が0.1を超えるパラメータを調査対象にする診断基準を記録した。
該当するものはなかった。この基準は工学的な警報条件であり、認識精度や任意入力でのカーネルの正しさを保証する上界ではない。
複数層・長い系列・実際の画像特徴・他の損失での照合は残る。

```sh
uv run --locked python -m exploration.check_mimo_gradients \
  --fixture results/mimo_trained_migration_cpu_v2 --output results/new_mimo_gradients
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_mimo_gradients \
  --root results/new_mimo_gradients
```

## 再実行

固定したルートuv環境から、まだ存在しない出力ディレクトリを使う。
最初のコマンドは順伝播・逆伝播の後にfixtureを保存するが、元環境の逐次API不一致も記録する。

```sh
uv run --locked python -m exploration.check_mimo_native \
  --chunk-size 8 --output results/new_mimo_fixture
uv run --no-sync --with 'apache-tvm-ffi==0.1.10' python -m exploration.check_mimo_decode \
  --fixture results/new_mimo_fixture/fixture.pt --output results/new_mimo_decode
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.mimo_fp64_reference \
  --fixture results/new_mimo_fixture/fixture.pt --decode results/new_mimo_decode \
  --output results/new_mimo_reference.json
```

1動画での視覚アダプタ接続と小規模な勾配照合を終え、次は[同条件SISOとの追加学習比較](MAMBA3_MIMO_PILOT.md)を準備した。
主学習20,000更新と既存の比較・費用測定・履歴診断の完了後に実行する計画であり、認識精度の改善結果はまだない。
MIMOという既存の方式へ置換したこと自体を研究の独自性とは数えない。
