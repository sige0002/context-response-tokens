# Mamba-3 MIMOをGB10で使うための実行確認

現在の主学習はSISOのまま固定する。MIMOへの置換は、認識精度を改善できるかを調べる次候補であり、
新しい原理や性能向上を立証したものではない。
今回確認したのは、ランダム重み・2系列×8観測の単体ブロックである。実画像の認識評価は未実施。

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

これは2系列の数値診断である。native勾配の独立照合、長系列、実画像、保存・復元前後の視覚出力の確認は残る。
実行中の学習とGPUを共有したため、速度比較にも使わない。
[importされたMambaソースとvendorの照合](results/mimo_native_provenance.json)も記録した。

## 視覚アダプタへの接続とCPU検証

[MIMOアダプタ](exploration/mimo_siglip.py)を、実行中の学習とは独立した探索用モジュールとして実装した。
既存の正規化・幅変換・残差ゲートを引き継ぎ、内部のSISOコアをMIMOへ移す。
元のコアと同じ働きをする経路を一つ残し、追加経路の書き込み・読み出しの係数をゼロから学習させる。
学習済みSISOの重みを引き継げる設計だが、学習済みチェックポイントを使った実画像の移行評価はまだ行っていない。

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
このモジュールのnative逐次経路、画像特徴、認識精度、保存・復元前後の予測の照合は次段階に残る。
主学習・比較待ちのジョブ・ルートuv環境には適用していない。

CPU検査だけを再実行する場合は、リポジトリのルートで次を実行する。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_mimo_siglip.py
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

次の判断に必要なのは、この経路を視覚アダプタへ接続したときの正しさと、同条件SISOに対する認識精度である。
MIMOという既存の方式へ置換したこと自体を研究の独自性とは数えない。
