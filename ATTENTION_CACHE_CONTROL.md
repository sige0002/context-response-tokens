# 時間AttentionのK/V実装による費用を分ける

2026-11-05。時間AttentionとMambaの比較に、同じ学習済みAttentionのK/V保存方法だけを
変えた実装対照を追加した。既存の学習・主比較のソースと条件は固定したまま扱う。

## 調べること

[現在の時間Attention](temporal_siglip.py)は各観測で`torch.cat`を使い、
過去のK/Vと新しいK/Vを連結する。Attentionの読み取りに加えて、過去K/Vの再コピーが発生する。
[予約バッファ版](exploration/buffered_attention.py)は、最大観測数Lの領域を確保し、
新しいK/Vだけを対応位置へ書く。Attentionへ渡すのは書き込み済みの範囲だけである。

投影行列・位置符号・全履歴へのAttention・因果マスクを維持し、元のパラメータオブジェクトを共有する。
履歴の削除、量子化、モデルの再学習は行わない。
これは既存の[固定容量K/Vキャッシュ](https://huggingface.co/docs/transformers/main/en/kv_cache#fixed-size-cache)と同種の
実装上の対照であり、新しい認識アーキテクチャとしての貢献ではない。
コンパイルや他の最適化まで含めた最速Attentionとの比較を完了したとも言えない。

## 状態と費用の数え方

224入力・196位置・3時間層・幅256・BF16では、使用したt観測分のK/Vは602,112t bytes。
動的版はこれに応じて保持量が増える。予約版の確保量は最初から602,112L bytesで、
別に3層のCPUカーソル計24 bytesを持つ。未使用部分も確保メモリに含める。
Lは測定動画の長さとして事前に与え、超過したらエラーにする。
この有限容量の一定確保量を、任意に長い履歴への定数メモリと呼ばない。

過去K/Vの再コピーを取り除いても、各観測のAttentionが全履歴を読む処理は残る。
速度改善量は未測定であり、コード構造だけから高速化率を主張しない。

## 実施した検査

[CPU4検査](results/buffered_attention_tests.log)で、長さが異なるchunkの連続処理、
バッファのアドレス維持、未使用領域をNaNで埋めても出力へ混ざらないこと、
容量超過・カーソル不一致・学習モードの拒否、パラメータ共有、状態の初期化を確認した。
測定順について、3反復で各方式が各実行位置に一度ずつ現れることも検査した。

[実画像の数値検査](results/buffered_attention_qualification_v2/summary.json)では、
SSv2の17本から得た224入力136観測を逐次処理した。
事前学習済みSigLIP 2、seed 71の未学習Attentionを使い、時間経路が実際に出力へ影響するよう
残差ゲートを0.1に設定した。136観測すべてで、元の実装と予約版の特徴は完全一致した。
重みも変更されなかった。予約K/Vは81,887,232 bytes、CPUカーソルは24 bytesだった。
これは未学習時間層の数値検査であり、学習済みモデルの認識精度や速度測定ではない。

[測定コード](exploration/benchmark_buffered_attention.py)は、正式な最終checkpointでも
各実動画の全フレームについて特徴差と分類器の予測一致を確認してから測定する。
許容絶対差は1e-5、予測クラスの相違は0を要求し、条件を満たさなければ結果を確認するため停止する。
全フレームの差・状態量・予測クラスを保存する。これはSSv2全件の再評価を代替しない。

## 固定した測定条件

[測定計画](results/buffered_attention_cost_control/plan.json)はseed 42の最終学習済み
AttentionとMambaを使い、次の3方式を比較する。

1. 動的なK/V連結を使う時間Attention。
2. 予約バッファを使う同一重みの時間Attention。
3. 既存の中間層Mamba。

RGBは既存の費用測定と同じ52・245・1,337フレームの実動画。
方式ごとに1周のwarmupと3周の測定を行い、各反復で方式の実行順を回転させる。
測定時にGPUへ置くエンコーダは一つだけで、他の方式はCPUへ置く。
各周で状態を初期化する。画像前処理と転送を含む時間、モデルのみのGPU時間、
現在プロセスのPyTorch割当ピーク、状態の確保量を分けて記録する。
動画デコード・モデルのGPU移動・数値検査は測定区間に含めない。

速度測定は他のGPU計算プロセスがあると拒否する。
[実際の主学習中の拒否検査](results/buffered_cost_concurrency_verification.json)でも、
測定結果や出力ディレクトリが作られないことを確認した。

[待機コード](exploration/run_buffered_cost.py)は、既存の遅延保持キューの終了を待つ。
主学習・動画比較・追加seed・既存費用測定が完了していることも確認する。
[待機と停止の検査](results/buffered_cost_queue_cancel_verification.json)では、
既存の5プロセスを維持したまま、自分の待機だけを停止できた。
[本キューの起動記録](results/buffered_cost_queue_activation.json)を保存した。
現時点で正式な速度測定は未実行であり、Mambaまたは予約版の性能優位は未確認。

## 再実行

ルートから実行する。数値検査にはSigLIP 2のキャッシュ、SSv2元動画と既存のデータ計画が必要。
測定には全件学習・最終評価を完了した両方式と費用測定用RGBが必要。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_buffered_attention.py
uv run --locked python -m exploration.check_buffered_attention --output results/new_buffered_qualification
uv run --locked python -m exploration.benchmark_buffered_attention \
  --priority results/temporal_priority_seed42_rgbfix \
  --selection shortest --output results/new_buffered_cost_shortest
```

未使用領域を読まないバッファなので、容量を大きくしても未使用キーのAttention計算は増えない。
ただし確保メモリは増え、長い使用履歴へのAttention自体の計算も増える。
ここを含めて、方式の原理と実装による費用を分けて報告する。
