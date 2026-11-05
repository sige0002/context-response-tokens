# Mamba-3の入力投影に特徴変化を渡す比較実験

現行モデルは8観測を増やすだけでは認識精度が改善せず、積分刻みの補正でも元の8観測に届かなかった。
次に、時間層への現在の入力だけでなく、直前の入力との差分を係数生成へ渡す構成を試す。
これは精度改善を調べる候補であり、新規性、SOTA、IROSへ投稿できる成果を立証したものではない。

## 比較する計算

各時間層の入力を`u_t`、元のMamba-3の入力投影を`P`とする。
追加する分岐は、幅128の中間層を持つ`F(a) = W2 SiLU(W1 a)`で、biasは使わない。

| 条件 | Mamba-3へ渡す投影 |
| --- | --- |
| appearance：同じ規模の対照 | `P(u_t) + F(u_t)` |
| difference：特徴変化を渡す候補 | `P(u_t) + F(u_t - u_(t-1))` |

エピソード先頭は`u_(t-1)=u_t`とし、候補の差分をゼロにする。
差分は同じパッチ座標間で取り、動いた物体を追跡して対応付けるものではない。
後段の時間層への入力は前段の履歴も含み得るので、appearance対照を「履歴のないモデル」とは呼ばない。

投影は`z, x, B, C, dt, A, trap, angles`を生成する部分であり、忘却ゲートだけを変更する設計ではない。
公式Mamba-3の正規化、係数の制約、GPU演算、出力投影を共通に使う。
第4・8・12層後の3時間層に分岐を置き、`W2=0`から始めるため、初期出力は元モデルと一致する。
逐次入力では各時間層で直前の入力1個を保持し、エピソード間では消去する。
前のclip全体の記憶領域を残さないよう、最後の入力はcloneして保持する。

[TDN](https://arxiv.org/abs/2012.10071v2)と[TEA](https://arxiv.org/abs/2004.01398v1)は、
時間・特徴差分を動作認識へ利用している。差分演算そのものを新しい原理として主張しない。
ここでは、凍結SigLIP2内のMamba-3への条件付けが、同じ規模の現在特徴分岐より有効かを調べる。

## 固定した学習と評価

[計画](results/transition_conditioning_step11000_v2/plan.json)は、性能を測る前に保存した。
主学習のソース・設定・状態は変更しない。

- 同じ11,000更新checkpoint、同じ追加分岐の初期パラメータを使う。
- SSv2の学習分割から、seed 701で選んだ重複のない8,192本を同じ順序で読む。
- 両条件を各512更新、microbatch 4、蓄積4、AdamW最大3e-5、warmup 32で学習する。
- 元の時間層、追加分岐、共通読み出しを正解ラベルで学習し、SigLIP2本体は凍結する。蒸留は使わない。
- 評価は各クラス4本、計696本に固定する。以前の348本を除外し、IDのhash順で選ぶ。
- 主指標は、8観測を1枚ずつ入力した後の現在フレームtop-1。clip入力の成績と予測の相違も報告する。
- 0更新と最終512更新を評価し、検証最高値のcheckpointを選ばない。
- 両条件の学習画像そのものと入力メタデータをhashで比較する。
- 保存モデルを読み戻し、全696本をclip・逐次の両方で再推論する。

全検証集合の集計値は以前から観測しているため、696本を完全に未使用のtestとは呼ばない。
これは1 seedの短い追加学習であり、元の20,000更新比較へ混ぜない。
主学習とGPUを共有するため、速度比較にも使わない。

## 実装検査と、最初に失敗した数値基準

[実装](exploration/transition_conditioning.py)、[学習コード](exploration/train_transition_conditioning.py)、
[CPU検査7件](results/transition_conditioning_tests_v2.log)を用意した。
CPUではゼロ初期化、非ゼロ分岐の因果性、分割入力との一致、勾配、状態の解放、評価集合の除外を確認した。

最初のGPU検査は、3更新後のclip・分割入力の特徴比較で、`atol=0.125, rtol=0.005`を超えて停止した。
[失敗ログ](results/transition_conditioning_qualification.log)と、
[変更前の実装hash](results/transition_conditioning_step11000_v1/source/manifest.json)を保持する。
この段階で本比較の学習は開始していない。

[切り分け結果](results/transition_conditioning_step11000_v1/numerical_diagnostic/summary.json)は次の通り。
同じ固定動画を8枚まとめて渡す場合と、3枚・1枚・4枚へ分割する場合を比べた。

| 構成 | 特徴の最大絶対差 | 特徴差のRMS | logitの最大絶対差 |
| --- | ---: | ---: | ---: |
| 追加分岐のない元のnativeモデル | 0.492740 | 0.005400 | 0.03125 |
| ゼロの追加分岐 | 0.492740 | 0.005400 | 0.03125 |
| 3更新後の候補 | 0.206294 | 0.005443 | 0.03125 |
| 同じ更新後の重みで追加分岐だけゼロ | 0.296613 | 0.005487 | 0.03125 |

この1動画では全構成で予測クラスが一致した。
また、学習後の各投影へ同じ入力を渡す分離検査では、clipと分割入力の差は3層すべてゼロだった。
差は既存のnative再帰にもあり、最初の閾値は元モデルにも成立していなかった。
この測定だけで、native演算のすべての誤差原因を特定したとは主張しない。

改訂した検査では閾値を広げず、その合否を結果として残す。
追加実装の合否は、ゼロ分岐が元モデルのclip・分割両経路と一致すること、学習後の投影が同じ入力で一致すること、
因果性、有限の勾配、凍結重み、保存・読み戻しで判定する。
**その検査が通っても、全モデルのclip・逐次特徴が数値的に一致したとは報告しない。**
精度への影響は、改訂計画の全696本の逐次評価で確認する。

## 再実行

固定11,000更新checkpointとSSv2元動画を用意し、リポジトリのルートから実行する。
出力先は新規のディレクトリを指定する。途中で失敗した出力へ上書き・無断再開しない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_transition_conditioning.py
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.train_transition_conditioning \
  --output results/new_transition_conditioning --plan-only
uv run --locked python -m exploration.train_transition_conditioning \
  --output results/new_transition_conditioning --qualify
uv run --locked python -m exploration.train_transition_conditioning \
  --output results/new_transition_conditioning --run
CUDA_VISIBLE_DEVICES='' uv run --locked python -m exploration.verify_transition_conditioning \
  --output results/new_transition_conditioning
```

[改訂後のGPU検査](results/transition_conditioning_step11000_v2/qualification/verification.json)では、
元モデルとのclip・分割両経路でのゼロ分岐の一致、3層の学習後投影の一致、因果性、保存・読み戻しを確認した。
元の全特徴の数値閾値は不合格のまま記録されている。
追加する直前入力の状態はbatch 1で301,056 bytesだった。
appearance対照は512更新と全696動画の保存モデル再実行を完了した。
[結果](results/transition_conditioning_step11000_v2/training/appearance/summary.json)と
[CPU再計数](results/transition_conditioning_step11000_v2/training/appearance/verification.json)を保存した。

| appearance対照 | 初期 | 512更新後 | 正解数の増減 |
| --- | ---: | ---: | ---: |
| 一括入力 | 323 / 696（46.41%） | 327 / 696（46.98%） | +4 |
| 逐次入力・主指標 | 323 / 696（46.41%） | 329 / 696（47.27%） | +6 |

逐次入力の初期との差は+0.86 pp、探索的な対応付き95%区間は[−1.44, 3.16] pp。
44本が新たに正解し、38本が不正解になった。明確な改善の証拠とは扱わない。
最終モデルの一括・逐次では予測が3本異なり、逐次が2本多く正解した。
各モードについて、保存モデルから再実行した全696本のlogitは元の実行と完全一致した。
4条件の計2,784予測を元ラベルから再計数した。SigLIP2の重みも不変だった。
この集合は各クラス4本の固定診断集合で、全24,777動画の主学習スコアとは直接比較しない。

difference候補の最終結果は未取得である。
候補の採否は、この対照を同じ更新数で上回るかを確認してから判断する。

[独立のCPU再計数](exploration/verify_transition_conditioning.py)は、元のmanifestのラベル、
初期重み・入力画像の一致、clipと逐次の予測、実行更新数、保存モデルの再実行記録を照合する。
両条件の検証画像も、メタデータと画素値それぞれのhashで照合する。
初期予測が同じでも画像が同じとは限らないため、片方だけ別の検証画像を使った記録を拒否する
[CPU検査を含む2件](results/transition_verification_tests_v2.log)を通過した。
片方だけが完了した場合は`--arm appearance`または`--arm difference`でその条件を検査できるが、
両条件の比較が完了した扱いにはしない。
この再計数自体が新しいGPU推論を行うわけではない。
