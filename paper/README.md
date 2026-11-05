# 原稿と結果表

[main.tex](main.tex)は方法・関連研究・評価条件と、確認済みの途中結果を記した英語の作業原稿。
6方式の最終比較・3 seed・本学習済みモデルの費用・長期認識は未完了で、投稿可能な完成原稿ではない。
著者欄は未設定。本文の研究上の貢献は、実際の比較結果を得てから確定する。

[build_tables.py](build_tables.py)は公開済みJSONから表を生成する。
公式ラベルで再計数した7,000更新までの全件検証と、4,000更新・348動画の履歴介入を分ける。
履歴介入は保存した予測クラスから正解数を数え直し、対応する検証記録のhashを照合する。
最終比較へ途中スコアを代入しない。[generated/evidence.json](generated/evidence.json)が各表の入力と生成元を記録する。
最終比較には別の[final_tables.py](final_tables.py)を用いる。完成済みの[正式集計](../TEMPORAL_PRIORITY.md)を
元予測・重み・実測記録まで再検証し、レビュー用の表を生成する。まだ完了した主比較はなく、実結果の表は生成していない。

## 再生成

リポジトリのルートで実行する。表の生成はCPUのみで、モデル重みや動画は不要。
学習用uv環境の定義は変更しない。PDF用依存はこのディレクトリのpixi環境に分離する。

```sh
uv run --locked python paper/build_tables.py
uv run --locked python paper/bootstrap_pixi.py
paper/.tools/pixi run --manifest-path paper/pixi.toml --locked pdf
```

生成先は`paper/build/main.pdf`。pixiが導入済みなら最後のコマンドの実行ファイルを`pixi`へ置き換えられる。
bootstrapはLinux aarch64の公式pixi 0.84.1を取得し、GitHub releaseのarchive・実行ファイルdigestと照合する。
OS全体にはインストールしない。[pixi.toml](pixi.toml)と`pixi.lock`でTeX環境を管理する。
TeXのフォント等はTectonicが初回実行で取得する。

## 実施した確認

上の手順でLetter・3ページのPDFを生成し、全ページの数式・表・参考文献を表示確認した。
フォントの代替を修正し、全フォントが埋め込まれていること、文字のはみ出しや未解決参照がないことを確認した。
参考文献の1箇所に行内余白の警告が残る。投稿サイトによるPDF適合検査は未実施。
[ビルド・表示確認の記録](../results/paper_build_verification.json)にソースと出力PDFのhashを保存した。

[表生成の3検査](../results/paper_tables_tests.log)は、予測数と合わない正解率、
異なるcheckpointの混入、部分集合を全件検証へ置き換える誤りを拒否する。
表の検証は新しいモデル評価の代わりにはならない。PDF本体・pixi環境・取得したツールはGitへ含めない。

## 最終比較を表へ取り込む手順

この処理は主比較キューと[AttentionのKVバッファ実装対照](../ATTENTION_CACHE_CONTROL.md)の完了後に使う。
現状の未完了キューでは出力を作らず停止する。学習やGPU推論を開始する処理ではない。

```sh
CUDA_VISIBLE_DEVICES='' uv run --project baselines/gmmix --locked python -m paper.final_tables collect \
  --priority results/temporal_priority_seed42_rgbfix \
  --buffered results/buffered_attention_cost_control \
  --output paper/generated/final
```

実行場所はリポジトリのルート。共通の検証器を読み込むため、既存のGMMix用uv環境を使う。
全14学習の最終checkpoint、元データmanifest、全件評価JSON、費用測定の生ログが必要である。
状態ファイルが`complete`というだけでは受理せず、既存検証器で次を再確認する。

- 6方式のseed 42と、主要4方式のseed 43・44。全件24,777動画、20,000更新、8観測、予定した学習条件。
- 一括・逐次推論と履歴介入の保存予測、checkpoint hash、対応のある正解差とbootstrap集計。
- 6方式×3動画の実測。各動画は52・245・1,337観測、1回warmup・3回測定。
- 動的KV・事前確保KV・Mambaの別測定。Attentionのフレームごとの出力比較と、予約したKV領域を含む状態容量。

出力の`final_result_bundle.json`には検証済み集計と参照したファイルのhashを保存し、
`evidence.json`には生成した表のhashと、費用表の時間ビンを記録する。
以下の10表を生成する。`main.tex`への挿入と、本文の解釈は実結果を確認してから行う。

| ファイル | 表示内容 |
| --- | --- |
| `comparison_final.tex` | seed 42の6方式。学習対象パラメータ数、逐次top-1、一括との予測不一致数 |
| `seeds_final.tex` | 主要4方式の各seed、平均、標本標準偏差 |
| `seed_differences_final.tex` | 同じseedでのMambaとの差。単位はpercentage point |
| `conditional_differences_final.tex` | seed 42の重みを固定した対応差と条件付き95%区間。seed間の不確実性とは別 |
| `costs_{shortest,median,longest}.tex` | 主6方式のGPU・前処理/転送込み時間、状態容量、PyTorchの最大割当量 |
| `buffered_costs_{shortest,median,longest}.tex` | KV実装対照の同じ指標。主測定とは別の実行として表示 |

費用表の時間は各動画の最後の時間ビンに含まれる観測の中央値である。
過去観測数32〜51、128〜244、256〜1,336のビンであり、最終1フレームだけの遅延ではない。
状態容量とPyTorch最大割当量は全測定内の最大値で、後者を総GPUメモリ・総システムメモリとは呼ばない。
時間分布の標本を独立した学習seedのようには扱わない。

取得済みbundleからの再描画だけならrootのuv環境で実行できる。

```sh
uv run --locked python -m paper.final_tables render \
  --bundle paper/generated/final/final_result_bundle.json \
  --output paper/generated/final_rerender
```

`render`は数値の整合性を検査するが、元の動画・checkpoint・生ログまでは検証し直さない。
出力先は新規ディレクトリに限定し、途中スコアへの自動フォールバックはしない。
表が揃っても、自然な履歴認識、研究上の貢献、投稿要件を満たしたという意味にはならない。

既存分を含む21テストが通過した。合成fixtureでは10表をIEEE書式へ配置して2ページのPDFを作り、
全ページを表示確認した。最初に見つかった列幅のはみ出しを修正し、再ビルドで解消した。
この合成PDFを研究原稿や実験結果へ混ぜていない。
[確認記録](../results/paper_final_tables_qualification.json)の実データでの検査は、未完了時の拒否までである。
**完了した全実験からの収集成功経路は、実験完了後に実行・検証する必要がある。**

## 書式の出典

[PaperCeptの公式LaTeX案内](https://ras.papercept.net/conferences/support/tex.php)が配布する
`ieeeconf.zip`から[ieeeconf.cls](ieeeconf.cls)を変更せず取得した。
元ファイル内の著作権・配布条件を保持している。[build_tools.json](build_tools.json)にURLとhashを保存した。
一般的なロボティクス会議向けのLetter書式として使い、IROS 2027固有の投稿要件の確認とは区別する。

2026-11-05に確認した[IEEE RASのイベント案内](https://www.ieee-ras.org/event/2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-70525/)は
論文締切を2027-03-01としている。そこからの会議サイトのリンクは2023年のサイトへ転送されたため、
2027年のページ数・匿名化・締切時刻等はまだ確認できていない。古い年の要件を2027年の要件として転記しない。
