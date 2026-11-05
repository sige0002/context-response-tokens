# 原稿と結果表

[main.tex](main.tex)は方法・関連研究・評価条件と、確認済みの途中結果を記した英語の作業原稿。
6方式の最終比較・3 seed・本学習済みモデルの費用・長期認識は未完了で、投稿可能な完成原稿ではない。
著者欄は未設定。本文の研究上の貢献は、実際の比較結果を得てから確定する。

[build_tables.py](build_tables.py)は公開済みJSONから表を生成する。
公式ラベルで再計数した7,000更新までの全件検証と、4,000更新・348動画の履歴介入を分ける。
履歴介入は保存した予測クラスから正解数を数え直し、対応する検証記録のhashを照合する。
最終比較へ途中スコアを代入しない。[generated/evidence.json](generated/evidence.json)が各表の入力と生成元を記録する。
最終比較を埋めるときは、[正式集計](../TEMPORAL_PRIORITY.md)の完成済み結果と照合して生成器を拡張する。

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

## 書式の出典

[PaperCeptの公式LaTeX案内](https://ras.papercept.net/conferences/support/tex.php)が配布する
`ieeeconf.zip`から[ieeeconf.cls](ieeeconf.cls)を変更せず取得した。
元ファイル内の著作権・配布条件を保持している。[build_tools.json](build_tools.json)にURLとhashを保存した。
一般的なロボティクス会議向けのLetter書式として使い、IROS 2027固有の投稿要件の確認とは区別する。

2026-11-05に確認した[IEEE RASのイベント案内](https://www.ieee-ras.org/event/2027-ieee-rsj-international-conference-on-intelligent-robots-and-systems-iros-70525/)は
論文締切を2027-03-01としている。そこからの会議サイトのリンクは2023年のサイトへ転送されたため、
2027年のページ数・匿名化・締切時刻等はまだ確認できていない。古い年の要件を2027年の要件として転記しない。
