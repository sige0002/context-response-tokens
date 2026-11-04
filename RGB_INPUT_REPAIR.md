# SSv2のRGB入力不具合と修正

## 結論と影響範囲

このGB10環境のPyAV 19.0.2／libswscale 10.1.102では、幅427のYUV420P画像を既定の設定でRGBへ変換すると、右端1列が不定な値になることを確認した。
同じ元動画・時刻・cropでも、モデルへ渡す画素が再現しない。元動画のSHA256、公式ラベル、時刻の照合だけでは検出できなかった。

旧全データ学習は4,370更新で停止した。旧checkpoint、予測、ログは診断証拠として保持する。
旧39.29%や履歴介入の値を、修正後の正式な比較結果として採用しない。
JPEGを保存した旧pilotも同じ変換を経由しているため、その画像キャッシュを新しい実験には使わない。
ただし、この不具合が低精度の全原因だと示したわけではない。修正後の精度と対照モデルとの比較が別途必要である。

## 発見と切り分け

- 3,500更新の同一checkpoint・同一1,024動画で、一括と逐次の予測が4本異なった。この比較内では同じ入力テンソルを両方へ渡している。
- 別の実行で保存した一括予測との間にも19本の差があり、追加調査を始めた。
- 差が出た動画を含む元と同じ21バッチ・84例を調べた。同じ入力テンソルでの反復推論と、逐次推論の前後の一括推論は完全一致し、モデルの全重み・永続bufferも変わらなかった。
- 別プロセスでは、時刻やcropが同じでも入力テンソルのハッシュが異なった。これはMambaの丸め誤差で説明できない。
- 元動画3本を再復号すると、可視範囲のY・U・V値は一致したが、既定のRGB変換の差は427画素幅の列426に集中した。

[入力の修復監査](results/video_rgb_repair_audit.json)、
[旧推論の再実行](results/video_repeatability_default_1/summary.json)、
[停止した実プロセスの記録](results/video_input_bug_stopped_jobs.json)を参照。
旧予測の実行間差を、そのままカーネル誤差と呼んではいけない。

## 修正内容

[video_data.frame_to_rgb](video_data.py)で、`BILINEAR | ACCURATE_RND | BITEXACT | FULL_CHR_H_INT`と変換スレッド数1を明示する。
画像の右端を切り捨てたり隣の画素で置換したりせず、元の画素をRGBへ変換する。
その後のSigLIP 2の224リサイズ・正規化は同じ設定を使う。クロマ変換方式の変更を含むため、旧入力との同一性は主張しない。

PyAVの[変換API](https://pyav.org/docs/stable/api/video.html#av.video.frame.VideoFrame.to_image)は変換オプションをVideoReformatterへ渡す。
実際に使用したフラグ定義は、この環境にインストールされた`av/video/reformatter.py`でも照合した。
ARM用の最適化経路の参考には[FFmpeg n9.0の公開ソース](https://github.com/FFmpeg/FFmpeg/blob/n9.0/libswscale/aarch64/swscale_unscaled.c)を使ったが、インストール済みバイナリとそのcommitが完全一致すると証明したわけではない。
不具合と修正の判定は上記の実測を根拠にしている。

## 修正検証

| 検査 | 結果・範囲 |
| --- | --- |
| 一様YUV `(100,128,128)`、427×240、BT.601 limited | 正解RGBは約`(98,98,98)`。既定変換では右端に`(0,0,0)`などが現れ、修正後は全画素が`(98,98,98)` |
| 奇数・偶数幅、幅1を含む7形状と4色 | 既知の色、全画素の一様性、反復一致を確認 |
| 右端だけ白い画像、奇数幅3種類 | 本物の右端を保つことを確認。切り捨て・隣列コピーによる回避を拒否 |
| 実動画64本 | 学習32本は学習crop付き、検証32本は固定crop。2回の単一ワーカー読み込みとspawn 2ワーカーで全入力・ラベルが一致 |
| 最初に失敗した実動画3本 | 修正後のRGBと224入力が一致。テンソル最大差0 |
| データ・学習・評価・キューのCPU検査 | 42件通過。後続の抽出キャッシュ検査などは別ログに保存 |

再現コマンド：

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q test_video_rgb.py test_video_data.py test_ssv2_preparation.py
CUDA_VISIBLE_DEVICES='' uv run --locked python audit_video_rgb.py --output results/new_rgb_audit.json
```

64本の検査を全193,690本の復号済み画素の監査と呼ばない。GPUの学習・再開検査も少数例での実装検査であり、認識精度の実験ではない。

## 同じ不具合の別経路

`variant-analysis`スキルの手順に沿い、保守対象のPythonコード全体で呼び出しを調べた。
`vendor`、`baselines`の外部実装と`results`内の保存ソースは、今回変更するコードから除いた。

| 段階 | 検索 | 判定 |
| --- | --- | --- |
| 完全一致 | `.to_image().save` | `extract_ssv2.py`に1件。修正前の既定変換をJPEG保存していた |
| API呼び出しの拡張 | `to_image`、`to_ndarray`、`reformat`、`to_rgb` | 本来のreader、上記抽出器、意図的に不具合を再現する監査コードを確認 |
| RGB形式まで拡張 | `swscale`、`rgb24` | テストでのRGB入力作成が追加されるが、YUVからの不正な変換ではない。追加の本番経路なし |

抽出器の問題は研究結果の妥当性に対する高い影響があり、確度は高い。共有する修正済み変換へ変更し、変換条件が未記録のキャッシュは拒否する。
監査コードの既定変換は再現用であり、実験の入力を作らないため修正対象から除いた。
さらに[CI向け検出ルール](rules/video_rgb_conversion.yml)を追加した。Semgrep自体は今回実行しておらず、検索と動作検査で確認した。

## 再実行方針

全方式の学習・評価で、同じRGB変換と明示した決定的実行設定を使用する。
GMMixの最初のRGB修正後の再学習検査は、重み差が事前基準を超えて失敗したため記録を保持した。
基準を緩めず、cuDNN・cuBLAS・PyTorchの設定を全方式で揃え、空間FP32の演算でTF32を明示的に無効にして再検査した。
RGB修正だけの試行と、決定的設定だけを追加した試行は基準超過のまま記録を残した。
最終設定では、GMMixの再開時の最大重み差は`1.19e-7`、独立再学習では`2.74e-6`で、事前の`5e-6`基準内だった。
Mamba-3では、この3更新検査の重み・検証損失が再開・独立再学習とも完全一致した。
これを長期間の学習や全カーネルのビット単位再現性の保証にはしない。

- [Mamba-3の実動画学習・再開検査](results/video_training_integration_rgbfix_fp32/verification.json)
- [独立した評価CLIと学習時評価の照合](results/video_training_integration_rgbfix_fp32/evaluation_verification.json)
- [GMMixの実動画学習・再開検査](results/gmmix_training_integration_rgbfix_fp32/verification.json)
- [入力・学習・評価・比較計画の57検査](results/video_rgb_protocol_final_tests.log)
- [点追跡の設定・評価処理の24検査](results/point_precision_protocol_tests_v2.log)

少数例での一括・逐次の分類予測は一致したが、BF16を使う時間層のlogit最大差`0.0078125`は残る。
全検証集合での一致はまだ確認していない。

点追跡でも同じ数値精度設定を使用し、修正後の動画checkpointから検査をやり直した。
[Mamba-3](results/point_training_integration_rgbfix_fp32/verification.json)と
[GMMix](results/gmmix_point_training_integration_rgbfix_fp32/verification.json)の双方で、
3更新の連続実行・中断再開・独立再実行の重み、損失、予測座標が完全一致した。
probeではエンコーダ全体が変化せず、adaptではSigLIP 2を固定したまま追加層が更新された。
実動画1本の逐次評価も行い、[Mamba-3](results/point_online_integration_rgbfix_fp32/mamba/raw_label_metric_audit.json)と
[GMMix](results/point_online_integration_rgbfix_fp32/gmmix/raw_label_metric_audit.json)の保存予測を元注釈から再採点した。
両者のAJはこの3更新試験では0であり、計測経路の一致を点追跡精度の改善と呼ばない。

新しい本学習は修正前の時間層・分類器を引き継がず、同じ事前学習SigLIP 2とseedから始める。
データ分割、224入力、8フレーム、20,000更新、教師ラベル、比較する5方式は維持する。
