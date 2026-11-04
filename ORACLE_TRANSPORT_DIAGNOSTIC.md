# 空間記憶の移送診断：2026-11-05

**grid上の直交移送と、単純に座標だけを更新する方式は採用を保留する。**
8シーンの診断では、通常のbilinear移送より平均誤差が大きかった。
これは学習した視覚エンコーダの比較結果ではなく、記憶移送の設計を選別するCPU実験である。
現在のSigLIP 2＋Mamba-3の学習・比較キューは変更していない。

## 対象と評価

- データ：準備済みKubricの**合成レンダリングRGB**、224×224。実世界の撮影動画ではない。
- 既存の固定validation splitに属する準備済みsceneをID順で8件選び、結果を見る前に固定。
  `0016, 0020, 0030, 0061, 0065, 0081, 0123, 0128`。
- 各sceneの軌跡をseed 17＋scene IDで分け、演算の推定8,192点、評価8,192点を重複なく使用。
  同じ物体・近接領域の点も含まれるため、独立した8,192サンプルとはみなさない。
- 記憶は14×14。初期RGBの各16×16パッチを固定乱数の行列で32次元へ写し、空間方向に標準化。
  独立な32次元ランダム指紋も評価。**SigLIP特徴でも学習済み意味表現でもない。**
- 初期フレーム以降の新規観測の書込みはなし。初期記憶を現在の対応位置から読む能力だけを測る。
- 24時点、stride 1 / 4。最後はframe 23 / 92。float64、CPU 2 threads。
- 全方式で同じ初期・現在可視の評価点を使う。連続可視は途中の元フレームすべてを確認する。
- 誤差は各sceneの二乗誤差合計／正解記憶値の二乗和。その**scene間の単純平均**を報告。
  小さいほど良い。認識top-1、追跡AJ、mIoUではない。

入力アーカイブ、注釈、変換metadata、ソースのSHA-256を検査した。
各実験はplanを保存してから実行し、結果にplanのhashを記録している。
小規模な探索用validation診断であり、未知testでの性能立証ではない。

## 1. 記憶そのものを移す

隣接する評価時点の正解対応を推定点から読み、`H_t = W_t H_(t-1)`を実行する。
固定座標、平均flowのbilinear warp、SO(196)のProcrustes解、ridge線形移送を比較した。
直交・ridgeのidentity prior係数は事前に0.001へ固定し、結果を見て探索していない。
密な直交行列は局所Givens回転の表現制約を緩めた対照だが、認識精度の上限ではない。

| 記憶 | 最終frame | 固定座標 | bilinear移送 | 直交移送 | ridge移送 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB非学習特徴 | 23 | 0.669535 | **0.312688** | 0.359691 | 0.313474 |
| RGB非学習特徴 | 92 | 1.226305 | **0.652209** | 0.746055 | 0.731483 |
| ランダム指紋 | 23 | 0.862115 | **0.429198** | 0.551336 | 0.482935 |
| ランダム指紋 | 92 | 1.483814 | **0.749340** | 1.007747 | 0.785752 |

frame 92のRGBでは直交移送は8シーン中7シーンでbilinearより悪い。
状態エネルギー比はbilinear 0.379に対して直交1.000で、直交性の最大残差は約6.44×10⁻¹⁵だった。
**エネルギーを保ったことと、正しい場所で読める情報を多く残したことは一致しなかった。**
全可視の評価点数はframe 23で42,995、frame 92で37,337。

一つの位置の値を拡大後の二位置で同じ値として読む写像は、任意の記憶に対して
grid上の直交行列だけでは実現できない。この反例は演算検査に含めた。
ただし、学習する読出しや潜在座標系を加えた可逆モデルすべてを否定する結果ではない。

## 2. 初期フレームから直接読む参考値

初期記憶は変更せず、frame 0→現在の正解対応から毎回読出しを作った。
推定点と評価点は実験1と同じ。初期・現在で可視の推定点を使うため、
隣接フレームの方式とは推定点の可視性と対応情報が異なる。

| RGB非学習特徴 | 最終frame | 毎回bilinear移送 | 初期→現在bilinear | 初期→現在ridge |
| --- | ---: | ---: | ---: | ---: |
| 全可視 | 23 | 0.312688 | 0.158584 | 0.076600 |
| 全可視 | 92 | 0.652209 | 0.288135 | 0.137673 |
| 遮蔽後の再出現 | 92 | 0.932245 | 0.374445 | 0.188024 |

全可視では両方式が両horizonで8/8シーン改善した。
しかし、**長距離の正解対応を直接もらう有利な条件**であり、
この改善を補間回数だけの効果、学習可能性、実装済みモデルの性能と解釈できない。

## 3. 同じ対応情報で、値の再補間を避ける

この交絡を確認するため、実験1と全く同じ隣接warp `W_t`を使い、
初期記憶は固定して座標場だけを更新した。

```text
値を毎回移送： H_t = W_t H_(t-1)
座標を更新：   C_t = W_t C_(t-1)
読出し：       H_read_t = bilinear_sample(H_0, C_t)
```

どちらも14×14の出力記憶を作って同じ評価点から読む。
座標方式は各位置に2スカラー余分に保持する。新しい観測、長距離の正解対応は与えていない。
bilinear対照を再計算し、全時点・全stratumの元スコアとの差が10⁻¹⁰以内であることを確認した。
評価点数と正解記憶のエネルギーも一致を要求した。

| 記憶 | 最終frame | 毎回bilinear移送 | 座標だけ更新 | 改善したscene |
| --- | ---: | ---: | ---: | ---: |
| RGB非学習特徴 | 23 | **0.312688** | 0.336670 | 5/8 |
| RGB非学習特徴 | 92 | **0.652209** | 0.839912 | 1/8 |
| ランダム指紋 | 23 | **0.429198** | 0.444570 | 3/8 |
| ランダム指紋 | 92 | **0.749340** | 0.953960 | 0/8 |

frame 92で遮蔽後に再出現したRGBの誤差も、0.932245から1.192740へ悪化した。
単純な座標更新を次の本命とする根拠は得られなかった。

考えられる理由は、座標の誤差も蓄積すること、単一の平均座標では複数の対応候補を表せないことである。
数学的にも、一般の特徴場で「座標を平均してから値を読む」と「値を読んで平均する」は異なる。
ただし、この診断は誤差原因の寄与率を測定していない。遮蔽だけが原因とは断定しない。

## 研究上の判断と先行研究

ノルム保存と再補間の削減だけでは、実際の対応読出しの改善に十分ではなかった。
次の採否実験では画像からの対応修正、新規観測による更新、対応の多義性を個別に比較する。
既存の固定比較へ、負けた候補をそのまま足して学習費用を使わない。

warp補間が細部をぼかす問題は[BasicVSR++](https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.pdf)でも論じられている。
空間記憶は[Spann3R](https://arxiv.org/abs/2408.16061)、
3D座標に結び付いた記憶は[Point3R](https://arxiv.org/html/2507.02863v1)に先例がある。
[Slot3R](https://arxiv.org/html/2610.12282v1)は、同じ空間アドレスで複数状態を保持し、
近いという理由だけで融合する問題を扱う。
したがって「座標と記憶を分離」「複数の証拠を保持」だけを独自性の根拠にはしない。
今回の診断は新しいSOTAアーキテクチャの発表ではなく、失敗条件を特定した研究記録である。

## コード・結果・再実行

| 実験 | 実装 | 固定plan | 結果 |
| --- | --- | --- | --- |
| 1 | [oracle_transport.py](exploration/oracle_transport.py) | [plan](results/oracle_transport_seed17/plan.json) | [summary](results/oracle_transport_seed17/summary.json) |
| 2 | [oracle_anchor_readout.py](exploration/oracle_anchor_readout.py) | [plan](results/oracle_anchor_seed17/plan.json) | [summary](results/oracle_anchor_seed17/summary.json) |
| 3 | [oracle_coordinate_readout.py](exploration/oracle_coordinate_readout.py) | [plan](results/oracle_coordinate_seed17/plan.json) | [summary](results/oracle_coordinate_seed17/summary.json) |

公開するsummaryには集計値と各scene出力のhashを含め、実験2・3には最後の時点のscene別結果も含める。
実験1の全4方式のscene別最終結果は[CSV](results/oracle_transport_endpoints.csv)に保存した。
元画像・軌跡と全時点のscene JSONはローカル生成物であり、リポジトリには含めない。
元データを同じ変換手順で準備した後、別の出力ディレクトリで再実行できる。
CPU検査は[9件通過](results/oracle_readout_tests.log)。3実験の実行はそれぞれ約18秒、7秒、7秒だった。

```sh
CUDA_VISIBLE_DEVICES='' uv run --locked python -m pytest -q exploration/test_oracle_transport.py exploration/test_oracle_readout.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --locked python -m exploration.oracle_transport --output results/reproduce_transport
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --locked python -m exploration.oracle_anchor_readout --original results/reproduce_transport --output results/reproduce_anchor
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --locked python -m exploration.oracle_coordinate_readout --original results/reproduce_transport --output results/reproduce_coordinate
```

`--plan-only`で先にplanを固定した場合は、同じコマンドへ`--resume`を付けて実行する。
公開planの絶対パスは実施記録であり、別PCではそのまま流用しない。
