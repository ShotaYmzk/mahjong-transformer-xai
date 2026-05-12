# Attention マスキング介入：効果・図一覧（ジャーナル執筆用メモ）

作成日: 2026/05/12  

**一覧**: [experiment-attn-group-mask-2026-05-12.md](experiment-attn-group-mask-2026-05-12.md)（詳細ログ・統計解釈） / [progress-2026-05-12.md](progress-2026-05-12.md)（作業経緯）

実験記録では **図1〜図9** と `figure/` 以下のファイル名で対応済み。このメモでも同じパスをそのまま使う。

---

## 1. 「Attention を隠す」とは何をしたか（方法）

摂動は **説明文ではなく、打牌方策の softmax 出力分布**への影響を測っている。

- **対象モデル**: `MahjongTransformerV2`（`outputs/impl1/hdf5_10epoch.pt`）。
- **介入**: 全層の self-attention **logits** で、対象グループに属する key 位置に **−1e9** を加え、そこへの質量を遮断する（`apply_attention_patch`）。
- **マスク単位**: イベント列を 8 の feature group に分け、[実験記録](experiment-attn-group-mask-2026-05-12.md) の定義に従う。
- **重要度**: 最終層 attention を平均し、グループ位置の質量を合算・正規化（`attention_patching` と実装共通）。

BERTScore で LLM 説明を評価する段階は別パイプラインとし、本章はモデル側 faithfulness が前提になる。

---

## 2. 介入の結果（k=1〜5、n=150）

**データ**: 天鳳 XML 先頭 5 ファイル、母集団 2772、`seed=42` で **150** サンプル。  
**random（条件 C）**: スイープでは **局面ごと 5 回**平均。実験1（k 固定）は **1 回**のみ。

曲線類は **図5（`figure/attn_k_sweep_aopc_curve.png`）** と **図8（`figure/attn_k_sweep_flip_pdrop.png`）** を論文フィギュア候補にする。

### 2.1 k 別サマリー（`outputs/results/attn_k_sweep_summary.json`）

| k | KL（top） | KL（bottom） | KL（random, 5×平均） | Gap | SNR | t（片側） | p |
|---|-----------|---------------|---------------------|-----|-----|-----------|---|
| 1 | 0.0709 | ~0 | 0.0228 | 0.0480 | **0.39** | 2.65 | 0.004 |
| **2** | 0.0944 | ~0 | 0.0402 | **0.0542** | 0.31 | 3.24 | 0.0007 |
| 3 | 0.0958 | ~0 | 0.0438 | 0.0520 | 0.27 | 3.61 | 0.0002 |
| 4 | 0.0963 | ~0 | 0.0718 | 0.0245 | 0.08 | 3.61 | 0.0002 |
| 5 | 0.0963 | 0.0005 | 0.0800 | 0.0163 | 0.05 | **5.44** | <0.0001 |

図9（`figure/attn_k_sweep_pvalues.png`）に p と t を棒で示している。

### 2.2 k=3 固定（実験1の可視化）

**図1（`figure/attn_mask_kl_comparison.png`）** と **図2（`figure/attn_mask_fliprate.png`）** が条件別の強さ。**図3（`figure/attn_mask_group_breakdown.png`）** は top がどのグループを潰したか。**図4（`figure/attn_group_heatmap.png`）** はサンプル横断の重要度ヒートマップ。

### 2.3 「どのパネルを Results に置くか」早見表

| 論文ドラフト用 | ファイルパス |
|----------------|---------------|
| プロトタイプ説明・k 固定での破壊度 | **図1** `figure/attn_mask_kl_comparison.png`、**図2** `figure/attn_mask_fliprate.png` |
| Attention が何を読んでいか | **図4** `figure/attn_group_heatmap.png` |
| Faithfulness と k の関係 | **図5** `figure/attn_k_sweep_aopc_curve.png` |
| top vs random の差が k でどう変わるか | **図6** `figure/attn_k_sweep_faithfulness_gap.png`（金=Gap argmax の k）|
| random のばらつきとの比（SNR） | **図7** `figure/attn_k_sweep_snr.png`（金=SNR argmax の k）|
| Flip / Prob drop の補助 | **図8** `figure/attn_k_sweep_flip_pdrop.png` |
| 統計検定パネル | **図9** `figure/attn_k_sweep_pvalues.png` |

---

## 3. k=3 採用とスコアの関係

| 観点 | データで突出する傾向 | k=3 を主報告に残す論拠の持ち方 |
|------|---------------------|----------------------------------|
| Faithfulness Gap | **k=2 が最大**（図6 `figure/attn_k_sweep_faithfulness_gap.png`） | 「差の最大化」だけが目的ではなく、上位 3 解釈カテゴリを一度に遮蔽する評価プロトタイプとして k=3 |
| SNR | **k=1 が最大**（図7 `figure/attn_k_sweep_snr.png`） | 同上。SNR と Gap は異なる単一目的最適を与えることを本文で明示 |
| paired t | **k=5 で t が最大**（図9 `figure/attn_k_sweep_pvalues.png`） | n が十分だとすべての k で有意になりやすい；過剰マスクとのトレードオフを議論 |

---

## 4. 再生成コマンド

- フル実行: `python experiments/run_attn_k_sweep_experiment.py` と `python experiments/run_attn_group_mask_experiment.py`（図は `figure/`、表は `outputs/results/`）。
- 図のみ: `python experiments/run_attn_k_sweep_experiment.py --plots-only`

---

## 5. メソッド記述時の注意

- random を **何回試行した平均か** を Methods で固定し、結果表のどちらの列かと対応させる。
- bottom-k がゼロ近傍なのは、系列に「下位グループ」のトークンが無いだけの効果であり、モデル異常とは限らない。
- **図6** と **図7** の「金」の k が異なること（本データ Gap→2、SNR→1）をキャプションで明示し、緑の **Adopted k=3** とは別次元の選択であることを書く。
