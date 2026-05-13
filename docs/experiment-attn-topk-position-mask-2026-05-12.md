# 実験記録: Attention 上位 k「位置」直接マスク（グループ集約なし）

作成日: 2026/05/12

### 再実行ログ（2026-05-12）

- **実施**: `python experiments/run_attn_topk_position_mask_experiment.py`（`seed=42`、`outputs/impl1/hdf5_10epoch.pt`、先頭 5 XML、**サンプル 150**、`k ∈ {1, 3, 5}`、条件 top / bottom / random）。
- **一次ソース**: JSONL `outputs/results/attn_topk_position_mask_20260512_150016.jsonl`、集計 `outputs/results/attn_topk_position_summary.json`、行一覧 CSV `outputs/results/attn_topk_position_results.csv`。
- **図**: `figure/attn_topk_position_kl_vs_k.png`、`figure/attn_topk_position_kl_bar_k3.png`、`figure/attn_topk_position_flip_pdrop_k3.png`。

[グループマスク実験](experiment-attn-group-mask-2026-05-12.md)と**同じコーパス・同じサンプル数・同じ checkpoint**で、マスク単位だけを「意味グループ」から「系列上のキー位置」に変えた対照実験である。

---

## 目的

Attention の重要度を **8 グループに束ねず**、最終層の位置別重要度スカラー（クエリ・ヘッド平均）に対して **上位 k タイムステップの key を直接マスク**する。その結果の KL / Flip / Prob drop をグループ版と並べて読む。

---

## 実験設定（グループ版との共通点）

| 項目 | 値 |
|------|-----|
| XML | `/home/ubuntu/Documents/tenhou_xml_2023/` 先頭 5 ファイル |
| サンプル | **150**（2,772 行から `seed=42` でランダムサンプル） |
| Checkpoint | `outputs/impl1/hdf5_10epoch.pt` |
| デバイス（本実行） | CUDA |
| パディング | `sequence[:,0] == 8`（PADDING）の位置は重要度ソート・random 抽選から除外 |

---

## 本実験特有の設定

| 項目 | 内容 |
|------|------|
| **重要度** | 最終層 `attn_weights` を `(query, head)` で平均 → 長さ `S` のベクトル |
| **top-k** | 非 PAD 位置のうち重要度 **高い k 添字**の key をマスク |
| **bottom-k** | 同様に **低い k 添字** |
| **random-k** | 非 PAD から **k 添字**を決定的乱数で選択（`rng_for_random_positions`） |
| **マスク** | `experiments/run_attn_group_mask_experiment.py` と同じ `build_position_patch`（全層 logits に indices パッチ） |
| **k の水準** | **1, 3, 5**（グループ版の k スイープ 1〜5 とは設計が異なることに注意） |

各 JSONL レコードには `masked_positions`・`importance_scores_masked`・`k_effective` が入る（有効系列が短い局面は `k_effective < k`）。

---

## 結果サマリー（`attn_topk_position_summary.json` より）

### 条件別平均 KL / Flip / Prob drop

| k | 条件 | Mean KL | Std KL | Mean Flip | Mean Prob drop |
|---|------|---------|--------|-----------|----------------|
| 1 | top | 0.0280 | 0.0716 | 0.0667 | 0.00493 |
| 1 | bottom | 0.0002 | 0.00072 | 0.0 | -0.0004 |
| 1 | random | 0.00287 | 0.0113 | 0.02 | -0.0016 |
| **3** | **top** | **0.0656** | **0.316** | **0.12** | **0.0214** |
| 3 | bottom | 0.0022 | 0.0130 | 0.0067 | -0.00257 |
| 3 | random | 0.0133 | 0.0600 | 0.0267 | -0.00937 |
| 5 | top | 0.0754 | 0.332 | 0.1467 | 0.0304 |
| 5 | bottom | 0.0104 | 0.0598 | 0.0333 | -0.00356 |
| 5 | random | 0.0239 | 0.0996 | 0.06 | 0.00543 |

### 対応あり（paired）top vs random: Faithfulness gap と片側 t

| k | Gap (KL_top − KL_random) | t（片側対応あり） | p（片側） |
|---|--------------------------|-------------------|-----------|
| 1 | 0.0251 | 4.32 | 1.4e-5 |
| **3** | **0.0522** | **2.02** | **0.022** |
| 5 | 0.0515 | 2.20 | 0.015 |

---

## 図の対応

| 図 | パス | 内容 |
|----|------|------|
| 1 | `figure/attn_topk_position_kl_vs_k.png` | k = 1,3,5 に対する top / bottom / random の平均 KL の折れ線 |
| 2 | `figure/attn_topk_position_kl_bar_k3.png` | **k=3** 固定の条件別平均 KL（±SEM） |
| 3 | `figure/attn_topk_position_flip_pdrop_k3.png` | **k=3** の Flip rate と Prob drop の棒グラフ |

---

## グループマスク（k=3）との比較メモ

- [グループ実験・k=3 時の代表値](experiment-attn-group-mask-2026-05-12.md)（条件 C は random 5×平均）: KL top ≈ **0.096**、random ≈ **0.044**、Flip top ≈ **0.147**。
- **位置直接 k=3** 本実行: KL top ≈ **0.066**、random ≈ **0.013**、Flip top ≈ **0.12**。

**解釈**: グループ top-3 は「選ばれたグループに属する**全**イベント key」を一度に潰すため、**1 局面あたりのマスク点数が多く**なりやすい。一方、位置 top-3 は **ちょうど 3 タイムステップ**のみを潰す。介入強度が異なるため、KL の絶対値をそのまま横比較するより、「top > random > bottom の秩序」と「有意な Gap」が両設計で成立するかを並べて報告するのが安全である。本データでは **両方で top が random を上回る秩序**と、位置版でも k=3 で **片側 p≈0.022** の gap が得られている。

---

## 再実行手順

```bash
python experiments/run_attn_topk_position_mask_experiment.py \
  --max-samples 150 \
  --k-values 1 3 5 \
  --seed 42 \
  --n-xml-files 5
```

- `--no-plots`: JSONL のみ（集計 JSON・CSV・図なし）。
- 既存の結果から図だけ描き直す例:

```bash
python -c "import pandas as pd; from experiments.run_attn_topk_position_mask_experiment import save_plots; save_plots(pd.read_csv('outputs/results/attn_topk_position_results.csv'))"
```

---

## 関連ファイル

| パス | 役割 |
|------|------|
| `experiments/run_attn_topk_position_mask_experiment.py` | 本実験のエントリポイント |
| `outputs/results/attn_topk_position_mask_*.jsonl` | レコード逐次ログ（タイムスタンプ付き） |
| `outputs/results/attn_topk_position_summary.json` | 集計 |
| `outputs/results/attn_topk_position_results.csv` | 全行（表計算用） |
