# 実験記録: Attention グループマスキング実験 + k スイープ分析

作成日: 2026/05/12

### 再実行ログ（2026-05-12・更新）

- **実施**: `python experiments/run_attn_k_sweep_experiment.py` および `python experiments/run_attn_group_mask_experiment.py`（`seed=42`、`outputs/impl1/hdf5_10epoch.pt`、`MAX_SAMPLES=150`）。
- **検証**: 正常終了。図はすべてリポジトリ直下の **`figure/`** に出力（例: 図1 `figure/attn_k_sweep_aopc_curve.png`）。数値は `outputs/results/attn_k_sweep_summary.json` および `outputs/results/attn_group_mask_summary.json` が一次ソース。
- **将来のメモ**: 数値がドキュメントとずれる場合は、**別 checkpoint / 別 seed / 条件C（random 1 回 vs 5 回平均）** の取り違えを最初に疑う。

本記録では、**出力が目に見える図**と本文を対応させるため、図を参照する箇所では必ず **`figure/ファイル名.png`** を併記する。

| 図番号 | パス | 内容 |
|--------|------|------|
| 図1 | `figure/attn_mask_kl_comparison.png` | k=3 固定：条件別平均 KL |
| 図2 | `figure/attn_mask_fliprate.png` | k=3 固定：Flip Rate / Prob Drop |
| 図3 | `figure/attn_mask_group_breakdown.png` | k=3 固定：top マスクで選ばれたグループの頻度 |
| 図4 | `figure/attn_group_heatmap.png` | k=3 固定：150 局面 × 8 グループのベースライン重要度 |
| 図5 | `figure/attn_k_sweep_aopc_curve.png` | k スイープ：平均 KL の AOPC 風曲線 |
| 図6 | `figure/attn_k_sweep_faithfulness_gap.png` | k スイープ：Faithfulness Gap（金色=Gap 最大、緑枠=採用 k=3） |
| 図7 | `figure/attn_k_sweep_snr.png` | k スイープ：SNR（金色=SNR 最大、緑枠=採用 k=3） |
| 図8 | `figure/attn_k_sweep_flip_pdrop.png` | k スイープ：Flip / Prob Drop vs k |
| 図9 | `figure/attn_k_sweep_pvalues.png` | k スイープ：対応あり t 検定の p と t |

図5（`figure/attn_k_sweep_aopc_curve.png`）および図6（`figure/attn_k_sweep_faithfulness_gap.png`）が、k スイープの全体傾向を把握する入口になる。図1（`figure/attn_mask_kl_comparison.png`）と図2（`figure/attn_mask_fliprate.png`）は k=3 固定実験の要約に用いる。

---

## 目的

`MahjongTransformerV2` の Attention 重みが「本当に重要な特徴を捉えているか」を
実際の天鳳 XML ゲームログを使って定量的に検証する。

**仮説**: 重要度が高いグループをマスクすればモデル出力が大きく変化し、
重要度が低いグループをマスクしても変化しないはず（Faithfulness = 説明忠実性）。

LLM 統合なし。モデル出力分布の変化量（KL divergence, Flip Rate, Probability Drop）で評価する。

**口頭説明・Flip率の分子分母・kスイープ図の読み方・グループマスクが attention のどこに効くか**は、[guide-attention-mask-narrative.md](guide-attention-mask-narrative.md) に集約している（本ページは実験手順・数値一次ソースの中心）。

**対照実験（意味グループを使わず、系列上の重要度上位 k 位置だけをマスク）**は [experiment-attn-topk-position-mask-2026-05-12.md](experiment-attn-topk-position-mask-2026-05-12.md) を参照。

---

## 実験設定

### 共通設定

| 項目 | 値 |
|---|---|
| 使用 XML | `/home/ubuntu/Documents/tenhou_xml_2023/` 先頭 5 ファイル |
| 抽出サンプル数（全体）| 2,772 |
| 実験サンプル数 | **150**（母体 2,772 からランダム、seed=42）|
| チェックポイント | `outputs/impl1/hdf5_10epoch.pt` |
| モデル設定 | `d_model=64, n_heads=4, n_layers=2, epoch=10` |
| val top1 | 65.6% |
| ランダムシード | 42 |

### マスク条件

Attention logits（softmax 前）の指定 key ポジションを **−1e9** に設定し、全層で適用。
`apply_attention_patch(mode="indices")` を使用（既存コード変更なし）。

| 条件 | 内容 |
|---|---|
| 条件A（top-k）| Attention 重要度スコア上位 k グループに属する event ポジションをゼロマスク |
| 条件B（bottom-k）| 重要度下位 k グループを同様にマスク |
| 条件C（random-k）| ランダムに k グループをマスク（5 回試行平均）|

### 評価指標

| 指標 | 意味 |
|---|---|
| **KL divergence** | ベースライン softmax 分布 vs マスク後分布の乖離量 |
| **Decision Flip Rate** | Top-1 打牌予測が変化したサンプルの割合 |
| **Probability Drop** | Top-1 打牌の確率値の低下量 |
| **Faithfulness Gap** | KL(top-k) − KL(random-k)：「偶然より上乗せされた損傷量」|
| **SNR** | Gap / std(random KL)：信号対雑音比 |

---

## 図と「採用 k」の対応（混同防止）

図6（`figure/attn_k_sweep_faithfulness_gap.png`）、図7（`figure/attn_k_sweep_snr.png`）、図9（`figure/attn_k_sweep_pvalues.png`）を対に読むこと。

| ファイル | 何を示すか | 金色の強調 | 本研究の採用 k |
|----------|------------|------------|----------------|
| `figure/attn_k_sweep_faithfulness_gap.png` | 各 k の Faithfulness Gap | **Gap 最大** → 本データ **k=2** | 緑枠・凡例 **Adopted k=3**（解釈性・上位3グループ同時マスク） |
| `figure/attn_k_sweep_snr.png` | Gap ÷ std(random KL) | **SNR 最大** → 本データ **k=1** | 同上 |
| `figure/attn_k_sweep_pvalues.png` | 対応あり t（top vs random） | — | **t 最大は k=5**（t≈5.44、p&lt;0.0001）。**主報告 k=3**（t≈3.61、p=0.0002）は「検定だけ最大」ではないが設計として妥当 |

**なぜ金（データ駆動の極値）と採用 k=3 が食い違うのか**: Gap / SNR は「top と random の差の大きさ」を見ており、k が小さいほどランダムが本命グループを踏みにくく差が出やすい。一方 **t はサンプル数を増やすとすべての k で有意になりやすい**（本実験では n=150 で k=1 も p&lt;0.01）。それでも **k=3** を報告プロトタイプにするのは、SafetyVsOthers / TileEfficiency / PointSituation の **上位 3 解釈グループを確実に同時遮蔽できる最小の k**であり、k=5 のような広範マスクより介入の説明がしやすいからである。図6の金色（Gap）は **k=2**、図7の金色（SNR）は **k=1** であり、それらは本文の **k=3 採用**と両立しないのではなく、**別々の単一目的最適解**として併記する。

図の再生成（モデル不要・`outputs/results/attn_k_sweep_results.csv` が必要）: `python experiments/run_attn_k_sweep_experiment.py --plots-only`

---

## 実験 1: k=3 固定の基礎実験

条件別の概要は **図1（`figure/attn_mask_kl_comparison.png`）** と **図2（`figure/attn_mask_fliprate.png`）**。**図4（`figure/attn_group_heatmap.png`）** に局面×グループの重要度を示す。マスク選択のばらつきは **図3（`figure/attn_mask_group_breakdown.png`）**。

### 結果

**条件A・B** の数値は `run_attn_group_mask_experiment.py` と `run_attn_k_sweep_experiment.py`（k=3 の列）で一致する。

**条件C（random）** だけ実装が違うため、次のどちらかを明記する必要がある。

| 条件Cの定義 | 平均 KL | 標準偏差 | Flip Rate | 出典 |
|---|---|---|---|---|
| 各局面で random マスクを **1 回** だけサンプル | **0.0597** | 0.2858 | **8.0%** | `outputs/results/attn_group_mask_summary.json` |
| 各局面で **5 回** ランダムし KL/flip を平均（スイープ実装・**k スイープ図・表と同一**） | **0.0438** | 0.1957 | **8.53%** | `outputs/results/attn_k_sweep_summary.json` の k=3 |

| 条件 | 平均 KL | 標準偏差 | Flip Rate | Prob Drop |
|---|---|---|---|---|
| **条件A（top-3）** | **0.0958** | 0.3176 | **14.7%** | **0.0259** |
| 条件C（random-3、**5×平均＝図5〜9と整合**） | **0.0438** | 0.1957 | **8.53%** | 0.0110 |
| 条件C（random-3、**1回のみ**・group_mask JSON） | **0.0597** | 0.2858 | 8.0% | 0.0127 |
| 条件B（bottom-3）| **0.0000** | 0.0000 | **0.0%** | **0.0000** |

条件A > 条件C（いずれかの定義）> 条件B の順で影響が大きい。**仮説と一致する。**

### ベースライングループ重要度（150 サンプル平均、`attn_group_mask_results.csv` の baseline_* で集計）

| グループ | 平均スコア | 主なイベント |
|---|---|---|
| **SafetyVsOthers** | **0.6747** | 子（非親）の捨て牌 DISCARD |
| **TileEfficiency** | **0.1759** | 自分の捨て牌 DISCARD |
| **PointSituation** | **0.1136** | AGARI / RYUUKYOKU / INIT |
| OpponentActions | 0.0205 | 他家の NAKI |
| YakuPotential | 0.0140 | REACH ほか |
| DoraValue | 0.0013 | DORA |
| SafetyVsDealer | 0.0000 | （本バッチでは観察上ほぼゼロ）|
| ShantenReduction | 0.0000 | （本バッチでは観察上ほぼゼロ）|

SafetyVsOthers が最大（約 67%）。これは同一 XML セット・150 サンプルにおいて
子の捨て牌イベントが event_seq に多く含まれる局面が多く、モデルが相手の河を強く参照していることを示す。

---

## 実験 2: k スイープ（k = 1, 2, 3, 4, 5）

### 目的

k=3 が実験設計として適切かを統計的に検証する（曲線としては **図5〜図9**）。

### 定量結果（n=150、random 5回平均）

| k | KL_top | KL_bot | KL_rnd | Gap | SNR | t値 | p値（片側）| 有意 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0708 | 0.0000 | 0.0228 | 0.0480 | **0.39** | 2.652 | 0.0044 | *** |
| **2** | 0.0944 | 0.0000 | 0.0402 | **0.0542** | 0.31 | 3.244 | 0.0007 | *** |
| **3** | **0.0958** | 0.0000 | **0.0438** | 0.0520 | 0.27 | **3.606** | 0.0002 | *** |
| 4 | 0.0963 | 0.0000 | 0.0718 | 0.0245 | 0.08 | 3.611 | 0.0002 | *** |
| 5 | 0.0963 | 0.0005 | 0.0800 | 0.0163 | 0.05 | **5.442** | **<0.0001** | *** |

p値は対応のある t 検定（top-k vs random-k）の片側（top の KL がより大きい）である。記号 *** は p<0.01 水準。

---

## k スイープ分析の解釈

### 観察1: Faithfulness と秩序

**すべての k で KL(top-k) > KL(random-k) ≥ KL(bottom-k)** が成立する（bottom は k=5 でのみ微小な KL）。図5（`figure/attn_k_sweep_aopc_curve.png`）で、上位マスクによる損傷が一貫して大きいことが読み取れる。条件B がほぼゼロに近いのは、低位グループのトークンが系列に現れない局面がまだ残るためと解釈できる。

### 観察2: Faithfulness Gap のピークと SNR

Gap の平均は **k=2 で最大**（0.0542）。**SNR（Gap / std(random KL)）は k=1 が最大**（0.39）。図6（`figure/attn_k_sweep_faithfulness_gap.png`）と図7（`figure/attn_k_sweep_snr.png`）の金の位置が食い違う点に注意。k≥4 では random が重要グループを踏みやすく Gap が縮小する。

### 観察3: サンプル数と「有意性だけでは k を決められない」

n=150 では **k=1 でも** top>random が **p=0.0044** と強く有意（図9 `figure/attn_k_sweep_pvalues.png`）。したがって **採用 k を p 値だけで選ぶことは不十分**であり、Gap・介入の解釈単位・KL(top) の飽和を併記する必要がある。

### 観察4: k=3 の位置付け（t 最大は k=5）

**t 統計量は k=5 で最大**（t≈5.44、p<0.0001）。**k=3**（t≈3.61、p=0.0002）も十分有意だが「検定最強」ではない。それでも k=3 を主たるプロトコルとする理由は次の **トレードオフ**である。

1. **有意性**: k=3 も高い有意水準を満たす（上表・図9）。
2. **解釈性**: SafetyVsOthers / TileEfficiency / PointSituation の **上位3グループを同時にマスクできる最小の k** が k=3（図4 `figure/attn_group_heatmap.png` の平均構造とも整合）。
3. **介入量**: KL(top) は k≥3 で ~0.096 付近に近づくが、k=5 のように広範マスクより説明がしやすい。
4. **XAI の慣習**: 「top-3 特徴」評価は文献でも一般的。

---

## k=3 を採用する根拠（教授向け論拠）

### 論拠A: 「有意」の全 k 化と設計選択

サンプルを増やすと **すべての k で有意**になりやすく、単に p を最小化すると **過剰にマスクする k（例: 5）**が選ばれがちである。**検定だけでなく** Faithfulness Gap や読み手への説明粒度をセットで述べる。図9（`figure/attn_k_sweep_pvalues.png`）では k=5 の t が最大だが、k=4,5 で Gap が小さい（図6）こともセットで読む。

### 論拠B: Faithfulness Gap と解釈可能な束ね方

データ駆動の Gap 単独最大は k=2 だが、k=3 は Gap も 0.052 と十分大きく、**3 因子を同時に遮断できる**評価プロトタイプとなる（図6 `figure/attn_k_sweep_faithfulness_gap.png`）。

### 論拠C: 上位グループ遮蔽の最小単位

k=3 以上で、平均重要度の上位 3 カテゴリ（SafetyVsOthers / TileEfficiency / PointSituation）を **常に** top-3 マスクに含められる設計に合わせやすい。k=2 では 3 位カテゴリが必ずしもマスク対象にならない。

### 論拠D: XAI 文献との整合性

Samek et al. (2017)、Lundberg & Lee (2017, SHAP) 等で「top-3 特徴」がよく報告される。8 グループ中 3 組を対象とするのは過剰でも不足でもない介入量として説明しやすい。

---

## 生成ファイル一覧

### 実験 1（k=3 固定）

| 保存先 | ファイル | 内容 |
|---|---|---|
| `figure/` | `attn_mask_kl_comparison.png`（図1） | 条件 A/B/C の KL 棒グラフ（k=3）|
| `figure/` | `attn_mask_fliprate.png`（図2） | Flip Rate / Prob Drop 棒グラフ（k=3）|
| `figure/` | `attn_mask_group_breakdown.png`（図3） | 条件 A で選ばれたグループ頻度 |
| `figure/` | `attn_group_heatmap.png`（図4） | 150 局面 × 8 グループ ヒートマップ |
| `outputs/results/` | `attn_group_mask_results.csv` | 全サンプル × 条件の詳細（k=3）|
| `outputs/results/` | `attn_group_mask_summary.json` | 条件別集計（k=3）|

### 実験 2（k スイープ）

| 保存先 | ファイル | 内容 |
|---|---|---|
| `figure/` | `attn_k_sweep_aopc_curve.png`（図5） | **AOPC カーブ**: KL vs k（top/bottom/random、誤差帯付き）|
| `figure/` | `attn_k_sweep_faithfulness_gap.png`（図6） | **Faithfulness Gap バーチャート** |
| `figure/` | `attn_k_sweep_flip_pdrop.png`（図8） | Flip Rate / Prob Drop の k 別折れ線 |
| `figure/` | `attn_k_sweep_snr.png`（図7） | SNR の k 別バーチャート |
| `figure/` | `attn_k_sweep_pvalues.png`（図9） | 各 k の t 統計量・p 値チャート |
| `outputs/results/` | `attn_k_sweep_results.csv` | 全サンプル × 全 k × 全条件の生データ |
| `outputs/results/` | `attn_k_sweep_summary.csv` | k 別集計サマリー |
| `outputs/results/` | `attn_k_sweep_summary.json` | 集計・結論 JSON |

---

## 実験スクリプト

| スクリプト | 役割 |
|---|---|
| `experiments/run_attn_group_mask_experiment.py` | k=3 固定の基礎実験 |
| `experiments/run_attn_k_sweep_experiment.py` | k=1〜5 スイープ＋統計分析 |
| 同上 `python ... --plots-only` | 保存済み `attn_k_sweep_results.csv` から図だけ再描画（計算省略） |

**設定変更箇所**（スクリプト上部の定数）:

```python
XML_DIR      = Path("/home/ubuntu/Documents/tenhou_xml_2023")
CHECKPOINT   = _ROOT / "outputs/impl1/hdf5_10epoch.pt"
N_XML_FILES  = 5      # パースするXMLファイル数
MAX_SAMPLES  = 150    # 実験サンプル数上限
K_VALUES     = [1, 2, 3, 4, 5]
RANDOM_SEED  = 42
N_RANDOM_REPEATS = 5  # random条件の反復回数
```

---

## 注意事項・次のステップ

### 結果の制限事項

- **モデルが小規模**: 現チェックポイントは `d_model=64, n_layers=2`（小規模構成）。
  より大きいモデルでは attention が複雑になり、グループスコアの分散が増える可能性がある。
- **特定グループが未活性**: SafetyVsDealer / DoraValue / YakuPotential / OpponentActions は
  今回のサンプルでスコアが 0。リーチ・ドラ・副露を含む局面を意図的にサンプリングすれば
  全グループの挙動を確認できる。
- **bottom-k KL=0 の解釈**: これは「重要でない特徴を隠しても影響なし」という理想的な結果だが、
  「そもそも該当イベントが存在しない」という別要因も含む。局面種別でフィルタした再実験が有効。

### 次のステップ

1. **LLM 統合後の BERTScore 評価**: `explain_fn` に Gemini API を接続し、
   `AttentionPatchingEvaluator.run_batch()` で説明文変化の定量化。
2. **より大きなサンプル**: XML ファイル数または `MAX_SAMPLES` を増やした追加バッチ（再現には seed を固定）。
3. **局面種別フィルタ**: リーチ局面・副露局面を含めて全 8 グループが活性化するデータで再実験。
4. **論文用図の作成**: AOPC カーブと Faithfulness Gap を論文グレード（LaTeX 対応フォント）で再生成。
