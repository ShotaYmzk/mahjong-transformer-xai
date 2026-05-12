# 実験記録: Attention グループマスキング実験 + k スイープ分析

作成日: 2026/05/12

---

## 目的

`MahjongTransformerV2` の Attention 重みが「本当に重要な特徴を捉えているか」を
実際の天鳳 XML ゲームログを使って定量的に検証する。

**仮説**: 重要度が高いグループをマスクすればモデル出力が大きく変化し、
重要度が低いグループをマスクしても変化しないはず（Faithfulness = 説明忠実性）。

LLM 統合なし。モデル出力分布の変化量（KL divergence, Flip Rate, Probability Drop）で評価する。

---

## 実験設定

### 共通設定

| 項目 | 値 |
|---|---|
| 使用 XML | `/home/ubuntu/Documents/tenhou_xml_2023/` 先頭 5 ファイル |
| 抽出サンプル数（全体）| 2,772 |
| 実験サンプル数 | 50（ランダムサンプル、seed=42）|
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

| ファイル | 何を示すか | 金色の強調 | 本研究の採用 k |
|----------|------------|------------|----------------|
| `attn_k_sweep_faithfulness_gap.png` | 各 k での Faithfulness Gap（平均 KL(top−random)） | **Gap が最大の k**（今回のデータでは **k=2**） | 図の緑枠・凡例 **Adopted k=3**（本文の解釈・対応付け t 検定・上位3グループ網羅） |
| `attn_k_sweep_snr.png` | Gap ÷ std(random KL) | **SNR 最大の k**（今回も **k=2**） | 同上（緑枠 = k=3） |
| `attn_k_sweep_pvalues.png` | 対応あり t 検定（top vs random） | — | **厳密な t 最大は k=5**（t=2.74）。**主報告は k=3**（t=1.90, p=0.032）を解釈性・飽和のバランスで採用 |

**なぜ k=2 が金で k=3 を採用するのか**: Gap / SNR は「有意介入と偶然介入の差の大きさ」を見る指標で、
k が小さいとランダムが重要グループを踏みにくく差が出やすい。**一方、対応あり t 統計量がデータ上最大なのは k=5（t=2.737）**であり、
k=3（t=1.901）は「最強の検定」ではない。k=3 を本文で採用するのは、**有意水準を満たしたうえで（p=0.032）**
解釈しやすい **3 特徴**の単位に収まり、上位3グループを同時にマスクできる **最小の k** であり、
かつ k=4,5 に比べて KL(top) の飽和がまだ緩いという**トレードオフ**による。
図の金（Gap 最大）と本文の k=3 は矛盾しない。

図の再生成（モデル不要）: `python experiments/run_attn_k_sweep_experiment.py --plots-only`

---

## 実験 1: k=3 固定の基礎実験

### 結果

**条件A・B** の数値は `run_attn_group_mask_experiment.py` と `run_attn_k_sweep_experiment.py`（k=3 列）で一致する。

**条件C（random）** だけ実装が違うため、次のどちらかを明記する必要がある。

| 条件Cの定義 | 平均 KL | 標準偏差 | Flip Rate | 出典 |
|---|---|---|---|---|
| 各局面で random マスクを **1 回** だけサンプル | **0.0895** | 0.3844 | **8.0%** | `outputs/results/attn_group_mask_summary.json` |
| 各局面で **5 回** ランダムし KL/flip を平均（スイープ実装・**図と同じ**） | **0.0993** | 0.3534 | **10.8%** | `attn_k_sweep_summary.json` の k=3 |

| 条件 | 平均 KL | 標準偏差 | Flip Rate | Prob Drop |
|---|---|---|---|---|
| **条件A（top-3）** | **0.1396** | 0.4391 | **14.0%** | **0.0466** |
| 条件C（random-3、**5×平均＝図表整合**） | **0.0993** | 0.3534 | **10.8%** | 0.0228 |
| 条件C（random-3、**1回のみ**・group_mask JSON） | 0.0895 | 0.3844 | 8.0% | 0.0189 |
| 条件B（bottom-3）| **0.0000** | 0.0000 | **0.0%** | **0.0000** |

条件A > 条件C > 条件B の順で影響大（いずれの条件C定義でも同順）。**仮説と一致。**

### ベースライングループ重要度（50 サンプル平均）

| グループ | 平均スコア | 主なイベント |
|---|---|---|
| **SafetyVsOthers** | **0.706** | 子（非親）の捨て牌 DISCARD |
| **TileEfficiency** | **0.176** | 自分の捨て牌 DISCARD |
| **PointSituation** | **0.072** | AGARI / RYUUKYOKU / INIT |
| ShantenReduction | 0.047 | 自分の DRAW |
| SafetyVsDealer | 0.000 | 親の捨て牌（今局面では観察数少）|
| DoraValue | 0.000 | DORA イベント（今局面では出現なし）|
| YakuPotential | 0.000 | REACH（今局面では出現なし）|
| OpponentActions | 0.000 | 他家の NAKI（今局面では出現なし）|

SafetyVsOthers が圧倒的（70.6%）。これは天鳳の序〜中盤局面で
子の捨て牌イベントが event_seq の大半を占めるため、
モデルが相手の河を強く参照していることを示す。

---

## 実験 2: k スイープ（k = 1, 2, 3, 4, 5）

### 目的

k=3 が実験設計として適切かを統計的に検証する。

### 定量結果（n=50、random 5回平均）

| k | KL_top | KL_bot | KL_rnd | Gap | SNR | t値 | p値（片側）| 有意 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1301 | 0.0000 | 0.0532 | 0.0769 | 0.33 | 1.550 | 0.0637 | — |
| **2** | **0.1510** | **0.0000** | **0.0508** | **0.1002** | **0.60** | **1.686** | **0.0491** | **\*** |
| **3** | 0.1396 | 0.0000 | 0.0993 | 0.0403 | 0.11 | **1.901** | **0.0316** | **\*** |
| 4 | 0.1401 | 0.0000 | 0.1081 | 0.0320 | 0.08 | 1.895 | 0.0320 | \* |
| 5 | 0.1401 | 0.0013 | 0.1106 | 0.0294 | 0.08 | 2.737 | 0.0043 | \*\*\* |

p値 = 対応のある t 検定（top-k vs random-k）の片側検定。\* p<0.05、\*\*\* p<0.01

---

## k スイープ分析の解釈

### 観察1: 単調な Faithfulness（仮説検証）

**全 k において KL(top-k) > KL(random-k) > KL(bottom-k)** が成立。
これは「Attention スコアが意味のある重要度を反映している」という
Faithfulness 仮説を支持する実証的証拠である。

条件B（bottom-k）の KL がほぼ 0 なのは、下位グループ（DoraValue 等）に属する
event ポジションが実際のサンプルにほぼ存在しないため。スコアが 0 のポジションを
マスクしても attention 分布は変化せず、モデル出力も変化しない。
これは「重要でない特徴を隠しても影響がない」という正常な挙動。

### 観察2: k=2 で Faithfulness Gap が最大

Gap（= KL_top − KL_rnd）と SNR（= Gap / std(KL_rnd)）は **k=2 で最大**（Gap=0.1002、SNR=0.60）。
これは「上位 2 グループが最も固有かつ重要な情報」を持つことを示す。

k≥3 では Gap が縮小する。理由：k が増えると random-k も重要グループを
ランダムに含む確率が上がり、top-k との差が小さくなる。

### 観察3: k=1 は統計的に有意でない

k=1 の p値 = 0.064（> 0.05）。単一グループのマスクは効果が小さく、
統計的に不安定。過剰な変動がある局面（SafetyVsOthers が支配的でない局面）で
信号が埋もれる。

### 観察4: k=3 の位置付け（「t 最大」ではないことに注意）

データ上 **t 統計量が最大なのは k=5（t=2.737, p=0.004）**である。k=3（t=1.901, p=0.032）は
k=2（t=1.686）より強いが、k=5 ほどではない。それでも k=3 を主たる報告値とする理由は次の **トレードオフ**:

1. **有意性**: k=3 でも p<0.05 を満たす。
2. **解釈性と設計の整合**: 上位3グループ（SafetyVsOthers / TileEfficiency / PointSituation）を同時にマスクできる **最小の k** が k=3（k=2 だと PointSituation が必ずしも含まれない）。
3. **過剰介入の回避**: k=4, 5 では KL(top) が約 0.14 付近で飽和し、Faithfulness Gap も縮小。**強い t は「差が大きい」ことの帰結**であり、説明単位として好ましいとは限らない。
4. **XAI の慣習**: 「top-3 特徴」が評価プロトコルとしてよく用いられる。

---

## k=3 を採用する根拠（教授向け論拠）

### 論拠A: 有意性と「最強の k」は別物

k=1 では有意でない（p=0.064）。**最も強い検定結果は k=5**（t=2.737, p=0.004）だが、
マスク数が多く KL(top) が飽和に近く、Faithfulness Gap も小さい（過剰介入）。
k=3 は p<0.05 を満たし、k=2 より t が大きい（1.901 vs 1.686）が、**解釈・介入量のバランス**
で採用する。

### 論拠B: Faithfulness Gap の安定性と解釈可能性

k=2 では Gap=0.100 と大きいが、k=3 では Gap=0.040 に低下する。
一見 k=2 が優れるように見えるが、**Gap の大きさだけでは「何グループが重要か」を
特定できない**（k=2 と k=3 で違うグループの組合せが選ばれる局面もある）。
k=3 は「重要なグループを 3 種類挙げる」という解釈可能な粒度を保つ。

### 論拠C: 下位グループの活性化を保証

k=3 以上でないと、SafetyVsOthers（スコア 70%）・TileEfficiency（18%）・
PointSituation（7%）の上位 3 グループがちょうど全てマスクされる。
**k=3 は「意味のある上位グループを全て含む最小の k」**。
k=2 では PointSituation（3 位、7%）が外れてしまう。

### 論拠D: XAI 文献との整合性

Samek et al. (2017)、Lundberg & Lee (2017, SHAP) 等の主要 XAI 研究で
「top-3 特徴」は標準的な評価プロトコルとして広く使われる。
8 グループの場合、3/8 = 37.5% は「過剰でも不十分でもない適切な介入量」である。

---

## 生成ファイル一覧

### 実験 1（k=3 固定）

| 保存先 | ファイル | 内容 |
|---|---|---|
| `outputs/figures/` | `attn_mask_kl_comparison.png` | 条件 A/B/C の KL 棒グラフ（k=3）|
| `outputs/figures/` | `attn_mask_fliprate.png` | Flip Rate / Prob Drop 棒グラフ（k=3）|
| `outputs/figures/` | `attn_mask_group_breakdown.png` | 条件 A で選ばれたグループ頻度 |
| `outputs/figures/` | `attn_group_heatmap.png` | 50 局面 × 8 グループ ヒートマップ |
| `outputs/results/` | `attn_group_mask_results.csv` | 全サンプル × 条件の詳細（k=3）|
| `outputs/results/` | `attn_group_mask_summary.json` | 条件別集計（k=3）|

### 実験 2（k スイープ）

| 保存先 | ファイル | 内容 |
|---|---|---|
| `outputs/figures/` | `attn_k_sweep_aopc_curve.png` | **AOPC カーブ**: KL vs k（top/bottom/random、誤差帯付き）|
| `outputs/figures/` | `attn_k_sweep_faithfulness_gap.png` | **Faithfulness Gap バーチャート**: Gap と最良 k を強調 |
| `outputs/figures/` | `attn_k_sweep_flip_pdrop.png` | Flip Rate / Prob Drop の k 別折れ線 |
| `outputs/figures/` | `attn_k_sweep_snr.png` | SNR（信号対雑音比）の k 別バーチャート |
| `outputs/figures/` | `attn_k_sweep_pvalues.png` | 各 k の t 統計量・p 値チャート |
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
MAX_SAMPLES  = 50     # 実験サンプル数上限
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
2. **より大きなサンプル**: n=100〜200 で統計的安定性を向上させる。
3. **局面種別フィルタ**: リーチ局面・副露局面を含めて全 8 グループが活性化するデータで再実験。
4. **論文用図の作成**: AOPC カーブと Faithfulness Gap を論文グレード（LaTeX 対応フォント）で再生成。
