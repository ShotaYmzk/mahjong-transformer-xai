# Attention グループマスク：説明メモ（議論・数値の整理）

作成日: 2026/05/12

開発チャットで口頭説明した内容を、そのまま論文ドラフトやスライドに転用できる形で **`docs/` に固定**したもの。指標・数値の一次ソースは **変わらず** `outputs/results/*.json` / `*.csv` および実行スクリプト参照。

---

## 関連ドキュメントと実装の所在

| 話題 | どこを読むか |
|------|----------------|
| モデル **`MahjongTransformerV2`**、学習スクリプト（HDF5 / XML ストリーム / NPZ） | [model-development-guide.md](model-development-guide.md)「まず使うファイル → 学習」、[project-structure.md](project-structure.md) |
| 教師データ・**特徴量テンソル**の形状・意味（`static` 157 次元、`sequence` 60×6 など） | [year-scale-training-pipeline.md](year-scale-training-pipeline.md) §4、[data/observation_schema.py](../data/observation_schema.py) |
| **学習率スケジューラ（LR scheduler）** | **本プロジェクトでは未使用。** `scripts/train_transformer_v2_hdf5.py` などは **`torch.optim.AdamW`** と **`--lr` 固定のみ**。詳細は上記パイプライン §6。長時間運用は `scripts/watch_hdf5_training.sh` の **再起動ループ**（PyTorch のスケジューラではない）。 |
| グループマスク実験の数値・図一覧 | [experiment-attn-group-mask-2026-05-12.md](experiment-attn-group-mask-2026-05-12.md) |
| `attention_patching.py` の API・LLM 連携評価 | [xai-attention-patching.md](xai-attention-patching.md) |

---

## Decision Flip Rate：何の割合か（分子・分母）

**定義**（コード）: [`experiments/metrics/faithfulness.py`](../experiments/metrics/faithfulness.py) の `decision_flip_rate`。  
ベースライン logits とマスク後 logits で **Top-1 種別が変わったか** をバッチごとに `{0,1}` で見て **平均**する（1 サンプル 1 局面なら 0 か 1）。

実験は **`n_samples=150`**、k=3 固定での代表値は **`attn_group_mask_summary.json`** および **`attn_k_sweep_summary.csv`** と整合。

### 条件A（top-k グループをマスク）

- 各局面でマスク適用は **1 回**。`flip` は **常に 0 または 1**。
- **母数**: **150** 局面。
- **分子**: 平均値 `mean_flip_rate = 0.1467`（四捨五入後）と整合する **`22`**（**22 / 150** ≒ 14.67%。表記上の「約 14.7%」）。

### 条件B（bottom-k）

- 同様に 0/1。**0 / 150**。

### 条件C（random-k）— 定義が2通りあるので混同しない

| 実装 | ファイル | Flip の読み |
|------|----------|-------------|
| **random を 1 回だけ**サンプル | `run_attn_group_mask_experiment.py` → [`attn_group_mask_summary.json`](../outputs/results/attn_group_mask_summary.json) | **12 / 150 = 8.0%**（`mean_flip_rate: 0.08`） |
| **5 回試行して KL・flip を平均** | `run_attn_k_sweep_experiment.py` → [`attn_k_sweep_summary.csv`](../outputs/results/attn_k_sweep_summary.csv) の `random_flip_mean`（k=3 行など） | 各局面の `flip` は **0, 0.2, …, 1.0**（5 回の 0/1 の平均）。**150 局面の平均** が例えば **0.0853**（「約 8.53%」）。**ベルヌイ試行の総数としては 150×5 = 750 回**；`0.0853` が四桁丸めだと総フリップ数は **約 64 / 750** と整合 |

論文では **どちらの条件Cを書いているか** を明示すること（実験記録本体にも同様の注意あり）。

---

## k スイープの表・Faithfulness Gap 図は何を表すか

- **縦軸の k**: 「**何グループ分**マスクするか」（特徴グループは 8 通り、[attention_patching.py](../attention_patching.py) の `FEATURE_GROUP_NAMES`）。
- **KL (top)**: Attention 重要度が高いグループ由来の **キー位置**への注意を抑えたときの出力分布の変化（KL）。
- **KL (bottom)**: 重要度の低いグループのみを抑えたとき。分布がほぼ変わらなければ **介入が意味のある階層を拾えている sanity check**。
- **KL (random, 5×平均)**: ランダムに選んだ k グループを同様に抑えたときの KL を **複数試行で平均**。偶然のマスクの強さ。
- **Gap（Faithfulness Gap）**: **KL(top) − KL(random)**。「重要グループ」を潰したときだけ、偶然以上にモデル出力がずれるか。

**棒グラフ（図6 [`figure/attn_k_sweep_faithfulness_gap.png`](../figure/attn_k_sweep_faithfulness_gap.png)）**は Gap を k で見たもの。データ上 Gap が最大となる k と、本文で採用する **報告用 k（例: k=3）**は必ずしも一致しない（有意性・解釈のしやすさでのトレードオフ）。理由の詳細は [experiment-attn-group-mask-2026-05-12.md](experiment-attn-group-mask-2026-05-12.md) の「図と『採用 k』の対応」。

---

## 「グループ別平均重要度スコア」の表は何か

各局面について:

1. 最終ブロックの **attention 重み**から、ヘッド・クエリ平均で **系列位置ごとの重要度**を得る。
2. **`_default_feature_group_map(sequence, player_id)`** で、各時刻を **麻雀意味のグループに唯一割り当て**（例: 他人の DISCARD→`SafetyVsOthers`）。
3. グループに属する位置の重要度を **合計して正規化**し、そのグループのスコアにする。
4. **150 局面で平均**したものが、「グループ別の平均％」のような表になる。

これは **「そのサブセットの局面でモデルが系列のどの意味ラベルに注意を載せていたか」の内訳**であり、(1) 全体のモデル強度の内訳ではない、(2) 牌譜の取り方を変えると比率は変わる、(3) グループ割当はヒューリスティックである、と説明すると誤読が減る。一次メモと数値は [experiment-attn-group-mask-2026-05-12.md](experiment-attn-group-mask-2026-05-12.md) の「ベースライングループ重要度」。

---

## マスクは「グループ単位」か「attention だけ」か

**両方**正しい。**低レベル**では Transformer の **キー側の系列インデックス**に対して、**全層・全ヘッド**で attention logits にパッチを当てている（[`models/mahjong_transformer_v2.py`](../models/mahjong_transformer_v2.py) の `apply_attention_patch`）。  
**どのインデックスを選ぶか**が、単位セル単位ではなく **「麻雀意味グループ → 複数インデックスの和集合」**で決まる（[`experiments/run_attn_group_mask_experiment.py`](../experiments/run_attn_group_mask_experiment.py) の `evaluate_sample`、`build_position_patch`）。

流れの要約:

1. `group_positions[g]` = グループ g に属する **時刻インデックスのリスト**。
2. ベースラインでグループごとに importance を足して **top/bottom/random で k グループ選択**。
3. 選んだグループのインデックスを **和集合・重複除去**して `positions_to_mask`。
4. それらの **キー位置**だけ attention が接続されないようにする。

論文では **「意味のあるイベント塊単位での attention 介入」**と書ける。

---

## 実装ファイル早見（本テーマのみ）

```text
attention_patching.py              # FEATURE_GROUP_NAMES, _default_feature_group_map
experiments/run_attn_group_mask_experiment.py
experiments/run_attn_k_sweep_experiment.py
experiments/metrics/faithfulness.py
models/mahjong_transformer_v2.py   # attention_patch の適用箇所
```

