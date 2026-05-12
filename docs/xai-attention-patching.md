# XAI: Attention Patching 実装仕様

作成日: 2026/05/12

`attention_patching.py` と `visualize_attention.py` の設計・使い方・注意点をまとめる。

---

## 目的

`MahjongTransformerV2` の Attention 重みを特徴グループ単位でマスクし、
マスク前後で LLM 生成説明文がどれだけ変化するかを定量的に測ることで、
「モデルが本当にその特徴を見て判断しているか」を検証する（説明忠実性）。

---

## ファイル構成

```
attention_patching.py       ← 忠実性評価クラス（LLM連携・BERTScore・統計検定）
visualize_attention.py      ← 可視化クラス（棒グラフ・ヒートマップ）
```

**既存のどのファイルも変更していない。**

---

## 既存コードとの関係

```
experiments/interventions/attention_masks.py
  → attention logits に直接パッチを当てて「モデル出力」の変化を測る

experiments/metrics/faithfulness.py
  → KL divergence, flip rate, probability drop などの低レベル指標

experiments/visualize/attention_heatmap.py
  → attention 行列を PNG/CSV に保存するユーティリティ

attention_patching.py（新規）
  → attention weights をグループ集約→マスク→LLM説明文変化を測る（高レベル）

visualize_attention.py（新規）
  → グループスコアをグラフ表示（attention_patching.py を内部で使う）
```

---

## モデル構造の実態（コードを読んで確認）

| 項目 | 実際の値 |
|---|---|
| クラス名 | `MahjongTransformerV2`（`models/mahjong_transformer_v2.py`）|
| `d_model` | **128**（デフォルト）|
| `n_heads` | **8**（デフォルト）|
| `n_layers` | **4** |
| Attention 実装 | カスタム `HookedSelfAttention`（`nn.MultiheadAttention` ではない）|
| ブロック格納先 | `model.blocks`（`ModuleList[HookedTransformerBlock]`）|
| Attention weight 取得 | 常に返す（`need_weights` 問題なし）|
| forward 戻り値 | `(out_tensor, {"attn_logits", "attn_weights", "head_outputs"})` |

---

## `attention_patching.py` の設計

### データ構造

```python
FEATURE_GROUP_NAMES = [
    "ShantenReduction",   # 自分のツモ（手牌形状・シャンテン変化）
    "SafetyVsDealer",     # 親の捨て牌（放銃危険度）
    "SafetyVsOthers",     # 子の捨て牌（放銃危険度）
    "DoraValue",          # ドラ表示イベント
    "YakuPotential",      # リーチ宣言（役形成）
    "TileEfficiency",     # 自分の捨て牌（牌効率）
    "PointSituation",     # 和了・流局・配牌（点数状況）
    "OpponentActions",    # 他家の副露（鳴き）
]
```

`GameStateBatch` — モデル入力テンソルをまとめるデータクラス。

`SingleResult` — 1局面の評価結果（スコア・説明文・BERTScore・キーワードシフト）。

### `AttentionPatchingEvaluator` クラス

```python
evaluator = AttentionPatchingEvaluator(
    model,          # MahjongTransformerV2（eval mode）
    explain_fn,     # callable(game_state, scores) → str（LLM等を注入）
    k=3,            # マスクするグループ数
)
```

#### 主なメソッド

| メソッド | 役割 |
|---|---|
| `_register_hooks()` | `HookedSelfAttention` 各層に `register_forward_hook` を設置 |
| `_remove_hooks()` | フック全解除 |
| `_extract_raw_attention(batch)` | forward → 最終層 attn_weights `(B, H, S, S)` |
| `_compute_group_scores(attn, seq, pids)` | heads+query 軸で平均 → 8グループに集約・正規化 |
| `_mask_attention_scores(scores, cond, k)` | 3条件マスク（top/bottom/random）+ 再正規化 |
| `_compute_bertscore(hyps, refs)` | bert_score で F1 計算（GPU 自動選択）|
| `_compute_keyword_shift(orig, masked)` | 8キーワードの出現有無変化率 |
| `run_single(game_state)` | 1局面・全条件を実行 → `SingleResult` |
| `run_batch(game_states, n=100)` | バッチ実行 + t検定 + Spearman 相関 + CSV 出力 |

#### マスク条件

| 条件 | 内容 | 狙い |
|---|---|---|
| 条件A（`"top"`）| 重要度上位 k グループをゼロマスク | 重要な特徴を隠すと説明が大きく変わるはず |
| 条件B（`"bottom"`）| 重要度下位 k グループをゼロマスク | 重要でない特徴を隠しても説明はあまり変わらないはず |
| 条件C（`"random"`）| ランダム k グループをゼロマスク | ベースライン（条件Aと比較する）|

マスク後は常に再正規化（sum=1 を維持）。

#### 統計検定

- **対応のある t 検定**（`scipy.stats.ttest_rel`）: 条件A vs 条件C の BERTScore F1
- **Spearman 相関**（`scipy.stats.spearmanr`）: マスク強度 k（1〜5）vs BERTScore 低下量

### 特徴グループマッピング（デフォルト実装）

`_default_feature_group_map(sequence, player_id)` が event_seq の各タイムステップを
イベント種別（`data/observation_schema.EVENT_TYPES`）とプレイヤー ID で分類する。

```
EVENT_TYPES: INIT=0, DRAW=1, DISCARD=2, N=3, REACH=4, DORA=5, AGARI=6, RYUUKYOKU=7, PADDING=8
```

カスタムマッピングを使いたい場合は `feature_group_map` 引数に関数を渡す:

```python
def my_map(sequence: Tensor, player_id: int) -> dict[str, list[int]]:
    ...

evaluator = AttentionPatchingEvaluator(model, explain_fn, feature_group_map=my_map)
```

---

## `visualize_attention.py` の設計

### `AttentionVisualizer` クラス

`AttentionPatchingEvaluator` を内部に持ち、コード重複なしで同じ hook・グループ集約・
マスクを再利用する。

```python
vis = AttentionVisualizer(model)
```

#### 主なメソッド

| メソッド | 役割 |
|---|---|
| `_forward_and_get_attention(game_state)` | forward → 最終層 attn_weights `(B, H, S, S)` |
| `_compute_group_scores(attn, game_state)` | 8グループスコア dict |
| `plot_mask_comparison(game_state, k=3, save_path)` | マスク前後を4色の横並び棒グラフで表示 |
| `plot_group_heatmap(game_states, game_labels, save_path)` | N局面×8グループのヒートマップ |

#### 棒グラフの色分け

| 系列 | 色 |
|---|---|
| Original | steelblue（青）|
| Top-k masked（条件A）| tomato（赤）|
| Bottom-k masked（条件B）| mediumseagreen（緑）|
| Random-k masked（条件C）| darkorange（オレンジ）|

---

## 使い方：実際の checkpoint で実験する手順

### 1. モデル読み込み

```python
import torch
from models.mahjong_transformer_v2 import MahjongTransformerV2, MahjongTransformerConfig

ckpt = torch.load("outputs/impl1/hdf5_10epoch.pt", map_location="cpu")
cfg = MahjongTransformerConfig(**ckpt["config"])
model = MahjongTransformerV2(cfg)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

### 2. explain_fn を実装（Gemini 等を使う場合）

```python
import google.generativeai as genai  # API キーは別ファイルで設定

def explain_fn(game_state: dict, attention_scores: dict) -> str:
    prompt = f"以下のAttentionスコアに基づき麻雀の打牌方針を日本語で説明してください。\n{attention_scores}"
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text
```

### 3. 1局面の評価

```python
from attention_patching import AttentionPatchingEvaluator

evaluator = AttentionPatchingEvaluator(model, explain_fn, k=3)
result = evaluator.run_single(game_state)
print(result.bertscore_f1)     # {"top": 0.82, "bottom": 0.95, "random": 0.91}
print(result.keyword_shift)    # {"top": 0.25, "bottom": 0.125, "random": 0.0}
```

### 4. 100局面バッチ実行

```python
df, stats = evaluator.run_batch(
    game_states,         # list of game_state dicts
    n=100,
    output_csv="outputs/results/attention_patching_100.csv",
)
ttest = stats["ttest_top_vs_random"]
print(f"t={ttest.statistic:.4f}  p={ttest.pvalue:.4f}")
```

### 5. 可視化

```python
from visualize_attention import AttentionVisualizer

vis = AttentionVisualizer(model)
vis.plot_mask_comparison(game_state, k=3, save_path="figure/mask_comparison.png")
vis.plot_group_heatmap(game_states[:20], save_path="figure/group_heatmap.png")
```

---

## 依存ライブラリ

| ライブラリ | 用途 | 追加日 |
|---|---|---|
| `pandas` | DataFrame・CSV出力 | 2026/05/12 |
| `scipy` | t検定・Spearman相関 | 2026/05/12 |
| `bert-score` | BERTScore F1 計算 | 2026/05/12 |
| `seaborn` | ヒートマップ描画 | 2026/05/12 |

```bash
pip install bert-score scipy pandas seaborn
```

---

## 注意事項

- BERTScore は初回実行時に `bert-base-multilingual-cased` をダウンロードする（約700MB）。
- BERTScore 計算はサンプルごとに呼び出すと遅い。100局面×3条件=300回の呼び出しになるため、
  バッチ化を検討する（現在は1件ずつ）。
- ランダムマスク（条件C）は seed 管理していないため、`run_batch` を2回実行すると結果が変わる。
  論文用データは seed を固定すること（`random.seed(42)` を `run_batch` 前に呼ぶ）。
- `explain_fn` が外部 LLM API を叩く場合、100局面×4回（ベースライン+3条件）= 400回の API 呼び出しが発生する。
  コスト・レート制限に注意。
