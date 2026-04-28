# Mahjong Transformer XAI 実装1 基盤

修士論文のための麻雀AI実験基盤です。天鳳XML牌譜から「打牌選択」の教師データを作り、`MahjongTransformerV2` を学習し、Attention / Head / Activation への介入によってモデル挙動の変化を測定します。

特に本実装では、前回問題になった **相手手牌リーク** を最重要リスクとして扱っています。完全情報の牌譜XMLを内部で処理しつつ、モデル入力には「行動者本人の手牌」と「全員に見える公開情報」だけが残るように、状態表現を明確に分離しています。

## 研究上の位置づけ

本実装は、修士論文案のうち「実装1: Attention数値調整によるモデル挙動変化の検証」の基盤です。

目的は、麻雀AIの判断に対して以下を調べることです。

- Attention重みや中間活性が、打牌判断にどの程度因果的に関係しているか。
- Attention上位要素をマスクしたとき、打牌分布やTop-k accuracyがどう変わるか。
- Head ablationで重要なattention headを特定できるか。
- Activation patchingで局面中のどの位置・層が元の打牌を支えているかを測れるか。

現在は説明文生成までは入れず、まず「正しい観測情報だけで打牌モデルを作る」「内部表現へ介入できる」段階を実装しています。

## 最重要: 相手手牌リーク防止

天鳳XMLには `INIT` に全員の配牌が含まれ、終局時にも `AGARI` / `RYUUKYOKU` の `hai` に手牌情報が出ることがあります。そのため、何も考えずに特徴量化すると、モデルが相手手牌や未来情報を見て学習してしまいます。

この実装では、次の方針で防いでいます。

- `PrivateRoundState` は内部検算用に全員の手牌を持つ。
- `ObservedState` はモデル入力用で、相手手牌を表すフィールドを持たない。
- `DatasetRow` に保存されるのは、`static_features`, `sequence_features`, `hand_counts`, `aka_flags`, `valid_mask`, `label`, `metadata` のみ。
- `validate_no_private_leakage()` で、保存前にリークしそうなキーや不合法ラベルを検査する。
- 他家ツモ牌はイベント列では牌IDを隠し、「誰の手番か」だけ残す。
- 今から予測する打牌の手出し/ツモ切りは、入力ではなく分析用metadataにだけ保存する。
- 終局時の `hai` は途中局面サンプルには使わない。

入力に入れてよい情報は以下です。

- 行動者本人の手牌。
- 全員の河。
- 河にある各捨て牌の手出し/ツモ切りフラグ。
- 全員の副露情報。
- リーチ状態、点数、親、本場、供託、場風、巡目、ドラ表示牌。
- 公開済みイベント履歴。

入力に入れてはいけない情報は以下です。

- 他家の配牌、現在手牌、ツモ牌。
- 他家の待ちや将来の打牌。
- 未来のドラ、未来のリーチ確定、未来の副露、未来の河。
- 終局時手牌を途中局面に逆流させた情報。

## ディレクトリ構成

```text
.
├── configs/
│   └── impl1_pilot.yaml
├── data/
│   ├── __init__.py
│   └── observation_schema.py
├── experiments/
│   ├── interventions/
│   │   ├── activation_patching.py
│   │   ├── attention_masks.py
│   │   └── head_ablation.py
│   ├── metrics/
│   │   └── faithfulness.py
│   └── visualize/
│       ├── attention_heatmap.py
│       └── causal_trace.py
├── models/
│   └── mahjong_transformer_v2.py
├── scripts/
│   ├── build_discard_dataset.py
│   ├── build_discard_hdf5_shards.py
│   ├── evaluate_policy.py
│   ├── evaluate_stream_xml_topk.py
│   ├── mjx_crosscheck.py
│   ├── train_transformer_v2.py
│   ├── train_transformer_v2_hdf5.py
│   └── train_transformer_v2_stream_xml.py
├── tests/
│   └── test_observation_extraction.py
└── utils/
    ├── dataset_utils.py
    ├── feature_utils.py
    ├── game_state.py
    ├── geme_state.py
    ├── mahjong_parser.py
    ├── naki_utils.py
    ├── shanten.py
    ├── tile_utils.py
    └── xml_parser.py
```

## 主要モジュール

### `data/observation_schema.py`

天鳳XMLから学習サンプルを作る中核です。

主なクラスと関数:

- `PrivateRoundState`
  - XMLを正しく再生するための完全情報状態。
  - 全員の手牌を持ちますが、学習データには直接保存しません。

- `ObservedState`
  - 行動者視点の観測状態。
  - 本人手牌、河、副露、ドラ、点数など公開情報のみを保持します。

- `DatasetRow`
  - 1つの教師データ行。
  - モデルに渡す特徴量と打牌ラベルを保持します。

- `build_dataset_rows_from_xml()`
  - XMLファイルまたはディレクトリから `DatasetRow` を生成します。

- `validate_no_private_leakage()`
  - NPZ/HDF5保存前に、相手手牌や不正ラベルが混入していないか検査します。

### `models/mahjong_transformer_v2.py`

打牌34種分類モデルです。

入力:

- `static`: 局面の静的特徴量。
- `sequence`: 公開イベント履歴。
- `hand_counts`: 行動者本人の34種手牌カウント。
- `aka_flags`: 行動者本人の赤ドラ所持フラグ。
- `valid_mask`: 合法打牌マスク。

出力:

- 34種類の打牌logits。

特徴:

- `return_internals=True` で以下を取得できます。
  - attention logits
  - attention weights
  - head outputs
  - hidden states
- `attention_patch`, `head_ablation`, `activation_patch` をforward引数で受け取れます。
- `valid_mask` によって非合法打牌は `-1e9` にマスクされます。

### `experiments/interventions/`

内部表現への介入実験です。

- `attention_masks.py`
  - Top-k / Random-k / Bottom-k / Uniform attention mask。

- `head_ablation.py`
  - 特定layer/headの出力をゼロ化。
  - KL divergenceでhead重要度を測れます。

- `activation_patching.py`
  - clean runのhidden stateをcorrupted runへ差し戻すactivation patching。
  - STRの簡易実装として、本人手牌の萬子・筒子・索子ブロックを入れ替える関数も用意しています。

### `experiments/metrics/faithfulness.py`

介入実験用メトリクスです。

- `decision_flip_rate`
- `kl_divergence`
- `probability_drop`
- `logit_difference_delta`
- `aopc`

### `experiments/visualize/`

論文・進捗報告用の可視化出力です。

- `save_attention_heatmap()`
- `save_causal_trace_heatmap()`

`matplotlib` が使える場合はPNG、使えない場合はCSVで保存します。

## データセット作成

### 小規模NPZを作る

最初の動作確認用です。

```bash
python scripts/build_discard_dataset.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --output outputs/impl1/pilot_10.npz \
  --limit-files 10 \
  --report outputs/impl1/pilot_10_report.json
```

実行済みの例では、10ファイルから以下を抽出できました。

```json
{
  "files_processed": 10,
  "rounds_processed": 102,
  "samples": 5105,
  "draw_discards": 4915,
  "call_discards": 190,
  "skipped_invalid_label": 0,
  "skipped_parse_errors": 0,
  "leakage_errors": []
}
```

### 全件HDF5 shardを作る

XMLを毎epoch再パースするとCPUボトルネックになるため、現在はこちらを推奨します。

```bash
python -u scripts/build_discard_hdf5_shards.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --output-dir outputs/impl1/hdf5_shards_250k \
  --samples-per-shard 250000 \
  --compression lzf \
  --report outputs/impl1/hdf5_shards_250k_report.json \
  --progress-every-files 1000
```

`250,000` サンプルごとにHDF5 shardを保存します。1 shardをGPUへ載せて学習できるサイズにするためです。

現在の実行例:

```json
{
  "file_index": 3000,
  "total_files": 194369,
  "samples": 1404839,
  "shards": 5,
  "buffered_samples": 154839,
  "elapsed_sec": 215.79
}
```

## 学習

### NPZで小規模学習

```bash
python scripts/train_transformer_v2.py \
  --data outputs/impl1/pilot_10.npz \
  --output outputs/impl1/pilot_10_ckpt.pt \
  --epochs 1 \
  --batch-size 256 \
  --d-model 32 \
  --n-layers 1 \
  --n-heads 4
```

実行済みの例:

```json
{
  "loss": 2.1710,
  "top1": 0.2396,
  "top3": 0.5166,
  "legal_rate": 1.0
}
```

### HDF5で本学習

HDF5 shard作成後はこちらを使います。

```bash
python -u scripts/train_transformer_v2_hdf5.py \
  --shards-dir outputs/impl1/hdf5_shards_250k \
  --output outputs/impl1/hdf5_10epoch.pt \
  --epochs 10 \
  --batch-size 8192 \
  --d-model 64 \
  --n-layers 2 \
  --n-heads 4 \
  --d-ff 256 \
  --valid-shards 8 \
  --log outputs/impl1/hdf5_10epoch.jsonl \
  --metrics-csv outputs/impl1/hdf5_10epoch_metrics.csv
```

terminalには以下のように表示されます。

```text
[epoch 1] train loss=2.2142 top1=0.1694 top3=0.3929 top5=0.5962 top10=0.9464 | val loss=2.1672 top1=0.1993 top3=0.4455 top5=0.6488 top10=0.9594
```

論文用グラフには `metrics-csv` のCSVを使います。保存される主な列は以下です。

- `epoch`
- `train_loss`
- `train_top1`
- `train_top3`
- `train_top5`
- `train_top10`
- `train_legal_rate`
- `val_loss`
- `val_top1`
- `val_top3`
- `val_top5`
- `val_top10`
- `val_legal_rate`
- `elapsed_sec`

## 評価

### NPZ評価

```bash
python scripts/evaluate_policy.py \
  --data outputs/impl1/pilot_10.npz \
  --checkpoint outputs/impl1/pilot_10_ckpt.pt \
  --top-k 1,3,5,10
```

### XMLから直接Top-k validation

checkpointをXML subsetで評価できます。

```bash
python scripts/evaluate_stream_xml_topk.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --checkpoint outputs/impl1/full_stream_100epoch.pt \
  --offset-files 180000 \
  --limit-files 100 \
  --top-k 1,3,5,10 \
  --output outputs/impl1/topk_val.json
```

実行済みの例:

```json
{
  "samples": 9340,
  "loss": 1.3758,
  "legal_rate": 1.0,
  "top1": 0.5526,
  "top3": 0.7939,
  "top5": 0.8847,
  "top10": 0.9892
}
```

## 介入実験

### Attention Top-k Mask

`experiments/interventions/attention_masks.py` の `run_attention_mask()` を使います。

できる条件:

- `topk`
- `random`
- `bottomk`
- `uniform`

これはattention logits段階に介入してからsoftmaxするため、softmax後attentionを直接書き換えるより自然な介入です。

### Head Ablation

`experiments/interventions/head_ablation.py` の `run_head_ablation()` / `head_importance_scores()` を使います。

出力:

- clean logits
- ablated logits
- KL divergence
- clean prediction
- ablated prediction

### Activation Patching

`experiments/interventions/activation_patching.py` の `activation_patch_effect()` を使います。

出力:

- clean logits
- corrupted logits
- patched logits
- indirect effect

## Mjxとの関係

[mjx-project/mjx](https://github.com/mjx-project/mjx) は、麻雀AI研究向けのTenhou互換フレームワークです。合法手や観測情報の確認には有用です。

ただし、README上でbuild/API変更に関する注意があるため、本実装では中核依存にはしていません。`scripts/mjx_crosscheck.py` は、インストールされている場合のみ照合し、使えない場合はskip理由を出します。

```bash
python scripts/mjx_crosscheck.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --limit-files 1
```

現在の環境では `mjx` は未インストールだったため、照合はskipされました。

## 進捗報告で話すポイント

来週の進捗報告では、以下の流れで説明できます。

1. 修論目的
   - Attentionや内部活性が麻雀AIの打牌判断にどの程度関係しているかを、介入実験で検証する。

2. 最初の課題
   - 天鳳XMLには全員の配牌が含まれるため、相手手牌リークが起きやすい。
   - 前回の問題を踏まえ、完全状態と観測状態を分離した。

3. 今回作ったもの
   - リーク防止付きXML抽出器。
   - Hook対応MahjongTransformerV2。
   - Top-k accuracy / legal rate / lossの評価。
   - Attention mask, head ablation, activation patching。
   - HDF5 shard化による高速学習基盤。

4. 確認できたこと
   - 10 XMLで5105サンプルを抽出。
   - 不正ラベル0、リーク検査エラー0。
   - 小規模学習で合法手率1.0。
   - Top-k validationを出せるようになった。

5. 次にやること
   - HDF5全件変換を完了させる。
   - HDF5から10 epoch学習。
   - Top-k推移をCSVからグラフ化。
   - Attention mask / Head ablationでDecision Flip RateやKLを測る。
   - Activation patchingのヒートマップを作る。

## 現在の実行状況

現在は、`/home/ubuntu/Documents/tenhou_xml_2023` 全件からHDF5 shardを作成中です。

出力先:

```text
outputs/impl1/hdf5_shards_250k/
```

進捗レポート:

```text
outputs/impl1/hdf5_shards_250k_report.json
```

HDF5化が完了したら、`scripts/train_transformer_v2_hdf5.py` で10 epoch学習します。

## 注意

`outputs/` には巨大な学習データ、checkpoint、ログが入ります。GitHubにはアップロードしません。再現用のスクリプトとREADMEだけを管理対象にします。

また、`/home/ubuntu/Documents/tenhou_xml_2023` は元データなので、このリポジトリには含めません。
