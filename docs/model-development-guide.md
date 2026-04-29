# 強いモデルを作るための開発ガイド

作成日: 2026/04/29

この文書は、今後「完全版に近い、もっと強い麻雀AI」を作る時に、どのファイルを使い、どこに何を置き、どの順番で作業するかを迷わないためのガイド。

## 現在の到達点

現在のモデルは `MahjongTransformerV2` による打牌34種分類。

できること:

- 天鳳XMLから打牌判断サンプルを作る。
- 行動者本人の手牌と公開情報だけで学習する。
- 合法打牌マスクで切れない牌を除外する。
- Attentionやhidden stateを取り出す。
- Attention mask、Head ablation、Activation patchingで介入実験できる。
- Webビューアで判断、確率、Attention上位イベントを確認できる。

まだできていないこと:

- 鳴き、リーチ、和了、見逃しなどを含む統合アクション選択。
- 段位戦で実際に打つための対局エージェント。
- より大きなモデル・長い履歴・追加特徴量での本格性能比較。
- 説明文生成まで含む完全なXAIシステム。

## まず使うファイル

データ作成:

```text
data/observation_schema.py
scripts/build_discard_hdf5_shards.py
```

学習:

```text
models/mahjong_transformer_v2.py
scripts/train_transformer_v2_hdf5.py
```

評価:

```text
scripts/evaluate_policy.py
scripts/evaluate_stream_xml_topk.py
```

可視化:

```text
scripts/export_decision_report.py
web/
```

介入実験:

```text
experiments/interventions/
experiments/metrics/faithfulness.py
experiments/visualize/
```

リーク検査:

```text
data/observation_schema.py
tests/test_observation_extraction.py
```

## 標準の作業順

1. データ仕様を決める。

今のモデルで十分なら `data/observation_schema.py` は大きく変えない。新しい観測情報を入れる場合は、まず「実戦で見える情報か」を確認する。

2. 小規模データで抽出を確認する。

```bash
python scripts/build_discard_dataset.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --output outputs/impl1/pilot_10.npz \
  --limit-files 10 \
  --report outputs/impl1/pilot_10_report.json
```

3. リーク検査とテストを通す。

```bash
python -m pytest tests/test_observation_extraction.py
```

`pytest` が未インストールなら `pip install pytest` が必要。

4. 全件HDF5 shardを作る。

```bash
python -u scripts/build_discard_hdf5_shards.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --output-dir outputs/impl1/hdf5_shards_250k \
  --samples-per-shard 250000 \
  --compression lzf \
  --report outputs/impl1/hdf5_shards_250k_report.json \
  --progress-every-files 1000
```

5. HDF5から学習する。

```bash
python -u scripts/train_transformer_v2_hdf5.py \
  --shards-dir outputs/impl1/hdf5_shards_250k \
  --output outputs/impl1/hdf5_10epoch.pt \
  --epochs 10 \
  --batch-size 4096 \
  --val-batch-size 1024 \
  --d-model 64 \
  --n-layers 2 \
  --n-heads 4 \
  --d-ff 256 \
  --valid-shards 8 \
  --log outputs/impl1/hdf5_10epoch.jsonl \
  --metrics-csv outputs/impl1/hdf5_10epoch_metrics.csv
```

6. checkpointを評価する。

```bash
python scripts/evaluate_stream_xml_topk.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --checkpoint outputs/impl1/hdf5_10epoch.pt \
  --offset-files 180000 \
  --limit-files 100 \
  --top-k 1,3,5,10 \
  --output outputs/impl1/topk_val.json
```

7. ビューア用JSONを作り、人間が見て確認する。

```bash
python scripts/export_decision_report.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --checkpoint outputs/impl1/hdf5_10epoch.pt \
  --output web/public/reports/sample.json \
  --offset-files 180000 \
  --limit-files 5 \
  --limit-decisions 200 \
  --device cpu
```

## 新しい強いモデルを作る時の置き方

新しいモデル本体:

```text
models/mahjong_transformer_v3.py
```

新しい学習スクリプト:

```text
scripts/train_transformer_v3_hdf5.py
```

新しいデータ仕様:

```text
data/observation_schema_v2.py
```

新しい設定:

```text
configs/impl2_stronger_model.yaml
```

新しい出力:

```text
outputs/impl2/
```

新しいWeb表示が必要なら:

```text
web/src/
web/public/reports/
```

## 強化する時の候補

モデル側:

- `d_model`, `n_layers`, `n_heads`, `d_ff` を大きくする。
- イベント履歴長 `MAX_EVENT_HISTORY` を伸ばす。
- dropoutやweight decayを調整する。
- 局面特徴量とイベント特徴量の埋め込み方法を改善する。

データ側:

- 副露後打牌を含めるか分けて評価する。
- リーチ後ツモ切り制約をより厳密に扱う。
- 赤ドラ、ドラ、巡目、点棒状況の特徴を見直す。
- 学習・検証の牌譜分割を固定し、過学習を見やすくする。

アクション側:

- 打牌34種だけでなく、リーチ、鳴き、和了を含むアクション空間へ拡張する。
- ただし拡張時は、教師ラベルと合法手マスクの設計を先に固める。

評価側:

- Top-k accuracyだけでなく、実戦的な期待値評価を検討する。
- 局面種別ごとの精度を出す。
- Attentionが本当に因果的に効くかを介入実験で確認する。

## 絶対に守ること

他家の配牌、現在手牌、非公開ツモ牌、終局時手牌をモデル入力に入れない。

予測対象の打牌について、手出し/ツモ切り情報を入力に入れない。

データ仕様を変えたら、古いHDF5 shard、古いcheckpoint、古いreport JSONは使い回さない。

変更したら `docs/progress-YYYY-MM-DD.md` と `docs/troubleshooting.md` に必要な記録を残す。
