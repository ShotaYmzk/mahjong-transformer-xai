# Mahjong Transformer XAI

作成日: 2026/04/29

修士論文のための麻雀AI実験基盤です。

天鳳XML牌譜から打牌選択の教師データを作り、`MahjongTransformerV2` を学習し、Attention / Head / Activation への介入によってモデル判断の説明忠実性を調べます。

## 現在の状態

現在は「実装1: Attention数値調整によるモデル挙動変化の検証」の基盤段階です。

できていること:

- Tenhou XMLから打牌教師データを作成。
- 相手手牌や他家非公開ツモが学習入力へ混ざらないよう、完全状態と観測状態を分離。
- `MahjongTransformerV2` による34種打牌分類。
- HDF5 shardを使った大規模学習。
- Top-k accuracy / legal rate / lossの評価。
- Attention mask、Head ablation、Activation patchingの介入実験。
- 学習済みモデルの判断を確認するWebビューア。

まだできていないこと:

- 鳴き、リーチ、和了を含む統合アクションモデル。
- 実戦で打つための完全版AI。
- より強いモデル構成での本格比較。
- 説明文生成まで含む完全なXAIシステム。

## まず読むドキュメント

- [Docs Index](docs/index.md): ドキュメント全体の入口。
- [プロジェクト構成](docs/project-structure.md): どこに何のファイルがあるか。
- [強いモデルを作るための開発ガイド](docs/model-development-guide.md): 今後モデルを強くする時に使うファイル、置き場所、作業順。
- [作業記録の運用](docs/work-log-policy.md): 成果、対処法、トラブルをどこへ記録するか。
- [トラブルシューティング](docs/troubleshooting.md): 発生した問題と直し方。
- [AI判断可視化ビューア](docs/viewer-guide.md): `web/` の使い方。
- [2026/04/28 成果記録](docs/progress-2026-04-28.md): 実装1基盤を作った日の記録。

## 最重要ルール

モデル入力に、実戦中に見えない情報を入れない。

特に以下は入れない:

- 他家の配牌。
- 他家の現在手牌。
- 他家の非公開ツモ牌。
- 終局時手牌を途中局面へ逆流させた情報。
- 予測対象打牌の手出し/ツモ切りなどの未来情報。

リーク防止の中核は `data/observation_schema.py` と `tests/test_observation_extraction.py` にあります。

## よく使うコマンド

小規模データ作成:

```bash
python scripts/build_discard_dataset.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --output outputs/impl1/pilot_10.npz \
  --limit-files 10 \
  --report outputs/impl1/pilot_10_report.json
```

HDF5 shard作成:

```bash
python -u scripts/build_discard_hdf5_shards.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023 \
  --output-dir outputs/impl1/hdf5_shards_250k \
  --samples-per-shard 250000 \
  --compression lzf \
  --report outputs/impl1/hdf5_shards_250k_report.json \
  --progress-every-files 1000
```

HDF5から学習:

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

ビューア用JSON作成:

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

詳細な手順は [強いモデルを作るための開発ガイド](docs/model-development-guide.md) と [AI判断可視化ビューア](docs/viewer-guide.md) を参照してください。

## ディレクトリ概要

```text
data/          観測状態、教師データ作成、リーク検査
models/        Mahjong Transformer本体
scripts/       データ作成、学習、評価、レポート出力
experiments/   Attention/Head/Activation介入実験
utils/         XMLパース、牌変換、副露解析など
tests/         リーク防止や抽出処理のテスト
web/           判断可視化ビューア
outputs/       データセット、checkpoint、ログ、評価結果
docs/          成果記録、手順、トラブルシューティング
```

## 記録ルール

今後はREADMEを大きくしすぎず、作業内容を `docs/` に溜めます。

日ごとの成果は `docs/progress-YYYY-MM-DD.md`。

問題対応は `docs/troubleshooting.md`。

モデル開発手順の変更は `docs/model-development-guide.md`。

ファイル構成や置き場所の変更は `docs/project-structure.md`。

詳しくは [作業記録の運用](docs/work-log-policy.md) を参照してください。

## 注意

`outputs/` には巨大な学習データ、checkpoint、ログが入ります。GitHubにはアップロードしません。

`/home/ubuntu/Documents/tenhou_xml_2023` は元データなので、このリポジトリには含めません。

データ仕様を変えた場合、古いHDF5 shard、古いcheckpoint、古いreport JSONは使い回さず、再生成・再学習してください。
