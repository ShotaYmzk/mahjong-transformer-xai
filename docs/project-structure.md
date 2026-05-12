# Project Structure

作成日: 2026/04/29

このプロジェクトは、天鳳XML牌譜から打牌判断データを作り、麻雀AIを学習し、Attentionや内部表現への介入で判断根拠を調べるための実験基盤。

## 主要フォルダ

`data/`

学習サンプルを作る中核。現在の最重要ファイルは `data/observation_schema.py`。

`models/`

学習するモデル本体。現在の主モデルは `models/mahjong_transformer_v2.py`。

`scripts/`

データセット作成、学習、評価、レポート出力などの実行スクリプトを置く。

`experiments/`

Attention mask、Head ablation、Activation patchingなど、説明忠実性を調べる介入実験を置く。

`utils/`

牌ID変換、天鳳XMLパース、副露解析、シャンテン計算などの共通処理を置く。

`tests/`

観測情報抽出やリーク防止のテストを置く。現在は `tests/test_observation_extraction.py` が重要。

`web/`

学習済みモデルの判断を確認するビューア。使い方は [AI判断可視化ビューア](viewer-guide.md) を参照。

`outputs/`

生成したデータセット、checkpoint、学習ログ、評価結果を置く。巨大ファイルを含むためGit管理対象にしない。

`docs/`

READMEから切り出した詳細ドキュメント、成果記録、トラブルシューティング、運用メモを置く。大規模XML→HDF5→学習の**実装と実測規模の整理**は [year-scale-training-pipeline.md](year-scale-training-pipeline.md)。

## 主要ファイル

`data/observation_schema.py`

Tenhou XMLから打牌教師データを作る。`PrivateRoundState` はXML再生用の完全情報状態、`ObservedState` はモデル入力用の観測状態、`DatasetRow` は保存される1サンプル。

`models/mahjong_transformer_v2.py`

34種打牌分類モデル。`static`, `sequence`, `hand_counts`, `aka_flags`, `valid_mask` を受け取り、合法打牌のlogitsを返す。

`scripts/build_discard_hdf5_shards.py`

大量XMLをHDF5 shardに変換する。全件学習ではこの出力を使う。

`scripts/train_transformer_v2_hdf5.py`

HDF5 shardから本学習するためのスクリプト。強いモデルを作る時の主な学習入口。

`scripts/export_decision_report.py`

学習済みcheckpointとTenhou XMLから、ビューア用の判断レポートJSONを作る。

`tests/test_observation_extraction.py`

相手手牌や他家非公開ツモが学習入力へ混ざらないことを確認するテスト。

## XAI モジュール（2026/05/12 追加）

`attention_patching.py`（プロジェクトルート）

説明忠実性を検証する Attention Patching 評価クラス。
特徴グループ単位で Attention スコアをマスクし、LLM 説明文の変化を BERTScore と
キーワードシフトで定量計算する。t検定・Spearman相関も含む。
詳細は [xai-attention-patching.md](xai-attention-patching.md) を参照。

`visualize_attention.py`（プロジェクトルート）

マスク前後の特徴グループスコアを棒グラフで比較し、複数局面のスコアをヒートマップで
表示する可視化スクリプト。`attention_patching.py` の処理を import して再利用する。

---

## 置き場所の原則

新しいモデル本体は `models/` に置く。

モデルを学習するコマンドライン入口は `scripts/train_*.py` に置く。

データセット作成スクリプトは `scripts/build_*.py` に置く。

評価スクリプトは `scripts/evaluate_*.py` に置く。

説明忠実性や介入実験は `experiments/` に置く。
ただし、LLM 説明文との比較が必要な高レベルな評価モジュールはプロジェクトルートに置く
（`attention_patching.py` がその例）。

論文用の一時出力、checkpoint、ログ、HDF5 shardは `outputs/` に置く。

作業記録、手順、トラブル対応は `docs/` に置く。
