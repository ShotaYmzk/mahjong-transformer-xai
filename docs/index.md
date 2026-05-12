# Docs Index

作成日: 2026/04/29

このフォルダには、READMEに置ききれない運用記録、トラブルシューティング、実験手順、モデル開発方針を置く。

READMEはプロジェクトの入口だけにして、詳しい内容はこの `docs/` 配下へ分けて管理する。

## まず読む

- [プロジェクト構成](project-structure.md): どのフォルダに何があるか。
- **[年次規模牌譜パイプライン（XML→HDF5→学習）](year-scale-training-pipeline.md): 大規模コーパス処理の実装詳細と数値の出所。**
- [強いモデルを作るための開発ガイド](model-development-guide.md): 新しいAIを作る時に使うファイル、置き場所、作業順。
- [作業記録の運用](work-log-policy.md): 日々の成果や対処法をどこへ記録するか。

## 作業・実験

- [2026/04/28 成果記録](progress-2026-04-28.md): 実装1の基盤作成、HDF5化、学習状況の記録。
- [2026/05/12 成果記録](progress-2026-05-12.md): XAI研究フェーズ、Attention Patching 実装。
- [AI判断可視化ビューア](viewer-guide.md): `web/` の使い方、JSON出力、起動方法。

## XAI・説明可能性

- [Attention Patching 実装仕様](xai-attention-patching.md): `attention_patching.py` / `visualize_attention.py` の設計、使い方、注意事項。
- [Attention グループマスク実験・k スイープ分析 (2026/05/12)](experiment-attn-group-mask-2026-05-12.md): 実際のXMLで実施した k=1〜5 スイープ結果、k=3 採用の統計的根拠、生成ファイル一覧。

## 問題対応

- [トラブルシューティング](troubleshooting.md): 発生した問題、原因、直し方、再発防止。

## 今後の追加先

新しい日次成果は `docs/progress-YYYY-MM-DD.md` へ追加する。

特定問題の対処法は `docs/troubleshooting.md` へ追記する。

モデルやデータ作成の運用変更は `docs/model-development-guide.md` または `docs/project-structure.md` へ反映する。
