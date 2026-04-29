# Troubleshooting

作成日: 2026/04/29

発生した問題、原因、修正、再発防止をここに記録する。

新しい問題を直したら、下に `## YYYY/MM/DD 問題名` の形で追記する。

## 2026/04/29 Attentionに他家の非公開DRAWが表示される

### 症状

Attentionの「注目イベント」に、判断者以外の `DRAW P1 非公開`、`DRAW P2 非公開`、`DRAW P3 非公開` のようなイベントが表示された。

麻雀では他家が何をツモったかは観測できないため、学習入力や可視化にその情報が入っていないか確認が必要だった。

### 原因

`data/observation_schema.py` の `ObservedState.sequence_features()` では、他家の非公開ツモ牌の牌ID自体は `0` にマスクしていた。

しかし、`DRAW` イベントそのものは系列入力に残っていた。そのため、モデルは「この時点で他家がツモった」という非公開イベントを見られる状態だった。

また、`scripts/export_decision_report.py` のAttention表示も同じイベント列を使っていたため、注目イベントに他家の非公開 `DRAW` が表示されていた。

### 修正

`ObservedState.observable_events()` を追加した。

判断者本人以外の `private_tile=True` なイベントを観測イベント列から除外するようにした。

`sequence_features()` は `observable_events()` を使うように変更し、他家の非公開 `DRAW` が学習用系列特徴量に入らないようにした。

`scripts/export_decision_report.py` の `sequence_event_payloads()` も `observable_events()` を使うようにし、Attention表示と学習入力のイベント列を一致させた。

`validate_no_private_leakage()` には、`sequence_features` 内に「牌IDが伏せられたDRAWイベント」が残っていないかを検査するチェックを追加した。

### 確認方法

`tests/test_observation_extraction.py` で以下を確認する。

- P1の判断サンプルに、直前のP0の非公開 `DRAW` が残らないこと。
- レポート用 `sequence_events` にも、他家の非公開 `DRAW` が残らないこと。
- `validate_no_private_leakage()` が `sequence_features contains private draw events` を検出できること。

この環境では `pytest` がoptionalで未インストールだったため、同等のPython検証スクリプトで確認した。

### 注意点

既に生成済みの `.npz` / `.h5` データセット、学習済みcheckpoint、既存のreport JSONは古いイベント列を含んでいる可能性がある。

この修正を反映するには、データセットの再生成、モデルの再学習、report JSONの再出力が必要。
