# 年次規模牌譜（大規模コーパス）からの学習パイプライン

作成日: 2026/05/12

この文書は、**天鳳 XML 牌譜ディレクトリ全体**（運用上は例として `/home/ubuntu/Documents/tenhou_xml_2023`）から、リークのない打牌教師を作り、**HDF5 シャード**経由で `MahjongTransformerV2` を学習するまでの実装を、コードベースに即して整理したものです。  
「1 年分の牌譜」のように時間区切りで語る場合でも、**コードが保証するのは「指定パス以下の `*.xml` を列挙して処理する」こと**であり、カレンダー上の 1 年との一致はデータ取得側の運用に依存します。

---

## 1. 全体データフロー

```text
Tenhou *.xml（巨大ディレクトリ）
        │
        ▼  build_dataset_rows_from_xml / build_discard_hdf5_shards
DatasetRow（観測＋打牌ラベル）
        │
        ▼  rows_to_npz_dict + validate_no_private_leakage
NumPy 配列（shard 単位）
        │
        ▼  h5py → discard_shard_XXXXX.h5
HDF5 シャード群
        │
        ▼  train_transformer_v2_hdf5.py
checkpoint（.pt）+ metrics CSV/JSONL
```

代替経路（コーパス全体を HDF5 に倒さない場合）:

```text
*.xml → train_transformer_v2_stream_xml.py（ファイル単位でパース→学習→解放）
```

---

## 2. 元データ（XML）

| 項目 | 実装・慣例 |
|------|------------|
| 入力パス | ディレクトリなら `Path.glob("*.xml")` で**ソート済み**一覧（`data/observation_schema.py` の `iter_xml_files`） |
| 単一ファイル | `--input` に XML 1 ファイルを渡した場合はその 1 本のみ |
| リポジトリ外 | `README.md` 通り、牌譜本体は通常 Git に含めない |
| 本環境での実例 | `build_discard_hdf5_shards.py` のレポート上、`/home/ubuntu/Documents/tenhou_xml_2023` に対し **194,369 ファイル**を処理 |

**「1 年分」について**: フォルダ名が `tenhou_xml_2023` なら運用上「2023 年分を入れたディレクトリ」という意味合いになり得るが、**学習スクリプトは年フィルタをかけない**。中身はユーザが用意した全 `*.xml` である。

---

## 3. 教師一行（サンプル）の定義

### 3.1 何を予測するか

- **タスク**: 各決裁点で、**actor が切った牌**を **34 種（種別）** の多クラス分類で予測する。
- **ラベル**: 天鳳牌 ID を `tile_id_to_kind` で種別に射影した整数（`DatasetRow.label`）。カテゴリ数は `MahjongTransformerConfig.num_actions == 34`。

### 3.2 いつサンプルを切るか

`build_dataset_rows_from_xml`（`data/observation_schema.py`）は、各局のイベント列を走査し:

1. `T/U/V/W` 系でツモを `PrivateRoundState` に反映（モデル入力には観測可能な形だけが出る）。
2. `D/E/F/G` 系の **捨て牌**で、`state.pending_player` が一致し、かつ `include_call_discards` 周りの条件を満たす場合に **1 サンプル**を追加。
3. ツモ直後の通常打牌 vs 鳴き直後の打牌の両方を（デフォルトでは）含めうる（`decision_source` が metadata に入る）。

三麻 (`meta.is_sanma`) はスキップ。

### 3.3 観測状態とリーク防止

- `ObservedState` には**その家から見える情報だけ**が入る（`observation_schema.py` 冒頭 docstring）。
- `PrivateRoundState` は検証・遷移用に手全体を持ちうるが、**シリアライズされるのは `DatasetRow` 側**。
- バッチ書き出し後に `validate_no_private_leakage` が以下を検査する:
  - 禁止キーワードを含む配列キーが無いか
  - `labels` が各行の `valid_masks` で合法か
  - `static_features` 内の手牌ブロックと `hand_counts` の一致
  - シーケンスに「他家の private ツモ」相当のイベントが無いか

詳細テストは `tests/test_observation_extraction.py`。

---

## 4. 特徴量テンソル（モデル入力）

`rows_to_npz_dict` が返すキーと形状（バッチ長 `N`）は以下。

| キー | dtype | 形状 | 内容（要約） |
|------|-------|------|----------------|
| `static_features` | float32 | `(N, 157)` | 場況・スコア・自分の河・ドラ・副露公開牌などから組み立てたベクトル（`STATIC_FEATURE_DIM`） |
| `sequence_features` | float32 | `(N, 60, 6)` | 直近イベントを最大 **60** 件、各 **6** 次元（`MAX_EVENT_HISTORY`, `EVENT_FEATURE_DIM`） |
| `hand_counts` | float32 | `(N, 34)` | 自摸手の種別枚数 |
| `aka_flags` | float32 | `(N, 3)` | 赤 5m/5p/5s 保有フラグ |
| `valid_masks` | float32 | `(N, 34)` | その時点で切りうる種別（リーチ中はツモ切りのみ 1） |
| `labels` | int64 | `(N,)`` | 正解種別 |
| `metadata` | object | `(N,)` | ファイル名・局・イベント index など（学習テンソルには使わない） |

HDF5 では `metadata` を JSON 文字列の可変長配列として保存（`build_discard_hdf5_shards.py`）。

---

## 5. HDF5 シャード生成

**スクリプト**: `scripts/build_discard_hdf5_shards.py`

| CLI | 既定 | 役割 |
|-----|------|------|
| `--input` | 必須 | XML ファイルかディレクトリ |
| `--output-dir` | 必須 | シャード出力先 |
| `--samples-per-shard` | `250000` | 1 ファイルあたり最大サンプル数（GPU メモリと相談） |
| `--limit-files` | なし | デバッグ用に先頭 N ファイルのみ |
| `--compression` | `lzf` | `none` / `lzf` / `gzip` |
| `--report` | なし | 集計 JSON パス |
| `--progress-every-files` | `1000` | 標準出力に進捗 JSON を出す間隔 |

各 shard 書き込み前に **当該 shard 内**で `validate_no_private_leakage`。  
HDF5 属性: `num_samples`, `schema = "leak_safe_discard_v1"`。

### 5.1 実測レポート（`outputs/impl1/hdf5_shards_250k_report.json`）

※ 生成済みのレポートに基づく数値（再現には同コーパスが必要）。

| 指標 | 値 |
|------|-----|
| 処理ファイル数 | 194,369 |
| 総サンプル数 | 92,954,245 |
| シャード数 | 372 |
| 先頭シャード〜大半 | 各 250,000 サンプル |
| 最終シャード | 204,245 サンプル |
| パーススキップ | 0（本レポート時点） |
| 処理時間（elapsed_sec） | 約 14,860 秒（約 4.1 時間） |

---

## 6. HDF5 からの学習

**スクリプト**: `scripts/train_transformer_v2_hdf5.py`

### 6.1 シャードの train / valid 分割

- `Path.glob("*.h5")` を **ソート**。
- `--valid-shards` が `K` なら、**末尾 K 個を検証**、残りを学習に使う（コード上は `shards[:-K]` / `shards[-K:]`）。

実レポート構成では **372 シャード・valid-shards=8** のとき:

- 検証サンプル数 = 7×250,000 + 204,245 = **1,954,245**
- 学習サンプル数 = 92,954,245 − 1,954,245 = **91,000,000**（`hdf5_10epoch_metrics.csv` の `train_samples` / `val_samples` と一致）

※ **時系列ホールドアウトではなく「ファイル名ソート順の末尾シャード」**である点に注意（天鳳 XML のファイル名順が日付順とは限らない）。

### 6.2 1 epoch の内部

各 epoch:

1. 学習シャードの順番を `random.shuffle`。
2. **シャードごとに**全サンプルを GPU にロード（`load_shard` でテンソル化）。
3. シャード内を `batch_size` でミニバッチ。**学習時はシャード内でランダム順**（`torch.randperm`）。
4. 検証は valid シャードを順に、`train=False` で同様に集計。
5. 損失は全サンプル平均相当の加重平均、`top1`〜`top10` と **legal_rate**（argmax 予測が `valid_mask` 上で合法か）を記録。
6. チェックポイント: `--output` に `epoch`・`history` 等を保存。`.latest.pt` に中間保存あり。

### 6.3 主要ハイパーパラメータ（README / watchdog と整合）

| 引数 | README 例 | `watch_hdf5_training.sh` |
|------|-----------|---------------------------|
| `--epochs` | 10 | 10 |
| `--batch-size` | 4096 | 4096 |
| `--valid-batch-size` | 1024 | 1024 |
| `--lr` | 既定 3e-4 | 既定のまま |
| `--d-model` | 64 | 64 |
| `--n-layers` | 2 | 2 |
| `--n-heads` | 4 | 4 |
| `--d-ff` | 256 | 256 |
| `--valid-shards` | 8 | 8 |
| `--resume` | 任意 | **あり**（watchdog が落ちても続きから） |

Optimizer: **AdamW**（weight_decay=1e-4）、勾配クリップ 1.0。

### 6.4 学習結果の例（`outputs/impl1/hdf5_10epoch_metrics.csv`）

10 epoch 終了時点（同一 run）:

- **val_top1** ≈ **0.656**
- **val_top3** ≈ **0.898**
- **val_legal_rate** = **1.0**

（epoch 10 行: train_top1 ≈ 0.629, val_top1 ≈ 0.656）

---

## 7. ストリーミング XML 学習（代替実装）

**スクリプト**: `scripts/train_transformer_v2_stream_xml.py`

- **目的**: 年スケールで **巨大 NPZ/HDF5 を作らず**、XML を一定バッファ貯めてからバッチ学習。
- **検証ファイル**: デフォルトで「ソート済み XML 一覧の **180000 ファイル目以降**から最大 100 本」（`--validation-offset-files` / `--validation-limit-files`）。
- **モデル既定**: `d_model=32`, `n_layers=1` など **HDF5 本番用より小さい**（引数で揃えること）。

本番の 64-dim / 2-layer checkpoint は **HDF5 パイプライン**で学習したものが `outputs/impl1/hdf5_10epoch.pt`。

---

## 8. 評価・エクスポート

| スクリプト | 用途 |
|------------|------|
| `scripts/evaluate_stream_xml_topk.py` | 任意の checkpoint を XML サブセットに対して Top-k・loss 集計 |
| `scripts/export_decision_report.py` | ビューア用に局面単位の判断 JSON を出力（`--offset-files` でコーパス後半の「見ていない」局を取りやすい） |

---

## 9. 付帯スクリプト・成果物パス（参照用）

| パス | 内容 |
|------|------|
| `scripts/watch_hdf5_training.sh` | `train_transformer_v2_hdf5.py` の **resume 付き再起動ループ**（10 epoch 完了まで） |
| `outputs/impl1/hdf5_shards_250k/` | シャード `.h5`（Git 非推奨・巨大） |
| `outputs/impl1/hdf5_shards_250k_report.json` | シャード生成サマリー |
| `outputs/impl1/hdf5_10epoch.pt` | 本パイプライン由来 checkpoint（実験で多用） |
| `outputs/impl1/hdf5_10epoch_metrics.csv` | epoch ごとの train/val 指標 |
| `outputs/impl1/hdf5_10epoch.jsonl` | 同上の JSONL ログ（あれば） |

---

## 10. 変更時の注意

- `STATIC_FEATURE_DIM`・シーケンス仕様・ラベル定義を変えたら、**既存 HDF5 / checkpoint は互換ではない**。シャード再生成と再学習が必要。
- valid を「末尾 K シャード」にしているため、シャード生成時の **ファイル順・シャード境界**が変わると検証分布も変わる。

---

## 11. 関連ドキュメント

- [model-development-guide.md](model-development-guide.md) — 作業手順の再掲
- [progress-2026-04-28.md](progress-2026-04-28.md) — 基盤実装当時の経緯
- [project-structure.md](project-structure.md) — ディレクトリ全体像
