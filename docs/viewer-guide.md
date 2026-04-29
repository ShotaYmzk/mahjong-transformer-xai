# AI判断可視化ビューア

作成日: 2026/04/29

学習済み `MahjongTransformerV2` の打牌判断をWebビューアで確認する手順。

流れは、Tenhou XML牌譜とcheckpointから `web/public/reports/sample.json` を作り、ブラウザでビューアを開く。

## 依存関係

Python依存:

```bash
pip install -r requirements.txt
```

CPUで見るだけなら `--device cpu` を使う。GPU推論したい場合はCUDA対応のPyTorchが必要。

## レポートJSONを作る

単一の牌譜ファイルを解析する例:

```bash
python scripts/export_decision_report.py \
  --input /home/ubuntu/Documents/tenhou_xml_2023/2023010100gm-00a9-0000-058a3aaf.xml \
  --checkpoint outputs/impl1/hdf5_10epoch.pt \
  --output web/public/reports/sample.json \
  --limit-decisions 200 \
  --device cpu
```

ディレクトリ内の一部を解析する例:

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

主なオプション:

- `--input`: Tenhou XMLファイル、またはXMLディレクトリ。
- `--checkpoint`: 学習済みcheckpoint。
- `--output`: ビューアが読むJSON。通常は `web/public/reports/sample.json`。
- `--offset-files`: ディレクトリ指定時に先頭から何ファイル飛ばすか。
- `--limit-files`: ディレクトリ指定時に何ファイル読むか。
- `--limit-decisions`: 出力する判断局面数の上限。
- `--include-call-discards`: 鳴き直後の打牌も含める。
- `--device`: `cuda` または `cpu`。

## ビューアを起動する

Node.jsなしで静的HTMLを開く場合:

```bash
cd web/public
python -m http.server 5173
```

ブラウザで開く:

```text
http://127.0.0.1:5173/standalone.html
```

React/Vite版で開く場合:

```bash
cd web
npm install
npm run dev
```

表示されたURLをブラウザで開く。

## 見られる内容

ビューアでは以下を確認できる。

- AIが選んだ打牌。
- 実際の打牌。
- 合法打牌ごとの確率、logit、順位。
- 手牌、河、ドラ表示牌、点数。
- Attention上位イベント。
- 不一致局面、低確信局面への移動。

現在のモデルは34種の打牌分類なので、鳴き、リーチ、和了判断は候補として表示しない。

## 牌画像

`tenhou_discards_tiles_mapped/` の画像を `web/public/tiles/` にコピーして使う。

期待する配置:

```text
web/public/tiles/
├── self_bottom/
├── shimocha/
├── toimen/
└── kamicha/
```

画像が読み込めない場合でも、ビューアは文字牌表示にフォールバックする。

## よくある問題

`レポートを読み込めません`

`web/public/reports/sample.json` が存在するか確認する。`file://` で直接HTMLを開くとfetchに失敗することがあるため、HTTPサーバー経由で開く。

`No XML files selected`

`--input` のパスが間違っているか、ディレクトリ内に `*.xml` がない。

`npm: command not found`

Node.js/npmが未インストール。静的HTML版なら `cd web/public && python -m http.server 5173` で起動できる。
