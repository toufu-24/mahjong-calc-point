# mahjong-calc-point

麻雀の手牌を画像から認識し、役・翻・符・点数・支払いまで計算するブラウザアプリです。Flaskのローカルサーバーとして動作するため、PCだけでなく同じネットワーク上のスマートフォンからも利用できます。

## できること

- YOLOv5で画像内の牌を検出
- 任意の分類器で、検出した牌を34種の牌ラベルへ再分類
- 赤ドラの検出
- 画像上のガイドに合わせた手牌・副露・暗槓・ドラ表示牌の自動振り分け
- 認識結果の牌種・役割の修正、牌の追加・削除
- チー・ポン・明槓・暗槓とドラ表示牌の手入力
- 役、翻、符、点数、およびロン・ツモ時の支払いの計算

認識結果は必ず確認してください。画像の角度、照明、牌の重なり方によって誤認識や検出漏れが起こるため、必要に応じて画面上で修正してから計算します。

## 必要なもの

- Python 3.9以上
- [uv](https://docs.astral.sh/uv/)
- カメラを使う場合は、カメラにアクセスできるブラウザ

CPUでも起動できます。分類器の学習にはGPU（CUDA）またはMacのMPSを利用できます。

## ローカルで起動する

```bash
git clone https://github.com/toufu-24/mahjong-calc-point.git
cd mahjong-calc-point
uv sync
uv run python -m mahjong_calc_point.flask.app
```

ブラウザで <http://127.0.0.1:5000> を開きます。初回の画像認識ではモデルの読み込みに時間がかかる場合があります。

### スマートフォンから使う

PCのIPアドレスを使ってLAN向けに待ち受けます。

```bash
FLASK_RUN_HOST=0.0.0.0 FLASK_RUN_PORT=5000 uv run python -m mahjong_calc_point.flask.app
```

スマートフォンで `http://<PCのIPアドレス>:5000` を開きます。HTTPのLAN接続ではブラウザの制限によりページ内の「カメラ開始」を使えないことがあるため、「画像を選択」から撮影・アップロードしてください。ページ内カメラを使うにはHTTPSまたはlocalhostが必要です。

## 基本的な使い方

1. 牌を次のガイドゾーンに置いて撮影します。
   - 上段左: ドラ表示牌
   - 上段中央: 副露
   - 上段右: 暗槓
   - 下段: 手牌
2. 「認識」を押し、認識結果を確認します。
3. 「牌ごとに修正」から牌種と役割（手牌・和了・副露・暗槓・ドラ・除外）を修正します。
4. 認識できなかった牌は「牌を追加」から追加し、副露やドラ表示牌は専用の入力欄から補います。
5. 和了牌、親・リーチ・ツモなどの状況を指定し、「点数計算」を押します。

和了牌は手牌の中から1枚だけ「和了」に設定してください。副露・暗槓として認識された牌は、面子として妥当な組み合わせになっているか確認してください。

## 牌認識の仕組み

推論は次の二段構成です。

1. `src/mahjong_calc_point/yolo/best.pt` で牌のバウンディングボックスを検出する
2. `src/mahjong_calc_point/classifier/tile_classifier.pt` が存在する場合、検出領域を分類器に入力して牌種を再分類する

分類器がない場合、または分類器の確信度が低くYOLOのラベルの方が信頼できる場合は、YOLOのラベルにフォールバックします。分類器は任意ですが、牌種の認識精度を高めたい場合は学習して配置してください。分類器のチェックポイントはGit管理対象外です。

黒い牌や暗い画像に対応するため、既定では元画像・反転画像・コントラスト補正版の複数画像を推論します。不要な場合は `MAHJONG_ENABLE_BLACK_TILE_AUGMENTATION=0` で無効にできます。

## 主な環境変数

| 変数 | 既定値 | 用途 |
| --- | --- | --- |
| `FLASK_RUN_HOST` | `127.0.0.1` | Flaskの待ち受けアドレス |
| `FLASK_RUN_PORT` | `5000` | Flaskの待ち受けポート |
| `FLASK_DEBUG` | `0` | `1`でデバッグモードを有効化 |
| `MAHJONG_TILE_CLASSIFIER_PATH` | `src/mahjong_calc_point/classifier/tile_classifier.pt` | 分類器のチェックポイント |
| `MAHJONG_REQUIRE_TILE_CLASSIFIER` | 未設定 | `1`で分類器を必須にする |
| `MAHJONG_TILE_CLASSIFIER_DEVICE` | `auto` | 分類器の推論デバイス（`auto`、`cpu`、`cuda`、`mps`） |
| `MAHJONG_ENABLE_BLACK_TILE_AUGMENTATION` | `1` | 黒い牌向けの追加推論を有効化 |

認識のしきい値や重複除去の設定も環境変数で変更できます。詳しくは `src/mahjong_calc_point/detection.py` の `DEFAULT_*` 定数を参照してください。

## Renderへデプロイする

リポジトリをRenderに接続し、Blueprintとして `render.yaml` を読み込むと、`Dockerfile.render` を使ったWeb Serviceを作成できます。デプロイ後はRenderが発行するHTTPS URLを開いてください。HTTPSのため、対応ブラウザではページ内のカメラを利用できます。

`render.yaml` にはヘルスチェック（`/`）、`PORT=10000`、暗い牌向けの追加推論設定が含まれています。画像推論はCPU上で行う構成のため、画像サイズや同時利用数によって応答時間が変わります。

## 分類器を学習する

### データセット

Roboflow Universeの [mahjong Object Detection Dataset](https://universe.roboflow.com/project-xv49e/mahjong-x5dzz) をYOLOv5形式で取得します。データセットのライセンスはCC BY 4.0です。APIを使う場合はRoboflowのAPIキーを用意してください。

```bash
ROBOFLOW_API_KEY=... uv run python annotation/download_roboflow_dataset.py
uv run python annotation/build_classification_dataset.py
```

ブラウザからZIPをダウンロード済みの場合は、次のコマンドで展開できます。

```bash
uv run python annotation/download_roboflow_dataset.py \
  --zip ~/Downloads/your-roboflow-export.zip
```

生成されるデータセットは `annotation/train`、`annotation/valid`、`annotation/test`、`annotation/classification` に保存され、Git管理対象外です。

### GPU（CUDA）で学習する

CUDAとNVIDIA Container Toolkitが利用できる環境では、Dockerを使えます。

```bash
docker compose -f compose.gpu.yml build
docker compose -f compose.gpu.yml run --rm trainer \
  python -c "import torch; print(torch.cuda.is_available())"
```

`True` が表示されることを確認してから、データ取得・データ生成・学習を実行します。

```bash
ROBOFLOW_API_KEY=... docker compose -f compose.gpu.yml run --rm trainer \
  python annotation/download_roboflow_dataset.py
docker compose -f compose.gpu.yml run --rm trainer \
  python annotation/build_classification_dataset.py
docker compose -f compose.gpu.yml run --rm trainer
```

Dockerを使わず、CUDA版PyTorchを設定済みの環境で実行する場合は次のコマンドです。

```bash
uv run python annotation/train_tile_classifier.py \
  --epochs 20 --batch-size 128 --num-workers 2 \
  --device cuda --pretrained
```

### Mac（MPS）で学習する

```bash
uv run python annotation/train_tile_classifier.py \
  --epochs 20 --batch-size 128 --num-workers 0 \
  --device mps --pretrained
```

学習済み分類器は既定で `src/mahjong_calc_point/classifier/tile_classifier.pt` に保存されます。別の場所に保存・配置する場合は、起動前に次のように指定します。

```bash
MAHJONG_TILE_CLASSIFIER_PATH=/path/to/tile_classifier.pt \
  uv run python -m mahjong_calc_point.flask.app
```

学習コードだけを短時間確認する場合:

```bash
uv run python annotation/train_tile_classifier.py \
  --epochs 1 --batch-size 64 --num-workers 0 \
  --max-samples-per-class 8 --image-size 128 \
  --device cpu --progress-interval 1 --no-save
```

## 評価

Roboflowの検証データに対する検出・二段分類の比較評価を実行できます。

```bash
uv run python annotation/evaluate_two_stage.py --split valid
```

## 開発用ファイル

- `src/mahjong_calc_point/flask/app.py`: Flaskアプリと点数計算API
- `src/mahjong_calc_point/flask/templates/index.html`: ブラウザUI
- `src/mahjong_calc_point/detection.py`: 牌検出・分類・後処理
- `src/mahjong_calc_point/classification.py`: 牌分類器の読み込みと推論
- `annotation/`: データセット取得、crop生成、分類器学習、評価
- `test_images/`: 認識確認用のサンプル画像
