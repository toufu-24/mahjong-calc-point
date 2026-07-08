# mahjong-calc-point

麻雀の手牌画像をYOLOで検出し、点数計算フォームへ反映するローカルアプリです。

## 牌認識

推論は二段構成です。

1. YOLOv5モデル `src/mahjong_calc_point/yolo/best.pt` で牌のbboxを検出する
2. 分類器 `src/mahjong_calc_point/classifier/tile_classifier.pt` がある場合、bbox cropを34種の牌ラベルへ分類する

分類器が未配置の場合は、従来通りYOLOのクラスラベルへフォールバックします。分類器を必須にしたい場合は `MAHJONG_REQUIRE_TILE_CLASSIFIER=1` を指定してください。

## ブラウザで起動する

PC上で起動して同じWi-Fiのスマホから使う場合は、LAN向けに待ち受けます。

```bash
FLASK_RUN_HOST=0.0.0.0 FLASK_RUN_PORT=5000 uv run python -m src.mahjong_calc_point.flask.app
```

PCのIPアドレスを確認し、スマホのブラウザで `http://<PCのIPアドレス>:5000` を開きます。

スマホのカメラをブラウザから直接使うには、ブラウザの制限によりHTTPSまたはlocalhostが必要です。HTTPのLAN接続では「画像を選択」から撮影・アップロードする使い方が安定です。

## Roboflowデータから分類器を作る

Roboflow Universeの [mahjong Object Detection Dataset](https://universe.roboflow.com/project-xv49e/mahjong-x5dzz) をYOLOv5形式でダウンロードし、`annotation/train` と `annotation/valid` に展開します。

GPU環境では、まずリポジトリをcloneして依存を同期します。

```bash
git clone https://github.com/toufu-24/mahjong-calc-point.git
cd mahjong-calc-point
uv sync
```

CUDA版PyTorchを使う環境では、必要に応じてその環境の手順でPyTorchを入れ直してください。例:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

サーバーでDockerを使う場合は、ホスト側に `uv` は不要です。DockerとNVIDIA Container Toolkitが使える状態で実行します。

```bash
git clone https://github.com/toufu-24/mahjong-calc-point.git
cd mahjong-calc-point
docker compose -f compose.gpu.yml build
docker compose -f compose.gpu.yml run --rm trainer python -c "import torch; print(torch.cuda.is_available())"
```

`True` が出ればCUDAを使えます。以降のコマンドは `docker compose -f compose.gpu.yml run --rm trainer ...` の形で実行できます。

データセットを取得してcrop分類データを生成します。

```bash
ROBOFLOW_API_KEY=... uv run python annotation/download_roboflow_dataset.py
uv run python annotation/build_classification_dataset.py
```

Dockerの場合:

```bash
ROBOFLOW_API_KEY=... docker compose -f compose.gpu.yml run --rm trainer python annotation/download_roboflow_dataset.py
docker compose -f compose.gpu.yml run --rm trainer python annotation/build_classification_dataset.py
```

ブラウザでZIPをダウンロードした場合は、代わりに次のように展開できます。

```bash
uv run python annotation/download_roboflow_dataset.py --zip ~/Downloads/your-roboflow-export.zip
```

本番分類器をGPUで学習します。

```bash
uv run python annotation/train_tile_classifier.py --epochs 20 --batch-size 128 --num-workers 2 --device cuda --pretrained
```

Dockerの場合:

```bash
docker compose -f compose.gpu.yml run --rm trainer
```

VS Codeでコンテナ内に入って作業する場合は、Dev Containers拡張を入れてから `Dev Containers: Reopen in Container` を実行します。中では `/workspace/mahjong-calc-point` が作業ディレクトリです。

ターミナルで入るだけなら:

```bash
docker compose -f compose.gpu.yml up -d dev
docker compose -f compose.gpu.yml exec dev bash
```

MacでMPSを使う場合:

```bash
uv run python annotation/train_tile_classifier.py --epochs 20 --batch-size 128 --num-workers 0 --device mps --pretrained
```

まず学習コードだけ短く確認する場合:

```bash
uv run python annotation/train_tile_classifier.py --epochs 1 --batch-size 64 --num-workers 0 --max-samples-per-class 8 --image-size 128 --device cpu --progress-interval 1 --no-save
```

学習済み分類器は既定で `src/mahjong_calc_point/classifier/tile_classifier.pt` に保存されます。このファイルはGit管理対象外です。別の場所に置く場合は `MAHJONG_TILE_CLASSIFIER_PATH=/path/to/tile_classifier.pt` を指定してください。

比較評価は次のコマンドで実行できます。

```bash
uv run python annotation/evaluate_two_stage.py --split valid
```
