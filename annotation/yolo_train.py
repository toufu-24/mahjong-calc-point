import torch
from pathlib import Path
from yolov5 import train

# データセットのパス
dataset_path = Path("./")

# YOLOv5のトレーニング関数を呼び出す
train.run(
    data=str(dataset_path / "data.yaml"),  # データセットのパス
    imgsz=640,  # 画像サイズ
    batch_size=16,  # バッチサイズ
    epochs=100,  # エポック数
    weights="yolov5s.pt",  # 初期重みファイル
    device="0" if torch.cuda.is_available() else "cpu",  # GPUが使用可能かどうか
    project="runs/train",  # プロジェクトの保存先
    name="exp"  # 実験の名前
)


