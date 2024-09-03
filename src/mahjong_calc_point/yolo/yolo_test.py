import torch
# from yolov5 import detect
from yolov5 import detect as detect_yolo
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# モデルと画像のパス
model_path = Path("./best.pt")
image_path = Path("./img/dataset.jpg")

# 推論の実行
detect_yolo.run(
    weights=str(model_path),
    source=str(image_path),
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="0" if torch.cuda.is_available() else "cpu",
    save_txt=True,
    save_conf=False,
    save_crop=False,
    project="runs/detect",
    name="exp",
    exist_ok=True,  # フォルダが既に存在する場合は上書きする
)

# 推論結果の画像パス
output_image_path = Path("runs/detect/exp/") / image_path.name

# 画像の読み込みと表示
img = cv2.imread("img/dataset.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
