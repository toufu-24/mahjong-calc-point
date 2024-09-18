import torch
import cv2
import numpy as np

# モデルの読み込み
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="best.pt"
)  # 自分のモデルパスを指定
model.eval()

# カメラからフレームを取得
cap = cv2.VideoCapture(0)  # カメラデバイスIDが0の場合

if not cap.isOpened():
    print("カメラを開くことができませんでした")
    exit()

isBlack = bool(input("Is the background black? (y/n)") == "y")

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # 色反転
    if isBlack:
        frame = cv2.bitwise_not(frame)

    # YOLOv5で推論を実行
    results = model(frame)

    # 推論結果を取得して表示
    result_img = np.squeeze(results.render())  # 推論結果の画像
    cv2.imshow("YOLOv5 Detection", result_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
