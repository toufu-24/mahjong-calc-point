from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

MODEL_PATH = Path(__file__).resolve().parent / "yolo" / "best.pt"

_model: Any | None = None
_model_lock = threading.Lock()


def _load_model() -> Any:
    global _model
    if _model is None:
        import yolov5

        _model = yolov5.load(str(MODEL_PATH))
        _model.conf = 0.25
    return _model


def detect_tiles(image_bytes: bytes) -> dict[str, Any]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("画像を読み込めませんでした")

    height, width = image.shape[:2]
    model = _load_model()

    with _model_lock:
        results = model(image)

    names = getattr(results, "names", {})
    detections = []
    for row in results.xyxy[0].tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = row
        class_id = int(class_id)
        detections.append(
            {
                "label": names.get(class_id, str(class_id))
                if isinstance(names, dict)
                else str(names[class_id]),
                "confidence": float(confidence),
                "x": float(xmin),
                "y": float(ymin),
                "width": float(xmax - xmin),
                "height": float(ymax - ymin),
            }
        )

    detections.sort(key=lambda item: (item["y"], item["x"]))
    return {
        "image_width": width,
        "image_height": height,
        "detections": detections,
    }
