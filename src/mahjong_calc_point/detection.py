from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from mahjong_calc_point.classification import classifier_exists, crop_tile, load_classifier


os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

MODEL_PATH = Path(__file__).resolve().parent / "yolo" / "best.pt"

_model: Any | None = None
_model_lock = threading.Lock()


def _load_model() -> Any:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import yolov5

                with _torch_load_trusted_yolov5_checkpoint():
                    _model = yolov5.load(str(MODEL_PATH))
                _model.conf = 0.25
    return _model


@contextmanager
def _torch_load_trusted_yolov5_checkpoint() -> Iterator[None]:
    import torch

    original_load = torch.load
    supports_weights_only = "weights_only" in signature(original_load).parameters

    def load_with_yolov5_pickle(*args: Any, **kwargs: Any) -> Any:
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_yolov5_pickle
    try:
        yield
    finally:
        torch.load = original_load


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
    classifier = load_classifier()
    require_classifier = os.environ.get("MAHJONG_REQUIRE_TILE_CLASSIFIER") == "1"
    if require_classifier and classifier is None:
        raise RuntimeError(
            "tile classifier model is required but was not found. "
            "Set MAHJONG_TILE_CLASSIFIER_PATH or train "
            "src/mahjong_calc_point/classifier/tile_classifier.pt."
        )

    detections = []
    for row in results.xyxy[0].tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = row
        class_id = int(class_id)
        detector_label = (
            names.get(class_id, str(class_id))
            if isinstance(names, dict)
            else str(names[class_id])
        )
        label = detector_label
        classification_confidence: float | None = None
        classification_source = "detector_fallback"

        tile_crop = crop_tile(image, xmin, ymin, xmax, ymax)
        if classifier is not None and tile_crop is not None:
            classification = classifier.classify(tile_crop)
            label = classification.label
            classification_confidence = classification.confidence
            classification_source = "classifier"

        detections.append(
            {
                "label": label,
                "confidence": classification_confidence
                if classification_confidence is not None
                else float(confidence),
                "detector_label": detector_label,
                "detector_confidence": float(confidence),
                "classifier_confidence": classification_confidence,
                "classification_source": classification_source,
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
        "classifier_available": classifier is not None and classifier_exists(),
        "detections": detections,
    }
