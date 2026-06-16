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
from mahjong_calc_point.tiles import TILE_LABELS


os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

MODEL_PATH = Path(__file__).resolve().parent / "yolo" / "best.pt"
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.75
DEFAULT_DUPLICATE_CONTAINMENT_THRESHOLD = 0.9
DEFAULT_MIN_DETECTOR_CONFIDENCE = 0.25
DEFAULT_MIN_CLASSIFIER_CONFIDENCE = 0.35
DEFAULT_MIN_SIZE_RATIO = 0.35
DEFAULT_MAX_SIZE_RATIO = 2.85
DEFAULT_MAX_DETECTIONS = 24
VALID_TILE_LABELS = set(TILE_LABELS)

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

    raw_detection_count = len(detections)
    detections = _filter_valid_tile_detections(detections)
    detections = _suppress_duplicate_detections(detections)
    detections = _filter_size_outliers(detections)
    detections = _limit_detection_count(detections)
    detections.sort(key=lambda item: (item["y"], item["x"]))
    return {
        "image_width": width,
        "image_height": height,
        "classifier_available": classifier is not None and classifier_exists(),
        "raw_detection_count": raw_detection_count,
        "filtered_detection_count": raw_detection_count - len(detections),
        "detections": detections,
    }


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_duplicate_iou_threshold() -> float:
    value = _get_float_env(
        "MAHJONG_DUPLICATE_IOU_THRESHOLD", DEFAULT_DUPLICATE_IOU_THRESHOLD
    )
    return max(0.0, min(1.0, float(value)))


def _get_duplicate_containment_threshold() -> float:
    value = _get_float_env(
        "MAHJONG_DUPLICATE_CONTAINMENT_THRESHOLD",
        DEFAULT_DUPLICATE_CONTAINMENT_THRESHOLD,
    )
    return max(0.0, min(1.0, float(value)))


def _filter_valid_tile_detections(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    min_detector_confidence = _get_float_env(
        "MAHJONG_MIN_DETECTOR_CONFIDENCE", DEFAULT_MIN_DETECTOR_CONFIDENCE
    )
    min_classifier_confidence = _get_float_env(
        "MAHJONG_MIN_CLASSIFIER_CONFIDENCE", DEFAULT_MIN_CLASSIFIER_CONFIDENCE
    )

    filtered = []
    for detection in detections:
        label = str(detection.get("label", ""))
        if label not in VALID_TILE_LABELS:
            continue
        if float(detection.get("detector_confidence", 0.0)) < min_detector_confidence:
            continue
        classifier_confidence = detection.get("classifier_confidence")
        if (
            classifier_confidence is not None
            and float(classifier_confidence) < min_classifier_confidence
        ):
            continue
        filtered.append(detection)
    return filtered


def _suppress_duplicate_detections(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threshold = _get_duplicate_iou_threshold()
    containment_threshold = _get_duplicate_containment_threshold()
    if (threshold <= 0.0 and containment_threshold <= 0.0) or len(detections) < 2:
        return detections

    kept: list[dict[str, Any]] = []
    for detection in sorted(
        detections, key=_detection_score, reverse=True
    ):
        overlaps_kept_detection = any(
            _detection_iou(detection, kept_detection) >= threshold
            or _detection_containment(detection, kept_detection)
            >= containment_threshold
            for kept_detection in kept
        )
        if not overlaps_kept_detection:
            kept.append(detection)
    return kept


def _filter_size_outliers(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(detections) < 5:
        return detections

    areas = np.asarray(
        [
            max(0.0, float(detection["width"]))
            * max(0.0, float(detection["height"]))
            for detection in detections
        ],
        dtype=np.float32,
    )
    median_area = float(np.median(areas))
    if median_area <= 0.0:
        return detections

    min_ratio = max(
        0.0, _get_float_env("MAHJONG_MIN_TILE_SIZE_RATIO", DEFAULT_MIN_SIZE_RATIO)
    )
    max_ratio = max(
        min_ratio,
        _get_float_env("MAHJONG_MAX_TILE_SIZE_RATIO", DEFAULT_MAX_SIZE_RATIO),
    )
    filtered = []
    for detection, area in zip(detections, areas):
        ratio = float(area) / median_area
        if min_ratio <= ratio <= max_ratio:
            filtered.append(detection)
    return filtered


def _limit_detection_count(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_detections = max(
        0, _get_int_env("MAHJONG_MAX_TILE_DETECTIONS", DEFAULT_MAX_DETECTIONS)
    )
    if max_detections <= 0 or len(detections) <= max_detections:
        return detections
    return sorted(detections, key=_detection_score, reverse=True)[:max_detections]


def _detection_score(detection: dict[str, Any]) -> float:
    classifier_confidence = detection.get("classifier_confidence")
    if classifier_confidence is not None:
        return float(classifier_confidence) * float(
            detection.get("detector_confidence", 0.0)
        )
    return float(detection.get("detector_confidence", 0.0))


def _detection_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    intersection, first_area, second_area = _detection_intersection_and_areas(
        first, second
    )
    if intersection <= 0.0:
        return 0.0

    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _detection_containment(first: dict[str, Any], second: dict[str, Any]) -> float:
    intersection, first_area, second_area = _detection_intersection_and_areas(
        first, second
    )
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0.0:
        return 0.0
    return intersection / smaller_area


def _detection_intersection_and_areas(
    first: dict[str, Any], second: dict[str, Any]
) -> tuple[float, float, float]:
    first_box = _detection_box(first)
    second_box = _detection_box(second)

    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    first_area = max(0.0, first_box[2] - first_box[0]) * max(
        0.0, first_box[3] - first_box[1]
    )
    second_area = max(0.0, second_box[2] - second_box[0]) * max(
        0.0, second_box[3] - second_box[1]
    )
    return intersection, first_area, second_area


def _detection_box(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    x1 = float(detection["x"])
    y1 = float(detection["y"])
    return (
        x1,
        y1,
        x1 + float(detection["width"]),
        y1 + float(detection["height"]),
    )
