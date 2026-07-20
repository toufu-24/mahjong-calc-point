from __future__ import annotations

import logging
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
DEFAULT_MIN_DETECTOR_CONFIDENCE = 0.3
DEFAULT_MIN_CLASSIFIER_CONFIDENCE = 0.35
DEFAULT_MIN_SIZE_RATIO = 0.5
DEFAULT_MAX_SIZE_RATIO = 2.85
DEFAULT_MAX_DETECTIONS = 24
DEFAULT_ENABLE_BLACK_TILE_AUGMENTATION = True
DEFAULT_RED_FIVE_MIN_RATIO = 0.018
DEFAULT_CLASSIFIER_DETECTOR_FALLBACK_CONFIDENCE = 0.6
DEFAULT_CLASSIFIER_DETECTOR_FALLBACK_MIN_DETECTOR_CONFIDENCE = 0.75
RED_FIVE_LABELS = {"0m", "0p", "0s"}
VALID_TILE_LABELS = set(TILE_LABELS) | RED_FIVE_LABELS

logger = logging.getLogger(__name__)
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

    names: dict[int, str] | list[str] | tuple[str, ...] = {}
    classifier = load_classifier()
    require_classifier = os.environ.get("MAHJONG_REQUIRE_TILE_CLASSIFIER") == "1"
    if require_classifier and classifier is None:
        raise RuntimeError(
            "tile classifier model is required but was not found. "
            "Set MAHJONG_TILE_CLASSIFIER_PATH or train "
            "src/mahjong_calc_point/classifier/tile_classifier.pt."
        )

    detections = []
    for variant_name, variant_image in _iter_detection_images(image):
        with _model_lock:
            results = model(variant_image)
        names = getattr(results, "names", names)
        detections.extend(
            _detections_from_results(
                image,
                results,
                names,
                classifier,
                variant_name,
            )
        )

    raw_detection_count = len(detections)
    detections = _filter_valid_tile_detections(detections)
    detections = _suppress_duplicate_detections(detections)
    detections = _filter_size_outliers(detections)
    detections = _limit_detection_count(detections)
    detections.sort(key=lambda item: (item["y"], item["x"]))
    _log_detector_label_fallbacks(detections)
    return {
        "image_width": width,
        "image_height": height,
        "classifier_available": classifier is not None and classifier_exists(),
        "black_tile_augmentation_enabled": _black_tile_augmentation_enabled(),
        "raw_detection_count": raw_detection_count,
        "filtered_detection_count": raw_detection_count - len(detections),
        "detections": detections,
    }


def _detections_from_results(
    classification_image: np.ndarray,
    results: Any,
    names: dict[int, str] | list[str] | tuple[str, ...],
    classifier: Any | None,
    image_variant: str,
) -> list[dict[str, Any]]:
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

        tile_crop = crop_tile(classification_image, xmin, ymin, xmax, ymax)
        if classifier is not None and tile_crop is not None:
            classification = classifier.classify(tile_crop)
            label = classification.label
            classification_confidence = classification.confidence
            classification_source = "classifier"
            if _should_fallback_to_detector_label(
                detector_label,
                classification_confidence,
                float(confidence),
            ):
                label = detector_label
                classification_source = "detector_low_classifier_confidence_fallback"

        red_dora_detected = False
        if tile_crop is not None:
            red_label = _detect_red_five_label(label, tile_crop)
            if red_label is not None:
                label = red_label
                red_dora_detected = True

        reported_confidence = (
            classification_confidence
            if classification_source == "classifier"
            and classification_confidence is not None
            else float(confidence)
        )
        detections.append(
            {
                "label": label,
                "red_dora": red_dora_detected,
                "confidence": reported_confidence,
                "detector_label": detector_label,
                "detector_confidence": float(confidence),
                "classifier_confidence": classification_confidence,
                "classification_source": classification_source,
                "image_variant": image_variant,
                "x": float(xmin),
                "y": float(ymin),
                "width": float(xmax - xmin),
                "height": float(ymax - ymin),
            }
        )
    return detections


def _log_detector_label_fallbacks(detections: list[dict[str, Any]]) -> None:
    for detection in detections:
        if (
            detection.get("classification_source")
            != "detector_low_classifier_confidence_fallback"
        ):
            continue
        logger.warning(
            "tile classifier fallback to YOLO label: "
            "variant=%s label=%s classifier_confidence=%.3f "
            "detector_confidence=%.3f bbox=(%.1f, %.1f, %.1f, %.1f)",
            detection.get("image_variant", "unknown"),
            detection.get("label", "unknown"),
            float(detection.get("classifier_confidence", 0.0)),
            float(detection.get("detector_confidence", 0.0)),
            float(detection.get("x", 0.0)),
            float(detection.get("y", 0.0)),
            float(detection.get("width", 0.0)),
            float(detection.get("height", 0.0)),
        )


def _iter_detection_images(image: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
    yield "original", image
    if not _black_tile_augmentation_enabled():
        return

    yield "inverted", cv2.bitwise_not(image)
    yield "contrast", _enhance_dark_tile_contrast(image)


def _black_tile_augmentation_enabled() -> bool:
    value = os.environ.get("MAHJONG_ENABLE_BLACK_TILE_AUGMENTATION")
    if value is None:
        return DEFAULT_ENABLE_BLACK_TILE_AUGMENTATION
    return value.lower() not in {"0", "false", "no", "off"}


def _enhance_dark_tile_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _should_fallback_to_detector_label(
    detector_label: str,
    classifier_confidence: float,
    detector_confidence: float,
) -> bool:
    if detector_label not in VALID_TILE_LABELS:
        return False
    if classifier_confidence >= _get_float_env(
        "MAHJONG_CLASSIFIER_DETECTOR_FALLBACK_CONFIDENCE",
        DEFAULT_CLASSIFIER_DETECTOR_FALLBACK_CONFIDENCE,
    ):
        return False
    return detector_confidence >= _get_float_env(
        "MAHJONG_CLASSIFIER_DETECTOR_FALLBACK_MIN_DETECTOR_CONFIDENCE",
        DEFAULT_CLASSIFIER_DETECTOR_FALLBACK_MIN_DETECTOR_CONFIDENCE,
    )


def _detect_red_five_label(label: str, crop_bgr: np.ndarray) -> str | None:
    if label not in {"5m", "5p", "5s"}:
        return None
    if _red_pixel_ratio(crop_bgr) < _get_float_env(
        "MAHJONG_RED_FIVE_MIN_RATIO", DEFAULT_RED_FIVE_MIN_RATIO
    ):
        return None
    return f"0{label[1]}"


def _red_pixel_ratio(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 65, 45), (12, 255, 255))
    red_high = cv2.inRange(hsv, (168, 65, 45), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    non_background = gray < 245
    denominator = int(np.count_nonzero(non_background))
    if denominator <= 0:
        denominator = crop_bgr.shape[0] * crop_bgr.shape[1]
    return float(np.count_nonzero(red_mask)) / float(denominator)


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
            detection.get("classification_source") == "classifier"
            and classifier_confidence is not None
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
    if (
        detection.get("classification_source") == "classifier"
        and classifier_confidence is not None
    ):
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
