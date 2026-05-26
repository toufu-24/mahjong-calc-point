from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mahjong_calc_point.tiles import TILE_LABELS


DEFAULT_CLASSIFIER_PATH = (
    Path(__file__).resolve().parent / "classifier" / "tile_classifier.pt"
)
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CROP_PADDING_RATIO = 0.08
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_classifier: TileClassifier | None = None
_classifier_lock = threading.Lock()


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float


class TileClassifier:
    def __init__(
        self,
        model: Any,
        class_names: list[str],
        image_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        device: Any,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.device = device

    def classify(self, crop_bgr: np.ndarray) -> ClassificationResult:
        import torch

        input_tensor = self._preprocess(crop_bgr)
        with torch.inference_mode():
            logits = self.model(input_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_index = probabilities.max(dim=0)

        index = int(class_index.item())
        if index < 0 or index >= len(self.class_names):
            raise ValueError(f"classifier returned invalid class index: {index}")
        return ClassificationResult(
            label=self.class_names[index],
            confidence=float(confidence.item()),
        )

    def _preprocess(self, crop_bgr: np.ndarray) -> Any:
        import torch

        if crop_bgr.size == 0:
            raise ValueError("empty crop cannot be classified")

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            crop_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        array = resized.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = (array - self.mean) / self.std
        return torch.from_numpy(array).unsqueeze(0).to(self.device)


def get_classifier_path() -> Path:
    return Path(os.environ.get("MAHJONG_TILE_CLASSIFIER_PATH", DEFAULT_CLASSIFIER_PATH))


def get_crop_padding_ratio() -> float:
    value = os.environ.get(
        "MAHJONG_TILE_CROP_PADDING_RATIO", str(DEFAULT_CROP_PADDING_RATIO)
    )
    return max(0.0, float(value))


def classifier_exists() -> bool:
    return get_classifier_path().is_file()


def load_classifier() -> TileClassifier | None:
    global _classifier
    path = get_classifier_path()
    if not path.is_file():
        return None

    with _classifier_lock:
        if _classifier is None:
            _classifier = _load_classifier_from_path(path)
    return _classifier


def crop_tile(
    image: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    padding_ratio: float | None = None,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    padding_ratio = get_crop_padding_ratio() if padding_ratio is None else padding_ratio

    box_width = max(0.0, xmax - xmin)
    box_height = max(0.0, ymax - ymin)
    if box_width <= 1.0 or box_height <= 1.0:
        return None

    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    left = max(0, int(np.floor(xmin - pad_x)))
    top = max(0, int(np.floor(ymin - pad_y)))
    right = min(width, int(np.ceil(xmax + pad_x)))
    bottom = min(height, int(np.ceil(ymax + pad_y)))

    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _load_classifier_from_path(path: Path) -> TileClassifier:
    import torch

    device = _select_device()

    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        model.to(device)
        class_names = _load_sidecar_class_names(path)
        return TileClassifier(
            model=model,
            class_names=class_names,
            image_size=_load_sidecar_image_size(path),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            device=device,
        )
    except Exception:
        checkpoint = torch.load(str(path), map_location=device)

    if hasattr(checkpoint, "eval"):
        checkpoint.eval()
        checkpoint.to(device)
        return TileClassifier(
            model=checkpoint,
            class_names=_load_sidecar_class_names(path),
            image_size=_load_sidecar_image_size(path),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            device=device,
        )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"unsupported classifier checkpoint format: {path}")

    class_names = list(checkpoint.get("class_names") or checkpoint.get("labels") or [])
    if not class_names:
        class_names = TILE_LABELS.copy()
    _validate_class_names(class_names)

    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    mean = tuple(checkpoint.get("mean", IMAGENET_MEAN))
    std = tuple(checkpoint.get("std", IMAGENET_STD))

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"classifier checkpoint has no model_state_dict: {path}")

    architecture = str(checkpoint.get("architecture", "mobilenet_v3_small"))
    model = _build_model(architecture, len(class_names))
    model.load_state_dict(_normalize_state_dict_keys(state_dict))
    model.eval()
    model.to(device)

    return TileClassifier(
        model=model,
        class_names=class_names,
        image_size=image_size,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
        device=device,
    )


def _build_model(architecture: str, num_classes: int) -> Any:
    import torch.nn as nn
    from torchvision import models

    if architecture != "mobilenet_v3_small":
        raise ValueError(f"unsupported classifier architecture: {architecture}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = value
    return normalized


def _select_device() -> Any:
    import torch

    configured = os.environ.get("MAHJONG_TILE_CLASSIFIER_DEVICE", "auto").lower()
    if configured != "auto":
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_sidecar_class_names(path: Path) -> list[str]:
    metadata = _load_sidecar_metadata(path)
    class_names = list(metadata.get("class_names") or metadata.get("labels") or [])
    if not class_names:
        class_names = TILE_LABELS.copy()
    _validate_class_names(class_names)
    return class_names


def _load_sidecar_image_size(path: Path) -> int:
    metadata = _load_sidecar_metadata(path)
    return int(metadata.get("image_size", DEFAULT_IMAGE_SIZE))


def _load_sidecar_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _validate_class_names(class_names: list[str]) -> None:
    if class_names != TILE_LABELS:
        raise ValueError(
            "classifier class_names must match mahjong_calc_point.tiles.TILE_LABELS"
        )
