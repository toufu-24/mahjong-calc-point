from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mahjong_calc_point.detection import detect_tiles
from mahjong_calc_point.tiles import TILE_LABELS


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate current detector labels against two-stage labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root containing Roboflow YOLOv5 train/valid/test folders.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path(__file__).resolve().parent / "data.yaml",
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-images",
        type=int,
        help="Evaluate at most this many images for quick smoke tests.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N evaluated images. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.data_yaml.read_text())
    class_names = list(config["names"])
    if class_names != TILE_LABELS:
        raise ValueError("data.yaml names do not match TILE_LABELS")

    split_dir = _resolve_split_dir(args.dataset_root, config, args.split)
    image_dir, label_dir = _resolve_image_and_label_dirs(split_dir)

    totals = {
        "images": 0,
        "gt_boxes": 0,
        "matched_boxes": 0,
        "two_stage_correct": 0,
        "detector_correct": 0,
        "exact_images": 0,
    }

    print(f"split_dir: {split_dir}")
    print(f"image_dir: {image_dir}")
    print(f"label_dir: {label_dir}")

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt_boxes = _load_gt_boxes(label_path, width, height, class_names)
        if not gt_boxes:
            continue

        result = detect_tiles(image_path.read_bytes())
        detections = result["detections"]
        matches = _match_detections(gt_boxes, detections, args.iou_threshold)

        totals["images"] += 1
        totals["gt_boxes"] += len(gt_boxes)
        totals["matched_boxes"] += len(matches)

        image_exact = len(matches) == len(gt_boxes) == len(detections)
        for gt, detection in matches:
            if detection["label"] == gt["label"]:
                totals["two_stage_correct"] += 1
            else:
                image_exact = False
            if detection.get("detector_label") == gt["label"]:
                totals["detector_correct"] += 1

        if image_exact:
            totals["exact_images"] += 1

        if args.progress_interval and totals["images"] % args.progress_interval == 0:
            print(
                f"evaluated {totals['images']} images, "
                f"matched {totals['matched_boxes']}/{totals['gt_boxes']} boxes",
                flush=True,
            )
        if args.max_images and totals["images"] >= args.max_images:
            break

    _print_metrics(totals)


def _resolve_split_dir(dataset_root: Path, config: dict, split: str) -> Path:
    config_keys = [split]
    if split == "valid":
        config_keys = ["val", "valid"]

    candidates = []
    for root in _candidate_dataset_roots(dataset_root):
        for key in config_keys:
            configured = config.get(key)
            if configured:
                path = Path(configured)
                candidates.append(path)
                if not path.is_absolute():
                    candidates.append(root / path)
                candidates.append(root / path.name)
        candidates.append(root / split)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not find {split} split under {dataset_root}")


def _resolve_image_and_label_dirs(split_dir: Path) -> tuple[Path, Path]:
    if split_dir.name == "images":
        return split_dir, split_dir.parent / "labels"

    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if image_dir.is_dir() and label_dir.is_dir():
        return image_dir, label_dir

    if not image_dir.is_dir():
        image_dir = split_dir
    if not label_dir.is_dir():
        label_dir = split_dir
    return image_dir, label_dir


def _candidate_dataset_roots(dataset_root: Path) -> list[Path]:
    roots = [dataset_root]
    if dataset_root.is_dir():
        roots.extend(
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and _looks_like_detection_dataset_root(path)
        )
    return roots


def _looks_like_detection_dataset_root(path: Path) -> bool:
    return any(
        (path / split / "images").is_dir() and (path / split / "labels").is_dir()
        for split in ("train", "valid", "val", "test")
    )


def _load_gt_boxes(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_names: list[str],
) -> list[dict]:
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        box_width = float(parts[3]) * image_width
        box_height = float(parts[4]) * image_height
        boxes.append(
            {
                "label": class_names[class_id],
                "x1": x_center - box_width / 2,
                "y1": y_center - box_height / 2,
                "x2": x_center + box_width / 2,
                "y2": y_center + box_height / 2,
            }
        )
    return boxes


def _match_detections(
    gt_boxes: list[dict],
    detections: list[dict],
    iou_threshold: float,
) -> list[tuple[dict, dict]]:
    candidates = []
    for gt_index, gt in enumerate(gt_boxes):
        for detection_index, detection in enumerate(detections):
            iou = _iou(
                gt,
                {
                    "x1": detection["x"],
                    "y1": detection["y"],
                    "x2": detection["x"] + detection["width"],
                    "y2": detection["y"] + detection["height"],
                },
            )
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, detection_index))

    matches = []
    used_gt = set()
    used_detection = set()
    for _, gt_index, detection_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or detection_index in used_detection:
            continue
        used_gt.add(gt_index)
        used_detection.add(detection_index)
        matches.append((gt_boxes[gt_index], detections[detection_index]))
    return matches


def _iou(first: dict, second: dict) -> float:
    x1 = max(first["x1"], second["x1"])
    y1 = max(first["y1"], second["y1"])
    x2 = min(first["x2"], second["x2"])
    y2 = min(first["y2"], second["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0

    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    return intersection / (first_area + second_area - intersection)


def _print_metrics(totals: dict[str, int]) -> None:
    matched = max(1, totals["matched_boxes"])
    gt_boxes = max(1, totals["gt_boxes"])
    images = max(1, totals["images"])
    print(f"images: {totals['images']}")
    print(f"gt boxes: {totals['gt_boxes']}")
    print(f"matched boxes: {totals['matched_boxes']} ({totals['matched_boxes'] / gt_boxes:.4f})")
    print(
        "two-stage label accuracy on matched boxes: "
        f"{totals['two_stage_correct'] / matched:.4f}"
    )
    print(
        "detector label accuracy on matched boxes: "
        f"{totals['detector_correct'] / matched:.4f}"
    )
    print(f"exact image match: {totals['exact_images'] / images:.4f}")


if __name__ == "__main__":
    main()
