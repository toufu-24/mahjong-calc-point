from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tile classification dataset from a YOLOv5 dataset."
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
        help="YOLOv5 data.yaml with the 34 mahjong class names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "classification",
        help="Output root for ImageFolder-style classification data.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.08,
        help="Padding ratio added around each YOLO bbox before cropping.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=4,
        help="Skip boxes smaller than this many pixels in width or height.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.data_yaml)
    class_names = list(config["names"])

    total_crops = 0
    for split, split_dir in _find_split_dirs(args.dataset_root, config).items():
        count = build_split(
            split=split,
            split_dir=split_dir,
            output_root=args.output,
            class_names=class_names,
            padding=args.padding,
            min_size=args.min_size,
        )
        total_crops += count
        print(f"{split}: wrote {count} crops")

    print(f"done: wrote {total_crops} crops to {args.output}")


def build_split(
    split: str,
    split_dir: Path,
    output_root: Path,
    class_names: list[str],
    padding: float,
    min_size: int,
) -> int:
    image_dir, label_dir = _resolve_image_and_label_dirs(split_dir)

    crop_count = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip unreadable image: {image_path}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        height, width = image.shape[:2]
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            parsed = _parse_yolo_label(line)
            if parsed is None:
                continue

            class_id, x_center, y_center, box_width, box_height = parsed
            if class_id < 0 or class_id >= len(class_names):
                print(f"skip invalid class id {class_id}: {label_path}:{line_number}")
                continue

            xmin, ymin, xmax, ymax = _to_pixel_box(
                width=width,
                height=height,
                x_center=x_center,
                y_center=y_center,
                box_width=box_width,
                box_height=box_height,
                padding=padding,
            )
            if xmax - xmin < min_size or ymax - ymin < min_size:
                continue

            crop = image[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            class_name = class_names[class_id]
            output_dir = output_root / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_{line_number:03d}.jpg"
            cv2.imwrite(str(output_path), crop)
            crop_count += 1

    return crop_count


def _resolve_image_and_label_dirs(split_dir: Path) -> tuple[Path, Path]:
    if split_dir.name == "images":
        image_dir = split_dir
        label_dir = split_dir.parent / "labels"
        return image_dir, label_dir

    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if image_dir.is_dir() and label_dir.is_dir():
        return image_dir, label_dir

    if not image_dir.is_dir():
        image_dir = split_dir
    if not label_dir.is_dir():
        label_dir = split_dir
    return image_dir, label_dir


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _find_split_dirs(dataset_root: Path, config: dict) -> dict[str, Path]:
    split_candidates = {
        "train": ["train"],
        "valid": ["val", "valid"],
        "test": ["test"],
    }

    split_dirs = {}
    dataset_roots = _candidate_dataset_roots(dataset_root)
    for output_split, config_keys in split_candidates.items():
        candidates: list[Path] = []
        for root in dataset_roots:
            for key in config_keys:
                configured = config.get(key)
                candidates.extend(_split_dir_candidates(root, configured))
            candidates.append(root / output_split)

        for candidate in candidates:
            if candidate.is_dir():
                split_dirs[output_split] = candidate
                break

    if "train" not in split_dirs:
        raise FileNotFoundError(
            f"could not find train split under {dataset_root}; "
            "download the Roboflow YOLOv5 export first:\n"
            "ROBOFLOW_API_KEY=... uv run python "
            "annotation/download_roboflow_dataset.py"
        )
    return split_dirs


def _split_dir_candidates(dataset_root: Path, configured: str | None) -> list[Path]:
    candidates: list[Path] = []
    if configured:
        path = Path(configured)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(dataset_root / path)
        candidates.append(dataset_root / path.name)
    return candidates


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


def _parse_yolo_label(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    return (
        int(float(parts[0])),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
    )


def _to_pixel_box(
    width: int,
    height: int,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    padding: float,
) -> tuple[int, int, int, int]:
    pixel_width = box_width * width
    pixel_height = box_height * height
    center_x = x_center * width
    center_y = y_center * height

    xmin = center_x - pixel_width / 2
    ymin = center_y - pixel_height / 2
    xmax = center_x + pixel_width / 2
    ymax = center_y + pixel_height / 2

    pad_x = pixel_width * padding
    pad_y = pixel_height * padding
    left = max(0, int(xmin - pad_x))
    top = max(0, int(ymin - pad_y))
    right = min(width, int(xmax + pad_x + 0.9999))
    bottom = min(height, int(ymax + pad_y + 0.9999))
    return left, top, right, bottom


if __name__ == "__main__":
    main()
