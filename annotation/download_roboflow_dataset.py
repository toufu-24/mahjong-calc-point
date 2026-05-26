from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_DATASET = "project-xv49e/mahjong-x5dzz/2"
DEFAULT_FORMAT = "yolov5pytorch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Roboflow mahjong dataset as a YOLOv5 export."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Roboflow dataset path: workspace/project/version.",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        help="Roboflow export format. Use yolov5pytorch for YOLOv5 PyTorch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where train/valid/test/data.yaml should be written.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key. Defaults to ROBOFLOW_API_KEY.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        help="Use an already-downloaded Roboflow ZIP instead of calling the API.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove existing train/valid/test folders before extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip and not args.api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is required for API download. Get it from "
            "Roboflow, then run:\n"
            "ROBOFLOW_API_KEY=... uv run python annotation/download_roboflow_dataset.py"
            "\n\nOr download the ZIP in your browser and run:\n"
            "uv run python annotation/download_roboflow_dataset.py "
            "--zip ~/Downloads/your-roboflow-export.zip"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "roboflow_dataset.zip"
        extract_root = Path(temp_dir) / "extract"
        if args.zip:
            zip_path = args.zip.expanduser().resolve()
            if not zip_path.is_file():
                raise FileNotFoundError(zip_path)
        else:
            export_url = _request_export_url(args.dataset, args.format, args.api_key)
            print(f"downloading {args.dataset} ({args.format})")
            _download_file(export_url, zip_path)

        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_root)

        dataset_root = _find_extracted_dataset_root(extract_root)
        _copy_dataset(dataset_root, args.output, keep_existing=args.keep_existing)

    print(f"dataset ready: {args.output}")


def _request_export_url(dataset: str, model_format: str, api_key: str) -> str:
    encoded_dataset = "/".join(urllib.parse.quote(part) for part in dataset.split("/"))
    query = urllib.parse.urlencode({"api_key": api_key})
    url = f"https://api.roboflow.com/{encoded_dataset}/{model_format}?{query}"
    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))

    export = payload.get("export") or {}
    link = export.get("link") or payload.get("link")
    if not link:
        raise RuntimeError(
            "Roboflow export response did not contain a download link. "
            f"Response keys: {sorted(payload.keys())}"
        )
    return str(link)


def _download_file(url: str, output_path: Path) -> None:
    with urllib.request.urlopen(url) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def _find_extracted_dataset_root(extract_root: Path) -> Path:
    candidates = [extract_root]
    candidates.extend(path for path in extract_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "data.yaml").is_file() and _has_split(candidate):
            return candidate
    raise FileNotFoundError(
        "downloaded archive did not contain data.yaml plus train/valid folders"
    )


def _has_split(path: Path) -> bool:
    return any((path / name).is_dir() for name in ("train", "valid", "val"))


def _copy_dataset(source: Path, output: Path, keep_existing: bool) -> None:
    for name in ("train", "valid", "val", "test"):
        source_path = source / name
        if not source_path.exists():
            continue
        target_name = "valid" if name == "val" else name
        target_path = output / target_name
        if target_path.exists() and not keep_existing:
            shutil.rmtree(target_path)
        if not target_path.exists():
            shutil.copytree(source_path, target_path)

    data_yaml = source / "data.yaml"
    if data_yaml.is_file():
        shutil.copy2(data_yaml, output / "data.yaml")


if __name__ == "__main__":
    main()
