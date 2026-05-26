from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mahjong_calc_point.classification import IMAGENET_MEAN, IMAGENET_STD
from mahjong_calc_point.tiles import TILE_LABELS


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tile crop classifier.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "classification",
        help="ImageFolder-style dataset produced by build_classification_dataset.py.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path(__file__).resolve().parent / "data.yaml",
        help="YOLOv5 data.yaml used to lock class order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "src"
        / "mahjong_calc_point"
        / "classifier"
        / "tile_classifier.pt",
        help="Classifier checkpoint path used by the Flask app.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Training device.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        help="Limit each class to this many images for quick smoke tests.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print a progress line every N batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training/evaluation without writing a checkpoint.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize MobileNetV3 with ImageNet weights if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = _load_class_names(args.data_yaml)
    train_dataset = TileCropDataset(
        root=args.data_root / "train",
        class_names=class_names,
        transform=_build_transform(args.image_size, training=True),
        max_samples_per_class=args.max_samples_per_class,
    )
    valid_dataset = TileCropDataset(
        root=_resolve_valid_root(args.data_root),
        class_names=class_names,
        transform=_build_transform(args.image_size, training=False),
        max_samples_per_class=args.max_samples_per_class,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = _select_device(args.device)
    model = build_model(num_classes=len(class_names), pretrained=args.pretrained).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"device={device} train_samples={len(train_dataset)} "
        f"valid_samples={len(valid_dataset)} classes={len(class_names)}",
        flush=True,
    )

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            progress_interval=args.progress_interval,
        )
        valid_loss, valid_accuracy = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            progress_interval=args.progress_interval,
        )

        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_accuracy:.4f}"
        )

        if valid_accuracy >= best_accuracy:
            best_accuracy = valid_accuracy
            if not args.no_save:
                save_checkpoint(
                    output_path=args.output,
                    model=model,
                    class_names=class_names,
                    image_size=args.image_size,
                    valid_accuracy=valid_accuracy,
                )

    print(f"best valid accuracy: {best_accuracy:.4f}")
    if args.no_save:
        print("checkpoint save skipped")
    else:
        print(f"saved classifier: {args.output}")


class TileCropDataset(Dataset):
    def __init__(
        self,
        root: Path,
        class_names: list[str],
        transform: transforms.Compose,
        max_samples_per_class: int | None = None,
    ) -> None:
        self.root = root
        self.class_names = class_names
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.samples = self._collect_samples()
        if not self.samples:
            raise FileNotFoundError(f"no classification images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, class_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), class_index

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for class_index, class_name in enumerate(self.class_names):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            class_samples = []
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    class_samples.append((image_path, class_index))
            if self.max_samples_per_class is not None:
                class_samples = class_samples[: self.max_samples_per_class]
            samples.extend(class_samples)
        return samples


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = None
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    progress_interval: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_index, (inputs, labels) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += labels.size(0)
        if progress_interval and batch_index % progress_interval == 0:
            print(
                f"epoch {epoch:03d} train "
                f"{batch_index}/{len(loader)} "
                f"loss={total_loss / total_count:.4f} "
                f"acc={total_correct / total_count:.4f}",
                flush=True,
            )

    return total_loss / total_count, total_correct / total_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    progress_interval: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.inference_mode():
        for batch_index, (inputs, labels) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += labels.size(0)
            if progress_interval and batch_index % progress_interval == 0:
                print(
                    f"epoch {epoch:03d} valid "
                    f"{batch_index}/{len(loader)} "
                    f"loss={total_loss / total_count:.4f} "
                    f"acc={total_correct / total_count:.4f}",
                    flush=True,
                )

    return total_loss / total_count, total_correct / total_count


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    valid_accuracy: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "valid_accuracy": valid_accuracy,
        },
        output_path,
    )


def _build_transform(image_size: int, training: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if training:
        steps.extend(
            [
                transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.15)], p=0.5),
                transforms.RandomRotation(4),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def _load_class_names(data_yaml: Path) -> list[str]:
    config = yaml.safe_load(data_yaml.read_text())
    class_names = list(config["names"])
    if class_names != TILE_LABELS:
        raise ValueError("data.yaml names do not match mahjong_calc_point.tiles.TILE_LABELS")
    return class_names


def _resolve_valid_root(data_root: Path) -> Path:
    for name in ("valid", "val"):
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not find valid or val split under {data_root}")


def _select_device(configured: str) -> torch.device:
    if configured != "auto":
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
