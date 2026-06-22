from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data_inspection import CV_VERSION_DIRS, DATA_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    steps = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def split_path(data_root: Path = DATA_ROOT, cv: str = "CV1", fold: str = "fold-1", split: str = "train") -> Path:
    if cv not in CV_VERSION_DIRS:
        raise ValueError(f"Unknown CV version: {cv}. Expected one of {sorted(CV_VERSION_DIRS)}.")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")

    fold_dir = data_root / CV_VERSION_DIRS[cv] / fold
    suffix = "-as-training" if split == "train" else "-as-test"
    matches = sorted(path for path in fold_dir.iterdir() if path.is_dir() and path.name.endswith(suffix))

    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {split} directory in {fold_dir}, found {len(matches)}.")
    return matches[0]


def create_dataset(
    data_root: Path = DATA_ROOT,
    cv: str = "CV1",
    fold: str = "fold-1",
    split: str = "train",
    image_size: int = 224,
) -> datasets.ImageFolder:
    root = split_path(data_root=data_root, cv=cv, fold=fold, split=split)
    return datasets.ImageFolder(root=root, transform=make_transform(image_size=image_size, train=split == "train"))


def create_dataloaders(
    data_root: Path = DATA_ROOT,
    cv: str = "CV1",
    fold: str = "fold-1",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = create_dataset(data_root, cv, fold, "train", image_size)
    test_dataset = create_dataset(data_root, cv, fold, "test", image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
