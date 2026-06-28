from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet50(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        y_prob.extend(probabilities.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def fit_resnet(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_name: str = "resnet50_cv6_fold1_best.pt",
    seed: int = 123,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, object]]:
    set_seed(seed)
    device = get_device()
    model = build_resnet50(num_classes=num_classes, freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / checkpoint_name

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_metrics: dict[str, object] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"test_loss={row['test_loss']:.4f} | "
            f"test_acc={row['test_accuracy']:.4f}"
        )

        if row["test_accuracy"] > best_accuracy:
            best_accuracy = row["test_accuracy"]
            best_metrics = test_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "epoch": epoch,
                    "test_accuracy": best_accuracy,
                },
                checkpoint_path,
            )

    return model, history, best_metrics