"""Reproducible CUDA-aware training and validation for the custom CNN."""

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW

from .checkpoint import save_checkpoint
from .config import DataConfig, ExperimentConfig
from .data import make_gtsrb_loaders
from .model import CustomCNN


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def _run_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions: list[int] = []
    targets: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        predicted = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += int((predicted == labels).sum().item())
        total += labels.size(0)
        predictions.extend(predicted.cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return (
        total_loss / total,
        correct / total,
        float(f1_score(targets, predictions, average="macro", zero_division=0)),
    )


def _write_history(history: list[EpochMetrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(history[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in history)
    epochs = [item.epoch for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item.train_loss for item in history], label="train")
    axes[0].plot(epochs, [item.validation_loss for item in history], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [item.train_accuracy for item in history], label="train")
    axes[1].plot(epochs, [item.validation_accuracy for item in history], label="validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def train(
    data_config: DataConfig,
    experiment_config: ExperimentConfig,
    split_manifest: Path,
    output_dir: Path,
) -> tuple[list[EpochMetrics], torch.device]:
    seed_everything(experiment_config.seed)
    device = select_device()
    loaders = make_gtsrb_loaders(data_config, split_manifest)
    model = CustomCNN(num_classes=experiment_config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=experiment_config.learning_rate, weight_decay=experiment_config.weight_decay)
    use_amp = experiment_config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[EpochMetrics] = []
    best_accuracy = -1.0
    stale_epochs = 0
    config_payload = {"data": asdict(data_config), "experiment": asdict(experiment_config)}
    config_payload["data"]["root"] = str(data_config.root)
    for epoch in range(1, experiment_config.epochs + 1):
        train_loss, train_accuracy = _run_train_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp
        )
        validation_loss, validation_accuracy, validation_macro_f1 = _run_validation(
            model, loaders["validation"], criterion, device, use_amp
        )
        metrics = EpochMetrics(
            epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, validation_macro_f1
        )
        history.append(metrics)
        if validation_accuracy > best_accuracy + experiment_config.early_stopping_min_delta:
            best_accuracy = validation_accuracy
            stale_epochs = 0
            save_checkpoint(output_dir / "best_cnn.pth", model, config_payload, asdict(metrics))
        else:
            stale_epochs += 1
            if stale_epochs >= experiment_config.early_stopping_patience:
                break
    _write_history(history, output_dir)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    return history, device
