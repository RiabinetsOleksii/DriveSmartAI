"""Two-phase CUDA-aware transfer learning for MobileNetV3-Small."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW

from .checkpoint import save_checkpoint
from .config import DataConfig, ExperimentConfig
from .data import make_gtsrb_loaders
from .mobilenet import MobileNetV3Small
from .training import (
    _run_train_epoch,
    _run_validation,
    seed_everything,
    select_device,
)


@dataclass(frozen=True)
class MobileNetEpochMetrics:
    epoch: int
    phase: str
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float


def _write_history(history: list[MobileNetEpochMetrics], output_dir: Path) -> None:
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


def _make_optimizer(model: nn.Module, config: ExperimentConfig, learning_rate: float) -> AdamW:
    return AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )


def train_mobilenet(
    data_config: DataConfig,
    experiment_config: ExperimentConfig,
    split_manifest: Path,
    output_dir: Path,
    classifier_epochs: int = 2,
    fine_tune_blocks: int = 3,
    fine_tune_learning_rate: float = 1e-4,
) -> tuple[list[MobileNetEpochMetrics], torch.device, MobileNetV3Small]:
    if not 0 <= classifier_epochs < experiment_config.epochs:
        raise ValueError("classifier_epochs must be at least 0 and less than total epochs")
    seed_everything(experiment_config.seed)
    device = select_device()
    loaders = make_gtsrb_loaders(data_config, split_manifest)
    model = MobileNetV3Small(num_classes=experiment_config.num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    use_amp = experiment_config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.freeze_backbone()
    optimizer = _make_optimizer(model, experiment_config, experiment_config.learning_rate)
    history: list[MobileNetEpochMetrics] = []
    best_accuracy = -1.0
    stale_epochs = 0
    config_payload: dict[str, Any] = {
        "data": asdict(data_config),
        "experiment": asdict(experiment_config),
        "model": {
            "architecture": "MobileNetV3-Small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "classifier_epochs": classifier_epochs,
            "fine_tune_blocks": fine_tune_blocks,
            "fine_tune_learning_rate": fine_tune_learning_rate,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    }
    config_payload["data"]["root"] = str(data_config.root)

    for epoch in range(1, experiment_config.epochs + 1):
        if epoch == classifier_epochs + 1:
            model.unfreeze_last_blocks(fine_tune_blocks)
            optimizer = _make_optimizer(model, experiment_config, fine_tune_learning_rate)
        phase = "classifier" if epoch <= classifier_epochs else "fine-tuning"
        if phase == "classifier":
            model.features.eval()
        train_loss, train_accuracy = _run_train_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, use_amp
        )
        validation_loss, validation_accuracy, validation_macro_f1 = _run_validation(
            model, loaders["validation"], criterion, device, use_amp
        )
        metrics = MobileNetEpochMetrics(
            epoch, phase, train_loss, train_accuracy, validation_loss, validation_accuracy, validation_macro_f1
        )
        history.append(metrics)
        if validation_accuracy > best_accuracy + experiment_config.early_stopping_min_delta:
            best_accuracy = validation_accuracy
            stale_epochs = 0
            save_checkpoint(output_dir / "best_mobilenetv3.pth", model, config_payload, asdict(metrics))
        else:
            stale_epochs += 1
            if stale_epochs >= experiment_config.early_stopping_patience:
                break
    _write_history(history, output_dir)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    return history, device, model
