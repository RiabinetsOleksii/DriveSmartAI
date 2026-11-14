"""Final independent GTSRB test-set evaluation for trained models."""

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from .config import DataConfig
from .data import load_gtsrb, make_loader
from .mapping import CLASS_MAPPING, validate_class_mapping
from .mobilenet import MobileNetV3Small
from .model import CustomCNN
from .preprocessing import build_transform


def compute_metrics(
    targets: list[int],
    predictions: list[int],
    num_classes: int = 43,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and have equal length")
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    class_accuracy = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(num_classes, dtype=float),
        where=support != 0,
    )
    names = class_names or {class_id: str(class_id) for class_id in labels}
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "support": int(support[class_id]),
                "accuracy": float(class_accuracy[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
            }
            for class_id in labels
        ],
        "confusion_matrix": matrix.tolist(),
        "num_samples": len(targets),
    }


def _write_confusion_matrix(matrix: list[list[int]], model_name: str, output_dir: Path) -> None:
    matrix_array = np.asarray(matrix)
    csv_path = output_dir / f"{model_name}_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_class", *[f"pred_{class_id}" for class_id in range(43)]])
        for class_id, row in enumerate(matrix_array):
            writer.writerow([class_id, *row.tolist()])

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix_array, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"{model_name} confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(43),
        yticks=range(43),
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(figure)


def _write_per_class_metrics(metrics: dict[str, Any], model_name: str, output_dir: Path) -> None:
    with (output_dir / f"{model_name}_per_class.csv").open("w", newline="", encoding="utf-8") as file:
        rows = metrics["per_class"]
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    warmup_batches: int,
) -> tuple[list[int], list[int], dict[str, float]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if batch_index < warmup_batches:
            with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
                model(images)
            continue
        break

    measured_samples = 0
    elapsed_ms = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            _synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(images)
        if device.type == "cuda":
            end_event.record()
            end_event.synchronize()
            elapsed_ms += start_event.elapsed_time(end_event)
        else:
            elapsed_ms += (time.perf_counter() - start) * 1000
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
        measured_samples += labels.size(0)
    return targets, predictions, {
        "inference_ms": elapsed_ms,
        "inference_ms_per_image": elapsed_ms / measured_samples,
        "throughput_images_per_second": measured_samples / (elapsed_ms / 1000),
        "measured_samples": measured_samples,
    }


def _load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid checkpoint format: {path}")
    config = checkpoint.get("config", {})
    num_classes = int(config.get("experiment", {}).get("num_classes", 43))
    architecture = str(config.get("model", {}).get("architecture", checkpoint.get("model_class", "")))
    if architecture == "MobileNetV3-Small":
        model: nn.Module = MobileNetV3Small(num_classes=num_classes, pretrained=False)
    elif checkpoint.get("model_class") == "CustomCNN" or not architecture:
        model = CustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    image_size = int(config.get("data", {}).get("image_size", 64))
    return model, checkpoint, image_size


def evaluate_model(
    checkpoint_path: Path,
    model_name: str,
    root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    warmup_batches: int = 2,
) -> dict[str, Any]:
    model, checkpoint, image_size = _load_model(checkpoint_path, device)
    preprocessing_metadata = checkpoint.get("preprocessing", {})
    metadata_image_size = preprocessing_metadata.get("image_size")
    warnings: list[str] = []
    if metadata_image_size != [image_size, image_size]:
        warnings.append(
            f"checkpoint preprocessing image_size {metadata_image_size} does not match "
            f"config data image_size {[image_size, image_size]}; config value was used"
        )
    dataset = load_gtsrb(root, "test", transform=build_transform(image_size=image_size))
    loader = make_loader(
        dataset,
        config=DataConfig(root=root, image_size=image_size, batch_size=batch_size, num_workers=num_workers),
        shuffle=False,
    )
    targets, predictions, timing = _predict(
        model,
        loader,
        device,
        mixed_precision=device.type == "cuda",
        warmup_batches=warmup_batches,
    )
    metrics = compute_metrics(targets, predictions, class_names=CLASS_MAPPING)
    metrics.update(
        {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "image_size": image_size,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
            "preprocessing": preprocessing_metadata,
            "warnings": warnings,
            **timing,
        }
    )
    _write_per_class_metrics(metrics, model_name, output_dir)
    _write_confusion_matrix(metrics["confusion_matrix"], model_name, output_dir)
    (output_dir / f"{model_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def evaluate_models(
    root: Path,
    custom_checkpoint: Path,
    mobilenet_checkpoint: Path,
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 0,
    warmup_batches: int = 2,
) -> dict[str, Any]:
    validate_class_mapping()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        evaluate_model(custom_checkpoint, "custom_cnn", root, output_dir, device, batch_size, num_workers, warmup_batches),
        evaluate_model(
            mobilenet_checkpoint,
            "mobilenetv3",
            root,
            output_dir,
            device,
            batch_size,
            num_workers,
            warmup_batches,
        ),
    ]
    comparison_rows = [
        {
            key: result[key]
            for key in (
                "model",
                "checkpoint",
                "image_size",
                "parameters",
                "num_samples",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "inference_ms",
                "inference_ms_per_image",
                "throughput_images_per_second",
                "device",
            )
        }
        for result in results
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    comparison = {"dataset": "GTSRB official test", "results": comparison_rows}
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return {"device": str(device), "results": results, "comparison": comparison}
