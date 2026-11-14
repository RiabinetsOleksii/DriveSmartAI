"""Checkpoint serialization for reproducible model artifacts."""

from pathlib import Path
from typing import Any

import torch

from .mapping import CLASS_MAPPING
from .preprocessing import PreprocessingContract


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "config": config,
            "metrics": metrics,
            "class_mapping": CLASS_MAPPING,
            "preprocessing": PreprocessingContract().to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid checkpoint format: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
