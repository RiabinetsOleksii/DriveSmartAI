"""Configuration models for reproducible data preparation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("..") / "datasets" / "gtsrb"
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")


@dataclass(frozen=True)
class ExperimentConfig:
    num_classes: int = 43
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_classes != 43:
            raise ValueError("GTSRB must contain exactly 43 classes")
