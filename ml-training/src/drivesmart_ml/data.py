"""GTSRB datasets and Windows-safe DataLoader construction."""

from pathlib import Path
from typing import Callable

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import GTSRB

from .config import DataConfig
from .preprocessing import build_transform
from .split import SampleRecord


class RecordDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[SampleRecord], transform: Callable) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.class_id


def load_gtsrb(root: Path, split: str, download: bool = False, transform: Callable | None = None) -> GTSRB:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return GTSRB(root=str(root), split=split, download=download, transform=transform)


def make_loader(dataset: Dataset, config: DataConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def make_gtsrb_loaders(config: DataConfig, split_manifest: Path | None = None) -> dict[str, DataLoader]:
    if split_manifest is None:
        raise ValueError("split_manifest is required to construct train and validation loaders")
    manifest = pd.read_csv(split_manifest)
    required_columns = {"path", "class_id", "group_id", "split"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"split manifest must contain columns: {sorted(required_columns)}")
    if not set(manifest["split"].dropna()).issubset({"train", "validation"}):
        raise ValueError("split manifest contains an unsupported split")
    if not manifest["class_id"].between(0, 42).all():
        raise ValueError("split manifest contains labels outside 0..42")
    train_records = [
        SampleRecord(row.path, int(row.class_id), row.group_id)
        for row in manifest.itertuples()
        if row.split == "train"
    ]
    validation_records = [
        SampleRecord(row.path, int(row.class_id), row.group_id)
        for row in manifest.itertuples()
        if row.split == "validation"
    ]
    if not train_records or not validation_records:
        raise ValueError("split manifest must contain both train and validation records")
    root = Path(config.root)
    test = load_gtsrb(root, "test", transform=build_transform(image_size=config.image_size))
    train = RecordDataset(train_records, build_transform(training=True, image_size=config.image_size))
    validation = RecordDataset(validation_records, build_transform(image_size=config.image_size))
    return {
        "train": make_loader(train, config, shuffle=True),
        "validation": make_loader(validation, config, shuffle=False),
        "test": make_loader(test, config, shuffle=False),
    }
