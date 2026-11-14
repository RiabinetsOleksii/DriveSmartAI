"""Group-aware train/validation splitting for GTSRB training samples."""

from dataclasses import dataclass
from pathlib import Path
import re
import random

import pandas as pd


@dataclass(frozen=True)
class SampleRecord:
    path: str
    class_id: int
    group_id: str


def gtsrb_group_id(path: str | Path, class_id: int) -> str:
    """Derive a stable physical-sign group from the GTSRB track filename."""
    stem = Path(path).stem
    match = re.match(r"(?P<track>\d+)(?:[_-]\d+)?$", stem)
    track = match.group("track") if match else stem
    return f"{class_id}:{track}"


def records_from_training_tree(root: Path) -> list[SampleRecord]:
    image_root = root / "GTSRB" / "Final_Training" / "Images"
    if not image_root.exists():
        image_root = root
    records: list[SampleRecord] = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".ppm", ".png", ".jpg", ".jpeg"}:
            continue
        class_part = path.parent.name
        if not class_part.isdigit() or not 0 <= int(class_part) <= 42:
            continue
        class_id = int(class_part)
        records.append(SampleRecord(str(path), class_id, gtsrb_group_id(path, class_id)))
    if not records:
        raise ValueError(f"no GTSRB training images found under {image_root}")
    return records


def split_records(
    records: list[SampleRecord],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not records:
        raise ValueError("cannot split an empty record list")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    frame = pd.DataFrame([record.__dict__ for record in records])
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for offset, class_id in enumerate(sorted(frame["class_id"].unique())):
        class_frame = frame[frame["class_id"] == class_id]
        groups = sorted(class_frame["group_id"].unique())
        if len(groups) < 2:
            raise ValueError(f"class {class_id} needs at least two groups for a split")
        randomizer = random.Random(seed + offset)
        randomizer.shuffle(groups)
        validation_group_count = max(1, round(len(groups) * validation_fraction))
        validation_groups = set(groups[:validation_group_count])
        validation_indices.extend(class_frame[class_frame["group_id"].isin(validation_groups)].index.tolist())
        train_indices.extend(class_frame[~class_frame["group_id"].isin(validation_groups)].index.tolist())
    train = [records[index] for index in sorted(train_indices)]
    validation = [records[index] for index in sorted(validation_indices)]
    if {record.class_id for record in train} != set(range(43)):
        raise ValueError("train split does not represent all 43 classes")
    if {record.class_id for record in validation} != set(range(43)):
        raise ValueError("validation split does not represent all 43 classes")
    return train, validation


def write_split_manifest(
    train: list[SampleRecord],
    validation: list[SampleRecord],
    output_path: Path,
    seed: int,
    validation_fraction: float,
) -> None:
    if {sample.path for sample in train} & {sample.path for sample in validation}:
        raise ValueError("image overlap detected between train and validation")
    if {sample.group_id for sample in train} & {sample.group_id for sample in validation}:
        raise ValueError("group overlap detected between train and validation")
    rows = [
        {"path": sample.path, "class_id": sample.class_id, "group_id": sample.group_id, "split": split}
        for split, samples in (("train", train), ("validation", validation))
        for sample in samples
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    output_path.with_suffix(".json").write_text(
        pd.Series(
            {
                "seed": seed,
                "validation_fraction": validation_fraction,
                "group_strategy": "class_id:track_from_filename",
                "train_count": len(train),
                "validation_count": len(validation),
            }
        ).to_json(),
        encoding="utf-8",
    )
