"""Dataset audit utilities. They inspect files without changing the dataset."""

from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg"}


def _test_labels(root: Path) -> dict[str, int]:
    candidates = sorted(root.rglob("GT-final_test.csv"))
    if not candidates:
        return {}
    frame = pd.read_csv(candidates[0], sep=";")
    return {
        str(row.Filename): int(row.ClassId)
        for row in frame.itertuples()
        if pd.notna(row.Filename) and pd.notna(row.ClassId)
    }


def _class_id_for_path(path: Path, root: Path, test_labels: dict[str, int]) -> int | None:
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "training" and parts[index + 1].isdigit():
            class_id = int(parts[index + 1])
            return class_id if 0 <= class_id <= 42 else None
    if "Final_Test" in parts:
        return test_labels.get(path.name)
    return None


def audit_image_tree(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    test_labels = _test_labels(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        row: dict[str, object] = {"path": str(path), "format": None, "width": None, "height": None, "error": None, "class_id": None}
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                row.update(format=image.format, width=image.width, height=image.height)
            row["class_id"] = _class_id_for_path(path, root, test_labels)
        except (OSError, UnidentifiedImageError) as error:
            row["error"] = type(error).__name__
        rows.append(row)
    return pd.DataFrame(rows)


def class_counts(audit: pd.DataFrame) -> Counter[int]:
    valid = audit[audit["error"].isna() & audit["class_id"].notna()]
    return Counter(int(class_id) for class_id in valid["class_id"])
