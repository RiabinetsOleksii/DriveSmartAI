import pandas as pd
import pytest

from drivesmart_ml.data import make_gtsrb_loaders
from drivesmart_ml.config import DataConfig


def test_loader_rejects_manifest_without_required_split_columns(tmp_path) -> None:
    manifest = tmp_path / "split.csv"
    pd.DataFrame([{"path": "image.ppm", "class_id": 0}]).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="columns"):
        make_gtsrb_loaders(DataConfig(root=tmp_path), manifest)
