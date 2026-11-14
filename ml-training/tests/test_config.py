import pytest

from drivesmart_ml.config import DataConfig, ExperimentConfig


def test_default_config_is_gtsrb_compatible() -> None:
    config = DataConfig()
    assert config.image_size == 64
    assert ExperimentConfig().num_classes == 43


@pytest.mark.parametrize("kwargs", [{"image_size": 0}, {"batch_size": 0}, {"num_workers": -1}])
def test_invalid_data_config_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_only_gtsrb_class_count_is_accepted() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(num_classes=2)
