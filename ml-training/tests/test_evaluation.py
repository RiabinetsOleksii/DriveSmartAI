import pytest

from drivesmart_ml.evaluation import compute_metrics


def test_compute_metrics_returns_aggregate_per_class_and_confusion_matrix() -> None:
    metrics = compute_metrics(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 1, 2, 0],
        num_classes=3,
        class_names={0: "zero", 1: "one", 2: "two"},
    )
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["num_samples"] == 6
    assert len(metrics["per_class"]) == 3
    assert metrics["per_class"][0]["support"] == 2
    assert metrics["per_class"][1]["f1"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_compute_metrics_rejects_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0], [0, 1])
