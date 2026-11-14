import torch

from drivesmart_ml.checkpoint import load_checkpoint, save_checkpoint
from drivesmart_ml.model import CustomCNN


def test_custom_cnn_forward_shape_and_architecture() -> None:
    model = CustomCNN()
    output = model(torch.randn(4, 3, 64, 64))
    assert tuple(output.shape) == (4, 43)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert sum(1 for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)) == 6


def test_checkpoint_round_trip(tmp_path) -> None:
    model = CustomCNN()
    model.eval()
    path = tmp_path / "best_cnn.pth"
    save_checkpoint(path, model, {"seed": 42}, {"validation_accuracy": 0.5})
    restored = CustomCNN()
    restored.eval()
    checkpoint = load_checkpoint(path, restored)
    assert checkpoint["config"] == {"seed": 42}
    assert checkpoint["class_mapping"][0] == "Speed limit (20km/h)"
    inputs = torch.randn(2, 3, 64, 64)
    assert torch.equal(model(inputs), restored(inputs))
