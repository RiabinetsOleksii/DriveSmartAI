import torch

from drivesmart_ml.mobilenet import MobileNetV3Small


def test_mobilenet_forward_supports_gtsrb_shape() -> None:
    model = MobileNetV3Small(pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 43)
    assert model.classifier[3].out_features == 43


def test_mobilenet_transfer_learning_freeze_and_unfreeze() -> None:
    model = MobileNetV3Small(pretrained=False)
    model.freeze_backbone()
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    model.unfreeze_last_blocks(2)
    assert any(parameter.requires_grad for parameter in model.features[-2].parameters())
    assert all(not parameter.requires_grad for parameter in model.features[:-2].parameters())
