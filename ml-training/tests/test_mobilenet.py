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


def _batch_norm_layers(model: MobileNetV3Small) -> list[torch.nn.modules.batchnorm._BatchNorm]:
    return [module for module in model.features.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]


def test_classifier_phase_keeps_frozen_backbone_and_batch_norm_in_eval() -> None:
    model = MobileNetV3Small(pretrained=False)
    model.freeze_backbone()
    model.train()
    assert not model.features.training
    before = [(layer.running_mean.clone(), layer.running_var.clone()) for layer in _batch_norm_layers(model)]
    _ = model(torch.randn(8, 3, 64, 64))
    after = [(layer.running_mean, layer.running_var) for layer in _batch_norm_layers(model)]
    assert all(torch.equal(old_mean, new_mean) and torch.equal(old_var, new_var) for (old_mean, old_var), (new_mean, new_var) in zip(before, after))


def test_fine_tuning_keeps_frozen_batch_norm_stable_and_trains_last_blocks() -> None:
    model = MobileNetV3Small(pretrained=False)
    model.unfreeze_last_blocks(3)
    model.train()
    frozen_layers = [
        layer
        for layer in _batch_norm_layers(model)
        if all(not parameter.requires_grad for parameter in layer.parameters())
    ]
    trainable_layers = [
        layer
        for layer in _batch_norm_layers(model)
        if any(parameter.requires_grad for parameter in layer.parameters())
    ]
    assert frozen_layers and trainable_layers
    assert all(not layer.training for layer in frozen_layers)
    assert all(layer.training for layer in trainable_layers)
    frozen_before = [(layer.running_mean.clone(), layer.running_var.clone()) for layer in frozen_layers]
    loss = model(torch.randn(8, 3, 64, 64)).sum()
    loss.backward()
    frozen_after = [(layer.running_mean, layer.running_var) for layer in frozen_layers]
    assert all(torch.equal(old_mean, new_mean) and torch.equal(old_var, new_var) for (old_mean, old_var), (new_mean, new_var) in zip(frozen_before, frozen_after))
    assert any(parameter.grad is not None for parameter in model.features[-3:].parameters() if parameter.requires_grad)
