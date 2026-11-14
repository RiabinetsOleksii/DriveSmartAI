import torch
from PIL import Image

from drivesmart_ml.preprocessing import preprocess_image


def test_preprocessing_returns_rgb_chw_tensor() -> None:
    image = Image.new("L", (23, 17), color=128)
    tensor = preprocess_image(image, image_size=32)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_training_preprocessing_does_not_change_tensor_contract() -> None:
    image = Image.new("RGB", (23, 17), color=(100, 120, 140))
    tensor = preprocess_image(image, training=True, image_size=32)
    assert tensor.shape == (3, 32, 32)
    assert torch.isfinite(tensor).all()
