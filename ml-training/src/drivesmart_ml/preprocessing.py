"""The shared image preprocessing contract used by future model consumers."""

from dataclasses import dataclass
from typing import Final

import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE: Final[int] = 64
IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingContract:
    image_size: int = IMAGE_SIZE
    interpolation: str = "bilinear"
    channel_order: str = "RGB"
    tensor_layout: str = "NCHW"
    value_range_before_normalization: tuple[float, float] = (0.0, 1.0)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def to_dict(self) -> dict[str, object]:
        return {
            "version": "1.0",
            "color": self.channel_order,
            "image_size": [self.image_size, self.image_size],
            "resize_interpolation": self.interpolation,
            "tensor_layout": self.tensor_layout,
            "normalization": {
                "scale": "pixel / 255.0",
                "mean": list(self.mean),
                "std": list(self.std),
            },
            "training_only_augmentations": [
                "small_rotation",
                "small_translation",
                "brightness_contrast",
            ],
            "forbidden_augmentations": ["horizontal_flip"],
            "class_id_range": [0, 42],
        }


def build_transform(training: bool = False, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    operations: list[object] = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)]
    if training:
        operations.extend(
            [
                transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def preprocess_image(image: Image.Image, training: bool = False, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    return build_transform(training=training, image_size=image_size)(rgb_image)
