"""MobileNetV3-Small transfer-learning model for GTSRB."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = True,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.network = mobilenet_v3_small(weights=weights)
        classifier = self.network.classifier
        if dropout is not None:
            classifier[0] = nn.Dropout(p=dropout)
        classifier[3] = nn.Linear(classifier[3].in_features, num_classes)

    @property
    def features(self) -> nn.Sequential:
        return self.network.features

    @property
    def classifier(self) -> nn.Sequential:
        return self.network.classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def freeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_last_blocks(self, block_count: int = 3) -> None:
        if not 0 < block_count <= len(self.features):
            raise ValueError("block_count must be between 1 and the number of feature blocks")
        self.freeze_backbone()
        for block in list(self.features.children())[-block_count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

