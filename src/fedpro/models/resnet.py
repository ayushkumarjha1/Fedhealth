"""
Medical ResNet18 and ResNet34 Adapters for Deep Healthcare Imaging.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from fedpro.models.base import BaseMedicalModel

class MedicalResNet(BaseMedicalModel):
    """ResNet model adapted for single-channel or multi-channel medical scans."""
    
    def __init__(
        self, 
        version: str = "resnet18", 
        in_channels: int = 1, 
        num_classes: int = 2,
        pretrained: bool = False
    ):
        super().__init__()
        if version == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = resnet34(weights=weights)
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            
        # Adapt input conv if in_channels != 3
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
        # Replace final classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
