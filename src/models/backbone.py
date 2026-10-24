"""
ResNet-50 Backbone with Modular Pooling Head Integration.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.models.heads import GAPHead, ABMILHead, TopKHead, DARHead, OCDPHead


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet-50 backbone truncated before GAP/FC.
    Outputs feature map of shape [B, 2048, 7, 7] for 224x224 inputs.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # [B, 2048, H, W]


class ClassifierWithHead(nn.Module):
    """
    Complete model combining ResNet-50 backbone with a selected pooling head.
    """
    def __init__(
        self,
        head_type: str = "ocdp",
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.head_type = head_type.lower()
        self.backbone = ResNet50FeatureExtractor(pretrained=pretrained_backbone)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = 2048
        if self.head_type == "gap":
            self.head = GAPHead(in_features=in_features, num_classes=num_classes)
        elif self.head_type == "abmil":
            self.head = ABMILHead(in_features=in_features, num_classes=num_classes)
        elif self.head_type == "topk":
            self.head = TopKHead(in_features=in_features, num_classes=num_classes)
        elif self.head_type == "dar":
            self.head = DARHead(in_features=in_features, num_classes=num_classes)
        elif self.head_type == "ocdp":
            self.head = OCDPHead(in_features=in_features, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_context: bool = False
    ) -> Any:
        feats = self.backbone(x)
        if self.head_type == "ocdp":
            return self.head(feats, return_context=return_context)
        else:
            return self.head(feats)
