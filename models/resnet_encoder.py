"""
ResNet Encoder
==============
Backbone encoder for SimCLR. Uses torchvision's ResNet variants with
the classification head replaced by Identity for feature extraction.
"""

import torch.nn as nn
import torchvision.models as models


# Mapping of backbone names to torchvision constructors + feature dims
BACKBONE_REGISTRY = {
    "resnet18": (models.resnet18, 512),
    "resnet34": (models.resnet34, 512),
    "resnet50": (models.resnet50, 2048),
}


class ResNetEncoder(nn.Module):
    """
    ResNet backbone encoder for self-supervised learning.

    Removes the final FC layer and returns feature representations.

    Args:
        backbone: Name of the ResNet variant (e.g., "resnet18", "resnet50").
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()

        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
            )

        constructor, self.feature_dim = BACKBONE_REGISTRY[backbone]

        # Load backbone with or without pretrained weights
        weights = "IMAGENET1K_V1" if pretrained else None
        self.encoder = constructor(weights=weights)

        # Replace the classification head with Identity
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature representations of shape (B, feature_dim).
        """
        return self.encoder(x)

    @property
    def output_dim(self):
        """Dimensionality of the encoder's output features."""
        return self.feature_dim
