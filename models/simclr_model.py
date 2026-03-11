"""
SimCLR Model
============
Full SimCLR model combining encoder + projection head.
Provides methods for pretraining and downstream feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_encoder import ResNetEncoder
from models.projection_head import ProjectionHead


class SimCLR(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

    Combines a ResNet encoder with a non-linear projection head.
    During pretraining, both encoder and projection head are used.
    For downstream tasks, only the encoder is used (projection head discarded).

    Args:
        backbone: ResNet variant name (e.g., "resnet18").
        projection_hidden_dim: Hidden dimension of the projection head.
        projection_output_dim: Output dimension of the projection head.
        use_batch_norm: Whether to use batch norm in the projection head.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_hidden_dim: int = 2048,
        projection_output_dim: int = 128,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        # Encoder (backbone)
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=False)

        # Projection head
        self.projection = ProjectionHead(
            input_dim=self.encoder.output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x):
        """
        Forward pass through encoder + projection head.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            h: Encoder representations of shape (B, feature_dim).
            z: L2-normalized projected features of shape (B, projection_output_dim).
        """
        h = self.encoder(x)
        z = self.projection(h)
        z = F.normalize(z, dim=1)  # L2 normalize for cosine similarity
        return h, z

    def encode(self, x):
        """
        Extract encoder representations only (for downstream tasks).

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Encoder features of shape (B, feature_dim).
        """
        return self.encoder(x)

    @property
    def feature_dim(self):
        """Dimensionality of the encoder's output features."""
        return self.encoder.output_dim

    def save_encoder(self, path: str):
        """Save only the encoder weights (for downstream use)."""
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder saved to {path}")

    def load_encoder(self, path: str, strict: bool = True):
        """Load encoder weights from a checkpoint."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict, strict=strict)
        print(f"Encoder loaded from {path}")

    def save_full_model(self, path: str):
        """Save the full model (encoder + projection head)."""
        torch.save(self.state_dict(), path)
        print(f"Full SimCLR model saved to {path}")

    def load_full_model(self, path: str, strict: bool = True):
        """Load the full model weights."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
        print(f"Full SimCLR model loaded from {path}")
