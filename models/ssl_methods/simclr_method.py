"""
SimCLR Method Wrapper
=====================
Wraps the existing SimCLR model into the unified SSL method interface.
"""

import torch
import torch.nn as nn

from models.simclr_model import SimCLR
from utils.losses import NTXentLoss


class SimCLRMethod(nn.Module):
    """
    Unified interface wrapper for SimCLR.

    Args:
        backbone: ResNet variant name.
        projection_hidden_dim: Hidden dim of projection MLP.
        projection_output_dim: Output dim of projection MLP.
        temperature: NT-Xent loss temperature.
    """

    METHOD_NAME = "simclr"

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_hidden_dim: int = 2048,
        projection_output_dim: int = 128,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.model = SimCLR(
            backbone=backbone,
            projection_hidden_dim=projection_hidden_dim,
            projection_output_dim=projection_output_dim,
        )
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(self, x_i, x_j):
        """Forward pass: returns loss given two augmented views."""
        _, z_i = self.model(x_i)
        _, z_j = self.model(x_j)
        return self.criterion(z_i, z_j)

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def feature_dim(self):
        return self.model.feature_dim

    def encode(self, x):
        return self.model.encode(x)

    def get_trainable_params(self):
        """Return parameters for optimizer."""
        return self.model.parameters()

    def save_encoder(self, path: str):
        self.model.save_encoder(path)

    def load_encoder(self, path: str):
        self.model.load_encoder(path)

    def save_full_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_full_model(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
