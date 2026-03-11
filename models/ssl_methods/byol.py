"""
BYOL — Bootstrap Your Own Latent
=================================
Implements BYOL (Grill et al., 2020).

Key components:
  - Online network: encoder + projector + predictor
  - Target network: encoder + projector (momentum-updated, no predictor)
  - NO negative pairs needed — avoids collapse via asymmetric architecture
  - Loss: cosine similarity between online predictions and target projections

Reference:
  Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", NeurIPS 2020
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_encoder import ResNetEncoder
from models.projection_head import ProjectionHead


class BYOLPredictor(nn.Module):
    """
    BYOL predictor: MLP that sits on top of the online projector.
    Breaks symmetry between online and target networks.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent.

    Args:
        backbone: ResNet variant name.
        projection_hidden_dim: Hidden dim of projection MLP.
        projection_output_dim: Output dim of projection MLP.
        predictor_hidden_dim: Hidden dim of predictor MLP.
        momentum: Momentum coefficient for target network update.
    """

    METHOD_NAME = "byol"

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_hidden_dim: int = 2048,
        projection_output_dim: int = 128,
        predictor_hidden_dim: int = 512,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.momentum = momentum

        # Online network: encoder + projector + predictor
        self.online_encoder = ResNetEncoder(backbone=backbone, pretrained=False)
        self.online_projector = ProjectionHead(
            input_dim=self.online_encoder.output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )
        self.predictor = BYOLPredictor(
            input_dim=projection_output_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projection_output_dim,
        )

        # Target network: encoder + projector (no predictor, no gradients)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_target(self):
        """Exponential moving average update of the target network."""
        m = self.momentum
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.mul_(m).add_(param_o.data, alpha=1.0 - m)
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data.mul_(m).add_(param_o.data, alpha=1.0 - m)

    @staticmethod
    def _cosine_loss(p, z):
        """Negative cosine similarity loss (BYOL objective)."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def forward(self, x_i, x_j):
        """
        Forward pass: compute symmetric BYOL loss.

        Args:
            x_i: First augmented view (B, C, H, W).
            x_j: Second augmented view (B, C, H, W).

        Returns:
            Scalar BYOL loss.
        """
        # Online predictions
        p_i = self.predictor(self.online_projector(self.online_encoder(x_i)))
        p_j = self.predictor(self.online_projector(self.online_encoder(x_j)))

        # Target projections (no gradient)
        with torch.no_grad():
            self._momentum_update_target()
            z_i = self.target_projector(self.target_encoder(x_i))
            z_j = self.target_projector(self.target_encoder(x_j))

        # Symmetric loss
        loss = self._cosine_loss(p_i, z_j) + self._cosine_loss(p_j, z_i)
        return loss / 2.0

    @property
    def encoder(self):
        """Return the online encoder for downstream tasks."""
        return self.online_encoder

    @property
    def feature_dim(self):
        return self.online_encoder.output_dim

    def encode(self, x):
        return self.online_encoder(x)

    def get_trainable_params(self):
        """Online network params: encoder + projector + predictor."""
        return (
            list(self.online_encoder.parameters())
            + list(self.online_projector.parameters())
            + list(self.predictor.parameters())
        )

    def save_encoder(self, path: str):
        torch.save(self.online_encoder.state_dict(), path)
        print(f"BYOL encoder saved to {path}")

    def load_encoder(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.online_encoder.load_state_dict(state_dict)
        print(f"BYOL encoder loaded from {path}")

    def save_full_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_full_model(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
