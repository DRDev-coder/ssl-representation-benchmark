"""
Projection Head
===============
Non-linear projection head for SimCLR.
Maps encoder representations to the contrastive loss space.

Architecture: 2-layer MLP with ReLU activation + optional batch norm.
"""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP Projection Head for SimCLR.

    Projects encoder representations into a lower-dimensional space
    where the contrastive loss is applied.

    Args:
        input_dim: Dimensionality of encoder output features.
        hidden_dim: Hidden layer dimension (default: 2048 per SimCLR paper).
        output_dim: Output projection dimension (default: 128).
        use_batch_norm: Whether to apply batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim)]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the projection head.

        Args:
            x: Encoder features of shape (B, input_dim).

        Returns:
            Projected features of shape (B, output_dim).
        """
        return self.net(x)
