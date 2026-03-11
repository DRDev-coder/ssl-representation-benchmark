"""
NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
==========================================================
The contrastive loss function used in SimCLR.

For a batch of N images, SimCLR generates 2N augmented views.
The loss treats (z_i, z_j) from the same image as positive pairs
and all other 2(N-1) samples as negatives.

Reference: Chen et al., "A Simple Framework for Contrastive Learning
of Visual Representations", ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

    Args:
        temperature: Temperature scaling parameter (default: 0.5).
                     Lower = sharper distribution, harder negatives.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: L2-normalized projections from view 1, shape (N, D).
            z_j: L2-normalized projections from view 2, shape (N, D).

        Returns:
            Scalar loss value.
        """
        N = z_i.size(0)
        device = z_i.device

        # Concatenate both views: [z_i; z_j] -> shape (2N, D)
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Compute pairwise cosine similarity matrix (2N x 2N)
        similarity = F.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0), dim=2
        )  # (2N, 2N)

        # Extract positive pairs: (i, i+N) and (i+N, i)
        sim_ij = torch.diag(similarity, N)   # (N,) — similarities for (i, j)
        sim_ji = torch.diag(similarity, -N)   # (N,) — similarities for (j, i)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2N,)

        # Mask to exclude self-similarity (diagonal)
        mask = (~torch.eye(2 * N, dtype=torch.bool, device=device)).float()

        # Compute loss
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        loss = -torch.log(numerator / denominator.sum(dim=1))

        return loss.mean()
