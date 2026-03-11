"""
Alignment Metric
================
Measures the closeness of embeddings from positive pairs (two augmented
views of the same image). Lower alignment = better representation.

Reference: Wang & Isola, "Understanding Contrastive Representation
Learning through Alignment and Uniformity on the Hypersphere", ICML 2020.

    alignment(f; α) = E_{(x,x+)~p_pos} [ ||f(x) - f(x+)||^α ]

Default α = 2 (squared Euclidean distance).
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_alignment(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Compute alignment between two sets of L2-normalised embeddings.

    Args:
        z1: Embeddings from view 1, shape (N, D). Will be L2-normalised.
        z2: Embeddings from view 2, shape (N, D). Will be L2-normalised.
        alpha: Exponent for distance (default 2 = squared distance).

    Returns:
        Alignment score (float). Lower is better.
    """
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    return (z1 - z2).norm(dim=1).pow(alpha).mean().item()
