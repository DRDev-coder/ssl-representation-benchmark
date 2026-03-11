"""
Feature Collapse Detection
===========================
Detects representation collapse in SSL models by measuring:
  1. Embedding variance: per-dimension variance, averaged over all dims
  2. Covariance rank: effective rank of the covariance matrix (via SVD)

Low variance or low rank indicates the encoder has collapsed to a
degenerate solution (all embeddings are identical or span a low-dim subspace).
"""

import numpy as np


def compute_embedding_variance(features: np.ndarray) -> dict:
    """
    Compute statistics about embedding variance.

    Args:
        features: Shape (N, D) — one feature vector per sample.

    Returns:
        dict with keys:
            - mean_variance: average per-dimension variance
            - min_variance: minimum per-dimension variance
            - max_variance: maximum per-dimension variance
            - per_dim_variance: array of shape (D,)
    """
    var = np.var(features, axis=0)
    return {
        "mean_variance": float(np.mean(var)),
        "min_variance": float(np.min(var)),
        "max_variance": float(np.max(var)),
        "per_dim_variance": var,
    }


def compute_covariance_rank(features: np.ndarray, threshold: float = 0.01) -> dict:
    """
    Compute the effective rank of the embedding covariance matrix.

    Uses the singular values of the centered feature matrix. The effective
    rank is the number of singular values above `threshold * max_sv`.

    Args:
        features: Shape (N, D).
        threshold: Fraction of the largest singular value below which
                   a dimension is considered "dead".

    Returns:
        dict with keys:
            - effective_rank: number of significant singular values
            - total_dims: D
            - rank_ratio: effective_rank / D
            - singular_values: top-20 singular values (for inspection)
    """
    centered = features - features.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)

    cutoff = threshold * sv[0]
    effective_rank = int(np.sum(sv > cutoff))

    return {
        "effective_rank": effective_rank,
        "total_dims": features.shape[1],
        "rank_ratio": float(effective_rank / features.shape[1]),
        "singular_values": sv[:20].tolist(),
    }
