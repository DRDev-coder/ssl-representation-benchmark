"""
Representation Quality Metrics
==============================
Quantitative metrics for evaluating SSL-learned representations:
  - Alignment: similarity between positive-pair embeddings
  - Uniformity: distribution of embeddings on the hypersphere
  - CKA: Centered Kernel Alignment between representation spaces
  - Feature Collapse Detection: embedding variance and covariance rank
"""

from evaluation.representation_metrics.alignment import compute_alignment
from evaluation.representation_metrics.uniformity import compute_uniformity
from evaluation.representation_metrics.cka import compute_cka, compute_cka_matrix
from evaluation.representation_metrics.collapse_detection import (
    compute_embedding_variance,
    compute_covariance_rank,
)

__all__ = [
    "compute_alignment",
    "compute_uniformity",
    "compute_cka",
    "compute_cka_matrix",
    "compute_embedding_variance",
    "compute_covariance_rank",
]
