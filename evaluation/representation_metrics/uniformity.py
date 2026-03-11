"""
Uniformity Metric
=================
Measures how uniformly embeddings are distributed on the unit hypersphere.
Lower uniformity value = more uniform distribution (better).

Reference: Wang & Isola, "Understanding Contrastive Representation
Learning through Alignment and Uniformity on the Hypersphere", ICML 2020.

    uniformity(f; t) = log E_{(x,y)~p_data} [ exp(-t ||f(x) - f(y)||^2) ]

Default t = 2.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_uniformity(z: torch.Tensor, t: float = 2.0, max_samples: int = 5000) -> float:
    """
    Compute uniformity of L2-normalised embeddings.

    Args:
        z: Embeddings of shape (N, D). Will be L2-normalised.
        t: Temperature scaling factor (default 2).
        max_samples: Subsample to this many embeddings to avoid OOM on
                     large datasets (pairwise computation is O(N^2)).

    Returns:
        Uniformity score (float). Lower (more negative) is better.
    """
    z = F.normalize(z.float(), dim=1)

    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0])[:max_samples]
        z = z[idx]

    # Pairwise squared distances
    sq_pdist = torch.cdist(z, z, p=2).pow(2)

    # Mask diagonal (self-distances = 0)
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sq_pdist = sq_pdist[mask]

    return torch.log(torch.exp(-t * sq_pdist).mean()).item()
