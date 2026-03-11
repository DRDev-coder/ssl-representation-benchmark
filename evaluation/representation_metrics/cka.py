"""
Centered Kernel Alignment (CKA)
================================
Compares the similarity of two representation spaces. CKA is invariant
to orthogonal transformations and isotropic scaling, making it ideal
for comparing representations learned by different SSL methods.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.

Linear CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F · ||Y^T Y||_F)
"""

import numpy as np


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: H K H where H = I - 1/n * 11^T."""
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two sets of representations.

    Args:
        X: Representations from model A, shape (N, D1).
        Y: Representations from model B, shape (N, D2).

    Returns:
        CKA similarity score in [0, 1]. Higher = more similar.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    cross = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")

    if norm_x < 1e-10 or norm_y < 1e-10:
        return 0.0
    return float(cross / (norm_x * norm_y))


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Alias for linear_cka."""
    return linear_cka(X, Y)


def compute_cka_matrix(features_dict: dict) -> tuple:
    """
    Compute pairwise CKA matrix for multiple methods.

    Args:
        features_dict: {method_name: np.ndarray of shape (N, D)}.

    Returns:
        (method_names, cka_matrix) where cka_matrix is (M, M) np.ndarray.
    """
    methods = list(features_dict.keys())
    n = len(methods)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            score = linear_cka(features_dict[methods[i]], features_dict[methods[j]])
            matrix[i, j] = score
            matrix[j, i] = score

    return methods, matrix
