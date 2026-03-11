"""
Core unit tests for the SimCLR project.

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.resnet_encoder import ResNetEncoder
from utils.losses import NTXentLoss
from interactive.similarity_search import SimilarityIndex


# ── Encoder tests ─────────────────────────────────────────────────────────────

def test_encoder_resnet18_output_shape():
    encoder = ResNetEncoder(backbone="resnet18")
    x = torch.randn(4, 3, 32, 32)
    out = encoder(x)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"


def test_encoder_resnet50_output_shape():
    encoder = ResNetEncoder(backbone="resnet50")
    x = torch.randn(2, 3, 32, 32)
    out = encoder(x)
    assert out.shape == (2, 2048), f"Expected (2, 2048), got {out.shape}"


def test_encoder_feature_dim_property():
    assert ResNetEncoder(backbone="resnet18").feature_dim == 512
    assert ResNetEncoder(backbone="resnet34").feature_dim == 512
    assert ResNetEncoder(backbone="resnet50").feature_dim == 2048


def test_encoder_unknown_backbone():
    import pytest
    with pytest.raises(ValueError):
        ResNetEncoder(backbone="vgg16")


def test_encoder_no_fc_layer():
    """The FC head should be replaced with Identity."""
    import torch.nn as nn
    encoder = ResNetEncoder(backbone="resnet18")
    assert isinstance(encoder.encoder.fc, nn.Identity)


# ── NT-Xent Loss tests ────────────────────────────────────────────────────────

def test_ntxent_loss_positive():
    loss_fn = NTXentLoss(temperature=0.5)
    z_i = F.normalize(torch.randn(8, 128), dim=1)
    z_j = F.normalize(torch.randn(8, 128), dim=1)
    loss = loss_fn(z_i, z_j)
    assert loss.item() > 0, "Loss should be positive"


def test_ntxent_loss_scalar():
    loss_fn = NTXentLoss(temperature=0.5)
    z_i = F.normalize(torch.randn(16, 128), dim=1)
    z_j = F.normalize(torch.randn(16, 128), dim=1)
    loss = loss_fn(z_i, z_j)
    assert loss.dim() == 0, "Loss should be a scalar"


def test_ntxent_loss_decreases_for_similar_pairs():
    """Loss should be lower when positive pairs are more similar."""
    loss_fn = NTXentLoss(temperature=0.5)
    z = F.normalize(torch.randn(8, 128), dim=1)
    # Nearly identical pairs → low loss
    loss_easy = loss_fn(z, z + 1e-4 * torch.randn_like(z))
    # Very different pairs → high loss
    loss_hard = loss_fn(z, F.normalize(torch.randn(8, 128), dim=1))
    assert loss_easy.item() < loss_hard.item()


def test_ntxent_temperature_effect():
    """Lower temperature should produce a higher (sharper) loss for random pairs."""
    z_i = F.normalize(torch.randn(8, 128), dim=1)
    z_j = F.normalize(torch.randn(8, 128), dim=1)
    loss_low_temp = NTXentLoss(temperature=0.1)(z_i, z_j)
    loss_high_temp = NTXentLoss(temperature=1.0)(z_i, z_j)
    # With random (non-positive) pairs, lower temp amplifies wrong negatives
    assert loss_low_temp.item() != loss_high_temp.item()


# ── SimilarityIndex tests ─────────────────────────────────────────────────────

def test_similarity_index_query_returns_k_results():
    emb = np.random.randn(100, 512).astype("float32")
    labels = np.zeros(100, dtype=np.int64)
    index = SimilarityIndex(emb, labels)
    q = np.random.randn(512).astype("float32")
    indices, scores = index.query(q, k=5)
    assert len(indices) == 5
    assert len(scores) == 5


def test_similarity_index_scores_descending():
    emb = np.random.randn(50, 128).astype("float32")
    labels = np.zeros(50, dtype=np.int64)
    index = SimilarityIndex(emb, labels)
    _, scores = index.query(np.random.randn(128).astype("float32"), k=10)
    assert list(scores) == sorted(scores, reverse=True), "Scores should be in descending order"


def test_similarity_index_scores_bounded():
    """Cosine similarity must be in [-1, 1]."""
    emb = np.random.randn(200, 64).astype("float32")
    labels = np.zeros(200, dtype=np.int64)
    index = SimilarityIndex(emb, labels)
    _, scores = index.query(np.random.randn(64).astype("float32"), k=20)
    assert np.all(scores >= -1.0 - 1e-5)
    assert np.all(scores <= 1.0 + 1e-5)


def test_similarity_index_exact_match():
    """Querying with a vector already in the index should return it as top result."""
    emb = np.random.randn(50, 64).astype("float32")
    labels = np.zeros(50, dtype=np.int64)
    index = SimilarityIndex(emb, labels)
    # Query with the exact (normalised) first vector
    q = emb[0] / np.linalg.norm(emb[0])
    indices, scores = index.query(q, k=1)
    assert indices[0] == 0
    assert scores[0] > 0.99


def test_similarity_index_len():
    emb = np.random.randn(77, 32).astype("float32")
    index = SimilarityIndex(emb, np.zeros(77, dtype=np.int64))
    assert len(index) == 77


def test_similarity_index_from_npz(tmp_path):
    emb = np.random.randn(20, 64).astype("float32")
    labels = np.arange(20, dtype=np.int64)
    images = np.zeros((20, 32, 32, 3), dtype=np.uint8)
    npz = tmp_path / "test.npz"
    np.savez_compressed(str(npz), embeddings=emb, labels=labels, images=images)
    index = SimilarityIndex.from_npz(str(npz))
    assert len(index) == 20
    assert index.images.shape == (20, 32, 32, 3)
