"""
MoCo v2 — Momentum Contrast for Unsupervised Visual Representation Learning
============================================================================
Implements MoCo v2 (He et al., 2020; Chen et al., 2020 improvements).

Key components:
  - Query encoder (online) and key encoder (momentum-updated)
  - Dynamic dictionary (queue) of encoded keys for large negative pool
  - InfoNCE contrastive loss
  - Momentum update: key_params = m * key_params + (1-m) * query_params

Reference:
  He et al., "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
  Chen et al., "Improved Baselines with Momentum Contrastive Learning", arXiv 2020
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_encoder import ResNetEncoder
from models.projection_head import ProjectionHead


class MoCoV2(nn.Module):
    """
    MoCo v2 with momentum encoder and dictionary queue.

    Args:
        backbone: ResNet variant name.
        projection_hidden_dim: Hidden dim of projection MLP.
        projection_output_dim: Output dim of projection MLP (queue feature dim).
        momentum: Momentum coefficient for key encoder update.
        queue_size: Size of the negative key queue.
        temperature: InfoNCE temperature.
    """

    METHOD_NAME = "moco"

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_hidden_dim: int = 2048,
        projection_output_dim: int = 128,
        momentum: float = 0.999,
        queue_size: int = 65536,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size

        # Query encoder (online)
        self.encoder_q = ResNetEncoder(backbone=backbone, pretrained=False)
        self.projection_q = ProjectionHead(
            input_dim=self.encoder_q.output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )

        # Key encoder (momentum-updated, no gradients)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projection_k = copy.deepcopy(self.projection_q)

        # Stop gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projection_k.parameters():
            param.requires_grad = False

        # Queue: (projection_output_dim, queue_size)
        self.register_buffer("queue", torch.randn(projection_output_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Exponential moving average update of the key encoder."""
        m = self.momentum
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # If batch doesn't fit exactly, wrap around
        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward(self, x_i, x_j):
        """
        Forward pass: compute InfoNCE loss.

        Args:
            x_i: Query view (B, C, H, W).
            x_j: Key view (B, C, H, W).

        Returns:
            Scalar InfoNCE loss.
        """
        # Query features
        q = self.projection_q(self.encoder_q(x_i))
        q = F.normalize(q, dim=1)

        # Key features (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.projection_k(self.encoder_k(x_j))
            k = F.normalize(k, dim=1)

        # Positive logits: (B, 1)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)

        # Negative logits: (B, queue_size)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())

        # Logits: (B, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positive is always index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss

    @property
    def encoder(self):
        """Return the query encoder for downstream tasks."""
        return self.encoder_q

    @property
    def feature_dim(self):
        return self.encoder_q.output_dim

    def encode(self, x):
        return self.encoder_q(x)

    def get_trainable_params(self):
        """Only online encoder + projection are trained."""
        return list(self.encoder_q.parameters()) + list(self.projection_q.parameters())

    def save_encoder(self, path: str):
        torch.save(self.encoder_q.state_dict(), path)
        print(f"MoCo encoder saved to {path}")

    def load_encoder(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.encoder_q.load_state_dict(state_dict)
        print(f"MoCo encoder loaded from {path}")

    def save_full_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_full_model(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
