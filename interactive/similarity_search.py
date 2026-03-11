"""
Similarity Search
=================
Nearest-neighbour retrieval in embedding space using cosine similarity.

Usage as a library:
    from interactive.similarity_search import SimilarityIndex
    index = SimilarityIndex.from_npz("embeddings/cifar10_test.npz")
    indices, scores = index.query(query_embedding, k=5)

Usage from the command line (quick sanity check):
    python interactive/similarity_search.py --npz embeddings/cifar10_test.npz --query_idx 42 --k 5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class SimilarityIndex:
    """In-memory cosine-similarity search over a pre-computed embedding matrix."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray,
                 images: np.ndarray | None = None):
        # Normalise once so dot product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.labels = labels
        self.images = images

    # ── construction helpers ─────────────────────────────────────────────

    @classmethod
    def from_npz(cls, path: str) -> "SimilarityIndex":
        """Load from a .npz file produced by embedding_extractor.py."""
        data = np.load(path, allow_pickle=False)
        images = data["images"] if "images" in data else None
        return cls(data["embeddings"], data["labels"], images)

    # ── query ────────────────────────────────────────────────────────────

    def query(self, query_embedding: np.ndarray, k: int = 5):
        """
        Retrieve the k most similar items.

        Args:
            query_embedding: 1-D float array of shape (D,).
            k: Number of neighbours to return.

        Returns:
            indices:  int array of shape (k,) — row indices into the index.
            scores:   float array of shape (k,) — cosine similarity scores.
        """
        q = query_embedding.astype(np.float32).ravel()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        # Cosine similarity via matrix-vector dot product
        sims = self.embeddings @ q
        top_k = np.argsort(sims)[::-1][:k]
        return top_k, sims[top_k]

    def query_by_index(self, idx: int, k: int = 5):
        """Retrieve neighbours for an item already in the index."""
        return self.query(self.embeddings[idx], k=k + 1)  # +1 because the item matches itself

    def __len__(self):
        return len(self.embeddings)


# ── CLI ──────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity nearest-neighbour search")
    parser.add_argument("--npz", required=True, help="Path to .npz embedding file")
    parser.add_argument("--query_idx", type=int, default=0,
                        help="Index of the query image in the dataset")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbours")
    args = parser.parse_args()

    index = SimilarityIndex.from_npz(args.npz)
    print(f"Loaded index with {len(index)} embeddings "
          f"(dim={index.embeddings.shape[1]})")

    indices, scores = index.query_by_index(args.query_idx, k=args.k)

    print(f"\nQuery image #{args.query_idx}  "
          f"(class: {CIFAR10_CLASSES[index.labels[args.query_idx]]})\n")
    print(f"{'Rank':<6}{'Index':<8}{'Class':<14}{'Similarity'}")
    print("-" * 40)
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        cls = CIFAR10_CLASSES[index.labels[idx]] if index.labels[idx] < 10 else str(index.labels[idx])
        tag = " (self)" if idx == args.query_idx else ""
        print(f"{rank:<6}{idx:<8}{cls:<14}{score:.4f}{tag}")


if __name__ == "__main__":
    main()
