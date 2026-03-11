"""
Interactive Embedding Explorer — Streamlit App
===============================================
Upload an image (or pick one from the dataset), compute its SimCLR
embedding, and retrieve the most semantically similar images.

Launch:
    streamlit run interactive/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from augmentations.simclr_augmentations import get_eval_transform
from models.resnet_encoder import ResNetEncoder
from interactive.similarity_search import SimilarityIndex

# ── constants ────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CHESTXRAY_CLASSES = ["no_finding", "pathology"]

ENCODER_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ── cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource
def load_encoder(encoder_path: str, backbone: str, device: str):
    encoder = ResNetEncoder(backbone=backbone).to(device)
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


@st.cache_resource
def load_index(npz_path: str) -> SimilarityIndex:
    return SimilarityIndex.from_npz(npz_path)


# ── embedding for a single PIL image ────────────────────────────────────────

@torch.no_grad()
def embed_image(pil_image: Image.Image, encoder, transform, device) -> np.ndarray:
    img_rgb = pil_image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    feat = encoder(tensor)
    feat = F.normalize(feat, dim=1)
    return feat.cpu().numpy().ravel()


# ── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SimCLR Embedding Explorer", layout="wide")
    st.title("🔍 SimCLR Embedding Explorer")
    st.markdown(
        "Upload an image and discover what the self-supervised encoder considers "
        "semantically similar — **no labels were used during training**."
    )

    # ── sidebar controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        # Encoder selection
        encoder_files = sorted(
            f for f in os.listdir(ENCODER_DIR)
            if f.endswith(".pth") and "encoder" in f
        ) if os.path.isdir(ENCODER_DIR) else []

        if not encoder_files:
            st.error("No encoder checkpoints found in `checkpoints/`.  "
                     "Run pretraining first.")
            return

        encoder_name = st.selectbox("Encoder checkpoint", encoder_files)
        encoder_path = os.path.join(ENCODER_DIR, encoder_name)

        backbone = st.selectbox("Backbone", ["resnet18", "resnet34", "resnet50"])
        dataset = st.selectbox("Reference dataset", ["cifar10", "stl10", "chestxray"])
        k = st.slider("Number of neighbours", min_value=1, max_value=25, value=8)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"Device: **{device_str}**")

    # ── check / build embedding index ────────────────────────────────────────
    image_size = {"cifar10": 32, "stl10": 96, "chestxray": 224}.get(dataset, 32)
    npz_path = os.path.join(EMBEDDING_DIR, f"{dataset}_test.npz")

    if not os.path.exists(npz_path):
        st.warning(
            f"Embedding file not found at `{npz_path}`.  \n"
            "Generate it first:\n"
            "```\n"
            f"python interactive/embedding_extractor.py "
            f"--encoder {encoder_path} --dataset {dataset}\n"
            "```"
        )
        return

    index = load_index(npz_path)
    encoder = load_encoder(encoder_path, backbone, device_str)
    transform = get_eval_transform(image_size)

    class_names = {
        "cifar10": CIFAR10_CLASSES,
        "chestxray": CHESTXRAY_CLASSES,
    }.get(dataset, CIFAR10_CLASSES)

    # ── input mode ───────────────────────────────────────────────────────
    st.subheader("1 — Choose a query image")
    mode = st.radio("Source", ["Upload an image", "Pick from dataset"],
                    horizontal=True)

    query_embedding = None
    query_pil = None

    if mode == "Upload an image":
        uploaded = st.file_uploader("Upload a PNG / JPG image",
                                    type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            query_pil = Image.open(uploaded)
            st.image(query_pil, caption="Query image", width=150)
            query_embedding = embed_image(query_pil, encoder, transform, device_str)

    else:  # pick from dataset
        max_idx = len(index) - 1
        idx = st.number_input("Image index", min_value=0, max_value=max_idx,
                              value=0, step=1)
        if index.images is not None:
            query_pil = Image.fromarray(index.images[idx])
            label = index.labels[idx]
            cls_name = class_names[label] if label < len(class_names) else str(label)
            st.image(query_pil, caption=f"#{idx}  ({cls_name})", width=150)
        query_embedding = index.embeddings[idx]

    # ── retrieve neighbours ──────────────────────────────────────────────
    if query_embedding is not None:
        st.subheader("2 — Most similar images (by cosine similarity)")
        indices, scores = index.query(query_embedding, k=k)

        cols = st.columns(min(k, 8))
        for rank, (i, score) in enumerate(zip(indices, scores)):
            col = cols[rank % len(cols)]
            with col:
                if index.images is not None:
                    img = Image.fromarray(index.images[i])
                    label = index.labels[i]
                    cls = class_names[label] if label < len(class_names) else str(label)
                    st.image(img, use_container_width=True)
                    st.caption(f"**#{i}** {cls}\nsim = {score:.3f}")
                else:
                    st.write(f"#{i}  sim={score:.3f}")

        # ── similarity distribution ──────────────────────────────────────
        with st.expander("Similarity score distribution"):
            import matplotlib.pyplot as plt
            all_sims = (index.embeddings @ query_embedding).ravel()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(all_sims, bins=80, color="#2176AE", edgecolor="white",
                    linewidth=0.4, alpha=0.85)
            for s in scores[:3]:
                ax.axvline(s, color="#D64045", linewidth=1.2, linestyle="--")
            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Count")
            ax.set_title("Query vs. all dataset embeddings")
            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
