# ============================================================
# SimCLR Docker Environment
# ============================================================
# Multi-stage build for reproducibility and deployment.
#
# Build:
#   docker build -t simclr-project -f docker/Dockerfile .
#
# Run (with GPU):
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results simclr-project
#
# Run specific script:
#   docker run --gpus all simclr-project python -m training.train_simclr --epochs 10
# ============================================================

# ── Base image with CUDA support ─────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Dependencies ─────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application ──────────────────────────────────────────────
FROM deps AS app

# Copy project code
COPY configs/ configs/
COPY models/ models/
COPY augmentations/ augmentations/
COPY datasets/ datasets/
COPY training/ training/
COPY evaluation/ evaluation/
COPY utils/ utils/
COPY mlops/ mlops/
COPY scripts/ scripts/

# Create output directories
RUN mkdir -p data results checkpoints mlruns

# Expose MLflow UI port
EXPOSE 5000

# Default: run the full pipeline
CMD ["python", "-m", "training.train_simclr", "--dataset", "stl10", "--epochs", "200"]
