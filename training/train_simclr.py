"""
SimCLR Pretraining Script
=========================
Trains a ResNet encoder + projection head using the SimCLR contrastive
learning framework on unlabeled data.

Key features:
  - Hydra configuration management for reproducible experiments
  - Mixed precision training (AMP) for RTX 5060 efficiency
  - Gradient accumulation for effective larger batch sizes
  - Cosine annealing LR schedule with warmup
  - Proper dual-view augmentation (two independent augmentations per image)
  - MLflow experiment tracking with full config logging
  - Periodic checkpointing

Usage:
    # Default (STL-10, ResNet-18, Adam)
    python training/train_simclr.py

    # Override from CLI
    python training/train_simclr.py dataset=cifar10 model=resnet34 batch_size=256

    # Hyperparameter sweep (Hydra multirun)
    python training/train_simclr.py -m temperature=0.3,0.5,0.7 batch_size=128,256

    # Use experiment preset
    python training/train_simclr.py +experiments=simclr_stl10
"""

import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path (Hydra changes cwd)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.simclr_model import SimCLR
from datasets.stl10_dataset import get_stl10_pretrain
from datasets.cifar10_dataset import get_cifar10_pretrain
from utils.losses import NTXentLoss
from utils.device import get_device, print_memory_stats
from utils.seed import set_seed
from mlops.mlflow_logger import MLflowLogger
from mlops.mlflow_tracker import ExperimentTracker


def _resolve_path(path: str) -> str:
    """Resolve a relative path against project root."""
    if not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, loader_len):
    """
    Cosine annealing LR schedule with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        loader_len: Number of batches per epoch (for step-level scheduling).

    Returns:
        LambdaLR scheduler.
    """
    warmup_steps = warmup_epochs * loader_len
    total_steps = total_epochs * loader_len

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device,
    accumulation_steps, use_amp, epoch, log_every=10, logger=None,
):
    """
    Train for one epoch with optional gradient accumulation and AMP.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, ((x_i, x_j), _) in enumerate(loader):
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast("cuda", enabled=use_amp):
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j) / accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        batch_loss = loss.item() * accumulation_steps
        total_loss += batch_loss
        num_batches += 1

        # Logging
        if batch_idx % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Batch [{batch_idx}/{len(loader)}] | "
                f"Loss: {batch_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if logger is not None:
                global_step = epoch * len(loader) + batch_idx
                logger.log_metric("train/batch_loss", batch_loss, step=global_step)
                logger.log_metric("train/learning_rate", current_lr, step=global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_simclr(cfg: DictConfig):
    """
    Core SimCLR pretraining logic using Hydra config.

    Accesses configuration values directly from the Hydra DictConfig:
      - cfg.batch_size, cfg.epochs, cfg.temperature  (top-level)
      - cfg.dataset.name, cfg.dataset.image_size      (dataset config group)
      - cfg.model.backbone, cfg.model.projection_*    (model config group)
      - cfg.optimizer.lr, cfg.optimizer.weight_decay   (optimizer config group)
      - cfg.training.warmup_epochs, cfg.training.*     (training config group)

    Args:
        cfg: Hydra DictConfig composed from YAML config files.

    Returns:
        Trained SimCLR model.
    """
    # Setup
    set_seed(cfg.seed)
    device = get_device()

    # Resolve paths
    data_dir = _resolve_path(cfg.paths.data_dir)
    checkpoint_dir = _resolve_path(cfg.paths.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Read config values
    dataset_name = cfg.dataset.name
    image_size = cfg.dataset.image_size
    accumulation_steps = cfg.training.get("accumulation_steps", 1)
    log_every = cfg.training.get("log_every", 10)
    save_every = cfg.training.get("save_every", 50)

    print("\n" + "=" * 60)
    print("SimCLR PRETRAINING")
    print("=" * 60)
    print(f"Dataset:      {dataset_name}")
    print(f"Image size:   {image_size}")
    print(f"Backbone:     {cfg.model.backbone}")
    print(f"Batch size:   {cfg.batch_size}")
    print(f"Accum steps:  {accumulation_steps}")
    print(f"Effective BS: {cfg.batch_size * accumulation_steps}")
    print(f"Epochs:       {cfg.epochs}")
    print(f"Optimizer:    {cfg.optimizer.name}")
    print(f"LR:           {cfg.optimizer.lr}")
    print(f"Temperature:  {cfg.temperature}")
    print(f"AMP:          {cfg.use_amp}")
    print("=" * 60 + "\n")

    # ── Dataset ──────────────────────────────────────────────────
    if dataset_name == "stl10":
        train_loader = get_stl10_pretrain(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        train_loader = get_cifar10_pretrain(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    print(f"Training batches per epoch: {len(train_loader)}")

    # ── Model ────────────────────────────────────────────────────
    model = SimCLR(
        backbone=cfg.model.backbone,
        projection_hidden_dim=cfg.model.projection_hidden_dim,
        projection_output_dim=cfg.model.projection_output_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.1f}M")

    # ── Loss, Optimizer, Scheduler ───────────────────────────────
    criterion = NTXentLoss(temperature=cfg.temperature)

    if cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.get("momentum", 0.9),
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    warmup_epochs = cfg.training.get("warmup_epochs", 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=cfg.epochs,
        loader_len=len(train_loader),
    )

    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # ── Resume from checkpoint ───────────────────────────────────
    start_epoch = 0
    best_loss = float("inf")
    resume_from = cfg.training.get("resume_from", None)

    if resume_from and os.path.exists(_resolve_path(resume_from)):
        resume_path = _resolve_path(resume_from)
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ── MLflow Logger ────────────────────────────────────────────
    logger = None
    tracker = None
    if cfg.mlflow.enabled:
        logger = MLflowLogger(experiment_name=cfg.mlflow.experiment_name)
        logger.start_run(run_name=f"simclr_{dataset_name}_{cfg.epochs}ep")
        # Log full Hydra config to MLflow for reproducibility
        logger.log_hydra_config(cfg)
        # High-level tracker for dataset info & checkpoint artifacts
        tracker = ExperimentTracker(cfg)
        tracker._run = logger._run  # share the active run
        tracker.active = logger.active
        tracker.log_dataset_info(num_samples=len(train_loader.dataset))

    # ── Training Loop ────────────────────────────────────────────
    print("\nStarting training...\n")
    training_start = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()

        avg_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            accumulation_steps=accumulation_steps,
            use_amp=cfg.use_amp,
            epoch=epoch,
            log_every=log_every,
            logger=logger,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log epoch metrics
        if logger is not None:
            logger.log_metric("train/epoch_loss", avg_loss, step=epoch)
            logger.log_metric("train/epoch_time_s", epoch_time, step=epoch)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_path = os.path.join(checkpoint_dir, "simclr_encoder_best.pth")
            model.save_encoder(best_encoder_path)

            best_model_path = os.path.join(checkpoint_dir, "simclr_full_best.pth")
            model.save_full_model(best_model_path)

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"simclr_checkpoint_ep{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ── Final Save ───────────────────────────────────────────────
    total_time = time.time() - training_start
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best loss: {best_loss:.4f}")

    # Save final encoder
    final_encoder_path = os.path.join(checkpoint_dir, "simclr_encoder_final.pth")
    model.save_encoder(final_encoder_path)

    final_model_path = os.path.join(checkpoint_dir, "simclr_full_final.pth")
    model.save_full_model(final_model_path)

    if logger is not None:
        logger.log_metric("train/total_time_hours", total_time / 3600)
        logger.log_metric("train/best_loss", best_loss)
        logger.log_artifact(final_encoder_path)
        # Log model checkpoints as artifacts
        best_enc = os.path.join(checkpoint_dir, "simclr_encoder_best.pth")
        if os.path.isfile(best_enc):
            logger.log_artifact(best_enc, artifact_path="model_checkpoints")
        logger.log_artifact(final_encoder_path, artifact_path="model_checkpoints")
        logger.end_run()

    print_memory_stats()
    print("\nDone!")

    return model


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Hydra entry point for SimCLR pretraining."""
    print(OmegaConf.to_yaml(cfg))
    train_simclr(cfg)


if __name__ == "__main__":
    main()
