"""
Unified SSL Pretraining Script
==============================
Trains any SSL method (SimCLR, MoCo v2, BYOL) using the unified interface.

Usage:
    python training/train_ssl.py method=simclr dataset=cifar10
    python training/train_ssl.py method=moco dataset=cifar10 temperature=0.07
    python training/train_ssl.py method=byol dataset=cifar10
"""

import os
import sys
import time
import math
from pathlib import Path

import torch
from torch.amp import autocast, GradScaler

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ssl_methods import SimCLRMethod, MoCoV2, BYOL
from datasets.cifar10_dataset import get_cifar10_pretrain
from datasets.stl10_dataset import get_stl10_pretrain
from utils.device import get_device, print_memory_stats


def _get_pretrain_loader(cfg, data_dir):
    """Build pretrain dataloader for any supported dataset."""
    if cfg.dataset.name == "stl10":
        return get_stl10_pretrain(
            data_dir=data_dir, image_size=cfg.dataset.image_size,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    elif cfg.dataset.name == "chestxray":
        from datasets.chestxray_dataset import get_chestxray_pretrain
        chestxray_dir = os.path.join(PROJECT_ROOT, "CXR8")
        return get_chestxray_pretrain(
            data_dir=chestxray_dir, image_size=cfg.dataset.image_size,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        return get_cifar10_pretrain(
            data_dir=data_dir, image_size=cfg.dataset.image_size,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
from utils.seed import set_seed
from mlops.mlflow_logger import MLflowLogger
from mlops.mlflow_tracker import ExperimentTracker


SSL_REGISTRY = {
    "simclr": SimCLRMethod,
    "moco": MoCoV2,
    "byol": BYOL,
}


def _resolve_path(path: str) -> str:
    if not os.path.isabs(path):
        return os.path.join(PROJECT_ROOT, path)
    return path


def build_ssl_model(cfg: DictConfig):
    """Instantiate an SSL method from config."""
    method_name = cfg.method.name if hasattr(cfg.method, "name") else str(cfg.method)
    if method_name not in SSL_REGISTRY:
        raise ValueError(f"Unknown SSL method '{method_name}'. Choose from: {list(SSL_REGISTRY.keys())}")

    cls = SSL_REGISTRY[method_name]
    kwargs = dict(
        backbone=cfg.model.backbone,
        projection_hidden_dim=cfg.model.projection_hidden_dim,
        projection_output_dim=cfg.model.projection_output_dim,
    )

    if method_name == "simclr":
        kwargs["temperature"] = cfg.temperature
    elif method_name == "moco":
        kwargs["temperature"] = cfg.method.get("moco_temperature", 0.07)
        kwargs["momentum"] = cfg.method.get("moco_momentum", 0.999)
        kwargs["queue_size"] = cfg.method.get("moco_queue_size", 65536)
    elif method_name == "byol":
        kwargs["momentum"] = cfg.method.get("byol_momentum", 0.996)
        kwargs["predictor_hidden_dim"] = cfg.method.get("byol_predictor_hidden_dim", 512)

    return cls(**kwargs)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, loader_len):
    warmup_steps = warmup_epochs * loader_len
    total_steps = total_epochs * loader_len

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = get_device()

    method_name = cfg.method.name if hasattr(cfg.method, "name") else str(cfg.method)
    data_dir = _resolve_path(cfg.paths.data_dir)
    checkpoint_dir = _resolve_path(cfg.paths.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    accumulation_steps = cfg.training.get("accumulation_steps", 1)
    warmup_epochs = cfg.training.get("warmup_epochs", 10)
    save_every = cfg.training.get("save_every", 50)
    log_every = cfg.training.get("log_every", 10)

    print("\n" + "=" * 60)
    print(f"SSL PRETRAINING — {method_name.upper()}")
    print("=" * 60)
    print(f"Dataset:      {cfg.dataset.name}")
    print(f"Backbone:     {cfg.model.backbone}")
    print(f"Batch size:   {cfg.batch_size}")
    print(f"Epochs:       {cfg.epochs}")
    print(f"AMP:          {cfg.use_amp}")
    print("=" * 60 + "\n")

    # Dataset
    train_loader = _get_pretrain_loader(cfg, data_dir)

    # Model
    model = build_ssl_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.1f}M")

    # Optimizer
    trainable_params = model.get_trainable_params()
    if cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params, lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.get("momentum", 0.9),
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_params, lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, cfg.epochs, len(train_loader))
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # MLflow
    logger = None
    if cfg.mlflow.enabled:
        logger = MLflowLogger(experiment_name=f"ssl_{method_name}")
        logger.start_run(run_name=f"{method_name}_{cfg.dataset.name}_{cfg.model.backbone}")
        logger.log_hydra_config(cfg)
        # Log dataset info
        tracker = ExperimentTracker(cfg)
        tracker._run = logger._run
        tracker.active = logger.active
        tracker.log_dataset_info(num_samples=len(train_loader.dataset))

    # Training loop
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, ((x_i, x_j), _) in enumerate(train_loader):
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)

            with autocast("cuda", enabled=cfg.use_amp):
                loss = model(x_i, x_j) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            batch_loss = loss.item() * accumulation_steps
            total_loss += batch_loss
            num_batches += 1

            if batch_idx % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Batch [{batch_idx}/{len(train_loader)}] | Loss: {batch_loss:.4f} | LR: {lr:.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | Loss: {avg_loss:.4f} | Time: {elapsed/60:.1f}m")

        if logger:
            logger.log_metric(f"{method_name}/epoch_loss", avg_loss, step=epoch)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_encoder(os.path.join(checkpoint_dir, f"{method_name}_encoder_best.pth"))

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            model.save_full_model(os.path.join(checkpoint_dir, f"{method_name}_full_epoch{epoch+1}.pth"))

    # Final save
    model.save_encoder(os.path.join(checkpoint_dir, f"{method_name}_encoder_final.pth"))
    model.save_full_model(os.path.join(checkpoint_dir, f"{method_name}_full_final.pth"))

    total_time = (time.time() - start_time) / 3600
    print(f"\n{method_name.upper()} pretraining complete! Time: {total_time:.2f}h | Best loss: {best_loss:.4f}")

    if logger:
        logger.log_metric(f"{method_name}/best_loss", best_loss)
        logger.log_metric(f"{method_name}/training_hours", total_time)
        # Log model checkpoints as artifacts
        best_enc = os.path.join(checkpoint_dir, f"{method_name}_encoder_best.pth")
        final_enc = os.path.join(checkpoint_dir, f"{method_name}_encoder_final.pth")
        for p in [best_enc, final_enc]:
            if os.path.isfile(p):
                logger.log_artifact(p, artifact_path="model_checkpoints")
        logger.end_run()


if __name__ == "__main__":
    main()
