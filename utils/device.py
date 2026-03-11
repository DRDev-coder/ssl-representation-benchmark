"""
Device Utilities
================
Centralized device management with CUDA optimizations for RTX 5060.
"""

import torch
from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """Device configuration details."""
    device: torch.device
    device_name: str
    cuda_available: bool
    gpu_name: str = "N/A"
    gpu_memory_gb: float = 0.0


def get_device(verbose: bool = True) -> torch.device:
    """
    Get the best available device (CUDA > CPU).

    Also enables cuDNN benchmark mode for fixed-size inputs.

    Args:
        verbose: Whether to print device information.

    Returns:
        torch.device — CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Enable cuDNN benchmark for faster training with fixed input sizes
        torch.backends.cudnn.benchmark = True

        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"cuDNN benchmark: enabled")
    else:
        device = torch.device("cpu")
        if verbose:
            print("CUDA not available. Using CPU.")

    return device


def get_device_config() -> DeviceConfig:
    """Get detailed device configuration info."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceConfig(
            device=torch.device("cuda"),
            device_name="cuda",
            cuda_available=True,
            gpu_name=props.name,
            gpu_memory_gb=props.total_memory / (1024 ** 3),
        )
    else:
        return DeviceConfig(
            device=torch.device("cpu"),
            device_name="cpu",
            cuda_available=False,
        )


def print_memory_stats():
    """Print current GPU memory usage (useful for debugging OOM)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory: {allocated:.2f} GB allocated / "
              f"{reserved:.2f} GB reserved / {total:.1f} GB total")
