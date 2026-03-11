"""SimCLR Project Configuration Package.

This package provides two configuration systems:

1. **Hydra YAML configs** (configs/*.yaml) — PRIMARY:
   Composable YAML configs with CLI overrides, sweeps, and
   automatic output directories.

   Config groups:
     - dataset/    (stl10, cifar10)
     - model/      (resnet18, resnet34, resnet50)
     - training/   (simclr, supervised, linear_eval, finetune)
     - optimizer/  (adam, sgd)
     - experiments/ (preset override bundles)

   Example: python training/train_simclr.py dataset=cifar10 batch_size=256

2. **Dataclass configs** (configs/simclr_config.py) — LEGACY:
   Python dataclasses for programmatic configuration.
"""
