"""
MLflow Experiment Tracker
=========================
High-level experiment tracking API built on top of MLflowLogger.

Provides convenience functions for the common workflow:
    start_experiment  -> log_parameters -> log_metrics -> log_artifacts -> end_experiment

Automatically detects experiment names from dataset/method/training mode,
logs dataset statistics, Hydra configs, and model checkpoints.

Usage:
    from mlops.mlflow_tracker import ExperimentTracker

    tracker = ExperimentTracker(cfg)
    tracker.start_experiment()
    tracker.log_parameters()                 # logs full Hydra config
    tracker.log_dataset_info()               # logs dataset stats
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.log_model_checkpoint("checkpoints/encoder.pth")
    tracker.log_artifacts("results/")
    tracker.end_experiment()
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Convenience functions (procedural API)
# ---------------------------------------------------------------------------

_active_tracker: Optional["ExperimentTracker"] = None


def start_experiment(
    cfg=None,
    experiment_name: str = None,
    run_name: str = None,
    tracking_uri: str = None,
) -> "ExperimentTracker":
    """Create and start an ExperimentTracker instance.

    If *cfg* (Hydra DictConfig) is provided, the experiment name and run name
    are inferred automatically from the config.
    """
    global _active_tracker
    tracker = ExperimentTracker(
        cfg=cfg,
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
    )
    tracker.start_experiment()
    _active_tracker = tracker
    return tracker


def log_parameters(params: Dict[str, Any] = None, cfg=None) -> None:
    """Log parameters to the active experiment."""
    if _active_tracker is not None:
        _active_tracker.log_parameters(params=params, cfg=cfg)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log metrics to the active experiment."""
    if _active_tracker is not None:
        _active_tracker.log_metrics(metrics, step=step)


def log_artifacts(path: str, artifact_path: Optional[str] = None) -> None:
    """Log an artifact file or directory to the active experiment."""
    if _active_tracker is not None:
        _active_tracker.log_artifacts(path, artifact_path=artifact_path)


def end_experiment() -> None:
    """End the currently active experiment run."""
    global _active_tracker
    if _active_tracker is not None:
        _active_tracker.end_experiment()
        _active_tracker = None


# ---------------------------------------------------------------------------
# ExperimentTracker — OOP API
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """High-level experiment tracker wrapping MLflow.

    Parameters
    ----------
    cfg : DictConfig, optional
        Hydra config from which experiment name / run name / parameters are
        inferred automatically.
    experiment_name : str, optional
        Override experiment name (otherwise derived from cfg).
    run_name : str, optional
        Override run name (otherwise derived from cfg).
    tracking_uri : str, optional
        MLflow tracking URI.  Defaults to ``mlruns/`` under project root.
    """

    def __init__(
        self,
        cfg=None,
        experiment_name: str = None,
        run_name: str = None,
        tracking_uri: str = None,
    ):
        self.cfg = cfg
        self.active = MLFLOW_AVAILABLE
        self._run = None

        # Derive names from config if not provided
        self.experiment_name = experiment_name or self._infer_experiment_name(cfg)
        self.run_name = run_name or self._infer_run_name(cfg)

        if self.active:
            uri = tracking_uri
            if uri is None:
                mlruns_path = Path(PROJECT_ROOT) / "mlruns"
                mlruns_path.mkdir(parents=True, exist_ok=True)
                uri = mlruns_path.as_uri()
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.experiment_name)

    # ── lifecycle ────────────────────────────────────────────────────
    def start_experiment(self) -> "ExperimentTracker":
        """Start a new MLflow run."""
        if self.active:
            self._run = mlflow.start_run(run_name=self.run_name)
            print(f"[MLflow] Run started: {self.run_name} "
                  f"(experiment: {self.experiment_name})")

            # Automatically log cfg if available
            if self.cfg is not None:
                self.log_parameters(cfg=self.cfg)
        return self

    def end_experiment(self) -> None:
        """End the current MLflow run."""
        if self.active and self._run is not None:
            mlflow.end_run()
            self._run = None
            print("[MLflow] Run ended.")

    # ── parameters ───────────────────────────────────────────────────
    def log_parameters(
        self,
        params: Dict[str, Any] = None,
        cfg=None,
    ) -> None:
        """Log parameters.

        If *cfg* is a Hydra DictConfig it is flattened into dot-separated
        keys and logged together with an artifact copy of the full YAML.
        Plain dicts are logged directly.
        """
        if not self.active:
            return

        # Log plain dict
        if params:
            safe = {k: str(v)[:500] for k, v in params.items()}
            mlflow.log_params(safe)

        # Log Hydra config
        target_cfg = cfg or self.cfg
        if target_cfg is not None and OMEGACONF_AVAILABLE and isinstance(target_cfg, DictConfig):
            resolved = OmegaConf.to_container(target_cfg, resolve=True)
            flat = _flatten_dict(resolved)
            safe_flat = {k: str(v)[:500] for k, v in flat.items()}
            mlflow.log_params(safe_flat)

            # Save full config YAML as artifact
            yaml_str = OmegaConf.to_yaml(target_cfg)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix="hydra_config_", delete=False
            ) as f:
                f.write(yaml_str)
                tmp = f.name
            mlflow.log_artifact(tmp, artifact_path="configs")
            os.remove(tmp)

    # ── metrics ──────────────────────────────────────────────────────
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log one or more numeric metrics."""
        if self.active:
            mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if self.active:
            mlflow.log_metric(key, value, step=step)

    # ── artifacts ────────────────────────────────────────────────────
    def log_artifacts(
        self,
        path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a file or directory as an MLflow artifact."""
        if not self.active:
            return
        if os.path.isdir(path):
            mlflow.log_artifacts(path, artifact_path)
        elif os.path.isfile(path):
            mlflow.log_artifact(path, artifact_path)

    def log_model_checkpoint(self, checkpoint_path: str) -> None:
        """Log a model checkpoint (.pth) as an MLflow artifact."""
        if self.active and os.path.isfile(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, artifact_path="model_checkpoints")
            print(f"[MLflow] Logged checkpoint: {os.path.basename(checkpoint_path)}")

    # ── dataset info ─────────────────────────────────────────────────
    def log_dataset_info(
        self,
        name: str = None,
        num_samples: int = None,
        num_classes: int = None,
        image_size: int = None,
        extra: Dict[str, Any] = None,
    ) -> None:
        """Log dataset statistics as MLflow parameters."""
        if not self.active:
            return

        info: Dict[str, Any] = {}
        cfg = self.cfg

        info["dataset.name"] = name or (cfg.dataset.name if cfg else "unknown")
        if num_classes is not None:
            info["dataset.num_classes"] = num_classes
        elif cfg and hasattr(cfg.dataset, "num_classes"):
            info["dataset.num_classes"] = cfg.dataset.num_classes
        if image_size is not None:
            info["dataset.image_size"] = image_size
        elif cfg and hasattr(cfg.dataset, "image_size"):
            info["dataset.image_size"] = cfg.dataset.image_size
        if num_samples is not None:
            info["dataset.num_samples"] = num_samples
        if extra:
            for k, v in extra.items():
                info[f"dataset.{k}"] = v

        mlflow.log_params({k: str(v)[:500] for k, v in info.items()})

    # ── tags ─────────────────────────────────────────────────────────
    def set_tag(self, key: str, value: str) -> None:
        if self.active:
            mlflow.set_tag(key, value)

    # ── private helpers ──────────────────────────────────────────────
    @staticmethod
    def _infer_experiment_name(cfg) -> str:
        """Derive experiment name from config: SSL_Benchmark_{DATASET}."""
        if cfg is None:
            return "default"
        dataset = "unknown"
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "name"):
            dataset = cfg.dataset.name
        elif hasattr(cfg, "mlflow") and hasattr(cfg.mlflow, "experiment_name"):
            return cfg.mlflow.experiment_name
        return f"SSL_Benchmark_{dataset.upper()}"

    @staticmethod
    def _infer_run_name(cfg) -> str:
        """Derive run name from config: {method}_{dataset}_{training_mode}."""
        if cfg is None:
            return "run"
        parts = []
        # method
        if hasattr(cfg, "method"):
            m = cfg.method.name if hasattr(cfg.method, "name") else str(cfg.method)
            parts.append(m)
        # dataset
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "name"):
            parts.append(cfg.dataset.name)
        # training mode
        if hasattr(cfg, "training") and hasattr(cfg.training, "name"):
            parts.append(cfg.training.name)
        return "_".join(parts) if parts else "run"

    def __enter__(self):
        self.start_experiment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_experiment()
        return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict with dot-separated keys."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items
