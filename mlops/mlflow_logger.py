"""
MLflow Logger
=============
Centralized MLflow experiment tracking wrapper.
Provides clean API for logging metrics, parameters, and artifacts
across all training and evaluation scripts.

Supports logging Hydra DictConfig objects for full experiment
reproducibility.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: mlflow not installed. Logging disabled. Install with: pip install mlflow")

try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


class MLflowLogger:
    """
    Wrapper around MLflow for experiment tracking.

    Handles:
      - Experiment creation/selection
      - Run lifecycle (start/end)
      - Metric/param/artifact logging
      - Graceful fallback when MLflow is unavailable

    Usage:
        logger = MLflowLogger(experiment_name="simclr_pretrain")
        logger.start_run(run_name="experiment_1")
        logger.log_params({"lr": 3e-4, "batch_size": 256})
        logger.log_metric("loss", 0.5, step=10)
        logger.end_run()
    """

    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.active = MLFLOW_AVAILABLE
        self._run = None

        if self.active:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # Default: local mlruns directory in project root
                # Use file:// URI so Windows drive letters aren't misread as URI schemes
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                mlruns_path = Path(project_root) / "mlruns"
                mlruns_path.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(mlruns_path.as_uri())

            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        if self.active:
            self._run = mlflow.start_run(run_name=run_name)
            print(f"MLflow run started: {run_name} (experiment: {self.experiment_name})")
        return self

    def end_run(self):
        """End the current MLflow run."""
        if self.active and self._run is not None:
            mlflow.end_run()
            self._run = None
            print("MLflow run ended.")

    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        if self.active:
            mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters at once."""
        if self.active:
            mlflow.log_params(params)

    def log_hydra_config(self, cfg, prefix: str = ""):
        """
        Log a Hydra DictConfig to MLflow as flattened parameters.

        Recursively flattens nested config into dot-separated keys
        (e.g., 'model.backbone', 'optimizer.lr') and logs each as a
        parameter. Also saves the full YAML config as an artifact.

        Args:
            cfg: Hydra DictConfig or dict to log.
            prefix: Optional prefix for parameter names.
        """
        if not self.active:
            return

        if not OMEGACONF_AVAILABLE:
            print("WARNING: omegaconf not installed. Cannot log Hydra config.")
            return

        # Flatten config to dot-separated keys
        resolved = OmegaConf.to_container(cfg, resolve=True)
        flat_params = self._flatten_dict(resolved, prefix)

        # MLflow has a 500-char limit on param values
        safe_params = {}
        for k, v in flat_params.items():
            str_val = str(v)
            if len(str_val) > 500:
                str_val = str_val[:497] + "..."
            safe_params[k] = str_val

        mlflow.log_params(safe_params)

        # Save full config YAML as artifact
        import tempfile
        config_yaml = OmegaConf.to_yaml(cfg)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="hydra_config_", delete=False
        ) as f:
            f.write(config_yaml)
            temp_path = f.name
        mlflow.log_artifact(temp_path, artifact_path="configs")
        os.remove(temp_path)

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "") -> dict:
        """Recursively flatten a nested dict with dot-separated keys."""
        items = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(MLflowLogger._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if self.active:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        if self.active:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if self.active and os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log a directory of artifacts."""
        if self.active and os.path.exists(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if self.active:
            mlflow.set_tag(key, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
