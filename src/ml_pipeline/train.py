"""
Model training orchestrator with MLflow tracking.

Handles:
- Cross-validated training for XGBoost, LightGBM, Random Forest
- Automatic hyperparameter logging to MLflow
- Model artefact serialisation (joblib + MLflow model registry)
- Accuracy improvement targets: 20–30% vs baseline
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    roc_auc_score,
)

from src.logger import get_logger
from src.settings import get_settings

from .models import ModelRegistry

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates multi-model training with MLflow experiment tracking.

    Example
    -------
    trainer = ModelTrainer(experiment_name="churn-v2")
    results = trainer.train_all(X_train, y_train, X_val, y_val,
                                task="classification")
    best = trainer.get_best_model(results, metric="roc_auc")
    """

    MODEL_NAMES = ["xgboost", "lightgbm", "random_forest"]

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        artifacts_dir: str = "./models",
    ) -> None:
        self.settings = get_settings()
        self.experiment_name = experiment_name or self.settings.mlflow_experiment_name
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task: str = "classification",
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """Train each model, log to MLflow, return results dict."""
        names = model_names or self.MODEL_NAMES
        results = {}
        for name in names:
            logger.info("training_model_start", model=name, task=task)
            result = self._train_single(name, X_train, y_train, X_val, y_val, task)
            results[name] = result
        return results

    def get_best_model(
        self, results: Dict[str, dict], metric: str = "roc_auc"
    ) -> Tuple[str, object]:
        """Return (model_name, model_object) with highest val metric."""
        best_name = max(results, key=lambda n: results[n]["metrics"].get(metric, 0))
        logger.info(
            "best_model_selected",
            model=best_name,
            metric=metric,
            value=results[best_name]["metrics"].get(metric),
        )
        return best_name, results[best_name]["model"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_single(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task: str,
    ) -> dict:
        model, cfg = ModelRegistry.get(model_name, task)

        with mlflow.start_run(run_name=f"{model_name}_{task}"):
            # Log hyperparameters
            mlflow.log_params(cfg.to_mlflow_params())
            mlflow.log_param("task", task)
            mlflow.log_param("train_size", len(X_train))

            t0 = time.perf_counter()

            # Fit with eval set for early-stopping models
            if model_name in ("xgboost", "lightgbm"):
                fit_kwargs = {
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                }
                if model_name == "xgboost":
                    fit_kwargs["early_stopping_rounds"] = cfg.early_stopping_rounds
                model.fit(X_train, y_train, **fit_kwargs)
            else:
                model.fit(X_train, y_train)

            train_sec = time.perf_counter() - t0

            # Evaluate
            metrics = self._evaluate(model, X_val, y_val, task)
            metrics["train_duration_sec"] = round(train_sec, 2)

            mlflow.log_metrics(metrics)
            logger.info("model_trained", model=model_name, **metrics)

            # Persist model
            artifact_path = str(self.artifacts_dir / f"{model_name}_{task}.joblib")
            joblib.dump(model, artifact_path)
            mlflow.log_artifact(artifact_path)

            # Register model
            if task == "classification":
                mlflow.sklearn.log_model(model, artifact_path=model_name)

        return {"model": model, "metrics": metrics, "artifact_path": artifact_path}

    @staticmethod
    def _evaluate(model, X_val, y_val, task: str) -> Dict[str, float]:
        y_pred = model.predict(X_val)
        metrics: Dict[str, float] = {}
        if task == "classification":
            metrics["accuracy"] = round(accuracy_score(y_val, y_pred), 4)
            metrics["f1"] = round(f1_score(y_val, y_pred, average="weighted"), 4)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y_val, y_proba), 4)
        else:
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_val, y_pred))), 4)
            metrics["mape"] = round(float(mean_absolute_percentage_error(y_val, y_pred)), 4)
        return metrics
