"""
Online and batch model predictor.

Loads a trained model artifact and feature pipeline, then
exposes a consistent predict() interface used by the FastAPI
serving layer and batch scoring jobs alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.logger import get_logger
from .features import FeatureEngineer

logger = get_logger(__name__)


class ModelPredictor:
    """
    Wraps a persisted model + feature pipeline for inference.

    Parameters
    ----------
    model_path : str
        Path to a joblib-serialised sklearn/XGBoost/LightGBM model.
    feature_pipeline_path : str | None
        Path to a FeatureEngineer joblib file.  If None, raw arrays
        are expected (already preprocessed).
    task : str
        "classification" | "regression"
    """

    def __init__(
        self,
        model_path: str,
        feature_pipeline_path: Optional[str] = None,
        task: str = "classification",
    ) -> None:
        self.task = task
        self.model = joblib.load(model_path)
        logger.info("model_loaded", path=model_path, type=type(self.model).__name__)

        self.feature_engineer: Optional[FeatureEngineer] = None
        if feature_pipeline_path and Path(feature_pipeline_path).exists():
            self.feature_engineer = FeatureEngineer.load(feature_pipeline_path)
            logger.info("feature_pipeline_loaded", path=feature_pipeline_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
        return_proba: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference and return predictions (+ probabilities for classifiers).

        Parameters
        ----------
        data : DataFrame | ndarray | list[dict]
        return_proba : bool
            If True and task == "classification", also return class probabilities.

        Returns
        -------
        dict with keys: "predictions", "probabilities" (optional)
        """
        X = self._coerce_input(data)
        if self.feature_engineer:
            pipeline_task = "churn" if self.task == "classification" else "demand"
            X = self.feature_engineer.transform(X, task=pipeline_task)

        preds = self.model.predict(X)
        result: Dict[str, Any] = {"predictions": preds.tolist()}

        if self.task == "classification" and return_proba and hasattr(
            self.model, "predict_proba"
        ):
            probas = self.model.predict_proba(X)
            result["probabilities"] = probas.tolist()
            result["churn_probability"] = probas[:, 1].tolist()

        logger.info("prediction_complete", n_samples=len(preds))
        return result

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper for single-record scoring (API use case)."""
        return self.predict([record])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_input(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(data, list):
            return pd.DataFrame(data)
        return data
