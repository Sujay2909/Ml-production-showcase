"""
Sklearn-compatible feature engineering pipeline.

Wraps pandas DataFrames with a consistent fit/transform interface
so the same preprocessor can be logged to MLflow and reloaded at
serving time.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a subset of columns from a DataFrame."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Columns missing from input: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numerical columns at [q_low, q_high] percentiles."""

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99) -> None:
        self.q_low = q_low
        self.q_high = q_high
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in X.select_dtypes(include=[np.number]).columns:
            lo = X[col].quantile(self.q_low)
            hi = X[col].quantile(self.q_high)
            self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


# ---------------------------------------------------------------------------
# Feature engineer
# ---------------------------------------------------------------------------


CHURN_FEATURES = [
    "total_events",
    "unique_products",
    "total_revenue",
    "avg_order_value",
    "avg_session_sec",
    "days_since_last_event",
    "customer_tenure_days",
    "active_days",
    "engagement_score",
]

DEMAND_FEATURES = [
    "daily_units",
    "daily_revenue",
    "daily_unique_buyers",
    "revenue_rolling_7d",
    "units_rolling_7d",
    "revenue_rolling_14d",
    "units_rolling_14d",
    "revenue_rolling_30d",
    "units_rolling_30d",
]


class FeatureEngineer:
    """
    Builds and persists sklearn preprocessing pipelines for each
    task: churn prediction and demand forecasting.
    """

    def __init__(self) -> None:
        self.churn_pipeline: Pipeline | None = None
        self.demand_pipeline: Pipeline | None = None

    def build_churn_pipeline(self) -> Pipeline:
        logger.info("building_churn_feature_pipeline")
        self.churn_pipeline = Pipeline(
            [
                ("select", ColumnSelector(CHURN_FEATURES)),
                ("clip", OutlierClipper()),
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]
        )
        return self.churn_pipeline

    def build_demand_pipeline(self) -> Pipeline:
        logger.info("building_demand_feature_pipeline")
        self.demand_pipeline = Pipeline(
            [
                ("select", ColumnSelector(DEMAND_FEATURES)),
                ("clip", OutlierClipper()),
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ]
        )
        return self.demand_pipeline

    def fit_transform(self, df: pd.DataFrame, task: str = "churn") -> np.ndarray:
        pipeline = self.churn_pipeline if task == "churn" else self.demand_pipeline
        if pipeline is None:
            raise RuntimeError(f"Pipeline for '{task}' not built. Call build_*_pipeline() first.")
        logger.info("fitting_feature_pipeline", task=task, rows=len(df))
        return pipeline.fit_transform(df)

    def transform(self, df: pd.DataFrame, task: str = "churn") -> np.ndarray:
        pipeline = self.churn_pipeline if task == "churn" else self.demand_pipeline
        if pipeline is None:
            raise RuntimeError(f"Pipeline for '{task}' not built.")
        return pipeline.transform(df)

    def save(self, path: str) -> None:
        joblib.dump({"churn": self.churn_pipeline, "demand": self.demand_pipeline}, path)
        logger.info("feature_pipelines_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        fe = cls()
        data = joblib.load(path)
        fe.churn_pipeline = data["churn"]
        fe.demand_pipeline = data["demand"]
        logger.info("feature_pipelines_loaded", path=path)
        return fe
