"""Unit tests for ML pipeline: feature engineering, models, training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_pipeline.features import (
    CHURN_FEATURES,
    ColumnSelector,
    FeatureEngineer,
    OutlierClipper,
)
from src.ml_pipeline.models import ModelRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def churn_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "total_events": np.random.randint(1, 500, n),
            "unique_products": np.random.randint(1, 20, n),
            "total_revenue": np.random.exponential(500, n),
            "avg_order_value": np.random.exponential(50, n),
            "avg_session_sec": np.random.exponential(120, n),
            "days_since_last_event": np.random.randint(0, 365, n),
            "customer_tenure_days": np.random.randint(30, 1000, n),
            "active_days": np.random.randint(1, 90, n),
            "engagement_score": np.random.uniform(0, 1, n),
        }
    )


@pytest.fixture
def churn_labels(churn_df):
    np.random.seed(42)
    return np.random.randint(0, 2, len(churn_df))


# ---------------------------------------------------------------------------
# ColumnSelector
# ---------------------------------------------------------------------------


class TestColumnSelector:
    def test_selects_correct_columns(self, churn_df):
        cols = ["total_events", "unique_products"]
        sel = ColumnSelector(cols)
        result = sel.fit_transform(churn_df)
        assert list(result.columns) == cols

    def test_raises_on_missing_column(self, churn_df):
        sel = ColumnSelector(["nonexistent_column"])
        with pytest.raises(ValueError, match="missing"):
            sel.fit_transform(churn_df)


# ---------------------------------------------------------------------------
# OutlierClipper
# ---------------------------------------------------------------------------


class TestOutlierClipper:
    def test_clips_extreme_values(self, churn_df):
        clipper = OutlierClipper(q_low=0.05, q_high=0.95)
        clipped = clipper.fit_transform(churn_df)
        numeric_cols = churn_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            orig_max = churn_df[col].quantile(0.95)
            assert clipped[col].max() <= orig_max + 1e-6

    def test_fit_stores_bounds(self, churn_df):
        clipper = OutlierClipper()
        clipper.fit(churn_df)
        assert len(clipper.bounds_) == len(churn_df.select_dtypes(include=[np.number]).columns)


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class TestFeatureEngineer:
    def test_build_and_fit_churn(self, churn_df):
        fe = FeatureEngineer()
        fe.build_churn_pipeline()
        X = fe.fit_transform(churn_df, task="churn")
        assert X.shape[0] == len(churn_df)
        assert X.shape[1] == len(CHURN_FEATURES)

    def test_raises_without_build(self, churn_df):
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.fit_transform(churn_df, task="churn")

    def test_save_load_roundtrip(self, churn_df, tmp_path):
        fe = FeatureEngineer()
        fe.build_churn_pipeline()
        fe.fit_transform(churn_df, task="churn")
        path = str(tmp_path / "fe.joblib")
        fe.save(path)

        fe2 = FeatureEngineer.load(path)
        X2 = fe2.transform(churn_df, task="churn")
        assert X2.shape == (len(churn_df), len(CHURN_FEATURES))


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "random_forest"])
    def test_get_classifier(self, model_name):
        model, cfg = ModelRegistry.get(model_name, task="classification")
        assert model is not None
        assert cfg is not None

    @pytest.mark.parametrize("model_name", ["xgboost", "lightgbm", "random_forest"])
    def test_get_regressor(self, model_name):
        model, cfg = ModelRegistry.get(model_name, task="regression")
        assert model is not None

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get("unsupported_model")

    def test_end_to_end_fit_predict(self, churn_df, churn_labels):
        fe = FeatureEngineer()
        fe.build_churn_pipeline()
        X = fe.fit_transform(churn_df, task="churn")

        model, _ = ModelRegistry.get("random_forest", task="classification")
        model.fit(X, churn_labels)
        preds = model.predict(X)
        assert len(preds) == len(churn_labels)
        assert set(preds).issubset({0, 1})
