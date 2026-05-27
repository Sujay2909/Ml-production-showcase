"""
Model definitions for XGBoost, LightGBM, and Random Forest.

Each model config is a dataclass that carries hyperparameters and
can be serialised to MLflow params.  A ModelRegistry class provides
a typed lookup so callers don't hard-code strings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------------------------
# Hyperparameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class XGBoostConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"
    use_label_encoder: bool = False
    random_state: int = 42
    n_jobs: int = -1

    def to_mlflow_params(self) -> Dict[str, Any]:
        return {f"xgb_{k}": v for k, v in asdict(self).items()}


@dataclass
class LightGBMConfig:
    n_estimators: int = 500
    max_depth: int = -1
    num_leaves: int = 63
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1

    def to_mlflow_params(self) -> Dict[str, Any]:
        return {f"lgbm_{k}": v for k, v in asdict(self).items()}


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1

    def to_mlflow_params(self) -> Dict[str, Any]:
        return {f"rf_{k}": v for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Central factory for creating model instances with default configs.

    Supports:
    - xgboost_classifier / xgboost_regressor
    - lightgbm_classifier / lightgbm_regressor
    - random_forest_classifier / random_forest_regressor
    """

    @staticmethod
    def get(model_name: str, task: str = "classification", **override_kwargs):
        """
        Parameters
        ----------
        model_name : str
            One of: xgboost | lightgbm | random_forest
        task : str
            classification | regression
        **override_kwargs :
            Override default hyperparameters.
        """
        name = model_name.lower()
        if name == "xgboost":
            cfg = XGBoostConfig(**{k: v for k, v in override_kwargs.items() if hasattr(XGBoostConfig, k)})
            params = {k: v for k, v in asdict(cfg).items() if k != "early_stopping_rounds"}
            if task == "classification":
                return xgb.XGBClassifier(**params), cfg
            return xgb.XGBRegressor(**params), cfg

        if name == "lightgbm":
            cfg = LightGBMConfig(**{k: v for k, v in override_kwargs.items() if hasattr(LightGBMConfig, k)})
            params = {k: v for k, v in asdict(cfg).items() if k != "early_stopping_rounds"}
            if task == "classification":
                return lgb.LGBMClassifier(**params), cfg
            return lgb.LGBMRegressor(**params), cfg

        if name == "random_forest":
            cfg = RandomForestConfig(**{k: v for k, v in override_kwargs.items() if hasattr(RandomForestConfig, k)})
            params = asdict(cfg)
            if task == "classification":
                return RandomForestClassifier(**params), cfg
            return RandomForestRegressor(**params), cfg

        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: xgboost, lightgbm, random_forest"
        )
