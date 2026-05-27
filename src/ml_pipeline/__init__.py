"""ML pipeline: feature engineering, model training, and prediction."""

from .features import FeatureEngineer
from .models import ModelRegistry
from .predict import ModelPredictor

# ModelTrainer is NOT imported here — it requires mlflow + a tracking server.
# Import it directly when needed: from src.ml_pipeline.train import ModelTrainer

__all__ = ["FeatureEngineer", "ModelRegistry", "ModelPredictor"]
