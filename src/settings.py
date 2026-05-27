"""Application-wide settings loaded from env vars / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = Field("development", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # PostgreSQL
    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("mlplatform", alias="POSTGRES_DB")
    postgres_user: str = Field("mluser", alias="POSTGRES_USER")
    postgres_password: str = Field("changeme", alias="POSTGRES_PASSWORD")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_db: int = Field(0, alias="REDIS_DB")
    redis_ttl_seconds: int = Field(300, alias="REDIS_TTL_SECONDS")

    # MLflow
    mlflow_tracking_uri: str = Field("http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("ml-platform", alias="MLFLOW_EXPERIMENT_NAME")

    # Spark / Delta
    spark_master: str = Field("local[*]", alias="SPARK_MASTER")
    delta_table_path: str = Field("./data/delta", alias="DELTA_TABLE_PATH")

    # Model serving
    model_registry_path: str = Field("./models", alias="MODEL_REGISTRY_PATH")
    serving_host: str = Field("0.0.0.0", alias="SERVING_HOST")
    serving_port: int = Field(8000, alias="SERVING_PORT")

    # NLP
    hf_model_name: str = Field("bert-base-uncased", alias="HF_MODEL_NAME")
    spacy_model: str = Field("en_core_web_sm", alias="SPACY_MODEL")
    nlp_max_length: int = Field(512, alias="NLP_MAX_LENGTH")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
