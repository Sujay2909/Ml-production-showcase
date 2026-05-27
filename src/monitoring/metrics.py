"""
Prometheus metrics collection for model and pipeline monitoring.

Exposes model accuracy, prediction latency, cache hit rate,
and data drift signals — mirrors the Tableau dashboard KPIs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
)

from src.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Singleton metrics registry.  Initialise once per process.

    Usage
    -----
    mc = MetricsCollector()
    mc.start_server(port=9090)

    with mc.track_prediction_latency("churn"):
        result = model.predict(X)
    mc.record_prediction("churn", label=1)
    mc.update_model_accuracy("churn", accuracy=0.923)
    """

    def __init__(self) -> None:
        # Prediction throughput
        self.prediction_counter = Counter(
            "ml_predictions_total",
            "Total predictions served",
            ["model_name", "label"],
        )
        self.prediction_errors = Counter(
            "ml_prediction_errors_total",
            "Prediction errors",
            ["model_name", "error_type"],
        )

        # Latency
        self.prediction_latency = Histogram(
            "ml_prediction_latency_seconds",
            "Model inference latency",
            ["model_name"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        )

        # Model quality
        self.model_accuracy = Gauge(
            "ml_model_accuracy",
            "Tracked model accuracy on validation data",
            ["model_name"],
        )
        self.model_roc_auc = Gauge(
            "ml_model_roc_auc",
            "Model ROC-AUC on validation data",
            ["model_name"],
        )

        # Data pipeline
        self.pipeline_records_processed = Counter(
            "etl_records_processed_total",
            "Records processed by ETL pipeline",
            ["pipeline_name"],
        )
        self.pipeline_reliability = Gauge(
            "etl_pipeline_reliability",
            "Ratio of valid to total records",
            ["pipeline_name"],
        )

        # Cache
        self.cache_hits = Counter("cache_hits_total", "Redis cache hits", ["model_name"])
        self.cache_misses = Counter("cache_misses_total", "Redis cache misses", ["model_name"])

        # Drift
        self.feature_drift_score = Gauge(
            "ml_feature_drift_score",
            "PSI-based feature drift score (>0.2 = significant drift)",
            ["model_name", "feature_name"],
        )

    # ------------------------------------------------------------------
    # Context manager for latency tracking
    # ------------------------------------------------------------------
    @contextmanager
    def track_prediction_latency(self, model_name: str):
        with self.prediction_latency.labels(model_name=model_name).time():
            yield

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_prediction(self, model_name: str, label: str = "unknown") -> None:
        self.prediction_counter.labels(model_name=model_name, label=str(label)).inc()

    def record_error(self, model_name: str, error_type: str = "unknown") -> None:
        self.prediction_errors.labels(model_name=model_name, error_type=error_type).inc()

    def update_model_accuracy(self, model_name: str, accuracy: float) -> None:
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
        if accuracy < 0.85:
            logger.warning(
                "accuracy_below_threshold",
                model=model_name,
                accuracy=accuracy,
                threshold=0.85,
            )

    def update_roc_auc(self, model_name: str, roc_auc: float) -> None:
        self.model_roc_auc.labels(model_name=model_name).set(roc_auc)

    def update_pipeline_stats(self, pipeline_name: str, records: int, reliability: float) -> None:
        self.pipeline_records_processed.labels(pipeline_name=pipeline_name).inc(records)
        self.pipeline_reliability.labels(pipeline_name=pipeline_name).set(reliability)

    def record_cache_hit(self, model_name: str) -> None:
        self.cache_hits.labels(model_name=model_name).inc()

    def record_cache_miss(self, model_name: str) -> None:
        self.cache_misses.labels(model_name=model_name).inc()

    def update_drift_score(self, model_name: str, feature_name: str, psi_score: float) -> None:
        self.feature_drift_score.labels(model_name=model_name, feature_name=feature_name).set(
            psi_score
        )
        if psi_score > 0.2:
            logger.warning(
                "significant_feature_drift",
                model=model_name,
                feature=feature_name,
                psi=psi_score,
            )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    def start_server(self, port: int = 9090) -> None:
        logger.info("prometheus_metrics_server_started", port=port)
        start_http_server(port)
