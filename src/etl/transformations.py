"""
Feature transformations for ML-ready datasets.

Covers customer-level aggregations, demand signals, and
churn-indicator feature engineering using PySpark SQL.
"""
from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.logger import get_logger
from .spark_session import get_spark

logger = get_logger(__name__)


class FeatureTransformer:
    """
    Transform raw event data into aggregated, ML-ready feature sets.

    Produces:
    - customer_features:  churn risk signals, engagement scores
    - demand_features:    rolling demand aggregates per product/region
    - compliance_flags:   boolean indicators for downstream NLP triage
    """

    LOOKBACK_DAYS = [7, 14, 30, 90]

    def __init__(self, spark: Optional[SparkSession] = None) -> None:
        self.spark = spark or get_spark()

    # ------------------------------------------------------------------
    # Customer churn features
    # ------------------------------------------------------------------

    def build_customer_features(self, events_df: DataFrame) -> DataFrame:
        """
        Aggregate event-level data into per-customer feature vectors
        suitable for churn and CLV modelling.
        """
        logger.info("building_customer_features")
        base = self._customer_aggregates(events_df)
        with_recency = self._add_recency_features(base, events_df)
        with_engagement = self._add_engagement_score(with_recency)
        logger.info("customer_features_built", columns=with_engagement.columns)
        return with_engagement

    def _customer_aggregates(self, df: DataFrame) -> DataFrame:
        return df.groupBy("customer_id", "region").agg(
            F.count("event_id").alias("total_events"),
            F.countDistinct("product_id").alias("unique_products"),
            F.sum("revenue").alias("total_revenue"),
            F.avg("revenue").alias("avg_order_value"),
            F.avg("session_duration_sec").alias("avg_session_sec"),
            F.max("timestamp").alias("last_event_ts"),
            F.min("timestamp").alias("first_event_ts"),
            F.countDistinct(F.date_format("timestamp", "yyyy-MM-dd")).alias(
                "active_days"
            ),
        )

    def _add_recency_features(
        self, agg_df: DataFrame, events_df: DataFrame
    ) -> DataFrame:
        """Days since last event — key churn predictor."""
        today = F.current_timestamp()
        return agg_df.withColumn(
            "days_since_last_event",
            F.datediff(today, F.col("last_event_ts")),
        ).withColumn(
            "customer_tenure_days",
            F.datediff(today, F.col("first_event_ts")),
        )

    def _add_engagement_score(self, df: DataFrame) -> DataFrame:
        """Composite engagement score (0–1, higher = more engaged)."""
        # Normalise recency inversely: fewer days → higher score
        max_recency = 180  # treat 180+ days inactive as fully churned
        return df.withColumn(
            "recency_score",
            F.greatest(
                F.lit(0.0),
                F.lit(1.0) - (F.col("days_since_last_event") / max_recency),
            ),
        ).withColumn(
            "engagement_score",
            (
                F.col("recency_score") * 0.4
                + (F.col("active_days") / F.lit(90)).cast("double") * 0.3
                + F.least(F.col("total_events") / F.lit(100), F.lit(1.0)) * 0.3
            ),
        )

    # ------------------------------------------------------------------
    # Demand forecasting features
    # ------------------------------------------------------------------

    def build_demand_features(self, events_df: DataFrame) -> DataFrame:
        """
        Rolling demand aggregates per (product_id, region, date).
        Generates lag/window features used by LightGBM demand model.
        """
        logger.info("building_demand_features")
        daily = (
            events_df.groupBy(
                F.date_format("timestamp", "yyyy-MM-dd").alias("date"),
                "product_id",
                "region",
            )
            .agg(
                F.sum("quantity").alias("daily_units"),
                F.sum("revenue").alias("daily_revenue"),
                F.countDistinct("customer_id").alias("daily_unique_buyers"),
            )
            .orderBy("product_id", "region", "date")
        )

        w = Window.partitionBy("product_id", "region").orderBy("date")
        for days in [7, 14, 30]:
            range_w = w.rowsBetween(-days, -1)
            daily = daily.withColumn(
                f"revenue_rolling_{days}d", F.sum("daily_revenue").over(range_w)
            ).withColumn(
                f"units_rolling_{days}d", F.sum("daily_units").over(range_w)
            )
        logger.info("demand_features_built")
        return daily

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to_pandas(self, df: DataFrame):
        """Collect a Spark DataFrame to Pandas for sklearn/XGBoost."""
        logger.info("collecting_to_pandas", partitions=df.rdd.getNumPartitions())
        return df.toPandas()
