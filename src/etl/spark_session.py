"""Singleton SparkSession factory with Delta Lake support."""

from __future__ import annotations

from pyspark.sql import SparkSession

from src.logger import get_logger
from src.settings import get_settings

logger = get_logger(__name__)
_session: SparkSession | None = None


def get_spark() -> SparkSession:
    """Return a cached SparkSession configured for Delta Lake."""
    global _session
    if _session is not None and not _session.sparkContext._jsc.sc().isStopped():
        return _session

    settings = get_settings()
    logger.info("initializing_spark_session", master=settings.spark_master)

    _session = (
        SparkSession.builder.master(settings.spark_master)
        .appName("ml-platform-etl")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    _session.sparkContext.setLogLevel("WARN")
    logger.info("spark_session_ready", version=_session.version)
    return _session
