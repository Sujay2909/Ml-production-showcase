"""
Delta Lake writer with merge (upsert), append, and overwrite modes.

Handles schema evolution and provides VACUUM + OPTIMIZE helpers
to keep Delta tables healthy in production.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession

from src.logger import get_logger
from src.settings import get_settings
from .spark_session import get_spark

logger = get_logger(__name__)


class WriteMode(str, Enum):
    APPEND = "append"
    OVERWRITE = "overwrite"
    MERGE = "merge"


class DeltaWriter:
    """
    Wraps Delta Lake write operations with logging and reliability guards.

    Usage
    -----
    writer = DeltaWriter()
    writer.write(df, table_name="customer_features", mode=WriteMode.MERGE,
                 merge_keys=["customer_id"])
    """

    def __init__(self, spark: Optional[SparkSession] = None) -> None:
        self.spark = spark or get_spark()
        self.settings = get_settings()
        self.base_path = self.settings.delta_table_path

    def write(
        self,
        df: DataFrame,
        table_name: str,
        mode: WriteMode = WriteMode.APPEND,
        partition_columns: Optional[List[str]] = None,
        merge_keys: Optional[List[str]] = None,
    ) -> None:
        path = f"{self.base_path}/{table_name}"
        logger.info("delta_write_start", table=table_name, mode=mode, path=path)

        if mode == WriteMode.MERGE:
            if not merge_keys:
                raise ValueError("merge_keys required for MERGE mode")
            self._merge(df, path, merge_keys)
        else:
            writer = df.write.format("delta").mode(mode.value).option(
                "mergeSchema", "true"
            )
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            writer.save(path)

        count = df.count()
        logger.info("delta_write_complete", table=table_name, rows_written=count)

    def _merge(
        self, df: DataFrame, path: str, merge_keys: List[str]
    ) -> None:
        """MERGE (upsert): update matching rows, insert new ones."""
        condition = " AND ".join(
            [f"target.{k} = source.{k}" for k in merge_keys]
        )
        if DeltaTable.isDeltaTable(self.spark, path):
            target = DeltaTable.forPath(self.spark, path)
            (
                target.alias("target")
                .merge(df.alias("source"), condition)
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            # First write — just create the table
            df.write.format("delta").save(path)

    def optimize(self, table_name: str, z_order_cols: Optional[List[str]] = None) -> None:
        """Run OPTIMIZE (and optional Z-ORDER) to compact small files."""
        path = f"{self.base_path}/{table_name}"
        logger.info("delta_optimize_start", table=table_name)
        z = f"ZORDER BY ({', '.join(z_order_cols)})" if z_order_cols else ""
        self.spark.sql(f"OPTIMIZE delta.`{path}` {z}")
        logger.info("delta_optimize_complete", table=table_name)

    def vacuum(self, table_name: str, retention_hours: int = 168) -> None:
        """Remove files older than retention_hours (default 7 days)."""
        path = f"{self.base_path}/{table_name}"
        logger.info("delta_vacuum_start", table=table_name, retention_hours=retention_hours)
        self.spark.sql(f"VACUUM delta.`{path}` RETAIN {retention_hours} HOURS")
        logger.info("delta_vacuum_complete", table=table_name)
