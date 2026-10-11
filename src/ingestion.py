"""
Data Ingestion Module for Hotel Booking Demand Dataset.

This module initializes a PySpark session, defines an explicit schema
using StructType for the 32 columns of the Antonio et al. (2019) dataset,
loads raw CSV data into a Spark DataFrame, and saves it to Parquet.
"""

import os
import sys
from pathlib import Path

# Configure Windows Hadoop environment for Spark
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HADOOP_DIR = PROJECT_ROOT / "hadoop"
if HADOOP_DIR.exists():
    os.environ["HADOOP_HOME"] = str(HADOOP_DIR)
    os.environ["hadoop.home.dir"] = str(HADOOP_DIR)
    os.environ["PATH"] = str(HADOOP_DIR / "bin") + ";" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)


def get_spark_session(app_name: str = "HotelDemandIngestion", memory: str = "4g") -> SparkSession:
    """
    Create or get a local SparkSession configured for Windows and Java 21 compatibility.
    
    Args:
        app_name: Spark application name.
        memory: Driver and executor memory allocation.
        
    Returns:
        SparkSession instance.
    """
    java_options = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    )
    
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.extraJavaOptions", java_options)
        .config("spark.executor.extraJavaOptions", java_options)
    )
    return builder.getOrCreate()


def get_hotel_schema() -> StructType:
    """
    Define explicit PySpark StructType schema for the 32 columns of hotel_bookings.csv.
    
    Returns:
        StructType schema.
    """
    return StructType([
        StructField("hotel", StringType(), True),
        StructField("is_canceled", IntegerType(), True),
        StructField("lead_time", IntegerType(), True),
        StructField("arrival_date_year", IntegerType(), True),
        StructField("arrival_date_month", StringType(), True),
        StructField("arrival_date_week_number", IntegerType(), True),
        StructField("arrival_date_day_of_month", IntegerType(), True),
        StructField("stays_in_weekend_nights", IntegerType(), True),
        StructField("stays_in_week_nights", IntegerType(), True),
        StructField("adults", IntegerType(), True),
        StructField("children", DoubleType(), True),
        StructField("babies", IntegerType(), True),
        StructField("meal", StringType(), True),
        StructField("country", StringType(), True),
        StructField("market_segment", StringType(), True),
        StructField("distribution_channel", StringType(), True),
        StructField("is_repeated_guest", IntegerType(), True),
        StructField("previous_cancellations", IntegerType(), True),
        StructField("previous_bookings_not_canceled", IntegerType(), True),
        StructField("reserved_room_type", StringType(), True),
        StructField("assigned_room_type", StringType(), True),
        StructField("booking_changes", IntegerType(), True),
        StructField("deposit_type", StringType(), True),
        StructField("agent", StringType(), True),
        StructField("company", StringType(), True),
        StructField("days_in_waiting_list", IntegerType(), True),
        StructField("customer_type", StringType(), True),
        StructField("adr", DoubleType(), True),
        StructField("required_car_parking_spaces", IntegerType(), True),
        StructField("total_of_special_requests", IntegerType(), True),
        StructField("reservation_status", StringType(), True),
        StructField("reservation_status_date", StringType(), True)
    ])


def load_raw_data(spark: SparkSession, csv_path: str | Path):
    """
    Load raw CSV data into a PySpark DataFrame with explicit schema.
    
    Args:
        spark: Active SparkSession.
        csv_path: Path to hotel_bookings.csv.
        
    Returns:
        PySpark DataFrame.
    """
    schema = get_hotel_schema()
    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("nullValue", "NULL")
        .option("nanValue", "NA")
        .schema(schema)
        .load(str(csv_path))
    )
    return df


def save_as_parquet(df, output_path: str | Path, mode: str = "overwrite") -> None:
    """
    Persist Spark DataFrame to Parquet format with resilient Windows fallback.
    
    Args:
        df: PySpark DataFrame.
        output_path: Target directory path.
        mode: Save mode ('overwrite' or 'append').
    """
    output_str = str(output_path)
    try:
        df.write.mode(mode).parquet(output_str)
    except Exception as e:
        print(f"Spark direct parquet notice ({e}), writing via PyArrow/Pandas engine...")
        pdf = df.toPandas()
        pdf.to_parquet(output_str, index=False)


if __name__ == "__main__":
    csv_file = Path("data/raw/hotel_bookings.csv")
    if not csv_file.exists():
        csv_file = Path("hotel_bookings.csv")
    
    print(f"Initializing Spark session and loading: {csv_file}")
    spark = get_spark_session("HotelDemand_Ingestion_Main")
    df = load_raw_data(spark, csv_file)
    row_count = df.count()
    col_count = len(df.columns)
    print(f"Successfully ingested {row_count:,} rows and {col_count} columns.")
    
    output_parquet = Path("data/processed/raw_ingested.parquet")
    print(f"Writing raw ingested data to Parquet: {output_parquet}")
    save_as_parquet(df, output_parquet)
    print("Ingestion complete.")
    spark.stop()
