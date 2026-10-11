"""
Data Cleaning Module for Hotel Booking Demand Dataset.

Performs robust data cleansing using PySpark:
1. Handles missing values:
   - children -> 0
   - country -> 'Unknown'
   - agent -> '0'
   - company -> '0'
2. Removes invalid rows:
   - Zero total guests (adults + children + babies == 0)
   - Zero total nights (stays_in_weekend_nights + stays_in_week_nights == 0)
3. Treats outliers:
   - ADR: removes negative ADR (< 0) and extreme anomaly (adr > 1000 or adr > 5000)
   - lead_time: validates non-negative lead time
4. Constructs ISO arrival_date (yyyy-MM-dd) from arrival_date_year, month, and day.
5. Persists cleaned dataset to Parquet.
"""

import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HADOOP_DIR = PROJECT_ROOT / "hadoop"
if HADOOP_DIR.exists():
    os.environ["HADOOP_HOME"] = str(HADOOP_DIR)
    os.environ["hadoop.home.dir"] = str(HADOOP_DIR)
    os.environ["PATH"] = str(HADOOP_DIR / "bin") + ";" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType


MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}


def clean_bookings_spark(df: DataFrame) -> DataFrame:
    """
    Cleans the PySpark DataFrame following strict business rules.
    
    Args:
        df: Raw PySpark DataFrame.
        
    Returns:
        Cleaned PySpark DataFrame.
    """
    # 1. Impute missing values
    df_cleaned = (
        df
        .fillna({"children": 0.0, "country": "Unknown", "agent": "0", "company": "0"})
        .withColumn("children", F.col("children").cast("integer"))
    )
    
    # 2. Filter invalid transactions: zero guests or zero nights
    total_guests_expr = F.col("adults") + F.col("children") + F.col("babies")
    total_nights_expr = F.col("stays_in_weekend_nights") + F.col("stays_in_week_nights")
    
    df_cleaned = df_cleaned.filter(
        (total_guests_expr > 0) & (total_nights_expr > 0)
    )
    
    # 3. Handle outliers in ADR and lead_time
    # Filter out negative ADR and extreme erroneous price spikes (e.g. single 5400 outlier)
    df_cleaned = df_cleaned.filter(
        (F.col("adr") >= 0.0) & (F.col("adr") <= 1000.0) & (F.col("lead_time") >= 0)
    )
    
    # 4. Construct arrival_date
    # Map month name to 2-digit month string
    month_case = F.when(F.col("arrival_date_month") == "January", "01")
    for month_name, month_num in MONTH_MAP.items():
        if month_name == "January":
            continue
        month_case = month_case.when(
            F.col("arrival_date_month") == month_name, f"{month_num:02d}"
        )
    month_case = month_case.otherwise("01")
    
    df_cleaned = (
        df_cleaned
        .withColumn("month_num_str", month_case)
        .withColumn("day_num_str", F.lpad(F.col("arrival_date_day_of_month").cast("string"), 2, "0"))
        .withColumn("arrival_date_str", F.concat_ws("-", F.col("arrival_date_year"), F.col("month_num_str"), F.col("day_num_str")))
        .withColumn("arrival_date", F.to_date(F.col("arrival_date_str"), "yyyy-MM-dd"))
        .drop("month_num_str", "day_num_str", "arrival_date_str")
    )
    
    return df_cleaned


def clean_bookings_pandas(df_raw):
    """
    Pandas equivalent of clean_bookings_spark for benchmarking and rapid exploration.
    
    Args:
        df_raw: Raw Pandas DataFrame.
        
    Returns:
        Cleaned Pandas DataFrame.
    """
    df = df_raw.copy()
    
    # Impute missing values
    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df["agent"] = df["agent"].fillna("0").astype(str)
    df["company"] = df["company"].fillna("0").astype(str)
    
    # Filter zero guests and zero nights
    total_guests = df["adults"] + df["children"] + df["babies"]
    total_nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df = df[(total_guests > 0) & (total_nights > 0)]
    
    # Filter ADR and lead_time outliers
    df = df[(df["adr"] >= 0.0) & (df["adr"] <= 1000.0) & (df["lead_time"] >= 0)]
    
    # Date construction
    month_mapped = df["arrival_date_month"].map(MONTH_MAP).astype(str).str.zfill(2)
    day_str = df["arrival_date_day_of_month"].astype(str).str.zfill(2)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" + month_mapped + "-" + day_str,
        format="%Y-%m-%d"
    )
    return df


if __name__ == "__main__":
    from src.ingestion import get_spark_session, load_raw_data, save_as_parquet
    import pandas as pd
    
    spark = get_spark_session("HotelDemand_Cleaning")
    raw_path = Path("data/raw/hotel_bookings.csv")
    if not raw_path.exists():
        raw_path = Path("hotel_bookings.csv")
        
    raw_df = load_raw_data(spark, raw_path)
    initial_count = raw_df.count()
    print(f"Raw record count: {initial_count:,}")
    
    cleaned_df = clean_bookings_spark(raw_df)
    cleaned_count = cleaned_df.count()
    removed_count = initial_count - cleaned_count
    print(f"Cleaned record count: {cleaned_count:,} (Removed {removed_count:,} invalid/outlier rows)")
    
    cleaned_output = Path("data/processed/cleaned_bookings.parquet")
    save_as_parquet(cleaned_df, cleaned_output)
    print(f"Saved cleaned parquet to: {cleaned_output}")
    spark.stop()
