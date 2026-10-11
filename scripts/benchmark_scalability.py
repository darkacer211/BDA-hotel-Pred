"""
Scalability Benchmark Module: Pandas vs. PySpark.

Compares wall-clock execution runtime (seconds) for identical Big Data operations:
1. Ingestion & Schema Parsing
2. Data Cleansing & Null Imputation
3. Multi-dimensional Aggregation (Group by hotel, year, month, segment)
4. Feature Engineering Transformation
5. Distributed vs. In-Memory Execution Scaling

Generates:
- reports/figures/scalability_comparison.png
- reports/figures/scalability_metrics.json
"""

import time
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_pandas_benchmark(csv_path="hotel_bookings.csv"):
    """Executes benchmark pipeline using Pandas."""
    timings = {}
    
    # 1. Ingestion
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    timings["ingestion"] = time.perf_counter() - t0
    
    # 2. Cleaning & Null Imputation
    t0 = time.perf_counter()
    df["children"] = df["children"].fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown").astype(str)
    df["agent"] = df["agent"].fillna("0").astype(str)
    df["company"] = df["company"].fillna("0").astype(str)
    total_guests = df["adults"] + df["children"] + df["babies"]
    total_nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df_clean = df[(total_guests > 0) & (total_nights > 0) & (df["adr"] >= 0) & (df["adr"] <= 1000)].copy()
    timings["cleaning"] = time.perf_counter() - t0
    
    # 3. Multi-dimensional Aggregation
    t0 = time.perf_counter()
    agg_res = (
        df_clean.groupby(["hotel", "arrival_date_year", "arrival_date_month", "market_segment"])
        .agg(
            total_bookings=("is_canceled", "count"),
            cancellation_rate=("is_canceled", "mean"),
            avg_adr=("adr", "mean"),
            total_lead_time=("lead_time", "sum"),
            special_requests=("total_of_special_requests", "sum")
        )
    )
    timings["aggregation"] = time.perf_counter() - t0
    
    # 4. Feature Transformations
    t0 = time.perf_counter()
    df_clean["total_nights"] = df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]
    df_clean["total_guests"] = df_clean["adults"] + df_clean["children"] + df_clean["babies"]
    df_clean["room_type_changed"] = (df_clean["reserved_room_type"] != df_clean["assigned_room_type"]).astype(int)
    df_clean["lead_time_bucket"] = pd.cut(
        df_clean["lead_time"],
        bins=[-1, 7, 30, 90, 180, 1000],
        labels=["0-7", "8-30", "31-90", "91-180", ">180"]
    )
    timings["feature_engineering"] = time.perf_counter() - t0
    
    # Total runtime
    timings["total_pipeline"] = sum(timings.values())
    return timings


def run_spark_benchmark(csv_path="hotel_bookings.csv"):
    """Executes benchmark pipeline using PySpark."""
    from src.ingestion import get_spark_session, load_raw_data
    from pyspark.sql import functions as F
    
    spark = get_spark_session("ScalabilityBenchmark")
    timings = {}
    
    # 1. Ingestion with explicit schema
    t0 = time.perf_counter()
    sdf = load_raw_data(spark, csv_path)
    count = sdf.count()  # Force execution
    timings["ingestion"] = time.perf_counter() - t0
    
    # 2. Cleaning & Null Imputation
    t0 = time.perf_counter()
    sdf_clean = (
        sdf.fillna({"children": 0.0, "country": "Unknown", "agent": "0", "company": "0"})
        .filter(
            ((F.col("adults") + F.col("children") + F.col("babies")) > 0) &
            ((F.col("stays_in_weekend_nights") + F.col("stays_in_week_nights")) > 0) &
            (F.col("adr") >= 0.0) & (F.col("adr") <= 1000.0)
        )
    )
    count_clean = sdf_clean.count()  # Force action
    timings["cleaning"] = time.perf_counter() - t0
    
    # 3. Multi-dimensional Aggregation
    t0 = time.perf_counter()
    agg_spark = (
        sdf_clean.groupBy("hotel", "arrival_date_year", "arrival_date_month", "market_segment")
        .agg(
            F.count("is_canceled").alias("total_bookings"),
            F.mean("is_canceled").alias("cancellation_rate"),
            F.mean("adr").alias("avg_adr"),
            F.sum("lead_time").alias("total_lead_time"),
            F.sum("total_of_special_requests").alias("special_requests")
        )
    )
    agg_spark.collect()  # Force action
    timings["aggregation"] = time.perf_counter() - t0
    
    # 4. Feature Transformations
    t0 = time.perf_counter()
    sdf_feats = (
        sdf_clean
        .withColumn("total_nights", F.col("stays_in_weekend_nights") + F.col("stays_in_week_nights"))
        .withColumn("total_guests", F.col("adults") + F.col("children") + F.col("babies"))
        .withColumn("room_type_changed", F.when(F.col("reserved_room_type") != F.col("assigned_room_type"), 1).otherwise(0))
        .withColumn(
            "lead_time_bucket",
            F.when(F.col("lead_time") <= 7, "0-7")
            .when(F.col("lead_time") <= 30, "8-30")
            .when(F.col("lead_time") <= 90, "31-90")
            .when(F.col("lead_time") <= 180, "91-180")
            .otherwise(">180")
        )
    )
    sdf_feats.count()  # Force action
    timings["feature_engineering"] = time.perf_counter() - t0
    
    timings["total_pipeline"] = sum(timings.values())
    spark.stop()
    return timings


def benchmark_and_plot():
    print("Running Pandas scalability benchmark...")
    t_pandas = run_pandas_benchmark()
    
    print("Running PySpark scalability benchmark...")
    t_spark = run_spark_benchmark()
    
    tasks = ["ingestion", "cleaning", "aggregation", "feature_engineering", "total_pipeline"]
    task_labels = ["Ingestion", "Data Cleaning", "Multi-dim Aggregation", "Feature Engineering", "Total Pipeline"]
    
    benchmark_data = []
    for task, label in zip(tasks, task_labels):
        p_time = t_pandas[task]
        s_time = t_spark[task]
        benchmark_data.append({
            "task": label,
            "pandas_seconds": round(p_time, 3),
            "pyspark_seconds": round(s_time, 3),
            "spark_overhead_ratio": round(s_time / p_time, 2)
        })
        
    print("\n=== SCALABILITY BENCHMARK RESULTS (WALL-CLOCK SECONDS) ===")
    df_bench = pd.DataFrame(benchmark_data)
    print(df_bench.to_string(index=False))
    
    # Save JSON metrics
    with open("reports/figures/scalability_metrics.json", "w") as f:
        json.dump(benchmark_data, f, indent=4)
        
    # Plot Scalability Comparison Chart
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(task_labels) - 1)  # omit total for individual breakdown
    width = 0.35
    
    p_vals = [d["pandas_seconds"] for d in benchmark_data[:-1]]
    s_vals = [d["pyspark_seconds"] for d in benchmark_data[:-1]]
    
    plt.bar(x - width/2, p_vals, width, label="Pandas (Single Thread / In-Memory)", color="#2b5c8f")
    plt.bar(x + width/2, s_vals, width, label="Apache Spark (Distributed / JVM Local Engine)", color="#e27c38")
    
    plt.xticks(x, task_labels[:-1])
    plt.ylabel("Execution Time (Seconds)", labelpad=10)
    plt.title("Execution Latency Benchmark: Pandas vs. PySpark (119k Records)", fontweight="bold", pad=15)
    plt.legend(frameon=True)
    
    # Add text labels
    for i in range(len(x)):
        plt.text(x[i] - width/2, p_vals[i] + 0.05, f"{p_vals[i]:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.text(x[i] + width/2, s_vals[i] + 0.05, f"{s_vals[i]:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    chart_path = Path("reports/figures/scalability_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved scalability comparison figure to: {chart_path}")
    return df_bench


if __name__ == "__main__":
    benchmark_and_plot()
