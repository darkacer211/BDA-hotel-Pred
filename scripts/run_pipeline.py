"""
Master Orchestration Script for Hotel Demand Forecasting and Cancellation Pipeline.

Executes all pipeline stages sequentially:
1. PySpark Data Ingestion (schema definition & raw parquet)
2. PySpark Data Cleansing (outliers, null imputation, ISO date)
3. Feature Engineering (Portugal holidays, lead time buckets, lag/rolling features)
4. EDA Visual Generation (7 figures saved to reports/figures/)
5. Machine Learning Modeling (Cancellation Classification & 90-day Demand Forecasting)
6. Big Data Scalability Benchmark (Pandas vs. PySpark wall-clock analysis)
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("🚀 STARTING HOTEL DEMAND & CANCELLATION BIG DATA PIPELINE")
    print("=" * 70)
    
    # Step 1: Ingestion
    print("\n[STEP 1/6] Running PySpark Ingestion...")
    from src.ingestion import get_spark_session, load_raw_data, save_as_parquet
    csv_file = PROJECT_ROOT / "hotel_bookings.csv"
    spark = get_spark_session("MasterPipeline_Ingestion")
    raw_df = load_raw_data(spark, csv_file)
    print(f"-> Ingested {raw_df.count():,} raw records with explicit schema.")
    
    # Step 2: Cleaning
    print("\n[STEP 2/6] Running PySpark Data Cleaning...")
    from src.cleaning import clean_bookings_spark
    cleaned_df = clean_bookings_spark(raw_df)
    clean_count = cleaned_df.count()
    print(f"-> Cleaned records: {clean_count:,}")
    save_as_parquet(cleaned_df, PROJECT_ROOT / "data/processed/cleaned_bookings.parquet")
    spark.stop()
    
    # Step 3: Feature Engineering
    print("\n[STEP 3/6] Running Feature Engineering...")
    from src.cleaning import clean_bookings_pandas
    from src.features import create_booking_features_pandas, create_daily_demand_series
    import pandas as pd
    
    raw_pdf = pd.read_csv(csv_file)
    clean_pdf = clean_bookings_pandas(raw_pdf)
    enriched_pdf = create_booking_features_pandas(clean_pdf)
    enriched_pdf.to_parquet(PROJECT_ROOT / "data/processed/cancellation_features.parquet", index=False)
    
    city_demand = create_daily_demand_series(clean_pdf, "City Hotel")
    resort_demand = create_daily_demand_series(clean_pdf, "Resort Hotel")
    city_demand.to_parquet(PROJECT_ROOT / "data/processed/daily_demand_city.parquet", index=False)
    resort_demand.to_parquet(PROJECT_ROOT / "data/processed/daily_demand_resort.parquet", index=False)
    print(f"-> Engineered {enriched_pdf.shape[1]} features across {len(enriched_pdf):,} records.")
    
    # Step 4: EDA Visuals
    print("\n[STEP 4/6] Generating EDA Visualizations...")
    from scripts.generate_eda_charts import generate_eda_figures
    generate_eda_figures()
    
    # Step 5: Models Training & Forecasting
    print("\n[STEP 5/6] Training Machine Learning & Forecasting Models...")
    from src.models import (
        prepare_classification_data,
        train_and_evaluate_cancellation_models,
        train_and_evaluate_forecasting
    )
    class_data = prepare_classification_data()
    train_and_evaluate_cancellation_models(class_data)
    train_and_evaluate_forecasting("City Hotel", 90)
    train_and_evaluate_forecasting("Resort Hotel", 90)
    
    # Step 6: Scalability Benchmark
    print("\n[STEP 6/6] Executing Scalability Benchmark (Pandas vs PySpark)...")
    from scripts.benchmark_scalability import benchmark_and_plot
    benchmark_and_plot()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print("All models serialized in models/, charts in reports/figures/")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
