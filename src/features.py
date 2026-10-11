"""
Feature Engineering Module for Hotel Demand and Cancellation Analysis.

Engineers:
1. Booking-level classification features:
   - total_nights: stays_in_weekend_nights + stays_in_week_nights
   - total_guests: adults + children + babies
   - day_of_week: 0 (Monday) to 6 (Sunday)
   - arrival_month_num: 1 to 12
   - is_weekend: 1 if arrival is Saturday/Sunday, else 0
   - is_portugal_holiday: 1 if arrival date is a Portugal national holiday
   - lead_time_bucket: ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '>180 days']
   - room_type_changed: 1 if reserved_room_type != assigned_room_type, else 0
   - has_special_requests: 1 if total_of_special_requests > 0, else 0
   - is_family: 1 if children + babies > 0, else 0

2. Time-series demand forecasting features:
   - Aggregated daily non-canceled bookings per hotel
   - Temporal lag features: lag_1, lag_7, lag_14
   - Rolling window averages: rolling_mean_7, rolling_mean_30
"""

from pathlib import Path
import pandas as pd
import numpy as np
import holidays


def get_portugal_holidays(years=(2015, 2016, 2017)):
    """Fetch Portugal national holidays as a set of ISO date strings."""
    pt_holidays = holidays.Portugal(years=list(years))
    return set(pt_holidays.keys())


def create_booking_features_pandas(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers tabular features on cleaned booking records using Pandas.
    
    Args:
        df_cleaned: Cleaned DataFrame with arrival_date column.
        
    Returns:
        DataFrame with enriched feature columns.
    """
    df = df_cleaned.copy()
    
    # 1. Stay durations and party composition
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    
    # 2. Calendar and timing features
    arrival_dt = pd.to_datetime(df["arrival_date"])
    df["day_of_week"] = arrival_dt.dt.dayofweek
    df["arrival_month_num"] = arrival_dt.dt.month
    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d in (5, 6) else 0)
    
    # 3. Holiday flag (Portugal)
    pt_holidays = get_portugal_holidays(years=(2015, 2016, 2017))
    df["arrival_date_only"] = arrival_dt.dt.date
    df["is_portugal_holiday"] = df["arrival_date_only"].apply(lambda d: 1 if d in pt_holidays else 0)
    df.drop(columns=["arrival_date_only"], inplace=True)
    
    # 4. Lead time buckets
    def bucket_lead_time(lt):
        if lt <= 7:
            return "0-7 days"
        elif lt <= 30:
            return "8-30 days"
        elif lt <= 90:
            return "31-90 days"
        elif lt <= 180:
            return "91-180 days"
        else:
            return ">180 days"
            
    df["lead_time_bucket"] = df["lead_time"].apply(bucket_lead_time)
    
    # 5. Room assignment change flag
    df["room_type_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    
    # 6. Additional behavioural features
    df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype(int)
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)
    
    return df


def create_daily_demand_series(df_cleaned: pd.DataFrame, hotel_type: str = None) -> pd.DataFrame:
    """
    Aggregates daily non-canceled bookings and computes lag and rolling features.
    
    Args:
        df_cleaned: Cleaned DataFrame.
        hotel_type: Filter for 'City Hotel', 'Resort Hotel', or None for overall.
        
    Returns:
        DataFrame indexed by date with booking count, lags, and rolling averages.
    """
    df = df_cleaned[df_cleaned["is_canceled"] == 0].copy()
    if hotel_type:
        df = df[df["hotel"] == hotel_type]
        
    daily = (
        df.groupby("arrival_date")
        .size()
        .reset_index(name="bookings_count")
    )
    daily["arrival_date"] = pd.to_datetime(daily["arrival_date"])
    daily = daily.sort_values("arrival_date").reset_index(drop=True)
    
    # Fill any missing dates with 0
    full_date_range = pd.date_range(
        start=daily["arrival_date"].min(),
        end=daily["arrival_date"].max(),
        freq="D"
    )
    daily = daily.set_index("arrival_date").reindex(full_date_range, fill_value=0).rename_axis("arrival_date").reset_index()
    
    # Demand features
    daily["day_of_week"] = daily["arrival_date"].dt.dayofweek
    daily["month"] = daily["arrival_date"].dt.month
    daily["is_weekend"] = daily["day_of_week"].apply(lambda d: 1 if d in (5, 6) else 0)
    
    # Lags
    daily["lag_1"] = daily["bookings_count"].shift(1)
    daily["lag_7"] = daily["bookings_count"].shift(7)
    daily["lag_14"] = daily["bookings_count"].shift(14)
    
    # Rolling averages
    daily["rolling_mean_7"] = daily["bookings_count"].shift(1).rolling(window=7, min_periods=1).mean()
    daily["rolling_mean_30"] = daily["bookings_count"].shift(1).rolling(window=30, min_periods=1).mean()
    
    # Add Portugal holiday flag
    pt_holidays = get_portugal_holidays()
    daily["is_holiday"] = daily["arrival_date"].dt.date.apply(lambda d: 1 if d in pt_holidays else 0)
    
    return daily


if __name__ == "__main__":
    from src.cleaning import clean_bookings_pandas
    
    csv_path = Path("hotel_bookings.csv")
    if not csv_path.exists():
        csv_path = Path("data/raw/hotel_bookings.csv")
        
    print(f"Reading dataset for feature generation: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    cleaned_df = clean_bookings_pandas(raw_df)
    
    print("Generating booking-level classification features...")
    enriched_df = create_booking_features_pandas(cleaned_df)
    print(f"Enriched DataFrame shape: {enriched_df.shape}")
    
    out_features = Path("data/processed/cancellation_features.parquet")
    enriched_df.to_parquet(out_features, index=False)
    print(f"Saved feature dataset to: {out_features}")
    
    print("Generating daily demand series for City Hotel and Resort Hotel...")
    city_demand = create_daily_demand_series(cleaned_df, "City Hotel")
    resort_demand = create_daily_demand_series(cleaned_df, "Resort Hotel")
    
    city_demand.to_parquet("data/processed/daily_demand_city.parquet", index=False)
    resort_demand.to_parquet("data/processed/daily_demand_resort.parquet", index=False)
    print("Daily demand series generated successfully.")
