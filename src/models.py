"""
Machine Learning and Time-Series Forecasting Models Module.

Implements:
1. Cancellation Classification Pipeline:
   - Logistic Regression baseline
   - Random Forest Classifier
   - XGBoost Classifier
   - Time-based chronological split
   - Leakage prevention (drops reservation_status, reservation_status_date)
   - SHAP Explainability
2. Demand Forecasting Pipeline:
   - Facebook Prophet with holiday and weekly/yearly regressors
   - SARIMA (Seasonal Autoregressive Integrated Moving Average)
   - 90-day held-out evaluation
"""

import pickle
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluate import evaluate_classifier, plot_confusion_matrix, plot_roc_curves, evaluate_forecast


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CANCELLATION CLASSIFICATION
# ---------------------------------------------------------------------------

def prepare_classification_data(parquet_path="data/processed/cancellation_features.parquet"):
    """
    Loads features, enforces leakage removal, and performs a chronological time-based split.
    
    Split date: 2017-04-01
    Train: July 2015 - March 2017 (~84,000 rows)
    Test: April 2017 - August 2017 (~34,000 rows)
    """
    df = pd.read_parquet(parquet_path)
    
    # 1. STRICT LEAKAGE PREVENTION: drop post-outcome columns
    leakage_cols = ["reservation_status", "reservation_status_date"]
    for col in leakage_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
    # Top countries to keep, grouping rare ones as 'Other'
    top_countries = df["country"].value_counts().nlargest(15).index.tolist()
    df["country_grouped"] = df["country"].apply(lambda c: c if c in top_countries else "Other")
    
    # Categorical and Numeric feature definitions
    categorical_features = [
        "hotel", "meal", "market_segment", "distribution_channel",
        "deposit_type", "customer_type", "lead_time_bucket",
        "reserved_room_type", "assigned_room_type", "country_grouped"
    ]
    
    numeric_features = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "is_repeated_guest",
        "previous_cancellations", "previous_bookings_not_canceled",
        "booking_changes", "days_in_waiting_list", "adr",
        "required_car_parking_spaces", "total_of_special_requests",
        "total_nights", "total_guests", "day_of_week",
        "arrival_month_num", "is_weekend", "is_portugal_holiday",
        "room_type_changed", "has_special_requests", "is_family"
    ]
    
    # 2. CHRONOLOGICAL TIME-BASED TRAIN/TEST SPLIT
    df["arrival_date"] = pd.to_datetime(df["arrival_date"])
    split_date = pd.to_datetime("2017-04-01")
    
    train_df = df[df["arrival_date"] < split_date].copy()
    test_df = df[df["arrival_date"] >= split_date].copy()
    
    X_train_raw = train_df[categorical_features + numeric_features]
    y_train = train_df["is_canceled"].values
    
    X_test_raw = test_df[categorical_features + numeric_features]
    y_test = test_df["is_canceled"].values
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Extract feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
    feature_names = cat_feature_names + numeric_features
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features
    }


def train_and_evaluate_cancellation_models(data_dict):
    """Trains Logistic Regression, Random Forest, and XGBoost, evaluates, and computes SHAP."""
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    feature_names = data_dict["feature_names"]
    
    results = []
    roc_curves = {}
    
    # 1. Logistic Regression Baseline
    print("Training Logistic Regression Baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    res_lr = evaluate_classifier(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
    results.append(res_lr)
    roc_curves["Logistic Regression"] = (y_test, y_prob_lr, res_lr["roc_auc"])
    print(f"Logistic Regression ROC-AUC: {res_lr['roc_auc']:.4f}, F1: {res_lr['f1']:.4f}")
    
    # 2. Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    res_rf = evaluate_classifier(y_test, y_pred_rf, y_prob_rf, "Random Forest")
    results.append(res_rf)
    roc_curves["Random Forest"] = (y_test, y_prob_rf, res_rf["roc_auc"])
    print(f"Random Forest ROC-AUC: {res_rf['roc_auc']:.4f}, F1: {res_rf['f1']:.4f}")
    
    # 3. XGBoost Classifier
    print("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    res_xgb = evaluate_classifier(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")
    results.append(res_xgb)
    roc_curves["XGBoost"] = (y_test, y_prob_xgb, res_xgb["roc_auc"])
    print(f"XGBoost ROC-AUC: {res_xgb['roc_auc']:.4f}, F1: {res_xgb['f1']:.4f}")
    
    # Determine best model
    best_res = max(results, key=lambda x: x["roc_auc"])
    best_name = best_res["model"]
    best_model = xgb_model if best_name == "XGBoost" else (rf if best_name == "Random Forest" else lr)
    y_pred_best = y_pred_xgb if best_name == "XGBoost" else (y_pred_rf if best_name == "Random Forest" else y_pred_lr)
    
    # Save visualizations
    print("Plotting Confusion Matrix and ROC Curves...")
    plot_confusion_matrix(y_test, y_pred_best, best_name, FIGURES_DIR / "confusion_matrix.png")
    plot_roc_curves(roc_curves, FIGURES_DIR / "roc_curves.png")
    
    # SHAP Explainability on best model
    print("Computing SHAP feature explanations...")
    sample_size = min(1500, len(X_test))
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[sample_idx]
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=12)
    plt.title("SHAP Feature Importance (Impact on Cancellation Probability)", fontweight="bold", pad=12)
    plt.tight_layout()
    shap_summary_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(shap_summary_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary to: {shap_summary_path}")
    
    # SHAP Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=12)
    plt.title("Mean Absolute SHAP Value (Global Feature Importance)", fontweight="bold", pad=12)
    plt.tight_layout()
    shap_bar_path = FIGURES_DIR / "shap_bar.png"
    plt.savefig(shap_bar_path, dpi=300)
    plt.close()
    print(f"Saved SHAP bar plot to: {shap_bar_path}")
    
    # Save models and preprocessor
    with open(MODELS_DIR / "best_cancellation_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(data_dict["preprocessor"], f)
        
    metadata = {
        "best_model": best_name,
        "feature_names": feature_names,
        "categorical_features": data_dict["categorical_features"],
        "numeric_features": data_dict["numeric_features"],
        "metrics": results
    }
    with open(MODELS_DIR / "cancellation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Classification models training and serialization complete.")
    return results


# ---------------------------------------------------------------------------
# TIME SERIES DEMAND FORECASTING
# ---------------------------------------------------------------------------

def train_and_evaluate_forecasting(hotel_type="City Hotel", forecast_days=90):
    """
    Fits Prophet and SARIMA models on daily demand, evaluating on held-out final 90 days.
    
    Args:
        hotel_type: 'City Hotel' or 'Resort Hotel'.
        forecast_days: Number of held-out days for evaluation.
        
    Returns:
        Dict of results and model objects.
    """
    parquet_path = f"data/processed/daily_demand_{hotel_type.lower().split()[0]}.parquet"
    daily_df = pd.read_parquet(parquet_path)
    daily_df["arrival_date"] = pd.to_datetime(daily_df["arrival_date"])
    daily_df = daily_df.sort_values("arrival_date").reset_index(drop=True)
    
    # Train / Test split on final 90 days
    train_df = daily_df.iloc[:-forecast_days].copy()
    test_df = daily_df.iloc[-forecast_days:].copy()
    
    print(f"\n--- Demand Forecasting: {hotel_type} ---")
    print(f"Train period: {train_df['arrival_date'].min().date()} to {train_df['arrival_date'].max().date()} ({len(train_df)} days)")
    print(f"Test period:  {test_df['arrival_date'].min().date()} to {test_df['arrival_date'].max().date()} ({len(test_df)} days)")
    
    # 1. PROPHET MODEL
    print(f"Training Prophet for {hotel_type}...")
    prophet_train = train_df[["arrival_date", "bookings_count"]].rename(
        columns={"arrival_date": "ds", "bookings_count": "y"}
    )
    m_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.90
    )
    m_prophet.add_country_holidays(country_name="PT")
    m_prophet.fit(prophet_train)
    
    future_df = m_prophet.make_future_dataframe(periods=forecast_days, freq="D")
    forecast_prophet = m_prophet.predict(future_df)
    
    prophet_test_pred = forecast_prophet.iloc[-forecast_days:]["yhat"].clip(lower=0).values
    prophet_metrics = evaluate_forecast(test_df["bookings_count"].values, prophet_test_pred, "Prophet")
    print(f"Prophet - RMSE: {prophet_metrics['rmse']}, MAE: {prophet_metrics['mae']}, MAPE: {prophet_metrics['mape']}%")
    
    # 2. SARIMA MODEL
    print(f"Training SARIMA for {hotel_type}...")
    train_series = train_df.set_index("arrival_date")["bookings_count"].asfreq("D")
    test_series = test_df.set_index("arrival_date")["bookings_count"].asfreq("D")
    
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 0, 7)
    sarima_model = SARIMAX(
        train_series,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_fitted = sarima_model.fit(disp=False)
    sarima_test_pred = sarima_fitted.forecast(steps=forecast_days).clip(lower=0).values
    sarima_metrics = evaluate_forecast(test_series.values, sarima_test_pred, "SARIMA")
    print(f"SARIMA  - RMSE: {sarima_metrics['rmse']}, MAE: {sarima_metrics['mae']}, MAPE: {sarima_metrics['mape']}%")
    
    # 3. FORECAST COMPARISON PLOT
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["arrival_date"], test_df["bookings_count"], label="Actual Demand", color="black", linewidth=2.2)
    plt.plot(test_df["arrival_date"], prophet_test_pred, label=f"Prophet Forecast (RMSE: {prophet_metrics['rmse']})", color="#0072B2", linewidth=2, linestyle="--")
    plt.plot(test_df["arrival_date"], sarima_test_pred, label=f"SARIMA Forecast (RMSE: {sarima_metrics['rmse']})", color="#D55E00", linewidth=2, linestyle=":")
    
    # Confidence interval from Prophet
    y_lower = forecast_prophet.iloc[-forecast_days:]["yhat_lower"].clip(lower=0).values
    y_upper = forecast_prophet.iloc[-forecast_days:]["yhat_upper"].clip(lower=0).values
    plt.fill_between(test_df["arrival_date"], y_lower, y_upper, color="#0072B2", alpha=0.15, label="Prophet 90% Confidence Interval")
    
    plt.title(f"90-Day Daily Booking Demand Forecast: {hotel_type} (Held-out Test Period)", fontweight="bold", pad=15)
    plt.xlabel("Arrival Date", labelpad=10)
    plt.ylabel("Non-Canceled Bookings / Day", labelpad=10)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    chart_path = FIGURES_DIR / f"demand_forecast_{hotel_type.lower().split()[0]}.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved forecast comparison chart to: {chart_path}")
    
    # Save Prophet model
    save_name = f"prophet_{hotel_type.lower().split()[0]}_hotel.pkl"
    with open(MODELS_DIR / save_name, "wb") as f:
        pickle.dump(m_prophet, f)
        
    return {
        "hotel": hotel_type,
        "prophet_metrics": prophet_metrics,
        "sarima_metrics": sarima_metrics,
        "test_dates": test_df["arrival_date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": test_df["bookings_count"].tolist(),
        "prophet_pred": prophet_test_pred.tolist(),
        "sarima_pred": sarima_test_pred.tolist()
    }


if __name__ == "__main__":
    print("=== TASK 5: CANCELLATION CLASSIFICATION ===")
    data = prepare_classification_data()
    print(f"Train samples: {data['train_size']:,}, Test samples: {data['test_size']:,}")
    class_results = train_and_evaluate_cancellation_models(data)
    
    print("\n=== TASK 6: DEMAND FORECASTING (90-DAY HELD OUT) ===")
    city_forecast = train_and_evaluate_forecasting("City Hotel", 90)
    resort_forecast = train_and_evaluate_forecasting("Resort Hotel", 90)
    
    forecast_results = [
        {"hotel": "City Hotel", **city_forecast["prophet_metrics"]},
        {"hotel": "City Hotel", **city_forecast["sarima_metrics"]},
        {"hotel": "Resort Hotel", **resort_forecast["prophet_metrics"]},
        {"hotel": "Resort Hotel", **resort_forecast["sarima_metrics"]}
    ]
    with open(MODELS_DIR / "forecasting_results.json", "w") as f:
        json.dump(forecast_results, f, indent=4)
        
    print("\n=== SUMMARY OF RESULTS ===")
    print("\n[Cancellation Prediction Results]")
    print(pd.DataFrame(class_results).to_string(index=False))
    print("\n[Demand Forecasting Results (Held-out 90 Days)]")
    print(pd.DataFrame(forecast_results).to_string(index=False))
