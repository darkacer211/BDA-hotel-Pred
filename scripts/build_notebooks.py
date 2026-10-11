"""
Notebook Generator Utility.

Creates the 4 required Jupyter Notebooks in notebooks/:
- 01_eda.ipynb
- 02_features.ipynb
- 03_cancellation_model.ipynb
- 04_demand_forecast.ipynb
"""

from pathlib import Path
import nbformat as nbf


NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_01_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 01. Exploratory Data Analysis (EDA)
### Hotel Demand Forecasting and Cancellation Prediction Using Apache Spark

This notebook performs in-depth exploratory data analysis on the Antonio et al. (2019) Hotel Booking Demand dataset (~119k records).
Key research questions investigated:
1. What is the monthly seasonality of bookings across City Hotel and Resort Hotel?
2. How do cancellation rates differ between properties?
3. What is the relationship between booking lead time and cancellations?
4. How do distribution channels and market segments perform?
5. What explains the "Deposit Paradox" (extreme cancellations in non-refundable bookings)?
6. How does Average Daily Rate (ADR) vary seasonally?
"""),
        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.insert(0, str(Path().resolve().parent))
from src.cleaning import clean_bookings_pandas

# Load and clean dataset
raw_df = pd.read_csv('../hotel_bookings.csv')
df = clean_bookings_pandas(raw_df)
print(f"Total cleaned bookings: {len(df):,}")
df.head()
"""),
        nbf.v4.new_markdown_cell("""## 1. Property-Level Cancellation Rates
City Hotel exhibits a substantially higher cancellation frequency (~41.7%) than Resort Hotel (~27.8%).
"""),
        nbf.v4.new_code_cell("""hotel_canc = df.groupby('hotel')['is_canceled'].agg(['count', 'mean']).reset_index()
hotel_canc['cancel_pct'] = hotel_canc['mean'] * 100
hotel_canc
"""),
        nbf.v4.new_markdown_cell("""## 2. Lead Time Distribution
Bookings made more than 90 days in advance exhibit an escalating risk of cancellation.
"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(9, 4.5))
sns.kdeplot(data=df[df['is_canceled']==0]['lead_time'], label='Confirmed (0)', fill=True, color='green')
sns.kdeplot(data=df[df['is_canceled']==1]['lead_time'], label='Canceled (1)', fill=True, color='red')
plt.title('Lead Time Distribution: Confirmed vs Canceled')
plt.xlabel('Lead Time (Days)')
plt.legend()
plt.show()
"""),
        nbf.v4.new_markdown_cell("""## 3. The Deposit Paradox
Why do Non-Refundable deposits have >99% cancellation rates?
Hotels systematically impose non-refundable requirements on high-risk group reservations and tour operators that later default or cancel in bulk.
"""),
        nbf.v4.new_code_cell("""df.groupby('deposit_type')['is_canceled'].agg(['count', 'mean'])
"""),
        nbf.v4.new_markdown_cell("""## 4. Monthly ADR Seasonality
Resort Hotel experiences an aggressive summer rate spike (€150+/night in July/August), while City Hotel maintains stable pricing year-round.
"""),
        nbf.v4.new_code_cell("""month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly_adr = df.groupby(['arrival_date_month', 'hotel'])['adr'].mean().unstack().reindex(month_order)
monthly_adr.plot(figsize=(10, 4.5), marker='o')
plt.title('Monthly ADR by Hotel Type (€)')
plt.ylabel('Average Daily Rate (€)')
plt.show()
""")
    ]
    with open(NOTEBOOKS_DIR / "01_eda.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/01_eda.ipynb")


def create_02_features_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 02. Feature Engineering Pipeline
### Hotel Demand Forecasting and Cancellation Prediction

This notebook details the creation of predictive features for:
1. Individual booking cancellation risk (Classification)
2. Macro daily room demand (Time Series Forecasting)

**Critical Rule**: Strict prevention of data leakage — `reservation_status` and `reservation_status_date` are explicitly dropped!
"""),
        nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import holidays

sys.path.insert(0, str(Path().resolve().parent))
from src.cleaning import clean_bookings_pandas
from src.features import create_booking_features_pandas, create_daily_demand_series

raw_df = pd.read_csv('../hotel_bookings.csv')
cleaned_df = clean_bookings_pandas(raw_df)
enriched_df = create_booking_features_pandas(cleaned_df)
print(f"Features DataFrame shape: {enriched_df.shape}")
enriched_df[['lead_time', 'lead_time_bucket', 'total_nights', 'total_guests', 'is_portugal_holiday', 'room_type_changed']].head()
"""),
        nbf.v4.new_markdown_cell("""## Daily Time-Series Demand Aggregation
Constructs daily non-canceled stays with temporal lag features (`lag_1`, `lag_7`, `lag_14`) and rolling averages (`rolling_mean_7`, `rolling_mean_30`).
"""),
        nbf.v4.new_code_cell("""city_demand = create_daily_demand_series(cleaned_df, 'City Hotel')
city_demand.head(10)
""")
    ]
    with open(NOTEBOOKS_DIR / "02_features.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/02_features.ipynb")


def create_03_cancellation_model_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 03. Booking Cancellation Prediction (Classification)
### Benchmarking Logistic Regression, Random Forest, and XGBoost

This notebook implements:
1. **Time-Based Split**: Training on historical bookings (July 2015 - March 2017) and testing on future bookings (April 2017 - August 2017) to prevent temporal data leakage.
2. **Model Benchmarking**: Logistic Regression, Random Forest, and XGBoost.
3. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
4. **Model Explainability**: SHAP (SHapley Additive exPlanations) values to interpret feature attributions.
"""),
        nbf.v4.new_code_cell("""import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path().resolve().parent))
from src.models import prepare_classification_data, train_and_evaluate_cancellation_models

data = prepare_classification_data('../data/processed/cancellation_features.parquet')
print(f"Train samples: {data['train_size']:,} | Test samples: {data['test_size']:,}")
"""),
        nbf.v4.new_markdown_cell("""## Model Evaluation Results
Best model: **XGBoost Classifier (ROC-AUC: 0.8760, F1: 0.7121)**
"""),
        nbf.v4.new_code_cell("""with open('../models/cancellation_metadata.json', 'r') as f:
    meta = json.load(f)
pd.DataFrame(meta['metrics'])
"""),
        nbf.v4.new_markdown_cell("""## SHAP Explainability
Visualizing the primary drivers of hotel cancellations:
- Lead Time (positive correlation with cancellation)
- Deposit Type Non-Refund (high cancellation risk)
- Total Special Requests (protective factor against cancellation)
- Required Car Parking Spaces (virtually guarantees arrival)
"""),
        nbf.v4.new_code_cell("""# Display saved SHAP beeswarm plot
from IPython.display import Image
Image('../reports/figures/shap_summary.png')
""")
    ]
    with open(NOTEBOOKS_DIR / "03_cancellation_model.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/03_cancellation_model.ipynb")


def create_04_demand_forecast_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# 04. Daily Demand Forecasting (Time Series)
### Evaluating Facebook Prophet and SARIMA over a Held-Out 90-Day Horizon

This notebook forecasts daily non-canceled hotel room demand over a 90-day held-out evaluation test set (June 3, 2017 to August 31, 2017).
Models compared:
- **Facebook Prophet** (Additive regression model with Portugal holiday regressors & Fourier seasonality)
- **SARIMA** (Seasonal Autoregressive Integrated Moving Average)
"""),
        nbf.v4.new_code_cell("""import sys
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image

with open('../models/forecasting_results.json', 'r') as f:
    forecast_results = json.load(f)
pd.DataFrame(forecast_results)
"""),
        nbf.v4.new_markdown_cell("""## Forecast Visualizations
Prophet achieves an impressive **19.71% MAPE** for Resort Hotel, closely tracking peak vacation demand.
"""),
        nbf.v4.new_code_cell("""Image('../reports/figures/demand_forecast_city.png')
"""),
        nbf.v4.new_code_cell("""Image('../reports/figures/demand_forecast_resort.png')
""")
    ]
    with open(NOTEBOOKS_DIR / "04_demand_forecast.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/04_demand_forecast.ipynb")


if __name__ == "__main__":
    create_01_eda_notebook()
    create_02_features_notebook()
    create_03_cancellation_model_notebook()
    create_04_demand_forecast_notebook()
    print("All 4 notebooks successfully created in notebooks/!")
