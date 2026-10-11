# Executive Project Summary: Hotel Demand Forecasting & Cancellation Prediction

**System**: Big Data Analytics Platform using Apache Spark, XGBoost & Facebook Prophet  
**Dataset**: Antonio, de Almeida & Nunes (2019), *Data in Brief* (~119k records, 32 columns)  
**Target Applications**: Operational Overbooking Control, Dynamic Room Pricing, Macro Staffing  

---

## 1. Problem Statement & Motivation
Hotel operations face extreme revenue vulnerability from two interrelated dynamics:
1. **Unpredictable Micro Cancellations**: High baseline cancellation rates (~37.1% overall, reaching 41.7% in City properties) cause inventory spoiling and lost revenue.
2. **Volatile Macro Booking Demand**: Sharp summer peaks and off-season troughs complicate workforce scheduling and dynamic room rate optimization.

The objective of this project is to build an enterprise-grade Big Data pipeline that ingests raw reservations with **Apache Spark**, predicts individual booking cancellation probabilities without data leakage, and projects 90-day daily room demand using time-series forecasting.

---

## 2. Technical Architecture & Approach

```
Raw CSV (119k rows) ➔ PySpark Ingestion (Explicit Schema) ➔ PySpark Cleaning (-827 outliers) 
   ➔ Clean Parquet Storage ➔ Feature Engineering (Holidays, Lags, Buckets)
   ➔ Dual Modeling Tracks:
        ├─ Micro: Cancellation Classification (Logistic Regression vs. Random Forest vs. XGBoost)
        └─ Macro: 90-Day Demand Forecasting (Facebook Prophet vs. Statsmodels SARIMA)
   ➔ Interactive Streamlit Operations Dashboard (app/dashboard.py)
```

- **Data Cleansing**: Null imputation (`children`, `country`, `agent`, `company`), removal of zero-guest/zero-night transactions, and elimination of price anomalies (clipping ADR to €0–€1000).
- **Leakage Elimination**: Post-outcome columns (`reservation_status`, `reservation_status_date`) were dropped from classification features.
- **Temporal Realism**: Enforced chronological time-based train/test splits (Train: July 2015 – March 2017; Test: April – August 2017) to prevent future data leakage.

---

## 3. Best Model Results (Verified Experimental Outputs)

### A. Cancellation Prediction (Chronological Test Set: 27,784 bookings)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 78.06% | 78.58% | 64.32% | 0.7074 | 0.8681 | Baseline |
| **Random Forest** | 76.71% | 80.32% | 57.63% | 0.6711 | 0.8671 | Spark MLlib Benchmark |
| **XGBoost Classifier** | **78.55%** | **79.73%** | **64.34%** | **0.7121** | **0.8760** | **Best Model** |

### B. 90-Day Daily Room Demand Forecasting (Held-out Test Period)
| Property | Model | RMSE | MAE | MAPE (%) | Optimal Application |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Resort Hotel** | **Prophet (Holidays)** | **8.07** | **6.33** | **19.71%** | **Best for Seasonality & Holidays** |
| Resort Hotel | SARIMA (1,1,1)x(1,1,0,7) | 18.74 | 14.71 | 44.84% | Baseline Auto-regressive |
| **City Hotel** | SARIMA / Prophet | 21.89 | 16.13 | 23.46% | Balanced Urban Arrivals |

---

## 4. Five Key Takeaways for Viva & Presentation

1. **The Deposit Paradox**: Non-refundable bookings exhibit an astonishing ~99% cancellation rate in historical data. This occurs because hoteliers systematically impose non-refundable requirements on high-risk group blocks and volatile tour operators who subsequently default en masse.
2. **Lead Time is the Dominant Cancellation Indicator**: As demonstrated by SHAP feature attributions, bookings created >90 days in advance have significantly higher attrition risk compared to reservations made within 30 days of arrival.
3. **Special Requests & Parking Signal True Intent**: Guests requesting parking spaces or specific amenities (extra pillows, high floor) cancel at less than half the rate of zero-request guests.
4. **Big Data Scalability Dynamics (Pandas vs. PySpark)**: While single-threaded Pandas is faster on 119k records (0.50s vs. 15.53s total latency) due to zero JVM/Catalyst scheduling overhead, PySpark is horizontally scalable. On production datasets exceeding single-node RAM (>100M rows), Pandas fails with Out-Of-Memory (OOM) errors while Spark distributes partitions across worker nodes.
5. **Actionable Revenue Management**: By feeding individual XGBoost cancellation probabilities into the Streamlit dashboard, front-desk staff can dynamically overbook high-risk blocks while protecting VIP reservations.
