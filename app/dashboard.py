"""
Enterprise Streamlit Operations Dashboard:
Hotel Demand Forecasting & Cancellation Intelligence Platform.

Theme: Dark Professional Theme (Obsidian / Indigo)
Styling: Zero decorative emojis, modern typography, glassmorphism cards.
"""

import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 1. PAGE SETUP & DESIGN SYSTEM (DARK THEME)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OccuPredict | Hotel Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Luxury Glassmorphism CSS (Emoji-Free)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d121f 0%, #080b12 90%);
        color: #F1F5F9;
    }

    /* Gradient Headers */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 60%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1.6rem;
    }

    /* Status Pill Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        margin-bottom: 0.8rem;
    }
    
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10B981;
    }

    /* Dark Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    .kpi-num {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-top: 0.35rem;
    }

    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
    }

    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    .delta-green { color: #34D399; }
    .delta-blue  { color: #60A5FA; }
    .delta-purple{ color: #C084FC; }
    .delta-amber { color: #FBBF24; }

    /* Risk Banners */
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 5px solid #EF4444;
        border-radius: 10px;
        padding: 1.1rem;
        color: #FCA5A5;
    }
    
    .risk-banner-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 1.1rem;
        color: #FCD34D;
    }
    
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 5px solid #10B981;
        border-radius: 10px;
        padding: 1.1rem;
        color: #6EE7B7;
    }

    .recom-tag {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    /* Sidebar adjustments for dark theme */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Style Plotly Charts */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_data
def load_demand_data(hotel_type):
    slug = "city" if hotel_type == "City Hotel" else "resort"
    parquet_path = Path(f"data/processed/daily_demand_{slug}.parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df["arrival_date"] = pd.to_datetime(df["arrival_date"])
        return df
    return None


@st.cache_resource
def load_ml_models():
    model_path = Path("models/best_cancellation_model.pkl")
    preproc_path = Path("models/preprocessor.pkl")
    meta_path = Path("models/cancellation_metadata.json")
    
    model, preprocessor, metadata = None, None, None
    if model_path.exists() and preproc_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preproc_path, "rb") as f:
            preprocessor = pickle.load(f)
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
    return model, preprocessor, metadata


@st.cache_resource
def load_prophet(hotel_type):
    slug = "city" if hotel_type == "City Hotel" else "resort"
    path = Path(f"models/prophet_{slug}_hotel.pkl")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.5rem; padding-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">OccuPredict</div>
        <div style="font-size: 0.72rem; color: #818CF8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Hospitality Revenue Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Property Filter")
    hotel_choice = st.selectbox(
        "Select Hotel:",
        ["City Hotel", "Resort Hotel"],
        help="City Hotel is located in Lisbon; Resort Hotel is in the Algarve."
    )
    
    st.markdown("#### Navigation")
    active_tab = st.radio(
        "View Section:",
        [
            "Executive Overview",
            "Demand Forecasting",
            "Cancellation Risk Predictor",
            "Exploratory Analytics",
            "Big Data Scalability Benchmark"
        ]
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; line-height: 1.5;">
        <strong>Dataset:</strong> Antonio et al. (2019)<br>
        <strong>Ingested Rows:</strong> 119,390<br>
        <strong>Cleaned Records:</strong> 118,563<br>
        <strong>Models:</strong> XGBoost & Prophet
    </div>
    """, unsafe_allow_html=True)

# Load resources
cancellation_model, preprocessor, metadata = load_ml_models()
demand_df = load_demand_data(hotel_choice)


# ---------------------------------------------------------------------------
# VIEW 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------------
if active_tab == "Executive Overview":
    st.markdown('<div class="hero-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">High-throughput hotel demand forecasting and booking cancellation intelligence pipeline.</div>', unsafe_allow_html=True)
    
    # 4 Clean Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-label">Ingested Records</div>
            <div class="kpi-num">119,390</div>
            <div class="kpi-delta delta-blue">PySpark Schema Enforced</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-label">Cleaned Valid Bookings</div>
            <div class="kpi-num">118,563</div>
            <div class="kpi-delta delta-green">827 Outliers Filtered</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-label">Overall Cancellation Rate</div>
            <div class="kpi-num">37.1%</div>
            <div class="kpi-delta delta-amber">City: 41.7% | Resort: 27.8%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-label">Best Model ROC-AUC</div>
            <div class="kpi-num">0.8760</div>
            <div class="kpi-delta delta-purple">XGBoost Classifier</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1.6, 1])
    
    with left_col:
        st.markdown("#### Historical Arrival Pulse & Moving Averages")
        if demand_df is not None:
            fig_pulse = go.Figure()
            fig_pulse.add_trace(go.Scatter(
                x=demand_df["arrival_date"],
                y=demand_df["bookings_count"],
                mode="lines",
                name="Daily Check-ins",
                line=dict(color="rgba(99, 102, 241, 0.4)", width=1.3)
            ))
            fig_pulse.add_trace(go.Scatter(
                x=demand_df["arrival_date"],
                y=demand_df["rolling_mean_7"],
                mode="lines",
                name="7-Day Moving Average",
                line=dict(color="#818CF8", width=2.8)
            ))
            fig_pulse.add_trace(go.Scatter(
                x=demand_df["arrival_date"],
                y=demand_df["rolling_mean_30"],
                mode="lines",
                name="30-Day Trend",
                line=dict(color="#EC4899", width=2.2, dash="dot")
            ))
            fig_pulse.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17, 24, 39, 0.4)",
                font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Confirmed Check-ins / Day"),
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pulse, width="stretch")
            
    with right_col:
        st.markdown("#### Architecture Blueprint")
        st.markdown("""
        <div class="metric-card" style="font-size: 0.86rem; line-height: 1.6; color: #CBD5E1;">
            <div style="font-weight: 700; color: #A5B4FC; margin-bottom: 0.4rem;">Pipeline Highlights:</div>
            • <strong>PySpark Ingestion</strong>: Reads raw CSV into a 32-column StructType schema with Parquet export.<br>
            • <strong>Data Cleaning</strong>: Handled null values (children, country, agent, company) and removed zero-guest rows.<br>
            • <strong>Zero Data Leakage</strong>: Dropped <code>reservation_status</code> and <code>reservation_status_date</code>.<br>
            • <strong>Chronological Split</strong>: Train (2015-07 to 2017-03), Test (2017-04 to 2017-08).<br>
            • <strong>Explainability</strong>: SHAP feature attributions explaining cancellation risk drivers.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("reports/figures/cancellation_by_hotel.png", width="stretch")


# ---------------------------------------------------------------------------
# VIEW 2: DEMAND FORECASTING
# ---------------------------------------------------------------------------
elif active_tab == "Demand Forecasting":
    st.markdown(f'<div class="hero-title">Daily Demand Forecasting: {hotel_choice}</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Held-out 90-day evaluation comparing Facebook Prophet and SARIMA models.</div>', unsafe_allow_html=True)
    
    # Model Scorecards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-label">Evaluation Horizon</div>
            <div class="kpi-num">90 Days</div>
            <div class="kpi-delta delta-blue">June 3 – August 31, 2017</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        resort_prophet = "8.07 / 6.33" if hotel_choice == "Resort Hotel" else "22.17 / 16.13"
        mape_val = "19.71%" if hotel_choice == "Resort Hotel" else "23.46%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Prophet RMSE / MAE</div>
            <div class="kpi-num">{resort_prophet}</div>
            <div class="kpi-delta delta-green">MAPE: {mape_val} (Optimal)</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        sarima_scores = "18.74 / 14.71" if hotel_choice == "Resort Hotel" else "21.89 / 17.63"
        sarima_mape = "44.84%" if hotel_choice == "Resort Hotel" else "29.42%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">SARIMA RMSE / MAE</div>
            <div class="kpi-num">{sarima_scores}</div>
            <div class="kpi-delta delta-amber">MAPE: {sarima_mape}</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        winner = "Facebook Prophet" if hotel_choice == "Resort Hotel" else "Prophet / SARIMA"
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-label">Selected Champion</div>
            <div class="kpi-num" style="font-size: 1.4rem; margin-top: 0.6rem;">{winner}</div>
            <div class="kpi-delta delta-purple">Holiday & Seasonality Adjusted</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    slug = "city" if hotel_choice == "City Hotel" else "resort"
    st.image(f"reports/figures/demand_forecast_{slug}.png", caption=f"Prophet vs SARIMA 90-Day Forecast Benchmark: {hotel_choice}", width="stretch")
    
    st.markdown("#### Interactive Forecast Timeline & Scenario Analysis")
    if demand_df is not None:
        min_date = demand_df["arrival_date"].min().date()
        max_date = demand_df["arrival_date"].max().date()
        
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            date_filter = st.slider(
                "Select Date Range to Inspect:",
                min_value=min_date,
                max_value=max_date,
                value=(pd.to_datetime("2016-03-01").date(), max_date)
            )
        with col_s2:
            simulate_delta = st.slider(
                "Simulate Demand Shock (%):",
                min_value=-30,
                max_value=30,
                value=0,
                step=5,
                format="%d%%"
            )
            
        filtered_df = demand_df[
            (demand_df["arrival_date"].dt.date >= date_filter[0]) &
            (demand_df["arrival_date"].dt.date <= date_filter[1])
        ].copy()
        
        if simulate_delta != 0:
            filtered_df["scenario_demand"] = filtered_df["bookings_count"] * (1 + simulate_delta / 100.0)
            
        fig_dyn = px.line(
            filtered_df,
            x="arrival_date",
            y="bookings_count",
            title=f"Confirmed Daily Check-ins: {hotel_choice} ({date_filter[0]} to {date_filter[1]})",
            labels={"arrival_date": "Date", "bookings_count": "Arrivals"},
            color_discrete_sequence=["#6366F1"]
        )
        if simulate_delta != 0:
            fig_dyn.add_scatter(
                x=filtered_df["arrival_date"],
                y=filtered_df["scenario_demand"],
                mode="lines",
                name=f"Scenario Shock ({simulate_delta:+d}%)",
                line=dict(color="#F43F5E", width=2, dash="dash")
            )
            
        fig_dyn.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_dyn, width="stretch")


# ---------------------------------------------------------------------------
# VIEW 3: CANCELLATION RISK PREDICTOR
# ---------------------------------------------------------------------------
elif active_tab == "Cancellation Risk Predictor":
    st.markdown('<div class="hero-title">Booking Cancellation Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Individual reservation cancellation assessment powered by XGBoost (ROC-AUC: 0.8760).</div>', unsafe_allow_html=True)
    
    st.markdown("##### Load Standard Booking Profiles:")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    
    if "archetype" not in st.session_state:
        st.session_state["archetype"] = "default"
        
    with col_b1:
        if st.button("Corporate Booking", width="stretch"):
            st.session_state.update({
                "lt": 5, "deposit": "No Deposit", "market": "Corporate", "adr": 135.0,
                "special": 1, "parking": 0, "prev_canc": 0, "prev_not": 3, "cust": "Transient",
                "nights_w": 0, "nights_d": 2, "guests": 1, "country": "PRT"
            })
    with col_b2:
        if st.button("Family Holiday", width="stretch"):
            st.session_state.update({
                "lt": 45, "deposit": "No Deposit", "market": "Direct", "adr": 175.0,
                "special": 2, "parking": 1, "prev_canc": 0, "prev_not": 0, "cust": "Transient",
                "nights_w": 2, "nights_d": 5, "guests": 4, "country": "GBR"
            })
    with col_b3:
        if st.button("Tour Group Block", width="stretch"):
            st.session_state.update({
                "lt": 220, "deposit": "Non Refund", "market": "Groups", "adr": 85.0,
                "special": 0, "parking": 0, "prev_canc": 2, "prev_not": 0, "cust": "Transient-Party",
                "nights_w": 2, "nights_d": 4, "guests": 2, "country": "PRT"
            })
    with col_b4:
        if st.button("Online Travel Agency", width="stretch"):
            st.session_state.update({
                "lt": 85, "deposit": "No Deposit", "market": "Online TA", "adr": 110.0,
                "special": 0, "parking": 0, "prev_canc": 0, "prev_not": 0, "cust": "Transient",
                "nights_w": 1, "nights_d": 2, "guests": 2, "country": "FRA"
            })
            
    lt_val = st.session_state.get("lt", 65)
    deposit_val = st.session_state.get("deposit", "No Deposit")
    market_val = st.session_state.get("market", "Online TA")
    adr_val = float(st.session_state.get("adr", 115.0))
    special_val = st.session_state.get("special", 1)
    parking_val = st.session_state.get("parking", 0)
    prev_canc_val = st.session_state.get("prev_canc", 0)
    prev_not_val = st.session_state.get("prev_not", 0)
    cust_val = st.session_state.get("cust", "Transient")
    nights_w_val = st.session_state.get("nights_w", 1)
    nights_d_val = st.session_state.get("nights_d", 3)
    country_val = st.session_state.get("country", "PRT")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_inputs, col_gauge = st.columns([1.3, 1])
    
    with col_inputs:
        with st.form("cancellation_radar_form"):
            st.markdown("#### Reservation Parameters")
            
            i_col1, i_col2 = st.columns(2)
            with i_col1:
                f_hotel = st.selectbox("Hotel Property:", ["City Hotel", "Resort Hotel"])
                f_lead_time = st.slider("Lead Time (Days):", 0, 400, lt_val)
                f_deposit = st.selectbox("Deposit Policy:", ["No Deposit", "Non Refund", "Refundable"], index=["No Deposit", "Non Refund", "Refundable"].index(deposit_val) if deposit_val in ["No Deposit", "Non Refund", "Refundable"] else 0)
                f_market = st.selectbox("Market Segment:", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"], index=["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"].index(market_val) if market_val in ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"] else 0)
                f_adr = st.number_input("Average Daily Rate (ADR in €):", min_value=0.0, max_value=800.0, value=adr_val, step=5.0)
                
            with i_col2:
                f_special = st.slider("Special Requests Count:", 0, 5, special_val)
                f_parking = st.selectbox("Required Parking Spaces:", [0, 1, 2], index=parking_val if parking_val in [0, 1, 2] else 0)
                f_prev_canc = st.number_input("Past Cancellations History:", 0, 10, prev_canc_val)
                f_prev_not = st.number_input("Past Completed Stays History:", 0, 20, prev_not_val)
                f_country = st.selectbox("Origin Country:", ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "USA", "Other"], index=["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "USA", "Other"].index(country_val) if country_val in ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "USA", "Other"] else 0)
                
            calc_btn = st.form_submit_button("Calculate Cancellation Risk", width="stretch")
            
    f_total_nights = nights_w_val + nights_d_val
    lt_b = "0-7 days" if f_lead_time <= 7 else ("8-30 days" if f_lead_time <= 30 else ("31-90 days" if f_lead_time <= 90 else ("91-180 days" if f_lead_time <= 180 else ">180 days")))
    
    sample_dict = {
        "hotel": [f_hotel],
        "meal": ["BB"],
        "market_segment": [f_market],
        "distribution_channel": ["TA/TO" if "TA" in f_market else ("Direct" if f_market == "Direct" else "Corporate")],
        "deposit_type": [f_deposit],
        "customer_type": [cust_val],
        "lead_time_bucket": [lt_b],
        "reserved_room_type": ["A"],
        "assigned_room_type": ["A"],
        "country_grouped": [f_country],
        "lead_time": [f_lead_time],
        "stays_in_weekend_nights": [nights_w_val],
        "stays_in_week_nights": [nights_d_val],
        "adults": [2],
        "children": [0],
        "babies": [0],
        "is_repeated_guest": [int(f_prev_not > 0)],
        "previous_cancellations": [f_prev_canc],
        "previous_bookings_not_canceled": [f_prev_not],
        "booking_changes": [0],
        "days_in_waiting_list": [0],
        "adr": [f_adr],
        "required_car_parking_spaces": [f_parking],
        "total_of_special_requests": [f_special],
        "total_nights": [f_total_nights],
        "total_guests": [2],
        "day_of_week": [3],
        "arrival_month_num": [7],
        "is_weekend": [0],
        "is_portugal_holiday": [0],
        "room_type_changed": [0],
        "has_special_requests": [int(f_special > 0)],
        "is_family": [0]
    }
    
    prob_val = 0.35
    if cancellation_model is not None and preprocessor is not None:
        try:
            X_samp = preprocessor.transform(pd.DataFrame(sample_dict))
            prob_val = float(cancellation_model.predict_proba(X_samp)[0, 1])
        except Exception as e:
            st.error(f"Inference notice: {e}")
            
    risk_pct = prob_val * 100.0
    
    with col_gauge:
        st.markdown("#### Risk Assessment Dial")
        
        gauge_color = "#10B981" if risk_pct < 30 else ("#F59E0B" if risk_pct < 65 else "#EF4444")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 42, 'color': '#FFFFFF', 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [30, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#F8FAFC", 'width': 3},
                    'thickness': 0.75,
                    'value': risk_pct
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_gauge, width="stretch")
        
        if risk_pct >= 65:
            st.markdown(f"""
            <div class="risk-banner-high">
                <span class="recom-tag" style="background:#DC2626; color:#FFF;">High Risk ({risk_pct:.1f}%)</span>
                <div style="font-weight:700; margin-top:0.25rem;">Operational Strategy: Overbooking Protection</div>
                <div style="font-size: 0.85rem; margin-top: 0.35rem; line-height:1.45;">
                    • Apply an overbooking buffer for this inventory slot.<br>
                    • Require mandatory pre-authorization or deposit.<br>
                    • Restrict advance room upgrades.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif risk_pct >= 30:
            st.markdown(f"""
            <div class="risk-banner-med">
                <span class="recom-tag" style="background:#D97706; color:#FFF;">Moderate Risk ({risk_pct:.1f}%)</span>
                <div style="font-weight:700; margin-top:0.25rem;">Operational Strategy: Confirmation Reminders</div>
                <div style="font-size: 0.85rem; margin-top: 0.35rem; line-height:1.45;">
                    • Send automated confirmation messages 7 and 2 days prior.<br>
                    • Offer welcome beverage upon arrival to confirm intent.<br>
                    • Standard cancellation policies apply.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-low">
                <span class="recom-tag" style="background:#059669; color:#FFF;">Low Risk ({risk_pct:.1f}%)</span>
                <div style="font-weight:700; margin-top:0.25rem;">Operational Strategy: Direct Upselling</div>
                <div style="font-size: 0.85rem; margin-top: 0.35rem; line-height:1.45;">
                    • Priority candidate for complimentary room upgrades.<br>
                    • Cross-sell dining and tour services.<br>
                    • Zero overbooking required for this booking.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("#### Global Feature Attribution (SHAP Analysis)")
    st.image("reports/figures/shap_summary.png", caption="SHAP Summary Plot: Primary Drivers of Cancellation Probabilities", width="stretch")


# ---------------------------------------------------------------------------
# VIEW 4: EXPLORATORY ANALYTICS
# ---------------------------------------------------------------------------
elif active_tab == "Exploratory Analytics":
    st.markdown('<div class="hero-title">Exploratory Data Analytics (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Empirical findings across 118,563 valid hotel booking records.</div>', unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.tabs(["Seasonality & Property", "Lead Time & Channels", "The Deposit Paradox", "Room Pricing Dynamics"])
    
    with t1:
        st.image("reports/figures/monthly_seasonality.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>Key Insight:</strong> Summer arrivals (June to August) surge significantly across both properties. City Hotel processes roughly double the volume of Resort Hotel year-round.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("reports/figures/cancellation_by_hotel.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>Key Insight:</strong> City Hotel faces a <strong>41.7% cancellation rate</strong> compared to <strong>27.8%</strong> for Resort Hotel, justifying higher overbooking buffers in urban properties.
        </div>
        """, unsafe_allow_html=True)
        
    with t2:
        st.image("reports/figures/lead_time_distribution.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>Key Insight:</strong> High lead time is the primary indicator of cancellation. Bookings made within 30 days almost always check in; bookings made >180 days in advance cancel over 60% of the time.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("reports/figures/market_segment_cancellation.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>Key Insight:</strong> Online Travel Agencies (OTAs) and Tour Groups exhibit higher cancellation rates (>40% and >60%), while Direct and Corporate guests are significantly more reliable (<20%).
        </div>
        """, unsafe_allow_html=True)
        
    with t3:
        st.image("reports/figures/deposit_type_impact.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>The Deposit Paradox:</strong> Non-refundable bookings paradoxically exhibit a <strong>~99% cancellation rate</strong>. This occurs because hotels enforce non-refundable terms on high-risk group reservations and tour operators who subsequently default or cancel en masse.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("reports/figures/country_distribution.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>Key Insight:</strong> Portugal accounts for the highest domestic booking volume but also has an elevated cancellation rate (~56%). Inbound European tourists from the UK, France, and Germany cancel at much lower rates (15%–20%).
        </div>
        """, unsafe_allow_html=True)
        
    with t4:
        st.image("reports/figures/adr_trends.png", width="stretch")
        st.markdown("""
        <div class="metric-card">
            <strong>Key Insight:</strong> Resort Hotel experiences high pricing volatility, spiking above €150/night in July and August during beach vacation season, whereas City Hotel maintains steady corporate rates (€100–€120/night).
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VIEW 5: SCALABILITY BENCHMARK
# ---------------------------------------------------------------------------
elif active_tab == "Big Data Scalability Benchmark":
    st.markdown('<div class="hero-title">Big Data Scalability Benchmark: Pandas vs. PySpark</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Wall-clock latency comparison and distributed computing architecture analysis.</div>', unsafe_allow_html=True)
    
    st.image("reports/figures/scalability_comparison.png", width="stretch")
    
    st.markdown("#### Measured Wall-Clock Latency Breakdown")
    metrics_path = Path("reports/figures/scalability_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            bench_data = json.load(f)
        b_df = pd.DataFrame(bench_data)
        st.dataframe(b_df.style.format({
            "pandas_seconds": "{:.3f}s",
            "pyspark_seconds": "{:.3f}s",
            "spark_overhead_ratio": "{:.1f}x"
        }), width="stretch")
        
    st.markdown("""
    #### Technical Evaluation Summary:
    <div class="metric-card" style="line-height: 1.6; font-size: 0.92rem; color: #CBD5E1;">
        <strong>1. Why is Pandas faster on 119k rows?</strong><br>
        At ~16 MB, this dataset fits completely inside single-machine CPU cache and RAM. Pandas runs vectorized C routines without inter-process communication. PySpark incurs JVM initialization, task serialization, and Catalyst query optimizer planning overhead.<br><br>
        <strong>2. Why is PySpark required in production?</strong><br>
        When scaling from 119k rows to <strong>100 million or 1 billion rows (50 GB – 10 TB)</strong>, Pandas fails with an Out-Of-Memory (OOM) error. PySpark partitions datasets horizontally across distributed clusters, enabling linear scale-out capability.
    </div>
    """, unsafe_allow_html=True)
