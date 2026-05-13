import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

rf_model = joblib.load(
    'models/random_forest_model.pkl'
)

gb_model = joblib.load(
    'models/gradient_boosting_model.pkl'
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="UAC Forecast Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv(
    "data/feature_engineered_uac_data.csv"
)

# -----------------------------
# TITLE
# -----------------------------

st.title(
    "Predictive Forecasting of Care Load & Placement Demand"
)

st.markdown(
    "Interactive healthcare forecasting dashboard"
)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Forecast Controls")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    1,
    30,
    7
)

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "ARIMA",
        "Random Forest",
        "Gradient Boosting"
    ]
)

current_hhs = st.sidebar.number_input(
    "Current HHS Care Load",
    min_value=1000,
    max_value=20000,
    value=7000
)

current_discharge = st.sidebar.number_input(
    "Current Daily Discharges",
    min_value=0,
    max_value=5000,
    value=300
)

current_transfer = st.sidebar.number_input(
    "Current Transfers",
    min_value=0,
    max_value=5000,
    value=250
)

# -----------------------------
# RUN BUTTON
# -----------------------------

run_forecast = st.sidebar.button(
    "Run Forecast"
)

# -----------------------------
# MAIN DASHBOARD
# -----------------------------

st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# KPI SECTION

st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average HHS Care",
        round(
            df['Children in HHS Care'].mean(),
            2
        )
    )

with col2:
    st.metric(
        "Maximum HHS Care",
        round(
            df['Children in HHS Care'].max(),
            2
        )
    )

with col3:
    st.metric(
        "Forecast Horizon",
        forecast_days
    )

# -----------------------------
# FORECAST LOGIC
# -----------------------------

if run_forecast:

    st.success(
        "Forecast Generated Successfully"
    )

    # Store predictions
    future_values = []

    # Initial value
    latest_value = current_hhs

    # Forecast loop
    for i in range(forecast_days):

        pressure = (
            current_transfer
            -
            current_discharge
        )

        # CORRECT 11-FEATURE INPUT
        features = pd.DataFrame({

            'lag_1': [latest_value],

            'lag_7': [latest_value],

            'lag_14': [latest_value],

            'rolling_mean_7': [latest_value],

            'rolling_mean_14': [latest_value],

            'rolling_std_7': [50],

            'net_pressure': [pressure],

            'day_of_week': [1],

            'month': [1],

            'quarter': [1],

            'intake_discharge_ratio': [

                current_transfer / (
                    current_discharge + 1
                )

            ]

        })

        # Random Forest
        if selected_model == "Random Forest":

            prediction = float(

                rf_model.predict(
                    features
                )[0]

            )

        # Gradient Boosting
        elif selected_model == "Gradient Boosting":

            prediction = float(

                gb_model.predict(
                    features
                )[0]

            )

        # ARIMA Placeholder
        else:

            prediction = float(
                latest_value + pressure
            )

        # Save prediction
        future_values.append(
            prediction
        )

        # Update latest value
        latest_value = prediction

    # -----------------------------
    # FORECAST GRAPH
    # -----------------------------

    st.subheader(
        "Forecasted HHS Care Load"
    )

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.plot(

        range(
            1,
            forecast_days + 1
        ),

        future_values,

        marker='o',

        linewidth=3

    )

    ax.set_xlabel(
        "Future Days"
    )

    ax.set_ylabel(
        "Predicted Care Load"
    )

    ax.set_title(
        f"{selected_model} Forecast"
    )

    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # FORECAST TABLE
    # -----------------------------

    forecast_df = pd.DataFrame({

        "Day":

        range(
            1,
            forecast_days + 1
        ),

        "Predicted Care Load":

        future_values

    })

    st.subheader(
        "Forecast Table"
    )

    st.dataframe(

        forecast_df,

        use_container_width=True

    )

    # -----------------------------
    # FORECAST CHART
    # -----------------------------

    st.subheader(
        "Forecasted HHS Care Load"
    )

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.plot(
        range(forecast_days),
        future_values,
        marker='o',
        linewidth=3
    )

    ax.set_xlabel(
        "Future Days"
    )

    ax.set_ylabel(
        "Predicted Care Load"
    )

    ax.set_title(
        f"{selected_model} Forecast"
    )

    ax.grid(True)

    st.pyplot(fig)

   