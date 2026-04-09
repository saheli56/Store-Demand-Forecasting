import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from store_demand_forecasting import (
    evaluate_forecasts,
    forecast_next_days,
    prepare_features,
    split_data,
    suggest_restocking,
    train_baseline,
    train_lightgbm,
    train_random_forest,
)

st.set_page_config(page_title="Store Demand Forecasting", layout="wide")

st.title("Store Demand Forecasting Dashboard")
st.markdown(
    "This app trains demand forecasting models from historical sales data, "
    "compares results, and visualizes future predictions for each store-item pair."
)

with st.sidebar:
    st.header("Data input")
    data_source = st.radio("Choose data source", ["Upload CSV", "Local file path"])
    uploaded_file = None
    local_path = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a sales CSV file", type="csv")
    else:
        local_path = st.text_input("Local CSV path", "dummy_sales.csv")

    forecast_days = st.slider("Forecast horizon", min_value=7, max_value=90, value=30, step=7)


def load_sales_dataframe(uploaded_file, local_path):
    if uploaded_file is not None:
        raw = uploaded_file.read()
        buffer = io.BytesIO(raw)
        df = pd.read_csv(buffer, parse_dates=["date"])
    elif local_path is not None and Path(local_path).exists():
        df = pd.read_csv(local_path, parse_dates=["date"])
    else:
        raise FileNotFoundError("No valid CSV file path or upload provided.")

    required_cols = {"date", "store_id", "item_id", "sales"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def plot_actual_vs_predicted(df, model, feature_cols, store_id, item_id):
    mask = (df["store_id"] == store_id) & (df["item_id"] == item_id)
    subset = df[mask].copy()
    if subset.empty:
        st.warning("No records found for that store-item combination.")
        return

    subset = subset.sort_values("date")
    X = subset[feature_cols]
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subset["date"], subset["sales"], label="Actual", marker="o", alpha=0.8)
    ax.plot(subset["date"], y_pred, label="Predicted", marker="x", alpha=0.8)
    ax.set_title(f"Actual vs Predicted sales for store {store_id}, item {item_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


if st.button("Run forecasting"):
    try:
        df = load_sales_dataframe(uploaded_file, local_path)
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")
        st.stop()

    st.success("Dataset loaded successfully.")
    st.write("### Dataset sample")
    st.dataframe(df.head(10))
    st.write(
        "### Dataset summary",
        {
            "Total rows": len(df),
            "Unique stores": df["store_id"].nunique(),
            "Unique items": df["item_id"].nunique(),
            "Date range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        },
    )

    with st.spinner("Training models and generating forecasts..."):
        df = train_baseline(df)
        df_features, feature_cols = prepare_features(df)
        X_train, X_valid, y_train, y_valid = split_data(df_features, feature_cols)
        rf_model = train_random_forest(X_train, y_train)
        lgb_model = train_lightgbm(X_train, y_train, X_valid, y_valid)

        y_pred_rf = rf_model.predict(X_valid)
        y_pred_lgb = lgb_model.predict(X_valid)
        metrics_rf = evaluate_forecasts(y_valid.values, y_pred_rf)
        metrics_lgb = evaluate_forecasts(y_valid.values, y_pred_lgb)
        chosen_model = lgb_model if metrics_lgb["rmse"] <= metrics_rf["rmse"] else rf_model

        forecast_df = forecast_next_days(df_features, chosen_model, feature_cols, forecast_days=forecast_days)
        restock_df = suggest_restocking(df)

    st.write("## Model comparison")
    comparison = pd.DataFrame([metrics_rf, metrics_lgb], index=["Random Forest", "LightGBM"])
    st.dataframe(comparison)

    st.write("## Forecast sample")
    st.dataframe(forecast_df.head(20))
    st.download_button(
        label="Download forecast sample as CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv",
    )

    st.write("## Restocking suggestions")
    st.dataframe(restock_df.head(20))

    st.write("## Explore store-item predictions")
    stores = sorted(df["store_id"].unique())
    items = sorted(df["item_id"].unique())
    if stores and items:
        selected_store = st.selectbox("Choose store", stores)
        selected_item = st.selectbox("Choose item", items)
        plot_actual_vs_predicted(df_features, chosen_model, feature_cols, selected_store, selected_item)
    else:
        st.warning("Dataset needs at least one store and one item to plot predictions.")

    st.write("### Notes")
    st.info(
        "The dashboard uses a rolling-feature tree model pipeline. "
        "LightGBM is selected when it has the lower RMSE on validation data."
    )
