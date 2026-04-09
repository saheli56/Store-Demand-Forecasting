import io
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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


def compute_shopkeeper_insights(df: pd.DataFrame) -> dict:
    summary = {}
    item_total = df.groupby("item_id")["sales"].sum().reset_index(name="total_sales")

    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
    recent_avg = recent.groupby("item_id")["sales"].mean().reset_index(name="recent_avg")
    item_summary = item_total.merge(recent_avg, on="item_id", how="left").fillna(0)
    item_summary["recent_share"] = item_summary["recent_avg"] / (item_summary["total_sales"] / 120 + 1e-9)

    best_sellers = item_summary.sort_values("total_sales", ascending=False).head(3)
    slow_movers = item_summary.sort_values("recent_share").head(3)

    out_of_stock = []
    recent_window = df[df["date"] >= df["date"].max() - pd.Timedelta(days=14)]
    no_recent_sales = recent_window.groupby("item_id")["sales"].sum().reset_index()
    for _, row in no_recent_sales.iterrows():
        if row["sales"] == 0:
            out_of_stock.append(int(row["item_id"]))

    summary["best_sellers"] = best_sellers["item_id"].astype(int).tolist()
    summary["slow_movers"] = slow_movers["item_id"].astype(int).tolist()
    summary["out_of_stock"] = out_of_stock
    summary["top_item"] = int(best_sellers.iloc[0]["item_id"])
    summary["top_sales"] = int(best_sellers.iloc[0]["total_sales"])
    summary["lowest_item"] = int(slow_movers.iloc[0]["item_id"])
    summary["lowest_recent_avg"] = float(slow_movers.iloc[0]["recent_avg"])
    return summary


def create_summary_metrics(insights: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Seller", f"Item {insights['top_item']}", f"{insights['top_sales']} total")
    out_label = ", ".join(str(x) for x in insights['out_of_stock']) if insights['out_of_stock'] else "None"
    col2.metric("Out of Stock", out_label)
    col3.metric("Slow Mover", f"Item {insights['lowest_item']}", f"Avg {insights['lowest_recent_avg']:.1f}")


def plot_interactive_top_sellers(df: pd.DataFrame) -> None:
    s = df.groupby('item_id')['sales'].sum().reset_index()
    fig = px.bar(s.sort_values('sales', ascending=False), x='item_id', y='sales',
                 title='Top-selling products', labels={'item_id': 'Item ID', 'sales': 'Total Sales'},
                 hover_data={'sales': ':,', 'item_id': True})
    fig.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_interactive_slow_movers(df: pd.DataFrame) -> None:
    s = df.groupby('item_id')['sales'].mean().reset_index()
    fig = px.bar(s.sort_values('sales'), x='item_id', y='sales',
                 title='Slow-moving products (avg daily sales)', labels={'item_id': 'Item ID', 'sales': 'Avg Sales'})
    fig.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def show_restock_table(restock_df: pd.DataFrame) -> None:
    disp = restock_df.copy()
    disp['demand_signal'] = disp['demand_signal'].round(2)
    disp = disp[['store_id', 'item_id', 'mean', 'std', 'sum', 'count', 'demand_signal']]
    st.dataframe(disp.style.format({
        'mean': '{:.1f}', 'std': '{:.1f}', 'sum': '{:.0f}', 'count': '{:.0f}', 'demand_signal': '{:.2f}'
    }))


def sparkline_figure(item_id: int, df: pd.DataFrame, days: int = 60):
    series = (
        df[df['item_id'] == item_id]
        .set_index('date')
        .resample('D')['sales']
        .sum()
        .fillna(0)
        .tail(days)
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series['date'], y=series['sales'], mode='lines', line=dict(color='#2E86AB')))
    fig.update_layout(width=250, height=90, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False))
    return fig


def show_top_item_cards(df: pd.DataFrame, top_n: int = 4):
    top_items = df.groupby('item_id')['sales'].sum().sort_values(ascending=False).head(top_n).index.tolist()
    cols = st.columns(top_n)
    for col, item in zip(cols, top_items):
        total = int(df[df['item_id'] == item]['sales'].sum())
        recent = (
            df[df['item_id'] == item]
            [df['date'] >= df['date'].max() - pd.Timedelta(days=30)]['sales'].mean()
        )
        with col:
            st.markdown(f"**Item {item}**")
            st.plotly_chart(sparkline_figure(item, df), use_container_width=True)
            st.caption(f"Total sales: {total} — Avg last 30d: {recent:.1f}")

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

    # Filters for shopkeeper clarity
    st.write("---")
    filter_col1, filter_col2 = st.columns([2, 3])
    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    with filter_col1:
        date_range = st.date_input("Date range", [min_date, max_date])
        store_opts = ['All'] + sorted(df['store_id'].unique())
        selected_store = st.selectbox("Store", store_opts, index=0)
    with filter_col2:
        st.markdown("Select the time window and store to focus the dashboard for easy action.")

    # apply filters
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['date'] >= start_d) & (df['date'] <= end_d)]
    if selected_store != 'All':
        df = df[df['store_id'] == selected_store]

    insights = compute_shopkeeper_insights(df)
    st.write("## Quick Actions & KPIs")
    create_summary_metrics(insights)

    # Additional professional KPIs
    total_sales = int(df['sales'].sum())
    avg_daily = float(df.groupby('date')['sales'].sum().mean())
    top_store = int(df.groupby('store_id')['sales'].sum().idxmax()) if not df.empty else None
    out_count = len(insights['out_of_stock'])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"{total_sales}")
    k2.metric("Avg Daily Sales", f"{avg_daily:.1f}")
    k3.metric("Top Store", f"Store {top_store}")
    k4.metric("Out-of-stock items", f"{out_count}")

    st.write("### Sales story")
    if insights["out_of_stock"]:
        st.warning(
            f"The following item IDs have had no sales in the last 14 days: {', '.join(str(x) for x in insights['out_of_stock'])}. "
            "Please check your inventory or reorder them quickly."
        )
    else:
        st.success("No likely out-of-stock items detected in the selected window.")

    st.write("### Top items — quick view")
    show_top_item_cards(df, top_n=4)

    st.write("### Top-selling products (interactive)")
    plot_interactive_top_sellers(df)

    st.write("### Slow-moving products (possible overstock risks)")
    plot_interactive_slow_movers(df)

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
