import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import chardet
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


def build_input_key(data_source: str, local_path: str, uploaded_file) -> str:
    uploaded_name = uploaded_file.name if uploaded_file is not None else ""
    uploaded_size = getattr(uploaded_file, "size", "")
    return f"{data_source}|{local_path}|{uploaded_name}|{uploaded_size}"


def get_cached_results(input_key: str):
    return st.session_state.get("run_results") if st.session_state.get("input_key") == input_key else None


def cache_results(input_key: str, results: dict):
    st.session_state["run_results"] = results
    st.session_state["input_key"] = input_key


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
            out_of_stock.append(str(row["item_id"]))

    summary["best_sellers"] = best_sellers["item_id"].astype(str).tolist()
    summary["slow_movers"] = slow_movers["item_id"].astype(str).tolist()
    summary["out_of_stock"] = out_of_stock
    summary["top_item"] = str(best_sellers.iloc[0]["item_id"])
    summary["top_sales"] = int(best_sellers.iloc[0]["total_sales"])
    summary["lowest_item"] = str(slow_movers.iloc[0]["item_id"])
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
    st.write("### Restock suggestions (higher demand_signal = higher priority)")

    # Styled HTML table with color bands based on demand_signal
    styler = disp.style.background_gradient(subset=['demand_signal'], cmap='YlOrRd')
    styler = styler.format({'mean': '{:.1f}', 'std': '{:.1f}', 'sum': '{:.0f}', 'count': '{:.0f}', 'demand_signal': '{:.2f}'})
    st.markdown(styler.to_html(), unsafe_allow_html=True)

    # Allow the shopkeeper to select items to include in a reorder list
    options = [f"{r['store_id']}-{r['item_id']}" for r in disp.to_dict(orient='records')]
    selected = st.multiselect("Select items to include in reorder list (store-item)", options, default=options)

    if selected:
        chosen_rows = []
        for s in selected:
            store_id, item_id = s.split('-')
            mask = (disp['store_id'].astype(str) == store_id) & (disp['item_id'].astype(str) == item_id)
            chosen_rows.extend(disp[mask].to_dict(orient='records'))
        chosen_df = pd.DataFrame(chosen_rows)
        if not chosen_df.empty:
            st.write("#### Reorder preview")
            st.dataframe(chosen_df)
            csv_bytes = chosen_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download reorder list as CSV", csv_bytes, file_name="reorder_list.csv", mime="text/csv")
    else:
        st.info("No items selected for reorder list.")


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
    st.markdown(
        "Upload your sales data here. Use drag-and-drop or click the button below to select a CSV file. "
        "If you prefer, choose the Local file path option and enter a path to a CSV file on disk."
    )
    data_source = st.radio("Choose data source", ["Upload CSV", "Local file path"], index=0, key="data_source")
    uploaded_file = None
    local_path = "dummy_sales.csv"
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a sales CSV file",
            type="csv",
            key="uploaded_file",
            help="Drag and drop a CSV file here or click Choose File.",
        )
        if uploaded_file is None:
            st.info("Drop a CSV file into this widget or click Choose File to upload.")
    else:
        local_path = st.text_input("Local CSV path", "dummy_sales.csv", key="local_path")

    forecast_days = st.slider("Forecast horizon", min_value=7, max_value=90, value=30, step=7, key="forecast_days")


def load_sales_dataframe(uploaded_file, local_path):
    # Helper: try multiple encodings and delimiter sniffing
    def _try_read_bytes(b: bytes):
        last_exc = None
        # Try chardet first
        try:
            det = chardet.detect(b)
            enc = det.get("encoding") or "utf-8"
            return pd.read_csv(io.BytesIO(b), encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_exc = e

        # Fallback encodings
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                return pd.read_csv(io.BytesIO(b), encoding=enc, sep=None, engine="python")
            except Exception as e:
                last_exc = e
        raise last_exc

    def _try_read_path(p: str):
        last_exc = None
        # Try chardet by reading initial bytes
        try:
            with open(p, "rb") as fh:
                raw = fh.read(100000)
            det = chardet.detect(raw)
            enc = det.get("encoding") or "utf-8"
            return pd.read_csv(p, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_exc = e

        # Fallback encodings
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                return pd.read_csv(p, encoding=enc, sep=None, engine="python")
            except Exception as e:
                last_exc = e
        raise last_exc

    if uploaded_file is not None:
        raw = uploaded_file.read()
        try:
            df = _try_read_bytes(raw)
        except Exception as e:
            raise ValueError(f"Unable to parse uploaded CSV: {e}")
    elif local_path is not None and Path(local_path).exists():
        try:
            df = _try_read_path(local_path)
        except Exception as e:
            raise ValueError(f"Unable to parse CSV at {local_path}: {e}")
    else:
        raise FileNotFoundError("No valid CSV file path or upload provided.")

    # Normalize possible column names to required schema
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        mapping = {}
        candidates = {
            "date": ["date", "order_date", "orderdate", "ship_date", "shipped_date", "timestamp", "time", "day", "invoice_date", "created_at"],
            "store_id": ["store_id", "store", "storeid", "shop_id", "customername", "customer_name", "territory", "country", "city", "ordernumber", "order_number"],
            "item_id": ["item_id", "item", "product_id", "product", "sku", "productcode", "product_code", "productline", "product_line", "model"],
            "sales": ["sales", "revenue", "amount", "total", "total_sales", "sale", "sales_amount"]
        }
        for std, cands in candidates.items():
            for cand in cands:
                if cand in cols:
                    mapping[cols[cand]] = std
                    break

        df = df.rename(columns=mapping)

        # Try to compute sales from quantity and price if sales is missing.
        if "sales" not in df.columns:
            qty_key = None
            price_key = None
            for cand in ["quantityordered", "quantity_ordered", "qty", "quantity", "units_sold"]:
                if cand in cols:
                    qty_key = cols[cand]
                    break
            for cand in ["priceeach", "price_each", "price", "unit_price", "amount"]:
                if cand in cols:
                    price_key = cols[cand]
                    break
            if qty_key and price_key:
                df["sales"] = pd.to_numeric(df[qty_key], errors="coerce") * pd.to_numeric(df[price_key], errors="coerce")
            elif qty_key and "sales" not in df.columns:
                df["sales"] = pd.to_numeric(df[qty_key], errors="coerce")

        # If item_id is still missing, use productcode or productline if available.
        if "item_id" not in df.columns:
            for cand in ["productcode", "product_code", "productline", "product_line", "item", "model"]:
                if cand in cols:
                    df["item_id"] = df[cols[cand]]
                    break

        # If store_id is still missing, use ordernumber or customer name or fall back to a single store.
        if "store_id" not in df.columns:
            if "ordernumber" in cols:
                df["store_id"] = pd.to_numeric(df[cols["ordernumber"]], errors="coerce")
            elif "customername" in cols:
                df["store_id"] = df[cols["customername"]].astype(str).factorize()[0] + 1
            else:
                df["store_id"] = 1

        return df

    try:
        df = _normalize_columns(df)
    except Exception as e:
        raise ValueError(
            f"Column normalization failed: {e}. "
            "Please upload a sales dataset with date and sales columns."
        )

    # Parse and coerce types when available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if "item_id" in df.columns:
        df["item_id"] = df["item_id"].astype(str).str.strip()
    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].astype(str).str.strip()
        df["store_id"] = pd.factorize(df["store_id"])[0] + 1

    before = len(df)
    drop_cols = [col for col in ["date", "item_id", "sales"] if col in df.columns]
    if drop_cols:
        df = df.dropna(subset=drop_cols)  # remove unusable rows
    after = len(df)
    dropped = before - after

    if dropped > 0:
        st.warning(f"Dropped {dropped} rows due to missing/invalid date, id, or sales values.")

    sort_cols = [col for col in ["store_id", "item_id", "date"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def validate_sales_dataset(df: pd.DataFrame) -> None:
    required = {"date", "sales", "item_id", "store_id"}
    missing = required - set(df.columns)
    if missing:
        st.error(
            "This file does not contain the required sales time-series columns. "
            f"Missing: {sorted(missing)}. "
            "For forecasting, upload a dataset with date, sales, item_id, and store_id."
        )
        st.write("Detected columns:", list(df.columns))
        st.info(
            "You can still preview the raw dataset here, but forecasting charts and demand predictions are only available for sales datasets."
        )
        st.stop()


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


input_key = build_input_key(data_source, local_path, uploaded_file)
if st.session_state.get("input_key") != input_key:
    st.session_state["input_key"] = input_key
    st.session_state["current_df"] = None
    st.session_state["run_results"] = None

if st.session_state.get("current_df") is None:
    try:
        if data_source == "Upload CSV" and uploaded_file is not None:
            st.session_state["current_df"] = load_sales_dataframe(uploaded_file, None)
        elif data_source == "Local file path" and local_path:
            st.session_state["current_df"] = load_sales_dataframe(None, local_path)
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")

if st.session_state.get("current_df") is not None:
    df = st.session_state["current_df"]
    st.success("Dataset loaded successfully.")
    st.write("### Dataset sample")
    st.dataframe(df.head(10))

    validate_sales_dataset(df)

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
        date_range = st.date_input("Date range", [min_date, max_date], key="date_range")
        store_opts = ['All'] + sorted(df['store_id'].unique())
        selected_store = st.selectbox("Store", store_opts, index=0, key="filter_store")
    with filter_col2:
        st.markdown("Select the time window and store to focus the dashboard for easy action.")

    filtered_df = df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df['date'] >= start_d) & (filtered_df['date'] <= end_d)]
    if selected_store != 'All':
        filtered_df = filtered_df[filtered_df['store_id'] == selected_store]

    insights = compute_shopkeeper_insights(filtered_df)
    st.write("## Quick Actions & KPIs")
    create_summary_metrics(insights)

    # Additional professional KPIs
    total_sales = int(filtered_df['sales'].sum())
    avg_daily = float(filtered_df.groupby('date')['sales'].sum().mean()) if not filtered_df.empty else 0.0
    top_store = int(filtered_df.groupby('store_id')['sales'].sum().idxmax()) if not filtered_df.empty else None
    out_count = len(insights['out_of_stock'])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales", f"{total_sales}")
    k2.metric("Avg Daily Sales", f"{avg_daily:.1f}")
    k3.metric("Top Store", f"Store {top_store}" if top_store is not None else "N/A")
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
    show_top_item_cards(filtered_df, top_n=4)

    st.write("### Top-selling products (interactive)")
    plot_interactive_top_sellers(filtered_df)

    st.write("### Slow-moving products (possible overstock risks)")
    plot_interactive_slow_movers(filtered_df)

    run = st.button("Run forecasting", key="run_forecasting")
    results = get_cached_results(input_key)
    if run:
        with st.spinner("Training models and generating forecasts..."):
            filtered_df = train_baseline(filtered_df)
            df_features, feature_cols = prepare_features(filtered_df)
            X_train, X_valid, y_train, y_valid = split_data(df_features, feature_cols)

            if X_train.empty or len(y_train) < 2:
                st.warning(
                    "Not enough structured time-series data to train a model. "
                    "Try a larger dataset or expand the selected date range."
                )
                forecast_df = pd.DataFrame(
                    {
                        "date": [],
                        "store_id": [],
                        "item_id": [],
                        "sales_forecast": [],
                    }
                )
                restock_df = suggest_restocking(filtered_df)
                metrics_rf = metrics_lgb = {"rmse": None, "mae": None, "smape": None}
                chosen_model = None
            else:
                rf_model = train_random_forest(X_train, y_train)
                lgb_model = train_lightgbm(X_train, y_train, X_valid, y_valid)

                y_pred_rf = rf_model.predict(X_valid) if not X_valid.empty else np.array([])
                y_pred_lgb = lgb_model.predict(X_valid) if not X_valid.empty else np.array([])
                metrics_rf = evaluate_forecasts(y_valid.values, y_pred_rf) if not X_valid.empty else {"rmse": None, "mae": None, "smape": None}
                metrics_lgb = evaluate_forecasts(y_valid.values, y_pred_lgb) if not X_valid.empty else {"rmse": None, "mae": None, "smape": None}
                chosen_model = lgb_model if metrics_lgb["rmse"] is not None and metrics_rf["rmse"] is not None and metrics_lgb["rmse"] <= metrics_rf["rmse"] else rf_model
                forecast_df = forecast_next_days(df_features, chosen_model, feature_cols, forecast_days=forecast_days)
                restock_df = suggest_restocking(filtered_df)

        results = {
            "df_features": df_features,
            "feature_cols": feature_cols,
            "metrics_rf": metrics_rf,
            "metrics_lgb": metrics_lgb,
            "forecast_df": forecast_df,
            "restock_df": restock_df,
            "chosen_model": chosen_model,
            "filtered_df": filtered_df,
        }
        cache_results(input_key, results)
    elif results is not None:
        df_features = results["df_features"]
        feature_cols = results["feature_cols"]
        metrics_rf = results["metrics_rf"]
        metrics_lgb = results["metrics_lgb"]
        forecast_df = results["forecast_df"]
        restock_df = results["restock_df"]
        chosen_model = results["chosen_model"]
        filtered_df = results["filtered_df"]
    else:
        st.info("Click Run forecasting to train the model and generate predictions.")
        results = None

    if results is not None:
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
        stores = sorted(filtered_df["store_id"].unique())
        items = sorted(filtered_df["item_id"].unique())
        if stores and items and chosen_model is not None:
            selected_store = st.selectbox("Choose store", stores, key="explore_store")
            selected_item = st.selectbox("Choose item", items, key="explore_item")
            plot_actual_vs_predicted(df_features, chosen_model, feature_cols, selected_store, selected_item)
        elif stores and items:
            st.info("Model training was skipped because the dataset is too small, so predictions are not available.")
        else:
            st.warning("Dataset needs at least one store and one item to plot predictions.")

    st.write("### Notes")
    st.info(
        "The dashboard uses a rolling-feature tree model pipeline. "
        "LightGBM is selected when it has the lower RMSE on validation data."
    )
