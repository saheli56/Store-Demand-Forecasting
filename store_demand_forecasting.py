import os
from typing import List, Tuple

import lightgbm as lgb
import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def load_sales_data(csv_path: str) -> pd.DataFrame:
    """Load historical sales data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    # Try chardet to detect encoding, then fall back to common encodings
    last_exc = None
    try:
        with open(csv_path, "rb") as fh:
            raw = fh.read(100000)
        det = chardet.detect(raw)
        enc = det.get("encoding") or "utf-8"
        df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
    except Exception as e:
        last_exc = e
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        for enc_try in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=enc_try, sep=None, engine="python")
                break
            except Exception as e2:
                last_exc = e2
        else:
            raise last_exc

    # Normalize column names (accept alternative names)
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

    if "sales" not in df.columns:
        qty_key = None
        price_key = None
        for cand in ["quantityordered", "quantity_ordered", "qty_ordered", "qty", "quantity", "units_sold", "order_qty", "order_quantity", "units"]:
            if cand in cols:
                qty_key = cols[cand]
                break
        for cand in ["priceeach", "price_each", "price", "unit_price", "amount"]:
            if cand in cols:
                price_key = cols[cand]
                break
        if qty_key and price_key:
            df["sales"] = pd.to_numeric(df[qty_key], errors="coerce") * pd.to_numeric(df[price_key], errors="coerce")
        elif qty_key:
            df["sales"] = pd.to_numeric(df[qty_key], errors="coerce")

    if "item_id" not in df.columns:
        for cand in ["productcode", "product_code", "productline", "product_line", "item", "model"]:
            if cand in cols:
                df["item_id"] = df[cols[cand]]
                break

    if "store_id" not in df.columns:
        if "ordernumber" in cols:
            df["store_id"] = pd.to_numeric(df[cols["ordernumber"]], errors="coerce")
        elif "customername" in cols:
            df["store_id"] = df[cols["customername"]].astype(str).factorize()[0] + 1
        else:
            df["store_id"] = 1

    required = {"date", "store_id", "item_id", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # Parse and clean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["store_id"] = df["store_id"].astype(str).str.strip()
    df["store_id"] = pd.factorize(df["store_id"])[0] + 1
    df = df.dropna(subset=["date", "item_id", "sales"])  # remove unusable rows
    
    # Handle sparsity: aggregate multiple sales per day, then resample daily to fill missing dates with 0.
    # A missing transaction in retail data generally implies zero demand/sales for that day.
    df = (
        df.groupby(["store_id", "item_id", "date"])["sales"]
        .sum()
        .reset_index()
    )
    
    # Reindex each store-item combination to have a complete daily date range
    max_date = df["date"].max()
    def fill_group(grp):
        if grp.empty:
            return grp
        idx = pd.date_range(start=grp["date"].min(), end=max_date, freq="D")
        grp = grp.set_index("date").reindex(idx)
        grp["sales"] = grp["sales"].fillna(0)
        return grp.rename_axis("date")
        
    df = (
        df.groupby(["store_id", "item_id"])
        .apply(fill_group, include_groups=False)
        .reset_index()
    )
    
    return df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-based features that help capture seasonality and trends."""
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.weekday
    df["day_of_year"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 7, 14],
    windows: List[int] = [7, 14, 30],
) -> pd.DataFrame:
    """Create lag features and rolling statistics by store-item group."""
    df = df.copy()
    group_cols = ["store_id", "item_id"]
    df = df.sort_values(group_cols + ["date"])

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)["sales"].shift(lag)

    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df.groupby(group_cols)["sales"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = (
            df.groupby(group_cols)["sales"].shift(1).rolling(window=window, min_periods=1).std()
        )
        df[f"rolling_max_{window}"] = (
            df.groupby(group_cols)["sales"].shift(1).rolling(window=window, min_periods=1).max()
        )
        df[f"rolling_min_{window}"] = (
            df.groupby(group_cols)["sales"].shift(1).rolling(window=window, min_periods=1).min()
        )

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering pipeline for model input."""
    df = create_date_features(df)
    df = create_lag_and_rolling_features(df)

    # Ensure identifiers are numeric for model training.
    df["item_id"] = pd.factorize(df["item_id"].astype(str))[0] + 1
    df["store_id"] = pd.factorize(df["store_id"].astype(str))[0] + 1

    # Drop the earliest rows that do not have lag feature values.
    features = [
        "day",
        "month",
        "year",
        "weekday",
        "day_of_year",
        "weekofyear",
        "is_weekend",
        "store_id",
        "item_id",
    ]
    features += [col for col in df.columns if col.startswith("lag_") or col.startswith("rolling_")]

    df = df.dropna(subset=[col for col in features if col not in ["store_id", "item_id"]])
    return df, features


def train_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add a naive baseline forecast such as last-day sales or weekly median."""
    df = df.copy()
    df["baseline_last_day"] = df.groupby(["store_id", "item_id"])["sales"].shift(1)
    df["baseline_weekly_median"] = (
        df.groupby(["store_id", "item_id"])["sales"].shift(1).rolling(window=7, min_periods=1).median()
    )
    return df


def split_data(df: pd.DataFrame, feature_cols: List[str], target_col: str = "sales") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into train and validation sets using the last calendar block.

    If the dataset is too small or the last 30-day split is empty, fall back to an 80/20 row split.
    """
    df = df.copy().sort_values("date")
    last_date = df["date"].max()
    cutoff_date = last_date - pd.Timedelta(days=30)

    train = df[df["date"] <= cutoff_date]
    valid = df[df["date"] > cutoff_date]

    if train.empty or valid.empty:
        if len(df) < 5:
            return df[feature_cols], pd.DataFrame(columns=feature_cols), df[target_col], pd.Series([], dtype=df[target_col].dtype)
        split_index = max(1, int(len(df) * 0.8))
        train = df.iloc[:split_index]
        valid = df.iloc[split_index:]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_valid = valid[feature_cols]
    y_valid = valid[target_col]

    return X_train, X_valid, y_train, y_valid


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train a Random Forest regressor as a strong tree-based baseline."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> lgb.LGBMRegressor:
    """Train a LightGBM model with early stopping using the sklearn wrapper."""
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_lightgbm_tweedie(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    tweedie_variance_power: float = 1.5,
) -> lgb.LGBMRegressor:
    """Train a LightGBM model with Tweedie objective for zero-inflated demand."""
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=tweedie_variance_power,
        metric="rmse",
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


class HurdleModel:
    """Two-stage model: predicts if demand > 0, then predicts quantity if > 0."""
    def __init__(self):
        self.classifier = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.1,
            num_leaves=31,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        self.regressor = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.1,
            num_leaves=31,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Stage 1: Classifier (Demand > 0)
        y_class = (y > 0).astype(int)
        self.classifier.fit(X, y_class)
        
        # Stage 2: Regressor (Only on positive demand)
        pos_mask = y > 0
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        if len(y_pos) > 0:
            self.regressor.fit(X_pos, y_pos)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # P(demand > 0)
        prob_positive = self.classifier.predict_proba(X)[:, 1]
        
        # E[demand | demand > 0]
        # We ensure regression predictions aren't negative
        pred_quantity = np.maximum(0, self.regressor.predict(X))
        
        # Final expected value = P(>0) * E[>0]
        return prob_positive * pred_quantity

def train_hurdle_model(X_train: pd.DataFrame, y_train: pd.Series) -> HurdleModel:
    model = HurdleModel()
    model.fit(X_train, y_train)
    return model


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    valid = denom != 0
    return np.mean(diff[valid] / denom[valid]) * 100 if np.any(valid) else 0.0


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error."""
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom if denom != 0 else 0.0

def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, SMAPE, and WAPE scores."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def plot_predictions(
    df: pd.DataFrame,
    store_id: int,
    item_id: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Sales",
) -> None:
    """Plot actual and predicted sales for one store-item combination."""
    mask = (df["store_id"] == store_id) & (df["item_id"] == item_id)
    subset = df[mask].copy()
    if subset.empty:
        print(f"No data found for store_id={store_id}, item_id={item_id}")
        return

    subset = subset.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(subset["date"], subset["sales"], label="Actual", marker="o", alpha=0.8)
    ax.plot(subset["date"].iloc[-len(y_pred) :], y_pred, label="Predicted", marker="x", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.tight_layout()
    plt.show()


def suggest_restocking(
    df: pd.DataFrame,
    threshold: float = 1.2,
    safety_days: int = 7,
    lead_time_days: int = 7,
    min_order_qty: int = 10,
    urgency_percentile: float = 0.8,
) -> pd.DataFrame:
    """Identify item-store pairs that may need restocking based on recent demand."""
    if df.empty or not all(c in df.columns for c in ["date", "store_id", "item_id", "sales"]):
        return pd.DataFrame(
            columns=[
                "store_id",
                "item_id",
                "mean",
                "std",
                "sum",
                "count",
                "demand_signal",
                "daily_mean",
                "daily_std",
                "lead_time_qty",
                "safety_qty",
                "suggested_qty",
                "reorder_status",
                "recent_14d_sum",
                "trend_ratio",
            ]
        )

    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    if recent.empty:
        recent = df.copy()

    window_days = max(1, (recent["date"].max() - recent["date"].min()).days + 1)
    if window_days < 30:
        window_days = 30

    summary = (
        recent.groupby(["store_id", "item_id"])["sales"]
        .agg(["mean", "std", "sum", "count"])
        .reset_index()
    )
    summary["daily_mean"] = summary["sum"] / window_days
    summary["daily_std"] = summary["std"].fillna(0)
    summary["demand_signal"] = summary["daily_mean"] + threshold * summary["daily_std"]
    summary["lead_time_qty"] = summary["daily_mean"] * lead_time_days
    summary["safety_qty"] = summary["daily_mean"] * safety_days + summary["daily_std"] * 2
    summary["suggested_qty"] = np.ceil(
        (summary["lead_time_qty"] + summary["safety_qty"]) / max(1, min_order_qty)
    ) * max(1, min_order_qty)
    summary["suggested_qty"] = summary["suggested_qty"].fillna(0).astype(int)

    recent_14 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=14)]
    recent_14_sum = (
        recent_14.groupby(["store_id", "item_id"])["sales"]
        .sum()
        .reset_index(name="recent_14d_sum")
    )
    summary = summary.merge(recent_14_sum, on=["store_id", "item_id"], how="left").fillna({"recent_14d_sum": 0})
    summary["trend_ratio"] = np.where(
        summary["daily_mean"] > 0,
        (summary["recent_14d_sum"] / 14) / (summary["daily_mean"] + 1e-9),
        0,
    )

    urgency_threshold = summary["demand_signal"].quantile(urgency_percentile)
    summary["reorder_status"] = np.where(
        (summary["demand_signal"] >= urgency_threshold) | (summary["trend_ratio"] > 1.2),
        "Urgent",
        "Normal",
    )
    summary = summary.sort_values(["reorder_status", "demand_signal"], ascending=[False, False])
    return summary


def forecast_next_days(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    forecast_days: int = 30,
) -> pd.DataFrame:
    """Generate future forecasts for each store-item combination."""
    # Work on a copy and build rolling features in a recursive fashion.
    history = df.copy().sort_values(["store_id", "item_id", "date"])
    future_rows = []
    last_date = history["date"].max()

    for day_offset in range(1, forecast_days + 1):
        next_date = last_date + pd.Timedelta(days=day_offset)
        step_rows = []
        for (store_id, item_id), group in history.groupby(["store_id", "item_id"]):
            last_row = group.iloc[-1:].copy()
            row = {
                "date": next_date,
                "store_id": store_id,
                "item_id": item_id,
                "day": next_date.day,
                "month": next_date.month,
                "year": next_date.year,
                "weekday": next_date.weekday(),
                "day_of_year": next_date.dayofyear,
                "weekofyear": int(next_date.isocalendar()[1]),
                "is_weekend": int(next_date.weekday() >= 5),
            }

            for lag in [1, 7, 14]:
                row[f"lag_{lag}"] = group["sales"].shift(lag).iloc[-1] if len(group) > lag else np.nan

            for window in [7, 14, 30]:
                shifted = group["sales"].shift(1).iloc[-window:]
                row[f"rolling_mean_{window}"] = shifted.mean() if not shifted.empty else np.nan
                row[f"rolling_std_{window}"] = shifted.std(ddof=0) if len(shifted) > 1 else 0.0
                row[f"rolling_max_{window}"] = shifted.max() if not shifted.empty else np.nan
                row[f"rolling_min_{window}"] = shifted.min() if not shifted.empty else np.nan

            step_rows.append(row)

        step_df = pd.DataFrame(step_rows)
        step_df = step_df[feature_cols]
        y_pred = model.predict(step_df)
        step_df["sales_forecast"] = y_pred
        step_df["date"] = next_date
        future_rows.append(step_df)

        # Append forecast values to history to use them for subsequent recursive steps.
        generated = step_df.copy()
        generated["sales"] = generated["sales_forecast"]
        history = pd.concat([history, generated], ignore_index=True, sort=False)

    forecast_df = pd.concat(future_rows, ignore_index=True, sort=False)
    return forecast_df[["date", "store_id", "item_id", "sales_forecast"]]


def run_pipeline(csv_path: str, forecast_days: int = 30) -> None:
    """Run the full demand forecasting pipeline end to end."""
    df = load_sales_data(csv_path)
    df = train_baseline(df)
    df_features, feature_cols = prepare_features(df)

    X_train, X_valid, y_train, y_valid = split_data(df_features, feature_cols)

    rf_model = train_random_forest(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train, X_valid, y_valid)

    y_pred_rf = rf_model.predict(X_valid)
    y_pred_lgb = lgb_model.predict(X_valid)

    print("Random Forest evaluation:", evaluate_forecasts(y_valid.values, y_pred_rf))
    print("LightGBM evaluation:", evaluate_forecasts(y_valid.values, y_pred_lgb))

    # Choose the better model by RMSE
    chosen_model = lgb_model if evaluate_forecasts(y_valid.values, y_pred_lgb)["rmse"] <= evaluate_forecasts(y_valid.values, y_pred_rf)["rmse"] else rf_model
    print("Using model:", type(chosen_model).__name__)

    forecast_df = forecast_next_days(df_features, chosen_model, feature_cols, forecast_days=forecast_days)
    print(f"Generated forecast for {forecast_days} future days. Sample rows:\n", forecast_df.head())

    restock_suggestions = suggest_restocking(df)
    print("Top restocking candidates:\n", restock_suggestions.head())

    # Example visualization for the first store-item pair in validation data
    if not X_valid.empty:
        first_store, first_item = X_valid.iloc[0]["store_id"], X_valid.iloc[0]["item_id"]
        plot_predictions(df, int(first_store), int(first_item), y_valid.iloc[:30], y_pred_lgb[:30])


if __name__ == "__main__":
    print("Store Demand Forecasting pipeline module loaded.")
    print("Use run_pipeline(csv_path, forecast_days=30) with a historical sales dataset.")
