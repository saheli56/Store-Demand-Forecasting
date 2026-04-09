import os
from typing import List, Tuple

import lightgbm as lgb
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

    df = pd.read_csv(csv_path, parse_dates=["date"])
    expected_cols = {"date", "store_id", "item_id", "sales"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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
    """Split the data into train and validation sets using the last calendar block."""
    df = df.copy()
    last_date = df["date"].max()
    cutoff_date = last_date - pd.Timedelta(days=30)

    train = df[df["date"] <= cutoff_date]
    valid = df[df["date"] > cutoff_date]

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


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    valid = denom != 0
    return np.mean(diff[valid] / denom[valid]) * 100 if np.any(valid) else 0.0


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, and SMAPE scores."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "smape": smape(y_true, y_pred),
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


def suggest_restocking(df: pd.DataFrame, threshold: float = 1.2) -> pd.DataFrame:
    """Identify item-store pairs that may need restocking based on recent demand."""
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    summary = (
        recent.groupby(["store_id", "item_id"])["sales"]
        .agg(["mean", "std", "sum", "count"])
        .reset_index()
    )
    summary["demand_signal"] = summary["mean"] + threshold * summary["std"]
    return summary.sort_values("demand_signal", ascending=False)


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
