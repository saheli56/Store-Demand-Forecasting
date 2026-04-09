# Store Demand Forecasting

This project provides a modular Python pipeline for building a store-item demand forecasting system using historical sales data.

## Goal
Predict daily product demand for multiple stores and items, capturing seasonality, trends, and store/item-specific behavior.

## Key Components
- Data loading and validation
- Date-based feature engineering
- Lag features and rolling statistics
- Baseline model logic
- Random Forest and LightGBM training
- Forecast evaluation with RMSE, MAE, and SMAPE
- Forecast generation for future days
- Restocking insight suggestions
- Visualizations for actual vs predicted demand

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib

## Usage
1. Place your CSV file in the workspace.
2. Ensure columns: `date`, `store_id`, `item_id`, `sales`.
3. Run the pipeline from Python:

```python
from store_demand_forecasting import run_pipeline

run_pipeline("path/to/sales_data.csv", forecast_days=30)
```

## Web App
Start the website with Streamlit:

```bash
streamlit run streamlit_app.py
```

The app lets you upload a CSV or select a local file, train models, compare RMSE/MAE/SMAPE, review forecast output, and explore store-item demand trends.

## Notes
- Replace `path/to/sales_data.csv` with your dataset location.
- The pipeline is designed for learning and can be extended with hyperparameter tuning, Prophet/ARIMA, or deep learning models.
