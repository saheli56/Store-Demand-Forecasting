# Store Demand Forecasting

An end-to-end machine learning pipeline for forecasting retail store demand across multiple products. This project ingests transactional sales data, engineers time-series features, handles extreme data sparsity, and utilizes advanced gradient boosting architectures to generate reliable multi-step predictions.

## Core Features

- **Robust Time-Series Pipeline:** Automatically processes raw transactional logs into continuous, chronological daily records. Efficiently handles missing dates by dynamic reindexing and zero-filling to maintain statistical integrity.
- **Advanced Forecasting Models:** Implements standard LightGBM for dense, continuous demand, alongside specialized formulations (LightGBM Tweedie Regression and dual-stage Hurdle classifiers) to effectively model zero-inflated, highly sparse retail environments without introducing false-positive volume spikes.
- **Automated Feature Engineering:** Generates temporal indicators, shifted lags, and rolling historical moments (mean, standard deviation, min, max) while enforcing strict temporal masking to guarantee zero data leakage.
- **Interactive Dashboard:** A Streamlit-based web application providing a user-friendly interface to upload datasets, select store-item combinations, and visualize historical sales trends against future forecasts.

## Technology Stack

- **Data Processing:** Pandas, NumPy
- **Machine Learning:** LightGBM, Scikit-Learn
- **Web Interface:** Streamlit

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Store-Demand-Forecasting
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To launch the interactive forecasting dashboard:

```bash
streamlit run streamlit_app.py
```

1. Access the application in your browser (typically `http://localhost:8501`).
2. Upload your historical sales dataset (`sales_data_sample.csv` or similar).
3. Select the desired store and item combination from the sidebar.
4. Set your forecasting horizon and view the generated predictions.

## Architecture Overview

The system architecture focuses on strict validation methodologies and computationally efficient inference:

- **Validation Strategy:** Time-aware chronological validation utilizing the final 30 days of data strictly segregated from the historical training window.
- **Sparsity Management:** Uses a conditional approach—employing Standard LightGBM for high-volume items and Tweedie regression for intermittent demand where >99% of dates represent zero sales. 
- **Latency:** Capable of generating thousands of multi-step forecasts in under 50ms (measured prediction time).