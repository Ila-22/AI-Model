# 🚦 AI-Model: Forecasting Categorized Traffic Accidents

This project provides a robust time-series forecasting pipeline for traffic accident data, categorized by type and cause (e.g., alcohol, hit-and-run). It combines classical feature engineering (lags, seasonality, year-over-year change) with traditional ML models like Linear Regression, Random Forest, and XGBoost.

## 🔍 Features

- Forecasts accident counts per month, per category (e.g., Alcohol, Hit-and-run)
- Includes lag features, cyclic seasonal encodings, and aggregate year-based metrics
- Model comparison framework with MAE, RMSE, and R² outputs
- Dedicated forecasting and visualization for custom periods (e.g., January 2021)
- REST API with FastAPI for serving trained models

---

## 📁 Project Structure

```text
AI-Model/
│
├── models/                        # All ML model wrappers
│   ├── BaselineLinearRegressor.py
│   ├── BaselineRandomForestRegressor.py
│   ├── BaselineXGBoostRegressor.py
│   └── GradBoostModel.py
│
├── tools/
│   └── DataProcessor.py          # Feature engineering utilities
│
├── app.py                        # FastAPI app for live predictions
├── main.ipynb                    # Main notebook for model training, plots, and analysis
├── df_clean.pkl                  # Preprocessed DataFrame (cached)
├── pipeline.pkl                  # Trained pipeline (latest model)
├── monatszahlen2501_*.csv        # Original input dataset
├── PostRequest.py                # Sample client to send POST requests to API
├── Procfile                      # Heroku entry point
├── requirements.txt              # Python dependencies
└── README.md                     # You're here!
