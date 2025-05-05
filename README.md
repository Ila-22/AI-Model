# AI-Model
A time-series analysis and forecasting pipeline for categorised traffic accidents

 - The pipeline includes seasonal, lag, and year-over-year features.

 - Developed with FastAPI, scikit-learn, and deployed on Heroku free tier.

# Project Structure
AI-Model/

├── app.py              # FastAPI application serving the forecasting model

├── df_clean.pkl        # Preprocessed DataFrame for feature lookups

├── pipeline.pkl        # Trained scikit-learn Pipeline for forecasting

├── requirements.txt    # Python dependencies

├── Procfile            # Heroku process file

└── PostRequest.py      # Script to submit DPS challenge mission

# Deployment
This app is deployed on Heroku at:
https://forecast11.herokuapp.com/forecast
