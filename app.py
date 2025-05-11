# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle

# request & response schemas     
class ForecastRequest(BaseModel):
    year: int = Field(..., example=2021)
    month: int = Field(..., ge=1, le=12, example=1)

class ForecastResponse(BaseModel):
    prediction: float

# load trained pipeline 
with open("model.pkl", "rb") as f: #pipeline.pkl
    pipeline = pickle.load(f)

# prep stats
df = pd.read_pickle("df_clean.pkl")  
ts = (
    df
    .query("Category=='Alcohol' and Accident_type=='Total'")
    .set_index("Month")['Value']
    .sort_index()
)
year2020_avg = ts[ts.index.year == 2020].mean()

# create the FastAPI app
app = FastAPI(title="Accident Forecast API")

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # 5) Build the feature row exactly as in your notebook
    target = pd.Period(f"{req.year}-{req.month:02d}", freq="M")
    m = req.month
    row = {
        "Category":                   "Alcohol",
        "Accident_type":              "Total",
        "Year":                       req.year,
        "Month":                      str(target),
        "PreviousYearValue":          ts.shift(12).get(target, np.nan),
        "lag_1":                      ts.shift(1).get(target, np.nan),
        "lag_2":                      ts.shift(2).get(target, np.nan),
        "ChangeFromPreviousMonth":    np.nan,
        "ChangeFromSameMonthLastYear":np.nan,
        "Year_Avg":                   year2020_avg,
        "month_num":                  m,
        "sin_month":                  np.sin(2*np.pi*m/12),
        "cos_month":                  np.cos(2*np.pi*m/12),
        "Season":                     target.to_timestamp().strftime("%B")
    }
    X_new = pd.DataFrame([row])
    X_new = X_new[pipeline.feature_names_in_]
    
    # run the prediction
    try:
        pred = pipeline.predict(X_new)[0]
    except Exception as e:
        raise HTTPException(500, f"Model prediction failed: {e}")
    
    # return the result
    return ForecastResponse(prediction=float(pred))
