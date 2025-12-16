from fastapi import FastAPI
from src.api.pydantic_models import CustomerData
import pandas as pd
import joblib

app = FastAPI(title="Credit Risk Probability API")

model = joblib.load("models/RandomForest.pkl")

@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])
    prob = model.predict_proba(df)[:,1]
    return {"risk_probability": float(prob[0])}
