from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn using XGBoost",
    version="1.0.0"
)


# --------------------------------------------------
# Load model and threshold
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "model",
    "customer_churn_model.pkl"
)

threshold_path = os.path.join(
    BASE_DIR,
    "model",
    "churn_threshold.pkl"
)

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running",
        "model": "Tuned XGBoost",
        "threshold": threshold
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert input to dictionary
    data = customer.model_dump()

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([data])

    # Predict probability
    churn_probability = model.predict_proba(input_df)[0][1]

    # Apply optimized threshold
    prediction = int(churn_probability >= threshold)

    # Human-readable result
    churn_result = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": prediction,
        "churn": churn_result,
        "churn_probability": round(float(churn_probability), 4),
        "decision_threshold": round(float(threshold), 4)
    }