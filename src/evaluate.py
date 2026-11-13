from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Scorecard API",
    version="1.0.0",
    description="Production-grade credit scoring and P&L optimization decision engine."
)

# Load serialized model package on startup
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "credit_scorecard_model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Please run training script first.")

model_package = joblib.load(MODEL_PATH)
pipeline = model_package['pipeline']
features = model_package['features']
threshold = model_package['optimal_threshold']

# Pydantic Schema with JSON field validation
class ApplicantData(BaseModel):
    age: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberRealEstateLoansOrLines: int
    NumberOfDependents: int
    Log_MonthlyIncome: float
    Log_Utilization: float
    Log_DebtRatio: float
    TotalPastDueCount: int
    IncomePerDependent: float
    IsHighUtilization: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfTimes90DaysLate: int

@app.get("/health")
def health_check():
    return {"status": "healthy", "threshold_loaded": threshold}

@app.post("/score")
def score_applicant(data: ApplicantData):
    try:
        # Convert incoming payload to dictionary
        input_dict = data.dict()
        
        # Map underscore keys back to the original hyphenated feature names expected by the pipeline
        key_mapping = {
            "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse"
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in input_dict:
                input_dict[new_key] = input_dict.pop(old_key)
        
        # Align features in exact training order
        input_df = pd.DataFrame([input_dict])[features]
        
        # Predict probability of default
        default_prob = pipeline.predict_proba(input_df)[0][1]
        
        # Calculate FICO-style score mapping
        base_score = 505.74
        fico_score = int(base_score - (default_prob * 150))
        fico_score = max(300, min(850, fico_score))
        
        # Evaluate decision against P&L optimized threshold..
        decision = "APPROVE" if default_prob < threshold else "DECLINE"
        
        return {
            "default_probability": round(float(default_prob), 4),
            "credit_score": fico_score,
            "decision": decision,
            "threshold_used": threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))