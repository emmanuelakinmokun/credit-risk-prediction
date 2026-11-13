from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# 1. Initialize FastAPI app
app = FastAPI(title="Credit Risk Scorecard API", version="1.0")

# 2. Load serialized model package on startup
model_package = joblib.load('/home/nika/Documents/crp/models/credit_scorecard_model.pkl')
pipeline = model_package['pipeline']
features = model_package['features']
threshold = model_package['optimal_threshold']

# 3. Define expected incoming JSON request schema
class ApplicantData(BaseModel):
    Log_Utilization: float
    NumberOfTimes90DaysLate: int
    TotalPastDueCount: int
    age: float
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    IncomePerDependent: float
    NumberRealEstateLoansOrLines: int
    NumberOfOpenCreditLinesAndLoans: int
    Log_DebtRatio: float
    IsHighUtilization: int
    Log_MonthlyIncome: float
    NumberOfDependents: int

@app.post("/score")
def score_applicant(data: ApplicantData):
    try:
        # 1. Convert Pydantic model to a dictionary
        input_dict = data.dict()
        
        # 2. Map underscore keys back to the original hyphenated feature names expected by the model
        key_mapping = {
            "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse"
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in input_dict:
                input_dict[new_key] = input_dict.pop(old_key)
        
        # 3. Create DataFrame using the exact feature order the pipeline was trained on
        input_df = pd.DataFrame([input_dict])[features]
        
        # 4. Predict probability and calculate score
        default_prob = pipeline.predict_proba(input_df)[0][1]
        
        base_score = 505.74
        fico_score = int(base_score - (default_prob * 150))
        fico_score = max(300, min(850, fico_score))
        
        decision = "APPROVE" if default_prob < threshold else "DECLINE"
        
        return {
            "default_probability": round(float(default_prob), 4),
            "credit_score": fico_score,
            "decision": decision,
            "threshold_used": threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))