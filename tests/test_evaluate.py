import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Test the API health endpoint to ensure the microservice is running and has successfully loaded the optimized model and P&L threshold.
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "threshold_loaded" in data

# Test the /score endpoint with a low-risk borrower profile to verify that the pipeline correctly calculates the default probability, maps it to a FICO-style score, and returns an 'APPROVE' decision.
def test_score_applicant_approved():
    payload = {
        "age": 42.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfDependents": 2,
        "Log_MonthlyIncome": 10.45,
        "Log_Utilization": 0.15,
        "Log_DebtRatio": 0.32,
        "TotalPastDueCount": 0,
        "IncomePerDependent": 25000.0,
        "IsHighUtilization": 0,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "default_probability" in data
    assert "credit_score" in data
    assert "decision" in data
    assert data["decision"] == "APPROVE"

def test_score_applicant_validation_error():
    # Sending invalid data type (string instead of float/int) for age
    payload = {
        "age": "forty-two",
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfDependents": 2,
        "Log_MonthlyIncome": 10.45,
        "Log_Utilization": 0.15,
        "Log_DebtRatio": 0.32,
        "TotalPastDueCount": 0,
        "IncomePerDependent": 25000.0,
        "IsHighUtilization": 0,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation failure)