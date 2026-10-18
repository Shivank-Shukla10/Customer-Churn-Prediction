from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data


def test_prediction():
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["churn_prediction"] in [0, 1]
    assert 0 <= data["churn_probability"] <= 1
    assert "churn" in data
    assert "decision_threshold" in data