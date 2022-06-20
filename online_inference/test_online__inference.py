from fastapi.testclient import TestClient
from online_inference import app

client = TestClient(app)

def test_predict_item():
    """We send a post request with a person who certainly has a disease"""
    response = client.post(
        "/predict/",
        json={
            "age": 65,
            "sex": 1,
            "cp": 0, 
            "trestbps": 138,
            "chol": 282, 
            "fbs": 1, 
            "restecg": 2, 
            "thalach": 174, 
            "exang": 0, 
            "oldpeak": 1.4,
            "slope": 1, 
            "ca": 1, 
            "thal": 0
        }
        )
    assert response.status_code == 200
    assert response.json() == "Artificial doctor's verdict is: True"

test_predict_item()