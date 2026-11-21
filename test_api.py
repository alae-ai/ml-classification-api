from fastapi.testclient import TestClient
from main import app, API_KEY

client = TestClient(app)

headers = {"x-api-key": API_KEY}

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Bienvenue dans l'API Iris"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    response = client.post("/predict",headers=headers,json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })

    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "setosa", "versicolor", "virginica"
    ]
    assert response.json()["status"] == "success"


def test_invalid_input():
    response = client.post("/predict",headers=headers,json={
        "sepal_length": -5,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })

    assert response.status_code == 422

def test_missing_field():
    response = client.post(
        "/predict",
        headers={"x-api-key": API_KEY},
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
        }
    )

    assert response.status_code == 422

def test_unauthorized_access():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 401