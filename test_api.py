"""Tests automatisés de l'API Iris (pytest + TestClient).

Politique d'accès testée :
- /predict et /predict/batch : ouverts ;
- /predict-secure : clé API obligatoire (401 sinon).
"""

from fastapi.testclient import TestClient

from main import API_KEY, CLASSES, app

client = TestClient(app)

VALID = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

AUTH = {"x-api-key": API_KEY}


# --------------------------------------------------------------------------
# Endpoints de santé
# --------------------------------------------------------------------------
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Bienvenue dans l'API Iris"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --------------------------------------------------------------------------
# /predict : ouvert, sans clé API
# --------------------------------------------------------------------------
def test_prediction_without_api_key():
    """L'endpoint /predict doit rester accessible sans clé API."""
    response = client.post("/predict", json=VALID)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in CLASSES
    assert 0 <= body["confidence"] <= 1
    assert body["status"] == "success"


def test_invalid_input():
    """Une mesure négative doit être refusée par Pydantic."""
    response = client.post("/predict", json={**VALID, "sepal_length": -5})
    assert response.status_code == 422


def test_invalid_type():
    """Une chaîne à la place d'un nombre doit être refusée."""
    response = client.post("/predict", json={**VALID, "sepal_length": "abc"})
    assert response.status_code == 422


def test_missing_field():
    """Un champ obligatoire manquant doit renvoyer 422."""
    payload = {key: value for key, value in VALID.items() if key != "petal_width"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# --------------------------------------------------------------------------
# /predict-secure : clé API obligatoire
# --------------------------------------------------------------------------
def test_secure_prediction_without_key():
    response = client.post("/predict-secure", json=VALID)
    assert response.status_code == 401
    assert response.json()["detail"] == "Clé API absente ou invalide"


def test_secure_prediction_with_wrong_key():
    response = client.post(
        "/predict-secure", json=VALID, headers={"x-api-key": "mauvaise-cle"}
    )
    assert response.status_code == 401


def test_secure_prediction_with_valid_key():
    response = client.post("/predict-secure", json=VALID, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in CLASSES
    assert body["status"] == "success"


# --------------------------------------------------------------------------
# /predict/batch : ouvert, plusieurs observations
# --------------------------------------------------------------------------
def test_batch_prediction():
    payload = {
        "observations": [
            VALID,
            {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 6.0,
                "petal_width": 2.5,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["predictions"]) == 2
    assert body["status"] == "success"
    assert body["model_version"]
    assert body["timestamp"]

    for item in body["predictions"]:
        assert item["prediction"] in CLASSES
        assert set(item["probabilities"]) == set(CLASSES)


def test_batch_empty_list():
    """Un lot vide doit être refusé (422) et non provoquer une erreur 500."""
    response = client.post("/predict/batch", json={"observations": []})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
def test_rate_limit():
    """La sixième requête en moins d'une minute doit renvoyer 429."""
    statuses = [client.get("/limited").status_code for _ in range(6)]

    assert statuses[:5] == [200] * 5
    assert statuses[5] == 429
