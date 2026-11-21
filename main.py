import os
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from pydantic import BaseModel, Field
import joblib

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from datetime import datetime, timezone

app = FastAPI(title="API intelligente Iris", version="1.0.0")

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

model = joblib.load("iris_model.pkl")
classes = ["setosa", "versicolor", "virginica"]
MODEL_VERSION = "1.0.0"

API_KEY = os.getenv("API_KEY", "alae-api-key")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)

class IrisBatchInput(BaseModel):
    observations: list[IrisInput]

def verify_api_key(
    x_api_key: str | None = Header(default=None)
):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API absente ou invalide"
        )

@app.get("/")
def home():
    return {"message": "Bienvenue dans l'API Iris"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
@limiter.limit("5/minute")
def predict(
    request: Request,
    data: IrisInput,
    _: None = Depends(verify_api_key)
):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": classes[prediction],
        "confidence": round(float(max(probabilities)), 4),
        "status": "success"
    }

@app.post("/predict/batch")
@limiter.limit("5/minute")

def predict_batch(
    request: Request,
    data: IrisBatchInput,
    _: None = Depends(verify_api_key)
):

    features = [
        [
            obs.sepal_length,
            obs.sepal_width,
            obs.petal_length,
            obs.petal_width
        ]
        for obs in data.observations
    ]

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    results = []

    for prediction, probs in zip(predictions, probabilities):

        results.append({
            "prediction": classes[int(prediction)],
            "probabilities": {
                classes[i]: round(float(probs[i]), 4)
                for i in range(len(classes))
            }
        })

    return {
        "predictions": results,
        "total": len(results),
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success"
    }

@app.post("/predict-secure")
def predict_secure(
    data: IrisInput,
    x_api_key: str | None = Header(default=None)
):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Clé API absente ou invalide"
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "prediction": classes[prediction],
        "status": "success"
    }

@app.get("/limited")
@limiter.limit("5/minute")
def limited(request: Request):
    return {"message": "Requête autorisée"}