"""API REST de classification des fleurs Iris.

Expose un modèle Random Forest entraîné avec scikit-learn (voir train_model.py).
L'authentification par clé API est réservée à l'endpoint de démonstration
/predict-secure ; les endpoints /predict et /predict/batch restent ouverts.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

MODEL_VERSION = "1.0.0"
MODEL_PATH = Path(__file__).with_name("iris_model.pkl")
CLASSES = ["setosa", "versicolor", "virginica"]
API_KEY = os.getenv("API_KEY", "alae-api-key")

app = FastAPI(
    title="API intelligente Iris",
    description=(
        "Classification de fleurs Iris à partir de quatre mesures "
        "(longueur et largeur du sépale et du pétale)."
    ),
    version=MODEL_VERSION,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------
# Schémas Pydantic
# --------------------------------------------------------------------------
class IrisInput(BaseModel):
    """Une observation Iris (mesures en centimètres)."""

    sepal_length: float = Field(..., gt=0, le=10, description="Longueur du sépale (cm)")
    sepal_width: float = Field(..., gt=0, le=10, description="Largeur du sépale (cm)")
    petal_length: float = Field(..., gt=0, le=10, description="Longueur du pétale (cm)")
    petal_width: float = Field(..., gt=0, le=10, description="Largeur du pétale (cm)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class IrisBatchInput(BaseModel):
    """Plusieurs observations à prédire en une seule requête."""

    observations: list[IrisInput] = Field(
        ..., min_length=1, description="Liste d'observations (au moins une)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "observations": [
                        {
                            "sepal_length": 5.1,
                            "sepal_width": 3.5,
                            "petal_length": 1.4,
                            "petal_width": 0.2,
                        },
                        {
                            "sepal_length": 6.3,
                            "sepal_width": 3.3,
                            "petal_length": 6.0,
                            "petal_width": 2.5,
                        },
                    ]
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Classe prédite")
    confidence: float = Field(..., ge=0, le=1, description="Probabilité de la classe prédite")
    status: str = "success"


class SecurePredictResponse(BaseModel):
    prediction: str = Field(..., description="Classe prédite")
    status: str = "success"


class BatchPrediction(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class BatchResponse(BaseModel):
    predictions: list[BatchPrediction]
    total: int
    model_version: str
    timestamp: str
    status: str = "success"


# --------------------------------------------------------------------------
# Sécurité et utilitaires
# --------------------------------------------------------------------------
def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Refuse la requête si la clé API est absente ou incorrecte."""
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API absente ou invalide")


ApiKey = Annotated[None, Depends(verify_api_key)]


def _predict(data: IrisInput) -> tuple[str, float]:
    """Retourne la classe prédite et sa probabilité."""
    features = [
        [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    ]
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    return CLASSES[prediction], round(float(max(probabilities)), 4)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", tags=["Santé"])
def home() -> dict[str, str]:
    """Message d'accueil de l'API."""
    return {"message": "Bienvenue dans l'API Iris"}


@app.get("/health", tags=["Santé"])
def health() -> dict[str, str]:
    """État de santé du service."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
@limiter.limit("5/minute")
def predict(request: Request, data: IrisInput) -> PredictResponse:
    """Prédit l'espèce d'une observation Iris (endpoint ouvert)."""
    prediction, confidence = _predict(data)
    return PredictResponse(prediction=prediction, confidence=confidence)


@app.post(
    "/predict-secure",
    response_model=SecurePredictResponse,
    tags=["Prédiction"],
    summary="Prédiction protégée par clé API",
)
@limiter.limit("5/minute")
def predict_secure(
    request: Request,
    data: IrisInput,
    _: ApiKey,
) -> SecurePredictResponse:
    """Prédit l'espèce d'une observation ; nécessite l'en-tête x-api-key."""
    prediction, _confidence = _predict(data)
    return SecurePredictResponse(prediction=prediction)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prédiction"])
@limiter.limit("5/minute")
def predict_batch(request: Request, data: IrisBatchInput) -> BatchResponse:
    """Prédit plusieurs observations avec les probabilités de chaque classe."""
    features = [
        [obs.sepal_length, obs.sepal_width, obs.petal_length, obs.petal_width]
        for obs in data.observations
    ]

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = [
        BatchPrediction(
            prediction=CLASSES[int(prediction)],
            probabilities={
                CLASSES[i]: round(float(probs[i]), 4)
                for i in range(len(CLASSES))
            },
        )
        for prediction, probs in zip(predictions, probabilities)
    ]

    return BatchResponse(
        predictions=results,
        total=len(results),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/limited", tags=["Démonstration"])
@limiter.limit("5/minute")
def limited(request: Request) -> dict[str, str]:
    """Endpoint de démonstration du rate limiting (5 requêtes / minute)."""
    return {"message": "Requête autorisée"}
