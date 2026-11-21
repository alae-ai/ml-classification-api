# API intelligente de classification Iris

API REST développée avec FastAPI permettant de classifier des fleurs Iris à l'aide d'un modèle Random Forest entraîné avec Scikit-learn.

## Fonctionnalités

- Classification des espèces Iris : Setosa, Versicolor et Virginica.
- Prédiction individuelle et multiple (Batch).
- Retour des probabilités pour chaque classe.
- Validation des données avec Pydantic.
- Authentification par clé API.
- Limitation du nombre de requêtes avec SlowAPI.
- Versionnement du modèle et horodatage des prédictions.
- Tests automatisés avec Pytest.

## Technologies utilisées

- Python
- FastAPI
- Scikit-learn
- Pydantic
- Joblib
- SlowAPI
- Pytest

## Installation

Cloner le projet et accéder au répertoire :

```bash
git clone <URL_DU_PROJET>
cd api_iris
```

Créer et activer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install fastapi uvicorn scikit-learn joblib pytest httpx slowapi
```

## Entraînement du modèle

```bash
python train_model.py
```

Cette commande entraîne le modèle Random Forest et génère `iris_model.pkl`.

## Configuration

Définir la clé API :

```bash
export API_KEY="my-secret-key"
```

## Exécution

```bash
python -m uvicorn main:app --reload
```

API disponible à l'adresse :

http://127.0.0.1:8000

Documentation interactive :

http://127.0.0.1:8000/docs

## Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | / | Accueil |
| GET | /health | Vérification de disponibilité |
| POST | /predict | Prédiction individuelle |
| POST | /predict/batch | Prédictions multiples |
| GET | /limited | Test du Rate Limiting |

Les endpoints de prédiction nécessitent le header `x-api-key`.

Limitation : 5 requêtes par minute et par adresse IP sur chaque endpoint concerné.

## Tests

Exécuter les tests automatisés :

```bash
pytest -v
```

## Sécurité

L'API utilise une clé API pour protéger les endpoints de prédiction et SlowAPI pour limiter les requêtes.

Pour un environnement de production, privilégier une authentification robuste (JWT/OAuth2), HTTPS, une gestion sécurisée des secrets et un stockage partagé comme Redis pour le Rate Limiting.
