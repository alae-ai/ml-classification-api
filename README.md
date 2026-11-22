# API intelligente de classification Iris

API REST développée avec **FastAPI** qui expose un classifieur **Random Forest**
entraîné avec **scikit-learn**. À partir de quatre mesures (longueur et largeur du
sépale et du pétale), elle prédit l'espèce d'une fleur Iris : **Setosa**,
**Versicolor** ou **Virginica**.

## Structure du projet

```
.
├── main.py            # application FastAPI (endpoints, schémas, sécurité)
├── train_model.py     # entraînement et sérialisation du modèle
├── test_api.py        # tests automatisés (pytest)
├── iris_model.pkl     # modèle entraîné (généré par train_model.py)
├── requirements.txt   # dépendances épinglées
└── README.md
```

## Prérequis

- Python 3.10 ou supérieur (testé avec Python 3.13)
- pip

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entraînement du modèle

```bash
python train_model.py
# Modèle Iris entraîné et sauvegardé.
```

Le script entraîne un `RandomForestClassifier` (100 arbres, `random_state=42`)
sur le jeu de données Iris et génère `iris_model.pkl`. Le fichier est déjà fourni :
cette étape n'est nécessaire que pour régénérer le modèle.

## Configuration

La clé API de démonstration est lue dans la variable d'environnement `API_KEY` :

```bash
export API_KEY="ma-cle-secrete"     # Windows : $env:API_KEY="ma-cle-secrete"
```

Sans variable définie, la valeur par défaut `alae-api-key` est utilisée.

## Exécution

```bash
python -m uvicorn main:app --reload
```

- API : http://127.0.0.1:8000
- Documentation interactive : http://127.0.0.1:8000/docs
- Schéma OpenAPI : http://127.0.0.1:8000/openapi.json

## Endpoints

| Méthode | Endpoint | Clé API | Description |
|---|---|---|---|
| GET | `/` | non | Message d'accueil |
| GET | `/health` | non | État de santé du service |
| POST | `/predict` | non | Prédiction d'une observation |
| POST | `/predict/batch` | non | Prédiction de plusieurs observations |
| POST | `/predict-secure` | **oui** | Prédiction protégée par clé API |
| GET | `/limited` | non | Démonstration du rate limiting |

Les endpoints de prédiction sont limités à **5 requêtes par minute et par adresse IP**.

## Exemples

### Prédiction simple (sans clé)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

```json
{"prediction": "setosa", "confidence": 1.0, "status": "success"}
```

### Prédiction protégée

```bash
# Sans clé : 401
curl -X POST http://127.0.0.1:8000/predict-secure \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# Avec la bonne clé : 200
curl -X POST http://127.0.0.1:8000/predict-secure \
  -H "Content-Type: application/json" \
  -H "x-api-key: alae-api-key" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Prédiction par lot

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"observations":[
        {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},
        {"sepal_length":6.3,"sepal_width":3.3,"petal_length":6.0,"petal_width":2.5}]}'
```

```json
{
  "predictions": [
    {"prediction": "setosa",
     "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}},
    {"prediction": "virginica",
     "probabilities": {"setosa": 0.0, "versicolor": 0.0, "virginica": 1.0}}
  ],
  "total": 2,
  "model_version": "1.0.0",
  "timestamp": "2026-11-21T12:09:44.120745+00:00",
  "status": "success"
}
```

## Validation des entrées

Les quatre mesures sont obligatoires, numériques, strictement positives et
inférieures ou égales à 10. Toute violation renvoie **422 Unprocessable Content**
avec le détail du champ fautif (type, valeur, contrainte).

## Tests

```bash
pytest -v
```

Les 12 tests couvrent :

- les endpoints de santé (`/`, `/health`) ;
- `/predict` accessible **sans** clé API ;
- les entrées invalides : valeur négative, type incorrect, champ manquant ;
- `/predict-secure` : clé absente, clé erronée, clé valide ;
- `/predict/batch` : lot valide, lot vide ;
- le rate limiting : 6ᵉ requête rejetée en 429.

> `test_rate_limit` consomme le quota de `/limited` : attendre une minute avant de
> relancer la suite dans le même processus.

## Sécurité et limites

- La clé API est une protection **pédagogique**. En production, préférer
  JWT/OAuth2, HTTPS et une gestion de secrets dédiée.
- Le rate limiting utilise un stockage mémoire local : en multi-instances,
  prévoir Redis pour partager les compteurs.
- Le modèle est chargé une seule fois au démarrage du service.
