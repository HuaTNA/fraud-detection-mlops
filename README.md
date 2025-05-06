# 🛡️ Fraud Detection MLOps System

A production-ready machine learning pipeline for real-time credit card fraud detection, built with **FastAPI**, **MLflow**, **XGBoost**, and Dockerized for scalable deployment.

---

## Features

- ML pipeline for fraud detection using XGBoost
- Model tracking & versioning with MLflow
- REST API for real-time prediction (FastAPI)
- Dockerized with `docker-compose` for local deployment
- SHAP-based explainability support (optional)

---

## Project Structure

```
fraud-detection-mlops/
├── api/                  # FastAPI backend
│   ├── main.py           # FastAPI app entrypoint
│   ├── routers/          # API endpoint modules
│   ├── services/         # Prediction logic
│   └── schemas/          # Pydantic input model
├── src/                  # Core ML logic
│   ├── data/             # Data loading & cleaning
│   ├── features/         # Feature engineering
│   ├── model/            # Training & evaluation
│   ├── utils/            # Logging, config, helpers
│   └── config.py
├── mlops/
│   ├── run_experiment.py # Train & log model
│   ├── register_model.py # Register + promote to Production
│   └── explain_shap.py   # SHAP model interpretation (optional)
├── model/                # Saved model artifacts (ignored in git)
├── data/                 # Raw/processed CSV files
├── tests/                # Unit + integration tests
├── docker/
│   └── docker-compose.yml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Launch with Docker

```bash
docker-compose up --build
```

- FastAPI runs at: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

---

## Model Training & Registration

### Train & Log a New Model

```bash
python src/model/train.py
```

This will:
- Train an XGBoost model
- Log metrics, parameters, and model to MLflow
- Optionally register a new model version

### Register Best Model to Production

```bash
python mlops/register_model.py
```

---

## API Usage

### Endpoint: `POST /predict`

```json
POST http://localhost:8000/predict
Content-Type: application/json

{
  "V1": -1.359807,
  "V2": -0.072781,
  ...
  "V28": 0.021,
  "Amount": 149.62
}
```

### Response:

```json
{
  "prediction": 0,
  "message": "Legitimate"
}
```

---

## Notes

- Tracking URI: `http://mlflow:5000` (inside Docker)
- If running scripts locally: add  
  ```python
  mlflow.set_tracking_uri("http://127.0.0.1:5000")
  ```
- Model registry is under name: `"fraud-xgb"`

---

## Future Improvements

- Add batch prediction support
- Integrate drift detection (e.g., EvidentlyAI)
- Auto-deploy latest Production model with CI/CD

---

## Author

**Hua Tan** — [LinkedIn](https://www.linkedin.com/in/hua-tan-751b31224/) | [GitHub](https://github.com/HuaTNA)
