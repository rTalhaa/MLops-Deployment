# Tox21 Molecular Toxicity MLOps Platform

[![CI Pipeline](https://github.com/rTalhaa/MLops-Deployment/actions/workflows/ci.yml/badge.svg)](https://github.com/rTalhaa/MLops-Deployment/actions/workflows/ci.yml)

Production-oriented MLOps project for molecular toxicity prediction using the Tox21 dataset. The system converts SMILES strings into Morgan fingerprints, trains and compares multiple machine learning models, tracks experiments with MLflow, serves the selected model through FastAPI, and exposes runtime metrics for Prometheus and Grafana.

The current deployed model predicts the `SR-ARE` toxicity endpoint and is packaged with the repository for reproducible local serving and CI validation.

## Table Of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Model Summary](#model-summary)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Model Card](#model-card)
- [Training Pipeline](#training-pipeline)
- [Experiment Tracking](#experiment-tracking)
- [Monitoring](#monitoring)
- [CI/CD](#cicd)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)

## System Overview

This project is designed as a compact, end-to-end MLOps workflow:

```text
Tox21 CSV dataset
  -> SMILES validation with RDKit
  -> 1024-bit Morgan fingerprint generation
  -> Multi-model training and evaluation
  -> MLflow experiment tracking
  -> Best model selection by ROC-AUC
  -> FastAPI model serving
  -> Docker Compose orchestration
  -> Prometheus metrics collection
  -> Grafana dashboarding
  -> GitHub Actions CI validation
```

Core capabilities:

- Train and compare Logistic Regression, Random Forest, and Extra Trees classifiers.
- Persist the best model and model metadata under `models/`.
- Serve predictions through a FastAPI application.
- Expose `/metrics` for Prometheus scraping.
- Run the API, Prometheus, and Grafana together with Docker Compose.
- Validate the project on every push and pull request with GitHub Actions.

## Architecture

```text
                  +--------------------+
                  | data/tox21.csv     |
                  +---------+----------+
                            |
                            v
                  +--------------------+
                  | src/train.py       |
                  | RDKit + sklearn    |
                  +----+----------+----+
                       |          |
                       v          v
              +-------------+  +----------------+
              | models/     |  | mlruns/        |
              | joblib/json |  | MLflow runs    |
              +------+------+  +----------------+
                     |
                     v
              +----------------+
              | app/main.py    |
              | FastAPI API    |
              +---+--------+---+
                  |        |
                  v        v
          +----------+  +--------------------+
          | /predict |  | /metrics           |
          +----------+  +---------+----------+
                                |
                                v
                       +----------------+
                       | Prometheus     |
                       +-------+--------+
                               |
                               v
                       +----------------+
                       | Grafana        |
                       +----------------+
```

## Model Summary

The training pipeline selects the best model using ROC-AUC on the held-out test split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7494 | 0.3443 | 0.6117 | 0.4406 | 0.7443 |
| Random Forest | 0.7923 | 0.3846 | 0.4787 | 0.4265 | 0.7592 |
| Extra Trees | 0.7700 | 0.3571 | 0.5319 | 0.4274 | 0.7648 |

Selected model:

```text
Model: Extra Trees
Target: SR-ARE
Feature type: 1024-bit Morgan fingerprints
Selection metric: ROC-AUC
Best ROC-AUC: 0.7648
```

The selected model is stored at `models/tox21_best_model.joblib`, with metadata in `models/model_metadata.json`.

## Repository Layout

```text
.
|-- .github/workflows/ci.yml      # GitHub Actions CI pipeline
|-- app/
|   |-- __init__.py
|   `-- main.py                   # FastAPI inference service
|-- data/
|   `-- tox21.csv                 # Training dataset
|-- models/
|   |-- model_comparison.csv      # Evaluation results
|   |-- model_metadata.json       # Selected model metadata
|   |-- tox21_best_model.joblib   # Model used by the API
|   `-- tox21_rf_model.joblib     # Additional trained artifact
|-- monitoring/
|   `-- prometheus.yml            # Prometheus scrape config
|-- src/
|   |-- load_test.py              # Request generator for monitoring demo
|   `-- train.py                  # Training and MLflow logging pipeline
|-- tests/
|   `-- test_api.py               # API tests
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Quick Start

### 1. Create Environment

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 is recommended because the Docker image and CI pipeline use Python 3.10.

### 2. Run Tests

```powershell
python -m pytest -q
```

### 3. Start The API Locally

```powershell
uvicorn app.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

### 4. Run The Full Stack

```powershell
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| FastAPI | `http://localhost:8000/docs` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

Stop the stack:

```powershell
docker compose down
```

## API Reference

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "dataset": "Tox21",
  "target": "SR-ARE",
  "best_model": "Extra Trees",
  "selection_metric": "roc_auc",
  "best_roc_auc": 0.7647705742720878,
  "task": "molecular_toxicity_prediction"
}
```

### Predict Toxicity

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "smiles": "CCO"
}
```

Response:

```json
{
  "smiles": "CCO",
  "target": "SR-ARE",
  "prediction": "Non-toxic",
  "toxicity_probability": 0.3764,
  "model_used": "Extra Trees",
  "model_version": "v1"
}
```

Invalid SMILES strings return HTTP `400`:

```json
{
  "detail": "Invalid SMILES string"
}
```

### Version Metadata

```http
GET /version
```

Example response:

```json
{
  "app_version": "1.0.0",
  "git_commit_sha": "unknown",
  "model_version": "v1",
  "best_model": "Extra Trees",
  "target": "SR-ARE",
  "dataset": "Tox21"
}
```

### Prometheus Metrics

```http
GET /metrics
```

Exposes Prometheus-compatible metrics for prediction counts, prediction errors, latency, and model loaded status.

## Model Card

See `MODEL_CARD.md` for intended use, limitations, evaluation summary, and operational notes for the selected `SR-ARE` toxicity classifier.

## Training Pipeline

Run the training workflow:

```powershell
python src/train.py
```

The script performs:

- Dataset loading from `data/tox21.csv`.
- Target selection for `SR-ARE`.
- Missing-label filtering.
- SMILES parsing and fingerprint generation with RDKit.
- Train/test split with stratification.
- Training for Logistic Regression, Random Forest, and Extra Trees.
- Metric logging to MLflow.
- Best model selection by ROC-AUC.
- Artifact export to `models/`.

Training outputs:

| File | Purpose |
|---|---|
| `models/tox21_best_model.joblib` | Serialized model loaded by FastAPI |
| `models/model_metadata.json` | Dataset, target, feature, and selected model metadata |
| `models/model_comparison.csv` | Comparison table for all trained models |
| `mlruns/` | Local MLflow experiment tracking output |

`mlruns/` is intentionally ignored by Git because it is local runtime output.

## Experiment Tracking

Start the MLflow UI:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

Experiment name:

```text
tox21_multi_model_molecular_toxicity
```

Tracked metrics:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Training time

## Monitoring

Prometheus scrapes the FastAPI service through `monitoring/prometheus.yml`.

Scrape target inside Docker Compose:

```text
tox21-api:8000
```

Generate prediction traffic:

```powershell
python src/load_test.py
```

Useful Prometheus queries:

| Query | Purpose |
|---|---|
| `sum(tox21_prediction_requests_total)` | Total predictions |
| `sum by (prediction) (tox21_prediction_requests_total)` | Prediction distribution |
| `sum(rate(tox21_prediction_requests_total[1m]))` | Request rate |
| `sum(tox21_prediction_errors_total)` | Prediction errors |
| `tox21_model_loaded` | Model loaded status |
| `rate(tox21_prediction_latency_seconds_sum[1m]) / rate(tox21_prediction_latency_seconds_count[1m])` | Average latency |
| `histogram_quantile(0.95, sum by (le) (rate(tox21_prediction_latency_seconds_bucket[5m])))` | P95 latency |

Grafana can use Prometheus as a data source at:

```text
http://prometheus:9090
```

Recommended dashboard panels:

- Total prediction requests
- Toxic vs non-toxic distribution
- Request rate
- Prediction errors
- Average latency
- P95 latency
- Model loaded status

## CI/CD

GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The CI pipeline runs on pushes and pull requests targeting `main`.

Pipeline stages:

1. Checkout repository.
2. Set up Python 3.10.
3. Install dependencies from `requirements.txt` with pip caching.
4. Verify required project files.
5. Compile Python files for syntax validation.
6. Run API tests with `pytest`.
7. Build the Docker image.

This validates the application from a clean GitHub runner instead of relying on locally installed packages.

### Continuous Delivery (Docker Image)

The repository includes a CD workflow that publishes the API image to GitHub Container Registry (GHCR):

```text
.github/workflows/cd.yml
```

On every push to `main` and on version tags like `v1.0.0`, it builds and pushes:

```text
ghcr.io/<owner>/tox21-api:latest
ghcr.io/<owner>/tox21-api:<branch>
ghcr.io/<owner>/tox21-api:<tag>
ghcr.io/<owner>/tox21-api:<semver>         # e.g. 1.0.0, 1.0, 1
ghcr.io/<owner>/tox21-api:sha-<short>
```

Optional continuous deployment trigger:

- Add a GitHub Actions secret named `DEPLOY_WEBHOOK_URL`.
- The workflow will `POST` to that URL after publishing the image (useful for Render/Railway deploy hooks).

## Operational Notes

- The API loads the model at startup from `models/tox21_best_model.joblib`.
- `models/model_metadata.json` is required for `/health` and prediction responses.
- The service assumes 1024-bit Morgan fingerprints with radius 2.
- The current Docker image is intended for local or demo deployment.
- Grafana dashboards are configured manually in the current version.
- This model is for MLOps demonstration and should not be used for clinical, regulatory, or safety-critical decisions without additional validation.

## Troubleshooting

### Port 8000 Is Already In Use

Check running containers:

```powershell
docker ps
```

Stop the API container if needed:

```powershell
docker stop tox21-api
```

Or identify the process using the port:

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Prometheus Target Is Down

Confirm Docker Compose is running:

```powershell
docker compose ps
```

Confirm `monitoring/prometheus.yml` contains:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "tox21-api"
    static_configs:
      - targets: ["tox21-api:8000"]
```

Restart the stack:

```powershell
docker compose down
docker compose up --build
```

### Grafana Shows No Data

Check:

- Prometheus target is `UP`.
- `python src/load_test.py` has generated traffic.
- Grafana time range includes recent activity.
- Prometheus data source URL is `http://prometheus:9090`.

### MLflow Has No Runs

Train the models first:

```powershell
python src/train.py
```

Then start MLflow:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

## License

No license file is currently included. Add one before distributing or reusing this project publicly.
