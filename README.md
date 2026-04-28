# Tox21 Molecular Toxicity Prediction MLOps Pipeline

This project implements an end-to-end MLOps pipeline for **molecular toxicity prediction** using the **Tox21 dataset**. The system trains multiple machine learning models on SMILES-based molecular fingerprints, tracks experiments using MLflow, deploys the best model through FastAPI, and monitors the deployed service using Prometheus and Grafana.

---

## 1. Project Overview

```text
Tox21 CSV Dataset
        ↓
SMILES Preprocessing + RDKit Parsing
        ↓
1024-bit Morgan Fingerprints
        ↓
Train 3 ML Models
        ↓
Track Experiments in MLflow
        ↓
Select Best Model by ROC-AUC
        ↓
Deploy Best Model with FastAPI
        ↓
Containerize with Docker Compose
        ↓
Monitor API with Prometheus
        ↓
Visualize Metrics in Grafana
```

---

## 2. Models Used

The pipeline trains and compares three relevant molecular machine learning models:

```text
1. Logistic Regression
2. Random Forest Classifier
3. Extra Trees Classifier
```

The best model is selected automatically using **ROC-AUC** and saved for deployment.

---

## 3. Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core implementation |
| RDKit | SMILES parsing and Morgan fingerprint generation |
| scikit-learn | Model training and evaluation |
| MLflow | Experiment tracking and model comparison |
| FastAPI | Model serving API |
| Docker | Containerization |
| Docker Compose | Multi-service orchestration |
| Prometheus | Metrics scraping and monitoring |
| Grafana | Metrics dashboard |
| GitHub Actions | CI/CD automation |

---

## 4. Project Structure

```text
tox21-mlops/
│
├── app/
│   └── main.py
│
├── data/
│   └── tox21.csv
│
├── models/
│   ├── tox21_best_model.joblib
│   ├── model_metadata.json
│   └── model_comparison.csv
│
├── monitoring/
│   └── prometheus.yml
│
├── src/
│   ├── train.py
│   └── load_test.py
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 5. Prerequisites

Install the following before running the project:

```text
Python 3.10 recommended
Docker Desktop
Git
VS Code or any code editor
```

Check installation:

```powershell
python --version
docker --version
docker compose version
git --version
```

---

## 6. Setup Instructions

### Step 1: Open the project folder

```powershell
cd tox21-mlops
```

### Step 2: Create virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If RDKit fails through `requirements.txt`, run:

```powershell
pip install rdkit-pypi
```

---

## 7. Dataset Setup

Place the original Tox21 CSV file inside the `data/` folder and rename it as:

```text
data/tox21.csv
```

The dataset should contain:

```text
smiles
mol_id
NR-AR
NR-AR-LBD
NR-AhR
NR-Aromatase
NR-ER
NR-ER-LBD
NR-PPAR-gamma
SR-ARE
SR-ATAD5
SR-HSE
SR-MMP
SR-p53
```

For this implementation, the selected prediction target is:

```text
SR-ARE
```

---

## 8. Train the Models

Run:

```powershell
python src/train.py
```

This script will:

```text
Load the Tox21 CSV dataset
Select the SR-ARE toxicity target
Remove missing target labels
Convert SMILES into 1024-bit Morgan fingerprints
Train Logistic Regression, Random Forest, and Extra Trees
Evaluate all models using Accuracy, Precision, Recall, F1-score, and ROC-AUC
Log all experiments in MLflow
Select the best model using ROC-AUC
Save the best model to models/tox21_best_model.joblib
Save metadata to models/model_metadata.json
Save comparison results to models/model_comparison.csv
```

Expected output:

```text
Training: Logistic Regression
Training: Random Forest
Training: Extra Trees

Best model selected:
Best Model: Extra Trees
Best ROC-AUC: ...
Saved best model to: models/tox21_best_model.joblib
```

---

## 9. View MLflow Experiment Tracking

Start MLflow:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

You should see:

```text
Experiment: tox21_multi_model_molecular_toxicity
Runs: Logistic Regression, Random Forest, Extra Trees
Metrics: accuracy, precision, recall, f1_score, roc_auc, training_time_seconds
```

Stop MLflow when finished:

```text
CTRL + C
```

---

## 10. Run FastAPI Locally

After training, run:

```powershell
uvicorn app.main:app --reload
```

Open FastAPI Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | Root API check |
| `/health` | Model and service health |
| `/predict` | Predict toxicity for a SMILES string |
| `/metrics` | Exposes Prometheus metrics |

Example request for `/predict`:

```json
{
  "smiles": "CCO"
}
```

Example response:

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

Stop FastAPI before running Docker Compose:

```text
CTRL + C
```

---

## 11. Run Full System with Docker Compose

Make sure port `8000` is free.

```powershell
docker compose down
docker compose up --build
```

This starts:

| Service | URL |
|---|---|
| FastAPI API | `http://localhost:8000/docs` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

---

## 12. Generate API Traffic

Prometheus and Grafana need API traffic to show meaningful metrics.

Run this in a separate terminal while Docker Compose is running:

```powershell
python src/load_test.py
```

This sends multiple SMILES requests to the `/predict` endpoint, including valid and invalid SMILES strings.

---

## 13. Prometheus Monitoring

Open:

```text
http://localhost:9090
```

Check targets:

```text
http://localhost:9090/targets
```

You should see:

```text
tox21-api    UP
```

Useful Prometheus queries:

### Total predictions

```promql
sum(tox21_prediction_requests_total)
```

### Prediction class distribution

```promql
sum by (prediction) (tox21_prediction_requests_total)
```

### Request rate

```promql
sum(rate(tox21_prediction_requests_total[1m]))
```

### Prediction errors

```promql
sum(tox21_prediction_errors_total)
```

### Average prediction latency

```promql
rate(tox21_prediction_latency_seconds_sum[1m]) / rate(tox21_prediction_latency_seconds_count[1m])
```

### P95 prediction latency

```promql
histogram_quantile(0.95, sum by (le) (rate(tox21_prediction_latency_seconds_bucket[5m])))
```

### Model loaded status

```promql
tox21_model_loaded
```

---

## 14. Grafana Dashboard Setup

Open:

```text
http://localhost:3000
```

Default login:

```text
Username: admin
Password: admin
```

Add Prometheus data source:

```text
Connections → Data sources → Add data source → Prometheus
```

Use this URL inside Grafana:

```text
http://prometheus:9090
```

Click:

```text
Save & test
```

Create a dashboard named:

```text
Tox21 Molecular Toxicity API Monitoring
```

Recommended panels:

| Panel | Query | Visualization |
|---|---|---|
| Total Prediction Requests | `sum(tox21_prediction_requests_total)` | Stat |
| Toxic vs Non-toxic Distribution | `sum by (prediction) (tox21_prediction_requests_total)` | Pie chart / Bar gauge |
| Prediction Request Rate | `sum(rate(tox21_prediction_requests_total[1m]))` | Time series |
| Prediction Errors | `sum(tox21_prediction_errors_total)` | Stat |
| Average Prediction Latency | `rate(tox21_prediction_latency_seconds_sum[1m]) / rate(tox21_prediction_latency_seconds_count[1m])` | Time series |
| P95 Prediction Latency | `histogram_quantile(0.95, sum by (le) (rate(tox21_prediction_latency_seconds_bucket[5m])))` | Time series |
| Model Loaded Status | `tox21_model_loaded` | Gauge |

---

## 15. GitHub Actions CI/CD

The GitHub Actions workflow is located at:

```text
.github/workflows/ci.yml
```

A simple CI workflow:

```yaml
name: CI Pipeline

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Check files
        run: |
          test -f app/main.py
          test -f src/train.py
          test -f Dockerfile
          test -f docker-compose.yml

      - name: Build Docker image
        run: docker build -t tox21-api .
```

---

## 16. How to Stop Everything

Stop Docker Compose:

```powershell
docker compose down
```

Stop local MLflow or FastAPI:

```text
CTRL + C
```

---

## 17. Troubleshooting

### Error: Port 8000 already allocated

This means Uvicorn or another Docker container is already using port 8000.

```powershell
docker ps
docker stop tox21-api
```

Or find the process using port 8000:

```powershell
netstat -ano | findstr :8000
taskkill /PID YOUR_PID /F
```

Then rerun:

```powershell
docker compose up --build
```

### Prometheus target is DOWN

Make sure `monitoring/prometheus.yml` contains:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "tox21-api"
    static_configs:
      - targets: ["tox21-api:8000"]
```

Then restart:

```powershell
docker compose down
docker compose up --build
```

### Grafana shows no data

Check:

```text
1. Prometheus target is UP
2. load_test.py has been run
3. Grafana time range is set to Last 15 minutes
4. Prometheus data source URL is http://prometheus:9090
```

### MLflow does not show runs

Run training first:

```powershell
python src/train.py
```

Then:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

---

## 18. Demo / Video Flow

For the final demo video, show this order:

```text
1. Project folder structure
2. Tox21 dataset in data/tox21.csv
3. Run python src/train.py
4. Show MLflow experiment with 3 model runs
5. Show best model saved in models/
6. Run docker compose up --build
7. Open FastAPI /docs
8. Send a /predict request
9. Run python src/load_test.py
10. Open Prometheus /targets and show tox21-api UP
11. Show Prometheus query: sum by (prediction) (tox21_prediction_requests_total)
12. Open Grafana dashboard
13. Explain that MLflow tracks training, while Prometheus + Grafana monitor the deployed API
```

---

## 19. Key Explanation

```text
MLflow:
Tracks training experiments and compares models.

Prometheus:
Collects live runtime metrics from the deployed FastAPI service.

Grafana:
Visualizes live prediction traffic, errors, latency, and model health.
```

---

## 20. Final Summary

This project demonstrates a complete MLOps workflow for computational drug design:

```text
Tox21 molecular toxicity prediction
SMILES to Morgan fingerprints
Multi-model training
MLflow experiment tracking
Best model selection
FastAPI model deployment
Docker Compose orchestration
Prometheus monitoring
Grafana visualization
GitHub Actions CI/CD
```
