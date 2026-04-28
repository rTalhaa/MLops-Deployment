import time
import json
import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from rdkit import Chem
from rdkit.Chem import AllChem


app = FastAPI(
    title="Molecular Toxicity Prediction API",
    description="MLOps demo API for computational drug design toxicity prediction",
    version="1.0",
)

MODEL_PATH = "models/tox21_best_model.joblib"
METADATA_PATH = "models/model_metadata.json"
APP_VERSION = "1.0.0"
MODEL_VERSION = "v1"
GIT_COMMIT_SHA = os.getenv("GIT_COMMIT_SHA", "unknown")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    model_metadata = json.load(file)


prediction_counter = Counter(
    "tox21_prediction_requests_total",
    "Total number of prediction requests",
    ["prediction"],
)

prediction_errors = Counter(
    "tox21_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)

prediction_latency = Histogram(
    "tox21_prediction_latency_seconds",
    "Prediction latency in seconds",
)

model_loaded_gauge = Gauge(
    "tox21_model_loaded",
    "Whether the model is loaded successfully",
)

model_loaded_gauge.set(1)


class MoleculeInput(BaseModel):
    smiles: str


def smiles_to_fingerprint(smiles: str, n_bits: int = 1024):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits,
    )

    arr = np.array(fp)
    return arr.reshape(1, -1)


@app.get("/")
def root():
    return {
        "message": "Molecular Toxicity Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "dataset": model_metadata["dataset"],
        "target": model_metadata["target_column"],
        "best_model": model_metadata["best_model_name"],
        "selection_metric": model_metadata["selection_metric"],
        "best_roc_auc": model_metadata["best_roc_auc"],
        "task": "molecular_toxicity_prediction",
    }


@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "git_commit_sha": GIT_COMMIT_SHA,
        "model_version": MODEL_VERSION,
        "best_model": model_metadata["best_model_name"],
        "target": model_metadata["target_column"],
        "dataset": model_metadata["dataset"],
    }


@app.post("/predict")
def predict(input_data: MoleculeInput):
    try:
        start_time = time.time()

        features = smiles_to_fingerprint(input_data.smiles)

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        latency = time.time() - start_time
        prediction_latency.observe(latency)

        label = "Toxic" if prediction == 1 else "Non-toxic"

        prediction_counter.labels(prediction=label).inc()

        return {
            "smiles": input_data.smiles,
            "target": model_metadata["target_column"],
            "prediction": label,
            "toxicity_probability": round(probability, 4),
            "model_used": model_metadata["best_model_name"],
            "model_version": MODEL_VERSION,
        }

    except ValueError as error:
        prediction_errors.labels(error_type="invalid_smiles").inc()
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        prediction_errors.labels(error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
