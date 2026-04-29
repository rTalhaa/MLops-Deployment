from fastapi.testclient import TestClient
import sklearn

from app.main import app


client = TestClient(app)


def test_root_endpoint_reports_service_running():
    response = client.get("/")

    assert response.status_code == 200
    assert "Molecular Toxicity Prediction API is running" in response.json()["message"]


def test_health_endpoint_reports_loaded_model_metadata():
    response = client.get("/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["dataset"] == "Tox21"
    assert payload["target"] == "SR-ARE"
    assert payload["selection_metric"] == "roc_auc"
    assert payload["training_sklearn_version"] == sklearn.__version__
    assert payload["runtime_sklearn_version"] == sklearn.__version__


def test_version_endpoint_reports_runtime_metadata():
    response = client.get("/version")
    payload = response.json()

    assert response.status_code == 200
    assert payload["app_version"] == "1.0.0"
    assert payload["model_version"] == "v1"
    assert payload["best_model"] == "Extra Trees"
    assert payload["target"] == "SR-ARE"
    assert payload["dataset"] == "Tox21"
    assert payload["training_sklearn_version"] == sklearn.__version__
    assert payload["runtime_sklearn_version"] == sklearn.__version__
    assert payload["trained_at_utc"] != "unknown"


def test_predict_endpoint_returns_toxicity_prediction_for_valid_smiles():
    response = client.post("/predict", json={"smiles": "CCO"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["smiles"] == "CCO"
    assert payload["target"] == "SR-ARE"
    assert payload["prediction"] in {"Toxic", "Non-toxic"}
    assert 0 <= payload["toxicity_probability"] <= 1


def test_predict_endpoint_rejects_invalid_smiles():
    response = client.post("/predict", json={"smiles": "not-a-smiles"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid SMILES string"


def test_metrics_include_attempt_and_outcome_counters():
    client.post("/predict", json={"smiles": "CCO"})
    client.post("/predict", json={"smiles": "not-a-smiles"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "tox21_prediction_attempts_total" in response.text
    assert 'tox21_prediction_outcomes_total{outcome="success"}' in response.text
    assert 'tox21_prediction_outcomes_total{outcome="invalid_input"}' in response.text
    assert 'tox21_prediction_status_codes_total{status_code="200"}' in response.text
    assert 'tox21_prediction_status_codes_total{status_code="400"}' in response.text
