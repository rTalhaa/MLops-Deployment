from fastapi.testclient import TestClient

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

