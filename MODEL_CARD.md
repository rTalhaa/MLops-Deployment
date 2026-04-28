# Model Card: Tox21 SR-ARE Toxicity Classifier

## Overview

This repository ships a molecular toxicity classification model trained on the Tox21 dataset to predict the `SR-ARE` endpoint from SMILES strings.

The deployed API converts each valid SMILES string into a 1024-bit Morgan fingerprint using RDKit and serves predictions through FastAPI.

## Model Details

| Field | Value |
|---|---|
| Selected model | Extra Trees |
| Task type | Binary classification |
| Target | `SR-ARE` |
| Input | SMILES string |
| Feature representation | 1024-bit Morgan fingerprints |
| Model artifact | `models/tox21_best_model.joblib` |
| Metadata file | `models/model_metadata.json` |
| API model version | `v1` |

## Training Data

| Field | Value |
|---|---|
| Dataset | Tox21 |
| Source format | CSV |
| Required input columns | `smiles`, `SR-ARE` |
| Invalid input handling | Invalid SMILES are rejected |

The training script removes rows with missing target values and skips molecules that RDKit cannot parse.

## Evaluation Summary

The following metrics come from the held-out evaluation saved in `models/model_comparison.csv`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7494 | 0.3443 | 0.6117 | 0.4406 | 0.7443 |
| Random Forest | 0.7923 | 0.3846 | 0.4787 | 0.4265 | 0.7592 |
| Extra Trees | 0.7700 | 0.3571 | 0.5319 | 0.4274 | 0.7648 |

Selected production model:

```text
Extra Trees
ROC-AUC: 0.7648
```

## Intended Use

This model is intended for:

- MLOps demonstration
- API serving and monitoring practice
- experiment tracking and model selection workflows
- educational or portfolio use in cheminformatics-flavored ML systems

## Out Of Scope

This model is not intended for:

- clinical decision support
- regulatory toxicology decisions
- laboratory replacement
- safety-critical deployment without domain validation

## Limitations

- Performance is specific to the selected `SR-ARE` endpoint, not all toxicity tasks.
- Input quality depends on valid SMILES strings.
- The model is trained on fingerprint features rather than richer graph or learned molecular representations.
- Evaluation metrics are moderate and should not be interpreted as production-grade scientific validation.
- The deployed API returns a label and probability estimate, but those outputs do not imply calibrated scientific certainty.

## Operational Notes

- `/health` reports loaded model metadata and service status.
- `/version` reports app version, model version, selected model, dataset, and commit SHA when available.
- `/metrics` exposes Prometheus metrics for request volume, error rate, latency, and model load state.

## Safety Note

This project demonstrates an end-to-end MLOps workflow, not a validated toxicology decision system. Any real-world biomedical or chemical safety use would require stronger data governance, model validation, calibration analysis, monitoring, and domain expert review.
