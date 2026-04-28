import os
import time
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/tox21.csv"
TARGET_COLUMN = "SR-ARE"
MODEL_PATH = "models/tox21_best_model.joblib"
METADATA_PATH = "models/model_metadata.json"

os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tox21_multi_model_molecular_toxicity")


def smiles_to_fingerprint(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits
    )

    return np.array(fingerprint)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def main():
    print("Loading real Tox21 dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Selected target: {TARGET_COLUMN}")

    df = df[["smiles", TARGET_COLUMN]].dropna()

    print(f"Rows after removing missing target values: {df.shape[0]}")

    features = []
    labels = []
    invalid_smiles = 0

    for _, row in df.iterrows():
        smiles = row["smiles"]
        label = row[TARGET_COLUMN]

        fingerprint = smiles_to_fingerprint(smiles)

        if fingerprint is None:
            invalid_smiles += 1
            continue

        features.append(fingerprint)
        labels.append(int(label))

    X = np.array(features)
    y = np.array(labels)

    print(f"Valid molecular samples: {X.shape[0]}")
    print(f"Invalid SMILES removed: {invalid_smiles}")
    print(f"Feature shape: {X.shape}")
    print(f"Positive class count: {int(y.sum())}")
    print(f"Negative class count: {int(len(y) - y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
    }

    results = []
    best_model = None
    best_model_name = None
    best_score = -1

    print("\nTraining multiple models...\n")

    for model_name, model in models.items():
        print(f"Training: {model_name}")

        with mlflow.start_run(run_name=model_name):
            start_time = time.time()

            model.fit(X_train, y_train)

            training_time = time.time() - start_time
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_param("dataset", "Tox21")
            mlflow.log_param("target_column", TARGET_COLUMN)
            mlflow.log_param("feature_type", "Morgan fingerprints")
            mlflow.log_param("fingerprint_bits", 1024)
            mlflow.log_param("model_name", model_name)

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
            mlflow.log_metric("training_time_seconds", training_time)

            mlflow.sklearn.log_model(model, "model")

            result = {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "training_time_seconds": training_time,
            }

            results.append(result)

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Training Time: {training_time:.2f} seconds")
            print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
            print("-" * 50)

            # Best model selection based on ROC-AUC
            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = model
                best_model_name = model_name

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_comparison.csv", index=False)

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "dataset": "Tox21",
        "target_column": TARGET_COLUMN,
        "feature_type": "Morgan fingerprints",
        "fingerprint_bits": 1024,
        "best_model_name": best_model_name,
        "selection_metric": "roc_auc",
        "best_roc_auc": best_score,
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nModel comparison:")
    print(results_df)

    print("\nBest model selected:")
    print(f"Best Model: {best_model_name}")
    print(f"Best ROC-AUC: {best_score:.4f}")
    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print("Saved comparison table to: models/model_comparison.csv")


if __name__ == "__main__":
    main()