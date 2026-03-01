from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Proper split: train/val/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Saved model to:", MODEL_PATH)
    print("Saved metrics to:", METRICS_PATH)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()