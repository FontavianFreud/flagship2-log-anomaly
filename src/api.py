from __future__ import annotations
from src.train import main as train_model

from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

app = FastAPI(title="ML Project Template API", version="0.1.0")


class PredictRequest(BaseModel):
    # Breast cancer dataset has 30 numeric features
    features: List[float] = Field(..., min_items=30, max_items=30)


class PredictResponse(BaseModel):
    prob: float
    pred: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not MODEL_PATH.exists():
        train_model()
        
    model = joblib.load(MODEL_PATH)
    x = np.array(req.features, dtype=float).reshape(1, -1)

    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= 0.5)
    return PredictResponse(prob=prob, pred=pred)