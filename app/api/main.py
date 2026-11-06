"""FastAPI backend for credit card fraud detection.

Endpoints
---------
GET  /health          - Health check
GET  /model/info      - Model information and metrics
POST /predict         - Single transaction prediction
POST /predict/batch   - Batch prediction
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..ml.predict import ModelBundle, get_model_info, load_bundle, predict_batch, predict_transaction
from ..ml.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Global bundle loaded at startup
_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global _bundle
    try:
        _bundle = load_bundle()
        logger.info("Model loaded: %s", _bundle.model_name)
    except FileNotFoundError as exc:
        logger.error("Failed to load model: %s", exc)
        _bundle = None
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="REST API for credit card fraud detection",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Pydantic schemas ----------

class TransactionFeatures(BaseModel):
    """Single transaction features (30 input features)."""

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_class: int
    risk_category: str
    threshold: float
    model_name: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    trained_at: str | None
    dataset_rows: int | None
    class_distribution: dict[str, int] | None
    test_metrics: dict[str, Any]
    feature_columns: list[str]
    target_column: str


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model information and test metrics."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = get_model_info(_bundle)
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures) -> PredictionResponse:
    """Predict fraud for a single transaction."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = transaction.model_dump()
        result = predict_transaction(_bundle, features)
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_endpoint(transactions: list[TransactionFeatures]) -> BatchPredictionResponse:
    """Predict fraud for a batch of transactions."""
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not transactions:
        raise HTTPException(status_code=422, detail="Empty batch")
    try:
        df = pd.DataFrame([t.model_dump() for t in transactions])
        result_df = predict_batch(_bundle, df)
        predictions = [
            PredictionResponse(
                fraud_probability=float(row["fraud_probability"]),
                predicted_class=int(row["predicted_class"]),
                risk_category=row["risk_category"],
                threshold=_bundle.threshold,
                model_name=_bundle.model_name,
            )
            for _, row in result_df.iterrows()
        ]
        return BatchPredictionResponse(predictions=predictions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
