"""Tests for the FastAPI backend."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.ml.schema import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan (model loading) executed."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_features() -> dict:
    df = pd.read_csv("creditcard.csv", nrows=1)
    return {col: float(df[col].iloc[0]) for col in FEATURE_COLUMNS}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "random_forest"
    assert "test_metrics" in data
    assert len(data["feature_columns"]) == 30


def test_predict_single(client, sample_features):
    resp = client.post("/predict", json=sample_features)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["predicted_class"] in (0, 1)
    assert data["risk_category"] in ("low", "medium", "high")


def test_predict_batch(client, sample_features):
    resp = client.post("/predict/batch", json=[sample_features, sample_features])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2


def test_predict_empty_batch(client):
    resp = client.post("/predict/batch", json=[])
    assert resp.status_code == 422


def test_predict_missing_feature(client, sample_features):
    bad = dict(sample_features)
    del bad["V1"]
    resp = client.post("/predict", json=bad)
    # Pydantic will reject missing required field with 422
    assert resp.status_code == 422
