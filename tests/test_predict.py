"""Tests for app.ml.predict."""

import pandas as pd
import pytest

from app.ml.predict import (
    ModelBundle,
    _risk_category,
    _validate_features,
    get_feature_importances,
    get_model_info,
    load_bundle,
    predict_batch,
    predict_transaction,
)
from app.ml.schema import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def bundle():
    """Load the real trained bundle once for all tests."""
    return load_bundle()


@pytest.fixture
def sample_features() -> dict:
    """A single transaction's features from the dataset."""
    df = pd.read_csv("creditcard.csv", nrows=1)
    return {col: float(df[col].iloc[0]) for col in FEATURE_COLUMNS}


def test_load_bundle(bundle):
    assert bundle.model is not None
    assert 0.0 <= bundle.threshold <= 1.0
    assert bundle.model_name == "random_forest"


def test_predict_transaction(bundle, sample_features):
    result = predict_transaction(bundle, sample_features)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["predicted_class"] in (0, 1)
    assert result["risk_category"] in ("low", "medium", "high")
    assert result["threshold"] == bundle.threshold
    assert result["model_name"] == bundle.model_name


def test_predict_transaction_missing_feature(bundle, sample_features):
    bad = dict(sample_features)
    del bad["V1"]
    with pytest.raises(ValueError, match="Missing required features"):
        predict_transaction(bundle, bad)


def test_predict_transaction_extra_feature(bundle, sample_features):
    bad = dict(sample_features)
    bad["extra"] = 1.0
    with pytest.raises(ValueError, match="Unexpected features"):
        predict_transaction(bundle, bad)


def test_predict_transaction_non_numeric(bundle, sample_features):
    bad = dict(sample_features)
    bad["Amount"] = "not_a_number"
    with pytest.raises(ValueError, match="non-numeric"):
        predict_transaction(bundle, bad)


def test_predict_batch(bundle, sample_features):
    df = pd.DataFrame([sample_features, sample_features])
    result = predict_batch(bundle, df)
    assert len(result) == 2
    assert "fraud_probability" in result.columns
    assert "predicted_class" in result.columns
    assert "risk_category" in result.columns
    assert result["predicted_class"].isin([0, 1]).all()


def test_predict_batch_missing_column(bundle, sample_features):
    df = pd.DataFrame([{k: v for k, v in sample_features.items() if k != "V2"}])
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch(bundle, df)


def test_risk_category_boundaries():
    assert _risk_category(0.0) == "low"
    assert _risk_category(0.19) == "low"
    assert _risk_category(0.20) == "medium"
    assert _risk_category(0.49) == "medium"
    assert _risk_category(0.50) == "high"
    assert _risk_category(1.0) == "high"


def test_get_model_info(bundle):
    info = get_model_info(bundle)
    assert info["model_name"] == "random_forest"
    assert "test_metrics" in info
    assert "feature_columns" in info
    assert len(info["feature_columns"]) == 30


def test_get_feature_importances(bundle):
    importances = get_feature_importances(bundle)
    assert importances is not None
    assert len(importances) == 30
    assert all(isinstance(v, float) for v in importances.values())
    # Importances should sum to ~1.0
    assert abs(sum(importances.values()) - 1.0) < 0.01
