"""Prediction module for credit card fraud detection.

Loads the trained model pipeline and threshold from ``artifacts/`` and
provides reusable prediction functions for single transactions and batches.

The model pipeline bundles preprocessing (StandardScaler + passthrough), so
inference uses the exact same preprocessing as training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import METADATA_PATH, METRICS_PATH, MODEL_PATH, THRESHOLD_PATH
from .schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Risk category thresholds (fraud probability)
RISK_LOW_MAX = 0.20
RISK_MEDIUM_MAX = 0.50


class ModelBundle:
    """Holds the loaded model, threshold, and metadata."""

    def __init__(self, model: Any, threshold: float, metadata: dict[str, Any]):
        self.model = model
        self.threshold = threshold
        self.metadata = metadata

    @property
    def model_name(self) -> str:
        return self.metadata.get("model_name", "unknown")


def load_bundle(
    model_path: Path = MODEL_PATH,
    threshold_path: Path = THRESHOLD_PATH,
    metadata_path: Path = METADATA_PATH,
) -> ModelBundle:
    """Load the trained model, threshold, and metadata from artifacts."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python -m app.ml.training` first."
        )
    model = joblib.load(model_path)

    threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = json.load(f).get("threshold", 0.5)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    logger.info("Loaded model '%s' with threshold %.2f", metadata.get("model_name"), threshold)
    return ModelBundle(model=model, threshold=threshold, metadata=metadata)


def _validate_features(features: dict[str, Any]) -> pd.DataFrame:
    """Validate feature names and build a single-row DataFrame.

    Raises ValueError if any required feature is missing or unexpected
    features are present.
    """
    provided = set(features.keys())
    required = set(FEATURE_COLUMNS)

    missing = required - provided
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    extra = provided - required
    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    # Build DataFrame with columns in schema order
    row = {col: features[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _risk_category(probability: float) -> str:
    """Map fraud probability to a risk category."""
    if probability < RISK_LOW_MAX:
        return "low"
    if probability < RISK_MEDIUM_MAX:
        return "medium"
    return "high"


def predict_transaction(bundle: ModelBundle, features: dict[str, Any]) -> dict[str, Any]:
    """Predict fraud for a single transaction.

    Parameters
    ----------
    bundle : ModelBundle
        Loaded model bundle.
    features : dict
        Mapping of feature name -> value. Must contain exactly the
        FEATURE_COLUMNS from schema (no more, no less).

    Returns
    -------
    dict
        {
            "fraud_probability": float,
            "predicted_class": int (0 or 1),
            "risk_category": str ("low" | "medium" | "high"),
            "threshold": float,
            "model_name": str,
        }

    Raises
    ------
    ValueError
        If feature names are invalid or values are non-numeric.
    """
    df = _validate_features(features)

    # Ensure all values are numeric
    for col in FEATURE_COLUMNS:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature '{col}' has non-numeric value: {features[col]!r}") from exc

    prob = float(bundle.model.predict_proba(df)[:, 1][0])
    predicted = int(prob >= bundle.threshold)

    return {
        "fraud_probability": prob,
        "predicted_class": predicted,
        "risk_category": _risk_category(prob),
        "threshold": bundle.threshold,
        "model_name": bundle.model_name,
    }


def predict_batch(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Predict fraud for a batch of transactions.

    Parameters
    ----------
    bundle : ModelBundle
        Loaded model bundle.
    df : pd.DataFrame
        DataFrame with FEATURE_COLUMNS. Extra columns are ignored;
        missing columns raise ValueError.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: fraud_probability, predicted_class,
        risk_category.
    """
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS].copy()
    probs = bundle.model.predict_proba(X)[:, 1]

    result = df.copy()
    result["fraud_probability"] = probs
    result["predicted_class"] = (probs >= bundle.threshold).astype(int)
    result["risk_category"] = [_risk_category(p) for p in probs]
    return result


def get_feature_importances(bundle: ModelBundle) -> dict[str, float] | None:
    """Return feature importances mapped to original feature names.

    The pipeline's ColumnTransformer outputs scaled features first
    (SCALED_FEATURES) then passthrough features (PASSTHROUGH_FEATURES),
    so importances are mapped back in that order. Returns None if the
    classifier does not expose feature_importances_.
    """
    try:
        classifier = bundle.model.named_steps["classifier"]
    except (AttributeError, KeyError):
        return None
    if not hasattr(classifier, "feature_importances_"):
        return None

    from .schema import PASSTHROUGH_FEATURES, SCALED_FEATURES

    output_names = SCALED_FEATURES + PASSTHROUGH_FEATURES
    importances = classifier.feature_importances_
    if len(importances) != len(output_names):
        return None
    return {name: float(imp) for name, imp in zip(output_names, importances)}


def get_model_info(bundle: ModelBundle) -> dict[str, Any]:
    """Return model information for API/dashboard display."""
    metrics_path = METRICS_PATH
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "model_name": bundle.model_name,
        "threshold": bundle.threshold,
        "trained_at": bundle.metadata.get("trained_at"),
        "dataset_rows": bundle.metadata.get("dataset_rows"),
        "class_distribution": bundle.metadata.get("class_distribution"),
        "test_metrics": metrics.get("test", {}),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": bundle.metadata.get("target_column", "Class"),
    }
