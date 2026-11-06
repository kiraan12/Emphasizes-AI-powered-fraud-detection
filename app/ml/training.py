"""Training pipeline for credit card fraud detection.

Orchestrates data loading, validation, splitting, SMOTE resampling,
model training, threshold tuning, evaluation, and artifact persistence.

Model selection strategy
------------------------
Three models are trained (Logistic Regression, Random Forest, XGBoost).
Each model's classification threshold is tuned on a held-out validation
split (carved from the training set) by maximizing F1-score on the fraud
class (class 1). The model with the highest validation F1 at its tuned
threshold is selected. The final test set is evaluated exactly once,
after selection, and is never used for tuning.

Data leakage prevention
-----------------------
- The dataset is split into train/test BEFORE any resampling.
- A validation split is carved from train BEFORE resampling.
- SMOTE is applied only to the training subset, never to validation or test.
- The preprocessing pipeline (StandardScaler) is fitted inside each model's
  Pipeline, so it only sees training data during ``fit``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ..config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    THRESHOLD_GRID,
    THRESHOLD_PATH,
    VAL_SIZE,
)
from .preprocessing import (
    build_lr_pipeline,
    build_rf_pipeline,
    build_training_pipeline,
)
from .schema import FEATURE_COLUMNS, TARGET_COLUMN
from .validation import clean_dataframe, validate_dataframe

logger = logging.getLogger(__name__)


def _load_and_validate(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and return a cleaned DataFrame."""
    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    report = validate_dataframe(df)
    logger.info("Validation: is_valid=%s, rows=%d, duplicates=%d",
                report.is_valid, report.total_rows, report.duplicate_rows)
    for warning in report.warnings:
        logger.warning("Validation warning: %s", warning)
    for error in report.errors:
        logger.error("Validation error: %s", error)
    if not report.is_valid:
        raise ValueError(f"Dataset validation failed: {report.errors}")
    return clean_dataframe(df)


def _build_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Return a dict of model name -> unfitted Pipeline."""
    return {
        "logistic_regression": build_lr_pipeline(random_state=random_state),
        "random_forest": build_rf_pipeline(random_state=random_state),
        "xgboost": build_training_pipeline(random_state=random_state),
    }


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find the threshold maximizing F1 on the fraud class.

    Returns (best_threshold, best_f1).
    """
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute evaluation metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def train_and_evaluate(
    data_path: Path = DATA_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Run the full training pipeline and save artifacts.

    Returns a dict with the selected model name, threshold, and metrics.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & clean
    df = _load_and_validate(data_path)
    logger.info("Cleaned dataset: %d rows, %d columns", len(df), len(df.columns))

    # 2. Features / target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    # 3. Train/test split (stratified, before any resampling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )
    logger.info("Train: %d rows (%d fraud), Test: %d rows (%d fraud)",
                len(X_train), int(y_train.sum()), len(X_test), int(y_test.sum()))

    # 4. Carve validation split from train (for threshold tuning only)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=random_state, stratify=y_train
    )
    logger.info("Train_sub: %d rows (%d fraud), Val: %d rows (%d fraud)",
                len(X_train_sub), int(y_train_sub.sum()),
                len(X_val), int(y_val.sum()))

    # 5. SMOTE on train_sub only
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_sub, y_train_sub)
    logger.info("After SMOTE: %d rows (%d fraud)",
                len(X_train_res), int(y_train_res.sum()))

    # 6. Train models
    models = _build_models(random_state=random_state)
    val_results: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train_res, y_train_res)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = _tune_threshold(y_val.to_numpy(), y_val_prob)
        val_metrics = _evaluate(y_val.to_numpy(), y_val_prob, threshold)
        val_results[name] = {
            "model": model,
            "threshold": threshold,
            "val_f1": val_f1,
            "val_metrics": val_metrics,
        }
        logger.info("%s: val_f1=%.4f threshold=%.2f", name, val_f1, threshold)

    # 7. Select best model by validation F1
    best_name = max(val_results, key=lambda k: val_results[k]["val_f1"])
    best = val_results[best_name]
    logger.info("Selected model: %s (val_f1=%.4f)", best_name, best["val_f1"])

    # 8. Final evaluation on test (once, after selection)
    y_test_prob = best["model"].predict_proba(X_test)[:, 1]
    test_metrics = _evaluate(y_test.to_numpy(), y_test_prob, best["threshold"])
    logger.info("Test metrics: %s", json.dumps({k: v for k, v in test_metrics.items()
                                                if k not in ("confusion_matrix", "classification_report")}, indent=2))

    # 9. Save artifacts
    joblib.dump(best["model"], MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best["threshold"], "model": best_name}, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump({
            "model": best_name,
            "threshold": best["threshold"],
            "validation": {k: {kk: vv for kk, vv in v["val_metrics"].items()
                               if kk not in ("confusion_matrix", "classification_report")}
                           for k, v in val_results.items()},
            "test": {k: v for k, v in test_metrics.items()
                     if k not in ("confusion_matrix", "classification_report")},
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "test_classification_report": test_metrics["classification_report"],
        }, f, indent=2, default=str)

    metadata = {
        "model_name": best_name,
        "threshold": best["threshold"],
        "random_state": random_state,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "class_distribution": {
            "legitimate": int((y == 0).sum()),
            "fraud": int((y == 1).sum()),
        },
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "all_models_validation_f1": {k: v["val_f1"] for k, v in val_results.items()},
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts saved to %s", artifacts_dir)

    return {
        "model_name": best_name,
        "threshold": best["threshold"],
        "test_metrics": test_metrics,
        "validation_f1": {k: v["val_f1"] for k, v in val_results.items()},
    }


def main() -> None:
    """CLI entry point: run training and print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = train_and_evaluate()
    print("\n=== Training Complete ===")
    print(f"Selected model: {result['model_name']}")
    print(f"Threshold: {result['threshold']:.2f}")
    print(f"Validation F1 by model: {result['validation_f1']}")
    tm = result["test_metrics"]
    print(f"Test Precision: {tm['precision']:.4f}")
    print(f"Test Recall:    {tm['recall']:.4f}")
    print(f"Test F1:        {tm['f1']:.4f}")
    print(f"Test ROC-AUC:   {tm['roc_auc']:.4f}")
    print(f"Test PR-AUC:    {tm['pr_auc']:.4f}")
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
