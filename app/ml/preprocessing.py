"""Preprocessing and training pipeline for credit card fraud detection."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import (
    PASSTHROUGH_FEATURES,
    SCALED_FEATURES,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def build_preprocessor():
    """Build the feature preprocessing pipeline.

    Scales ``Time`` and ``Amount`` with a ``StandardScaler`` and passes the
    PCA-derived ``V1``..``V28`` features through unchanged.
    """
    transformers = [
        ("scaler", StandardScaler(), SCALED_FEATURES),
        ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_training_pipeline(random_state=42):
    """Build the full training pipeline: preprocessing + XGBoost classifier."""
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError(
            "xgboost is required to build the training pipeline. "
            "Install it with: pip install xgboost"
        )

    preprocessor = build_preprocessor()

    classifier = XGBClassifier(
        subsample=0.6,
        reg_lambda=1.5,
        reg_alpha=0.01,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        gamma=0.1,
        colsample_bytree=1.0,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_lr_pipeline(random_state=42):
    """Build a Logistic Regression pipeline.

    No class_weight is used because SMOTE already balances the training data.
    """
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_rf_pipeline(random_state=42):
    """Build a Random Forest pipeline.

    No class_weight is used because SMOTE already balances the training data.
    """
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
