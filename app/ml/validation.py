"""Data validation for the credit card fraud detection dataset.

Validates a raw DataFrame against the schema defined in ``app.ml.schema``
and returns a structured report. Does not mutate the input DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .schema import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class ValidationReport:
    """Structured result of dataset validation."""

    is_valid: bool
    required_columns_present: bool
    missing_columns: list[str]
    extra_columns: list[str]
    total_rows: int
    missing_value_counts: dict[str, int]
    total_missing_values: int
    duplicate_rows: int
    target_present: bool
    target_unique_values: list[Any]
    target_valid: bool
    class_distribution: dict[int, int]
    fraud_rate: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a plain dictionary."""
        return {
            "is_valid": self.is_valid,
            "required_columns_present": self.required_columns_present,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "total_rows": self.total_rows,
            "missing_value_counts": self.missing_value_counts,
            "total_missing_values": self.total_missing_values,
            "duplicate_rows": self.duplicate_rows,
            "target_present": self.target_present,
            "target_unique_values": self.target_unique_values,
            "target_valid": self.target_valid,
            "class_distribution": self.class_distribution,
            "fraud_rate": self.fraud_rate,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Validate a DataFrame against the fraud detection schema.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset to validate. Not modified.

    Returns
    -------
    ValidationReport
        Structured validation result.
    """
    errors: list[str] = []
    warnings: list[str] = []

    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    present = set(df.columns)
    missing_columns = sorted(required - present)
    extra_columns = sorted(present - required)
    required_columns_present = len(missing_columns) == 0

    if not required_columns_present:
        errors.append(f"Missing required columns: {missing_columns}")

    if extra_columns:
        warnings.append(f"Extra columns present (will be ignored): {extra_columns}")

    # Missing values
    missing_value_counts = df.isnull().sum().to_dict()
    missing_value_counts = {k: int(v) for k, v in missing_value_counts.items() if v > 0}
    total_missing = int(sum(missing_value_counts.values()))
    if total_missing > 0:
        warnings.append(f"Dataset contains {total_missing} missing values")

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate rows")

    # Target validation
    target_present = TARGET_COLUMN in df.columns
    target_unique_values: list[Any] = []
    target_valid = False
    class_distribution: dict[int, int] = {}
    fraud_rate = 0.0

    if target_present:
        target_unique_values = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
        valid_labels = {0, 1}
        target_valid = set(target_unique_values).issubset(valid_labels)
        if not target_valid:
            errors.append(
                f"Target column '{TARGET_COLUMN}' has invalid labels: {target_unique_values}. "
                f"Expected subset of {sorted(valid_labels)}."
            )
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        class_distribution = {int(k): int(v) for k, v in counts.items()}
        total = len(df)
        fraud_rate = class_distribution.get(1, 0) / total if total > 0 else 0.0
    else:
        errors.append(f"Target column '{TARGET_COLUMN}' is missing")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        required_columns_present=required_columns_present,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        total_rows=int(len(df)),
        missing_value_counts=missing_value_counts,
        total_missing_values=total_missing,
        duplicate_rows=duplicate_rows,
        target_present=target_present,
        target_unique_values=target_unique_values,
        target_valid=target_valid,
        class_distribution=class_distribution,
        fraud_rate=fraud_rate,
        errors=errors,
        warnings=warnings,
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the DataFrame.

    Drops duplicate rows and rows with missing values in required columns.
    Does not modify the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with only required columns in schema order.
    """
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    cleaned = df[required].copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(subset=required)
    return cleaned
