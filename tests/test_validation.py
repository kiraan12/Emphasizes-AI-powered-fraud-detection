"""Tests for app.ml.validation."""

import pandas as pd
import pytest

from app.ml.schema import FEATURE_COLUMNS, TARGET_COLUMN
from app.ml.validation import clean_dataframe, validate_dataframe


def _make_valid_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal valid DataFrame."""
    data = {col: [0.0] * n for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = [0] * (n - 1) + [1]
    return pd.DataFrame(data)


def test_validate_valid_dataframe():
    df = _make_valid_df()
    report = validate_dataframe(df)
    assert report.is_valid
    assert report.required_columns_present
    assert report.target_valid
    assert report.total_rows == 10
    assert report.class_distribution == {0: 9, 1: 1}


def test_validate_missing_column():
    df = _make_valid_df().drop(columns=["V1"])
    report = validate_dataframe(df)
    assert not report.is_valid
    assert "V1" in report.missing_columns
    assert any("Missing required columns" in e for e in report.errors)


def test_validate_extra_column_is_warning():
    df = _make_valid_df()
    df["extra_col"] = 1.0
    report = validate_dataframe(df)
    assert report.is_valid  # extra columns are warnings, not errors
    assert "extra_col" in report.extra_columns
    assert any("Extra columns" in w for w in report.warnings)


def test_validate_invalid_target_labels():
    df = _make_valid_df()
    df.loc[0, TARGET_COLUMN] = 2
    report = validate_dataframe(df)
    assert not report.is_valid
    assert not report.target_valid
    assert any("invalid labels" in e for e in report.errors)


def test_validate_missing_values_and_duplicates():
    df = _make_valid_df(20)
    df.loc[0, "Amount"] = None
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row
    report = validate_dataframe(df)
    assert report.total_missing_values >= 1
    assert report.duplicate_rows >= 1
    assert any("missing values" in w for w in report.warnings)
    assert any("duplicate rows" in w for w in report.warnings)


def test_clean_dataframe_drops_duplicates_and_missing():
    df = _make_valid_df(10)
    df["Time"] = range(10)  # make rows unique so only the appended row is a duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 1 duplicate
    df.loc[1, "Amount"] = None  # 1 missing
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 9  # 11 rows - 1 dup - 1 missing
    assert list(cleaned.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert cleaned.isnull().sum().sum() == 0
    assert not cleaned.duplicated().any()
