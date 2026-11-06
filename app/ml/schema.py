"""Feature definitions for credit card fraud detection."""

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

NUMERIC_FEATURES = FEATURE_COLUMNS.copy()

SCALED_FEATURES = ["Time", "Amount"]

PASSTHROUGH_FEATURES = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in SCALED_FEATURES
]
