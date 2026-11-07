"""Central configuration for the fraud detection project."""

from pathlib import Path

# Project root (parent of the ``app`` package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = PROJECT_ROOT / "creditcard.csv"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # fraction of train used for threshold tuning

# Threshold tuning
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
