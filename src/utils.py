"""
Utility functions for the Student Dropout Risk Prediction System.
Provides logging setup, model save/load helpers, and common constants.
"""

import logging
import os
import joblib
import pandas as pd

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
RISK_THRESHOLD_HIGH = 0.60
RISK_THRESHOLD_MEDIUM = 0.35
CORRELATION_THRESHOLD = 0.90

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Primary dataset
PRIMARY_DATASET = os.path.join(RAW_DATA_DIR, "student_dropout_dataset_v3.csv")
TARGET_COLUMN = "Dropout"
ID_COLUMN = "Student_ID"


def setup_logging(level=logging.INFO):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("dropout_prediction")


def ensure_directories():
    """Create output directories if they don't exist."""
    for d in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_model(model, filename):
    """Save a trained model to the models directory."""
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    return filepath


def load_model(filename):
    """Load a model from the models directory."""
    filepath = os.path.join(MODELS_DIR, filename)
    return joblib.load(filepath)


def save_dataframe(df, filename, directory=None):
    """Save a DataFrame to CSV in the specified directory."""
    if directory is None:
        directory = PROCESSED_DATA_DIR
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    return filepath
