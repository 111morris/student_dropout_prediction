"""
Application Configuration.
Centralizes all settings, paths, and constants.
Uses environment variables for deployment flexibility.

To switch to PostgreSQL in production, set:
    DATABASE_URL=postgresql://user:password@host:5432/dbname
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────
# Backend dir → project root is one level up
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ──────────────────────────────────────────────────────────────
# Model Artifacts
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "best_model.joblib"))
SCALER_PATH = os.getenv("SCALER_PATH", str(MODELS_DIR / "scaler.joblib"))

# ──────────────────────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────────────────────
# Default: SQLite file in project root (great for development).
# For production, set DATABASE_URL env var to a PostgreSQL URL:
#   DATABASE_URL=postgresql://user:password@host:5432/student_ews
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT / 'student_ews.db'}",
)

# ──────────────────────────────────────────────────────────────
# Risk Score Thresholds (must match training pipeline)
# ──────────────────────────────────────────────────────────────
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.60"))
RISK_THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.35"))

# ──────────────────────────────────────────────────────────────
# API Settings
# ──────────────────────────────────────────────────────────────
API_TITLE = "Student Dropout Early Warning System"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "REST API for predicting student dropout risk. "
    "Designed as an Early Warning System (EWS) for schools."
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
