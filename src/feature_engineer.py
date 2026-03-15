"""
Feature Engineering Module.
Creates derived features, encodes categorical variables, and prepares
the final feature matrix for model training.
"""

import pandas as pd
import numpy as np
import logging

from src.utils import TARGET_COLUMN

logger = logging.getLogger("dropout_prediction.feature_engineer")


def create_derived_features(df):
    """
    Create engineered features based on domain knowledge:
    - GPA_CGPA_Diff: score decline signal (GPA - CGPA)
    - Study_Attendance_Ratio: engagement proxy
    - Income_per_Travel: financial-logistical burden
    - Overloaded_Flag: high stress + low study hours
    """
    logger.info("Creating derived features...")

    # Score decline signal
    if "GPA" in df.columns and "CGPA" in df.columns:
        df["GPA_CGPA_Diff"] = df["GPA"] - df["CGPA"]
        logger.info("  + GPA_CGPA_Diff (score decline signal)")

    # Engagement proxy
    if "Study_Hours_per_Day" in df.columns and "Attendance_Rate" in df.columns:
        # Avoid division by zero
        df["Study_Attendance_Ratio"] = df["Study_Hours_per_Day"] / (
            df["Attendance_Rate"] + 1e-6
        )
        logger.info("  + Study_Attendance_Ratio (engagement proxy)")

    # Financial-logistical burden
    if "Family_Income" in df.columns and "Travel_Time_Minutes" in df.columns:
        df["Income_per_Travel"] = df["Family_Income"] / (
            df["Travel_Time_Minutes"] + 1e-6
        )
        logger.info("  + Income_per_Travel (financial-logistical burden)")

    # Overloaded student flag
    if "Stress_Index" in df.columns and "Study_Hours_per_Day" in df.columns:
        df["Overloaded_Flag"] = (
            (df["Stress_Index"] > 7) & (df["Study_Hours_per_Day"] < 2)
        ).astype(int)
        logger.info("  + Overloaded_Flag (high stress + low study)")

    logger.info(f"✓ Created derived features. Shape: {df.shape}")
    return df


def encode_features(df):
    """
    Encode categorical features:
    - Binary (Yes/No, Male/Female): Label Encoding (0/1)
    - Multi-class (Department, Semester, Parental_Education): One-Hot Encoding
    """
    logger.info("Encoding categorical features...")

    # Binary encoding
    binary_maps = {
        "Gender": {"Male": 1, "Female": 0},
        "Internet_Access": {"Yes": 1, "No": 0},
        "Part_Time_Job": {"Yes": 1, "No": 0},
        "Scholarship": {"Yes": 1, "No": 0},
    }

    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            logger.info(f"  Binary encoded: {col}")

    # One-Hot encoding for multi-class categoricals
    multi_class_cols = []
    for col in ["Department", "Semester", "Parental_Education"]:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            multi_class_cols.append(col)

    if multi_class_cols:
        df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True, dtype=int)
        logger.info(f"  One-Hot encoded: {multi_class_cols}")

    logger.info(f"✓ Encoding complete. Shape: {df.shape}")
    return df


def prepare_features(df):
    """
    Full feature engineering pipeline:
    1. Create derived features
    2. Encode categoricals
    Returns (X, y) tuple.
    """
    logger.info("=" * 50)
    logger.info("STAGE: FEATURE ENGINEERING")
    logger.info("=" * 50)

    df = create_derived_features(df)
    df = encode_features(df)

    # Split features and target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    logger.info(f"✓ Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"✓ Target vector: {y.shape[0]} samples")

    return X, y
