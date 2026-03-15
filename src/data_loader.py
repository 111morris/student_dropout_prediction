"""
Data Loading & Validation Module.
Loads the primary dataset and performs initial validation checks.
"""

import pandas as pd
import numpy as np
import logging

from src.utils import PRIMARY_DATASET, TARGET_COLUMN, ID_COLUMN

logger = logging.getLogger("dropout_prediction.data_loader")


def load_dataset(filepath=None):
    """Load the primary CSV dataset and log basic info."""
    if filepath is None:
        filepath = PRIMARY_DATASET

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def validate_data(df):
    """
    Perform data validation checks:
    - Ensure target column exists
    - Check for valid value ranges
    - Log warnings for anomalous data
    Returns the validated DataFrame (rows with impossible values flagged).
    """
    logger.info("Running data validation checks...")
    issues = []

    # Check target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    # Check target is binary
    unique_targets = df[TARGET_COLUMN].unique()
    if not set(unique_targets).issubset({0, 1}):
        issues.append(f"Target column has unexpected values: {unique_targets}")

    # Validate numeric ranges
    range_checks = {
        "GPA": (0, 4.0),
        "Semester_GPA": (0, 4.0),
        "CGPA": (0, 4.0),
        "Attendance_Rate": (0, 100),
        "Stress_Index": (0, 10),
        "Age": (10, 60),
        "Study_Hours_per_Day": (0, 24),
    }

    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                issues.append(
                    f"  {col}: {len(out_of_range)} values outside [{min_val}, {max_val}]"
                )

    # Check for negative values where they shouldn't exist
    non_negative_cols = [
        "Family_Income",
        "Travel_Time_Minutes",
        "Assignment_Delay_Days",
    ]
    for col in non_negative_cols:
        if col in df.columns:
            negatives = df[df[col] < 0]
            if len(negatives) > 0:
                issues.append(f"  {col}: {len(negatives)} negative values found")

    if issues:
        logger.warning("Data validation issues found:")
        for issue in issues:
            logger.warning(f"  ⚠ {issue}")
    else:
        logger.info("✓ All validation checks passed")

    # Log missing values summary
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        logger.info(f"Missing values in {len(missing_cols)} columns:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    else:
        logger.info("✓ No missing values")

    # Log class distribution
    class_dist = df[TARGET_COLUMN].value_counts()
    dropout_rate = class_dist.get(1, 0) / len(df) * 100
    logger.info(
        f"Class distribution — Retained: {class_dist.get(0, 0)}, "
        f"Dropout: {class_dist.get(1, 0)} ({dropout_rate:.1f}%)"
    )

    return df
