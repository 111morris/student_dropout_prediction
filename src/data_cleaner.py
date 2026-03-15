"""
Data Cleaning Module.
Handles missing value imputation, duplicate removal, and multicollinearity checks.
"""

import pandas as pd
import numpy as np
import logging

from src.utils import ID_COLUMN, TARGET_COLUMN, CORRELATION_THRESHOLD

logger = logging.getLogger("dropout_prediction.data_cleaner")


def drop_id_column(df):
    """Drop the Student_ID column (identifier, not a feature)."""
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
        logger.info(f"Dropped '{ID_COLUMN}' column")
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    else:
        logger.info("✓ No duplicate rows found")
    return df


def impute_missing_values(df):
    """
    Impute missing values:
    - Numeric columns: median (robust to skewness in income/stress)
    - Categorical columns: mode
    """
    missing_before = df.isnull().sum().sum()
    if missing_before == 0:
        logger.info("✓ No missing values to impute")
        return df

    logger.info(f"Imputing {missing_before} total missing values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # Remove target from imputation lists
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)

    # Median imputation for numeric
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: {count} values → median ({median_val:.2f})")

    # Mode imputation for categorical
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            count = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: {count} values → mode ('{mode_val}')")

    missing_after = df.isnull().sum().sum()
    logger.info(f"✓ Missing values: {missing_before} → {missing_after}")

    return df


def check_multicollinearity(df, threshold=None):
    """
    Check for highly correlated numeric features.
    If correlation > threshold, recommend dropping one.
    Returns list of columns to drop.
    """
    if threshold is None:
        threshold = CORRELATION_THRESHOLD

    numeric_df = df.select_dtypes(include=[np.number])
    # Exclude target
    if TARGET_COLUMN in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[TARGET_COLUMN])

    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    cols_to_drop = []
    high_corr_pairs = []

    for col in upper_tri.columns:
        correlated = upper_tri[col][upper_tri[col] > threshold]
        for idx, val in correlated.items():
            high_corr_pairs.append((idx, col, val))
            # Drop the one with lower correlation to target
            if TARGET_COLUMN in df.columns:
                corr_with_target_1 = abs(df[idx].corr(df[TARGET_COLUMN]))
                corr_with_target_2 = abs(df[col].corr(df[TARGET_COLUMN]))
                drop_col = idx if corr_with_target_1 < corr_with_target_2 else col
            else:
                drop_col = col
            if drop_col not in cols_to_drop:
                cols_to_drop.append(drop_col)

    if high_corr_pairs:
        logger.info(f"High correlation pairs (threshold > {threshold}):")
        for c1, c2, val in high_corr_pairs:
            logger.info(f"  {c1} ↔ {c2}: {val:.3f}")
        logger.info(f"Columns to drop: {cols_to_drop}")
    else:
        logger.info(f"✓ No features correlated above {threshold}")

    return cols_to_drop, high_corr_pairs


def drop_correlated_features(df, cols_to_drop):
    """Drop the identified highly correlated columns."""
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped correlated features: {cols_to_drop}")
        logger.info(f"Remaining features: {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Full cleaning pipeline:
    1. Drop ID column
    2. Remove duplicates
    3. Impute missing values
    4. Check and handle multicollinearity
    """
    logger.info("=" * 50)
    logger.info("STAGE: DATA CLEANING")
    logger.info("=" * 50)

    df = drop_id_column(df)
    df = remove_duplicates(df)
    df = impute_missing_values(df)

    cols_to_drop, _ = check_multicollinearity(df)
    df = drop_correlated_features(df, cols_to_drop)

    logger.info(f"✓ Cleaned dataset shape: {df.shape}")
    return df
