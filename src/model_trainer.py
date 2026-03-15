"""
Model Training Module.
Trains multiple ML models using sklearn.Pipeline, performs Stratified K-Fold
cross-validation, and tunes classification thresholds for optimal recall.
"""

import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

from src.utils import RANDOM_STATE, TEST_SIZE, CV_FOLDS

logger = logging.getLogger("dropout_prediction.model_trainer")


def get_models():
    """
    Define model pipelines.
    - Logistic Regression gets StandardScaler (needs it)
    - Tree-based models skip scaling (don't need it)
    All use class_weight='balanced' to handle imbalance.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=3.25,  # approx ratio of negatives/positives
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
            )),
        ]),
    }
    return models


def split_data(X, y):
    """Stratified train/test split."""
    logger.info("=" * 50)
    logger.info("STAGE: TRAIN/TEST SPLIT")
    logger.info("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set:     {X_test.shape[0]} samples")
    logger.info(
        f"Train dropout rate: {y_train.mean():.3f} | "
        f"Test dropout rate: {y_test.mean():.3f}"
    )

    return X_train, X_test, y_train, y_test


def cross_validate_models(models, X_train, y_train):
    """
    Perform Stratified K-Fold cross-validation on all models.
    Returns a DataFrame with mean ± std for each metric.
    """
    logger.info("=" * 50)
    logger.info("STAGE: CROSS-VALIDATION")
    logger.info("=" * 50)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = {}

    for name, pipeline in models.items():
        logger.info(f"Cross-validating: {name}...")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        cv_results[name] = {
            metric: {
                "mean": scores[f"test_{metric}"].mean(),
                "std": scores[f"test_{metric}"].std(),
            }
            for metric in scoring
        }

        logger.info(
            f"  Accuracy: {cv_results[name]['accuracy']['mean']:.4f} "
            f"(±{cv_results[name]['accuracy']['std']:.4f})"
        )
        logger.info(
            f"  ROC-AUC:  {cv_results[name]['roc_auc']['mean']:.4f} "
            f"(±{cv_results[name]['roc_auc']['std']:.4f})"
        )

    return cv_results


def train_models(models, X_train, y_train):
    """Train all models on the full training set."""
    logger.info("=" * 50)
    logger.info("STAGE: MODEL TRAINING")
    logger.info("=" * 50)

    trained_models = {}
    for name, pipeline in models.items():
        logger.info(f"Training: {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        logger.info(f"  ✓ {name} trained successfully")

    return trained_models


def find_optimal_threshold(model, X_test, y_test, target_recall=0.85):
    """
    Find the classification threshold that maximizes F1
    while keeping recall above the target.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    best_threshold = 0.5
    best_f1 = 0

    for i in range(len(thresholds)):
        if recalls[i] >= target_recall:
            f1 = (
                2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i] + 1e-8)
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresholds[i]

    return best_threshold


def evaluate_on_test(trained_models, X_test, y_test):
    """
    Evaluate all trained models on the test set.
    Also performs threshold tuning for the best model.
    """
    logger.info("=" * 50)
    logger.info("STAGE: TEST SET EVALUATION")
    logger.info("=" * 50)

    # Baseline: majority class
    majority_acc = max(y_test.mean(), 1 - y_test.mean())
    logger.info(f"Baseline (majority class) accuracy: {majority_acc:.4f}")
    logger.info("-" * 50)

    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "beats_baseline": acc > majority_acc,
        }

        status = "✓ BEATS baseline" if acc > majority_acc else "✗ BELOW baseline"
        logger.info(f"{name}: Acc={acc:.4f} | Prec={prec:.4f} | "
                     f"Rec={rec:.4f} | F1={f1:.4f} | AUC={roc:.4f} | {status}")

    # Find best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    logger.info(f"\n★ Best model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.4f})")

    # Threshold tuning for best model
    best_model = trained_models[best_name]
    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test)
    logger.info(f"  Optimal threshold (recall-focused): {optimal_threshold:.3f}")

    if optimal_threshold != 0.5 and hasattr(best_model, "predict_proba"):
        y_proba_best = best_model.predict_proba(X_test)[:, 1]
        y_pred_tuned = (y_proba_best >= optimal_threshold).astype(int)
        tuned_rec = recall_score(y_test, y_pred_tuned, zero_division=0)
        tuned_prec = precision_score(y_test, y_pred_tuned, zero_division=0)
        tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
        logger.info(
            f"  Tuned metrics → Prec={tuned_prec:.4f} | "
            f"Rec={tuned_rec:.4f} | F1={tuned_f1:.4f}"
        )
        results[best_name]["optimal_threshold"] = optimal_threshold
        results[best_name]["tuned_recall"] = tuned_rec
        results[best_name]["tuned_precision"] = tuned_prec
        results[best_name]["tuned_f1"] = tuned_f1

    return results, best_name
