"""
Student Dropout Risk Prediction System — Main Pipeline
=====================================================
Orchestrates the full ML pipeline: load → validate → clean → EDA →
feature engineer → split → train → cross-validate → evaluate → save.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    setup_logging,
    ensure_directories,
    save_model,
    save_dataframe,
    RANDOM_STATE,
)
from src.data_loader import load_dataset, validate_data
from src.data_cleaner import clean_data
from src.feature_engineer import prepare_features
from src.eda import run_eda
from src.model_trainer import (
    get_models,
    split_data,
    cross_validate_models,
    train_models,
    evaluate_on_test,
)
from src.model_evaluator import run_evaluation


def main():
    """Execute the full student dropout prediction pipeline."""
    logger = setup_logging()
    ensure_directories()

    logger.info("=" * 60)
    logger.info("STUDENT DROPOUT RISK PREDICTION SYSTEM")
    logger.info("=" * 60)

    # ─── Step 1: Load & Validate ───────────────────────────────
    df = load_dataset()
    df = validate_data(df)

    # ─── Step 2: EDA (on raw data) ─────────────────────────────
    run_eda(df)

    # ─── Step 3: Clean ─────────────────────────────────────────
    df_clean = clean_data(df)

    # ─── Step 4: Feature Engineering ───────────────────────────
    X, y = prepare_features(df_clean)

    # Save processed dataset
    import pandas as pd
    processed = pd.concat([X, y], axis=1)
    save_dataframe(processed, "cleaned_dataset.csv")
    logger.info("✓ Processed dataset saved to data/processed/cleaned_dataset.csv")

    # ─── Step 5: Train/Test Split ──────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = list(X.columns)

    # ─── Step 6: Cross-Validation ──────────────────────────────
    models = get_models()
    cv_results = cross_validate_models(models, X_train, y_train)

    # ─── Step 7: Train on Full Training Set ────────────────────
    models = get_models()  # Fresh instances for full training
    trained_models = train_models(models, X_train, y_train)

    # ─── Step 8: Evaluate on Test Set ──────────────────────────
    results, best_name = evaluate_on_test(trained_models, X_test, y_test)

    # ─── Step 9: Full Evaluation (charts, reports, SHAP) ──────
    importance_df, risk_df = run_evaluation(
        results, best_name, cv_results, trained_models,
        X_test, y_test, feature_names,
    )

    # ─── Step 10: Save Best Model ──────────────────────────────
    logger.info("=" * 50)
    logger.info("STAGE: MODEL SAVING")
    logger.info("=" * 50)

    best_model = trained_models[best_name]
    model_path = save_model(best_model, "best_model.joblib")
    logger.info(f"✓ Best model ({best_name}) saved: {model_path}")

    # Save scaler if the best model has one
    if hasattr(best_model, "named_steps") and "scaler" in best_model.named_steps:
        scaler_path = save_model(best_model.named_steps["scaler"], "scaler.joblib")
        logger.info(f"✓ Scaler saved: {scaler_path}")

    # ─── Summary ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Model:    {best_name}")
    logger.info(f"ROC-AUC:       {results[best_name]['roc_auc']:.4f}")
    logger.info(f"F1 Score:      {results[best_name]['f1']:.4f}")
    logger.info(f"Recall:        {results[best_name]['recall']:.4f}")
    if "optimal_threshold" in results[best_name]:
        logger.info(f"Opt Threshold: {results[best_name]['optimal_threshold']:.3f}")
        logger.info(f"Tuned Recall:  {results[best_name]['tuned_recall']:.4f}")
    logger.info("")
    logger.info("Outputs:")
    logger.info("  data/processed/cleaned_dataset.csv")
    logger.info("  models/best_model.joblib")
    logger.info("  reports/evaluation_report.txt")
    logger.info("  reports/risk_scores.csv")
    logger.info("  reports/*.png (EDA & evaluation charts)")


if __name__ == "__main__":
    main()
