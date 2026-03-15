"""
Model Evaluation Module.
Generates evaluation visualizations: confusion matrices, ROC curves,
SHAP feature importance, and the risk score dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

from src.utils import REPORTS_DIR, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM

logger = logging.getLogger("dropout_prediction.model_evaluator")


def plot_confusion_matrices(results, y_test, save_dir=None):
    """Plot confusion matrix for each model."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Retained", "Dropout"],
            yticklabels=["Retained", "Dropout"],
        )
        ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "06_confusion_matrices.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def plot_roc_curves(results, y_test, save_dir=None):
    """Plot ROC curves for all models on one chart."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "07_roc_curves.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def plot_metrics_comparison(results, save_dir=None):
    """Bar chart comparing all metrics across models."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())

    data = []
    for name in model_names:
        for metric in metrics:
            data.append({
                "Model": name,
                "Metric": metric.upper().replace("_", "-"),
                "Score": results[name][metric],
            })

    df_metrics = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_metrics,
        x="Metric",
        y="Score",
        hue="Model",
        ax=ax,
        palette="viridis",
        edgecolor="white",
    )
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "08_metrics_comparison.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def compute_feature_importance(model, feature_names, best_name, save_dir=None):
    """
    Compute and plot feature importance.
    Uses SHAP for tree-based models, coefficients for Logistic Regression.
    """
    if save_dir is None:
        save_dir = REPORTS_DIR

    logger.info(f"Computing feature importance for {best_name}...")

    # Extract the actual estimator from the Pipeline
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    else:
        estimator = model

    importance_values = None
    method_used = ""

    # Try SHAP for tree-based models
    if best_name in ["XGBoost", "Random Forest", "Gradient Boosting"]:
        if hasattr(estimator, "feature_importances_"):
            importance_values = estimator.feature_importances_
            method_used = "feature_importances_ (Gini/Gain)"

    # Use coefficients for Logistic Regression
    elif hasattr(estimator, "coef_"):
        importance_values = np.abs(estimator.coef_[0])
        method_used = "absolute coefficients"

    if importance_values is None:
        logger.warning("Could not extract feature importance")
        return None

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values,
    }).sort_values("Importance", ascending=False)

    # Log top features
    logger.info(f"Feature importance method: {method_used}")
    logger.info("Top 10 features:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['Feature']:30s} {row['Importance']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(
        range(top_n),
        top_features["Importance"].values[::-1],
        color=colors[::-1],
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features["Feature"].values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance — {best_name}", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "09_feature_importance.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")

    return importance_df


def generate_risk_scores(model, X_test, save_dir=None):
    """
    Generate risk score dashboard output.
    Each student gets: Risk_Score, Risk_Level, Top_Risk_Factor
    """
    if save_dir is None:
        save_dir = REPORTS_DIR

    logger.info("Generating risk score dashboard...")

    if not hasattr(model, "predict_proba"):
        logger.warning("Model does not support predict_proba, skipping risk scores")
        return None

    y_proba = model.predict_proba(X_test)[:, 1]

    # Create risk levels
    risk_levels = []
    for p in y_proba:
        if p >= RISK_THRESHOLD_HIGH:
            risk_levels.append("🔴 High")
        elif p >= RISK_THRESHOLD_MEDIUM:
            risk_levels.append("🟡 Medium")
        else:
            risk_levels.append("🟢 Low")

    risk_df = pd.DataFrame({
        "Student_Index": range(len(y_proba)),
        "Risk_Score": np.round(y_proba, 4),
        "Risk_Level": risk_levels,
    })

    # Summary
    level_counts = risk_df["Risk_Level"].value_counts()
    logger.info("Risk Level Distribution:")
    for level, count in level_counts.items():
        pct = count / len(risk_df) * 100
        logger.info(f"  {level}: {count} ({pct:.1f}%)")

    filepath = os.path.join(save_dir, "risk_scores.csv")
    risk_df.to_csv(filepath, index=False)
    logger.info(f"✓ Risk scores saved: {filepath}")

    return risk_df


def generate_evaluation_report(results, best_name, cv_results, y_test, save_dir=None):
    """Generate a text-based evaluation report."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    filepath = os.path.join(save_dir, "evaluation_report.txt")

    with open(filepath, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("STUDENT DROPOUT RISK PREDICTION — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Baseline
        majority_acc = max(y_test.mean(), 1 - y_test.mean())
        f.write(f"Baseline (Majority Class) Accuracy: {majority_acc:.4f}\n\n")

        # Cross-Validation Results
        f.write("-" * 60 + "\n")
        f.write("CROSS-VALIDATION RESULTS (Stratified 5-Fold)\n")
        f.write("-" * 60 + "\n")
        if cv_results:
            f.write(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Recall':>10}\n")
            f.write("-" * 55 + "\n")
            for name, metrics in cv_results.items():
                acc = f"{metrics['accuracy']['mean']:.4f}±{metrics['accuracy']['std']:.4f}"
                roc = f"{metrics['roc_auc']['mean']:.4f}±{metrics['roc_auc']['std']:.4f}"
                rec = f"{metrics['recall']['mean']:.4f}±{metrics['recall']['std']:.4f}"
                f.write(f"{name:<25} {acc:>10} {roc:>10} {rec:>10}\n")
        f.write("\n")

        # Test Set Results
        f.write("-" * 60 + "\n")
        f.write("TEST SET RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
            f"{'F1':>7} {'AUC':>7} {'Beat BL':>8}\n"
        )
        f.write("-" * 68 + "\n")
        for name, res in results.items():
            beat = "Yes" if res.get("beats_baseline", False) else "No"
            f.write(
                f"{name:<25} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
                f"{res['recall']:>7.4f} {res['f1']:>7.4f} {res['roc_auc']:>7.4f} "
                f"{beat:>8}\n"
            )

        f.write(f"\n★ Best Model: {best_name}\n")
        f.write(f"  ROC-AUC: {results[best_name]['roc_auc']:.4f}\n")

        if "optimal_threshold" in results[best_name]:
            f.write(
                f"  Optimal Threshold: {results[best_name]['optimal_threshold']:.3f}\n"
            )
            f.write(
                f"  Tuned Recall: {results[best_name]['tuned_recall']:.4f}\n"
            )
            f.write(
                f"  Tuned Precision: {results[best_name]['tuned_precision']:.4f}\n"
            )

        # Classification Report for best model
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"CLASSIFICATION REPORT — {best_name}\n")
        f.write("-" * 60 + "\n")
        report = classification_report(
            y_test,
            results[best_name]["y_pred"],
            target_names=["Retained", "Dropout"],
        )
        f.write(report + "\n")

    logger.info(f"✓ Evaluation report saved: {filepath}")
    return filepath


def run_evaluation(results, best_name, cv_results, trained_models,
                   X_test, y_test, feature_names):
    """Run the full evaluation pipeline."""
    logger.info("=" * 50)
    logger.info("STAGE: MODEL EVALUATION")
    logger.info("=" * 50)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_metrics_comparison(results)

    importance_df = compute_feature_importance(
        trained_models[best_name], feature_names, best_name,
    )

    risk_df = generate_risk_scores(trained_models[best_name], X_test)

    generate_evaluation_report(results, best_name, cv_results, y_test)

    logger.info("✓ All evaluation outputs generated")

    return importance_df, risk_df
