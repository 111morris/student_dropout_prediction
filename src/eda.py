"""
Exploratory Data Analysis (EDA) Module.
Generates visualizations and saves them to the reports directory.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from src.utils import REPORTS_DIR, TARGET_COLUMN

logger = logging.getLogger("dropout_prediction.eda")

# Style configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_class_distribution(df, save_dir=None):
    """Plot dropout class distribution."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts = df[TARGET_COLUMN].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars = axes[0].bar(
        ["Retained (0)", "Dropout (1)"],
        counts.values,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(val),
            ha="center",
            fontweight="bold",
        )
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=["Retained", "Dropout"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.05),
        shadow=True,
    )
    axes[1].set_title("Dropout Rate")

    plt.tight_layout()
    filepath = os.path.join(save_dir, "01_class_distribution.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def plot_numeric_distributions(df, save_dir=None):
    """Plot distributions of numeric features."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)

    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(
            data=df,
            x=col,
            hue=TARGET_COLUMN,
            kde=True,
            ax=axes[i],
            palette=["#2ecc71", "#e74c3c"],
            alpha=0.6,
        )
        axes[i].set_title(col)
        axes[i].legend(labels=["Retained", "Dropout"], fontsize=8)

    # Hide unused axes
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Dropout Status", fontsize=15, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "02_numeric_distributions.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def plot_correlation_heatmap(df, save_dir=None):
    """Plot correlation heatmap of numeric features."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "03_correlation_heatmap.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def plot_categorical_vs_dropout(df, save_dir=None):
    """Plot dropout rates by categorical features."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not cat_cols:
        logger.info("  No categorical columns to plot")
        return

    n_cols = min(3, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        dropout_rates = df.groupby(col)[TARGET_COLUMN].mean().sort_values(ascending=False)
        dropout_rates.plot(
            kind="bar",
            ax=axes[i],
            color="#3498db",
            edgecolor="white",
        )
        axes[i].set_title(f"Dropout Rate by {col}")
        axes[i].set_ylabel("Dropout Rate")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].axhline(
            y=df[TARGET_COLUMN].mean(),
            color="red",
            linestyle="--",
            label="Overall avg",
        )
        axes[i].legend()

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Dropout Rate by Categorical Features", fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "04_categorical_dropout_rates.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def plot_top_correlations_with_target(df, save_dir=None):
    """Plot top features correlated with the target."""
    if save_dir is None:
        save_dir = REPORTS_DIR

    numeric_df = df.select_dtypes(include=[np.number])
    if TARGET_COLUMN not in numeric_df.columns:
        return

    target_corr = numeric_df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in target_corr.values]
    target_corr.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Correlation with Dropout", fontsize=14)
    ax.set_xlabel("Pearson Correlation")
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "05_target_correlations.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {filepath}")


def run_eda(df):
    """Run full EDA and save all visualizations."""
    logger.info("=" * 50)
    logger.info("STAGE: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 50)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    plot_class_distribution(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_vs_dropout(df)
    plot_top_correlations_with_target(df)

    logger.info(f"✓ All EDA visualizations saved to {REPORTS_DIR}")
