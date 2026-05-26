"""
Model Evaluation and Visualization
===================================
Provides utilities for comprehensive model evaluation, comparison, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
from pathlib import Path

from .config import REPORTS_DIR, TARGET_COLUMNS, MODEL_LABELS

sns.set_style("whitegrid")


def plot_model_comparison(report: dict, metric: str = "average_f1_score") -> plt.Figure:
    """
    Plot model comparison across metrics.

    Args:
        report: Training report dictionary
        metric: Metric to compare (e.g., 'average_f1_score', 'average_accuracy')

    Returns:
        Matplotlib figure
    """
    models = list(report["models"].keys())
    scores = [report["models"][m][metric] for m in models]
    labels = [report["models"][m]["label"] for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(range(len(models)))
    bars = ax.barh(labels, scores, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, i, f"{score:.4f}", va="center", fontweight="bold")

    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_target_metrics(report: dict, model_name: str) -> plt.Figure:
    """
    Plot metrics for each target across all stress types.

    Args:
        report: Training report
        model_name: Model to analyze

    Returns:
        Matplotlib figure
    """
    model_info = report["models"][model_name]
    targets = list(model_info["targets"].keys())
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics_to_plot):
        values = [model_info["targets"][t].get(metric, 0) for t in targets]
        colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))

        axes[idx].bar(targets, values, color=colors, edgecolor="black", linewidth=1.5)
        axes[idx].set_ylabel("Score", fontweight="bold")
        axes[idx].set_title(f"{metric.replace('_', ' ').title()}", fontweight="bold")
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(axis="y", alpha=0.3)

        # Add value labels
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    plt.suptitle(f"Metrics for {model_info['label']}", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_dict: dict, top_k: int = 15) -> plt.Figure:
    """
    Plot top K important features.

    Args:
        importance_dict: Dictionary of feature names and importance scores
        top_k: Number of top features to display

    Returns:
        Matplotlib figure
    """
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    features, importances = zip(*sorted_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, importance in zip(bars, importances):
        ax.text(importance + 0.01, bar.get_y() + bar.get_height() / 2, 
                f"{importance:.4f}", va="center", fontweight="bold")

    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title("Top Features by Importance", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    return fig


def plot_confusion_matrices(report: dict, model_name: str) -> plt.Figure:
    """
    Plot confusion matrices for each target (if data available).

    Args:
        report: Training report
        model_name: Model to analyze

    Returns:
        Matplotlib figure
    """
    model_info = report["models"][model_name]
    targets = list(model_info["targets"].keys())
    n_targets = len(targets)

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]

    for idx, target in enumerate(targets):
        target_metrics = model_info["targets"][target]

        if "true_positives" in target_metrics:
            tp = target_metrics["true_positives"]
            tn = target_metrics["true_negatives"]
            fp = target_metrics["false_positives"]
            fn = target_metrics["false_negatives"]

            cm = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar=False,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            axes[idx].set_title(f"{target.replace('_', ' ').title()}", fontweight="bold")
            axes[idx].set_ylabel("True Label", fontweight="bold")
            axes[idx].set_xlabel("Predicted Label", fontweight="bold")

    plt.suptitle(f"Confusion Matrices - {model_info['label']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def create_summary_report(report: dict) -> str:
    """
    Create a text summary of the training report.

    Args:
        report: Training report dictionary

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 80)
    summary.append("CROP STRESS PREDICTION - MODEL TRAINING SUMMARY")
    summary.append("=" * 80)

    summary.append(f"\nDataset Information:")
    summary.append(f"  • Training samples: {report.get('training_samples', 'N/A')}")
    summary.append(f"  • Testing samples: {report.get('testing_samples', 'N/A')}")
    summary.append(f"  • Selected features: {report.get('feature_count', len(report['selected_features']))}")
    summary.append(f"  • Soil parameters: {len(report['soil_parameters'])}")

    summary.append(f"\nModel Performance:")
    summary.append("-" * 80)

    for model_name, model_info in report["models"].items():
        summary.append(f"\n{model_info['label']} ({model_info.get('type', 'traditional').upper()})")
        summary.append(f"  Average F1-Score: {model_info['average_f1_score']:.4f}")
        summary.append(
            f"  Average Balanced Accuracy: {model_info.get('average_balanced_accuracy', model_info.get('average_accuracy', 'N/A'))}"
        )

        for target, metrics in model_info["targets"].items():
            summary.append(f"\n  {target.replace('_', ' ').title()}:")
            balanced_acc = metrics.get("balanced_accuracy", metrics["accuracy"])
            summary.append(f"    • Balanced Accuracy:  {balanced_acc:.4f}")
            summary.append(f"    • Precision: {metrics['precision']:.4f}")
            summary.append(f"    • Recall:    {metrics['recall']:.4f}")
            summary.append(f"    • F1-Score:  {metrics['f1_score']:.4f}")

    summary.append(f"\n{'=' * 80}")
    summary.append(f"Best Model: {report['best_model']['label']}")
    summary.append(f"Type: {report['best_model'].get('type', 'traditional').upper()}")
    summary.append(f"Average F1-Score: {report['best_model']['average_f1_score']:.4f}")
    if "average_balanced_accuracy" in report["best_model"]:
        summary.append(f"Average Balanced Accuracy: {report['best_model']['average_balanced_accuracy']:.4f}")
    summary.append(f"{'=' * 80}\n")

    return "\n".join(summary)


def export_metrics_to_csv(report: dict, output_path: Path = None) -> pd.DataFrame:
    """
    Export model metrics to CSV format.

    Args:
        report: Training report
        output_path: Path to save CSV (optional)

    Returns:
        Pandas DataFrame with metrics
    """
    rows = []

    for model_name, model_info in report["models"].items():
        for target, metrics in model_info["targets"].items():
            row = {
                "Model": model_info["label"],
                "Target": target,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "AUC": metrics.get("auc", "N/A"),
                "Specificity": metrics.get("specificity", "N/A"),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
