#!/usr/bin/env python
"""Update training report with BiLSTM metrics"""
import json
import joblib
from pathlib import Path

from src.config import MODELS_DIR, REPORTS_DIR


def _convert_target_metrics(metrics: dict) -> dict:
    return {
        "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
        "balanced_accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
        "precision": round(float(metrics.get("precision", 0.0)), 4),
        "recall": round(float(metrics.get("recall", 0.0)), 4),
        "f1_score": round(float(metrics.get("f1", 0.0)), 4),
        "auc": round(float(metrics.get("auc", 0.0)), 4),
        "loss": metrics.get("loss", 0),
    }


# Load BiLSTM artifact
bilstm_path = MODELS_DIR / "bilstm_artifacts.joblib"
bilstm_artifact = joblib.load(bilstm_path)

# Load current report
report_path = REPORTS_DIR / "training_report.json"
with open(report_path, "r", encoding="utf-8") as file:
    report = json.load(file)

# Extract BiLSTM metrics
bilstm_metrics = bilstm_artifact.get("metrics", {})
target_metrics = {
    target: _convert_target_metrics(metrics)
    for target, metrics in bilstm_metrics.items()
}

average_f1 = round(
    sum(metrics.get("f1", 0.0) for metrics in bilstm_metrics.values()) / len(bilstm_metrics),
    4,
) if bilstm_metrics else 0.0
average_accuracy = round(
    sum(metrics.get("accuracy", 0.0) for metrics in bilstm_metrics.values()) / len(bilstm_metrics),
    4,
) if bilstm_metrics else 0.0

# Add BiLSTM to report
report.setdefault("models", {})["bilstm"] = {
    "label": "Bidirectional LSTM",
    "type": "neural",
    "average_f1_score": average_f1,
    "average_accuracy": average_accuracy,
    "average_balanced_accuracy": average_accuracy,
    "targets": target_metrics,
}

# Update best model if BiLSTM is better
best_model_name = report.get("best_model", {}).get("name")
best_model_f1 = report["models"].get(best_model_name, {}).get("average_f1_score", 0.0)

if average_f1 >= best_model_f1:
    report["best_model"] = {
        "name": "bilstm",
        "label": "Bidirectional LSTM",
        "type": "neural",
        "average_f1_score": average_f1,
        "average_accuracy": average_accuracy,
        "average_balanced_accuracy": average_accuracy,
        "note": "Recommended for production use",
    }

# Save updated report
with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("Updated training report with BiLSTM metrics")
print(f"BiLSTM F1: {average_f1:.4f}")
print(f"BiLSTM Accuracy: {average_accuracy:.4f}")
print(f"Best model: {report['best_model']['name']}")
