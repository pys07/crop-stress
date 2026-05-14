#!/usr/bin/env python
"""Update training report with BiLSTM metrics"""
import json
import joblib
from pathlib import Path
from src.config import MODELS_DIR, REPORTS_DIR

# Load BiLSTM artifact
bilstm_path = MODELS_DIR / "bilstm_artifacts.joblib"
bilstm_artifact = joblib.load(bilstm_path)

# Load current report
report_path = REPORTS_DIR / "training_report.json"
with open(report_path, 'r') as f:
    report = json.load(f)

# Extract BiLSTM metrics
bilstm_metrics = bilstm_artifact.get("metrics", {})

# Add BiLSTM to report
report["models"]["bilstm"] = {
    "label": "Bidirectional LSTM",
    "type": "neural_network",
    "average_f1_score": sum(m.get("f1", 0) for m in bilstm_metrics.values()) / len(bilstm_metrics) if bilstm_metrics else 0,
    "average_accuracy": sum(m.get("accuracy", 0) for m in bilstm_metrics.values()) / len(bilstm_metrics) if bilstm_metrics else 0,
    "metrics_per_target": bilstm_metrics
}

# Update best model if BiLSTM is better
current_best_f1 = report["models"][report["best_model"]]["average_f1_score"]
bilstm_f1 = report["models"]["bilstm"]["average_f1_score"]

if bilstm_f1 > current_best_f1:
    report["best_model"] = "bilstm"

# Save updated report
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f'✅ Updated training report with BiLSTM metrics')
print(f'   BiLSTM F1: {report["models"]["bilstm"]["average_f1_score"]:.4f}')
print(f'   BiLSTM Accuracy: {report["models"]["bilstm"]["average_accuracy"]:.4f}')
print(f'   Best model: {report["best_model"]}')
