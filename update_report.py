#!/usr/bin/env python
"""
Update training report with BiLSTM metrics
"""
import json
import joblib
from pathlib import Path

# Load the current report
report_path = Path("reports/training_report.json")
with open(report_path) as f:
    report = json.load(f)

# Load the BiLSTM artifact
bilstm_artifact = joblib.load("models/bilstm_artifacts.joblib")

# Extract BiLSTM metrics
bilstm_metrics = bilstm_artifact.get("metrics", {})

# Add BiLSTM to the report
bilstm_entry = {
    "label": "Bidirectional LSTM",
    "type": "neural",
    "average_f1_score": round(sum(
        m.get("f1", 0) for m in bilstm_metrics.values() if isinstance(m, dict)
    ) / len(bilstm_metrics), 4) if bilstm_metrics else 0,
    "average_accuracy": round(sum(
        m.get("accuracy", 0) for m in bilstm_metrics.values() if isinstance(m, dict)
    ) / len(bilstm_metrics), 4) if bilstm_metrics else 0,
    "targets": bilstm_metrics
}

report["models"]["bilstm"] = bilstm_entry

# Update best model if BiLSTM has better average F1
current_best = report.get("best_model", "random_forest")
current_best_f1 = report["models"][current_best].get("average_f1_score", 0)
bilstm_f1 = bilstm_entry.get("average_f1_score", 0)

if bilstm_f1 > current_best_f1:
    report["best_model"] = "bilstm"

# Save updated report
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("✅ Training report updated with BiLSTM metrics")
print(f"BiLSTM Average F1: {bilstm_entry['average_f1_score']}")
print(f"BiLSTM Average Accuracy: {bilstm_entry['average_accuracy']}")
