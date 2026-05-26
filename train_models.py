"""
Train all crop stress prediction models
========================================
This script trains all models (Naive Bayes, Random Forest, Linear Regression, BiLSTM)
and generates comprehensive evaluation reports and visualizations.
"""

import sys
from pathlib import Path

from src.ml_utils import train_and_save_models, load_report
from src.evaluation import (
    create_summary_report,
    plot_model_comparison,
    plot_target_metrics,
    export_metrics_to_csv,
)


def main():
    """Train all models and generate reports."""
    print("\n" + "▶" * 40)
    print("CROP STRESS PREDICTION MODEL TRAINING")
    print("▶" * 40)

    try:
        # Train models
        report = train_and_save_models()

        # Generate summary report
        summary = create_summary_report(report)
        print(summary)

        # Export metrics to CSV
        reports_dir = Path(__file__).parent / "reports"
        csv_path = reports_dir / "model_metrics.csv"
        metrics_df = export_metrics_to_csv(report, csv_path)
        print(f"✓ Metrics exported to {csv_path.name}")

        # Display model comparison
        print("\nModel Rankings:")
        print("-" * 50)
        models = sorted(
            report["models"].items(),
            key=lambda x: x[1]["average_f1_score"],
            reverse=True,
        )
        for rank, (model_name, model_info) in enumerate(models, 1):
            print(
                f"{rank}. {model_info['label']:25} "
                f"F1: {model_info['average_f1_score']:.4f}  "
                f"Balanced Accuracy: {model_info.get('average_balanced_accuracy', model_info.get('average_accuracy', 'N/A'))}"
            )

        print("\n" + "▶" * 40)
        print("✓ Training completed successfully!")
        print("▶" * 40 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
