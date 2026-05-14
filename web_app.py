"""
Flask Web Application - Crop Stress Prediction
===============================================
Professional web interface running on localhost:8080
"""

from flask import Flask, render_template, request, jsonify, session
from functools import wraps
import json
import pandas as pd
from datetime import datetime, timedelta
import os

from src.config import MODEL_LABELS, SOIL_PARAMETER_COLUMNS, TARGET_COLUMNS, PRIMARY_MODEL
from src.cache_manager import get_cache_manager
from src.ml_utils import (
    get_feature_ranges,
    predict_stress,
    train_and_save_models,
)
from src.evaluation import create_summary_report

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'crop_stress_secret_key_2026'

# Get cache manager instance
cache = get_cache_manager()

def get_dataset():
    """Get cached dataset."""
    return cache.get_dataset()

def get_report():
    """Get cached report."""
    return cache.get_report()

def refresh_cache():
    """Refresh all caches."""
    cache.refresh_all()

def prettify(name: str) -> str:
    return name.replace("_", " ").replace("VPD", "VPD").title()

def compute_overview(df: pd.DataFrame, report: dict) -> dict:
    total_records = len(df)
    total_fields = int(df["field_id"].nunique()) if "field_id" in df.columns else 0
    if "date" in df.columns:
        date_min = df["date"].min()
        date_max = df["date"].max()
        if pd.isna(date_min) or pd.isna(date_max):
            date_range = "Date range unavailable"
        else:
            date_range = f"{date_min.date()} to {date_max.date()}"
    else:
        date_range = "Date range unavailable"
    stress_rates = {target: float(df[target].mean() * 100) for target in TARGET_COLUMNS}
    return {
        "total_records": total_records,
        "total_fields": total_fields,
        "date_range": date_range,
        "best_model": report["best_model"]["label"],
        "best_score": report["best_model"]["average_f1_score"],
        "stress_rates": stress_rates,
    }

@app.route('/')
def home():
    df = get_dataset()
    report = get_report()
    overview = compute_overview(df, report)
    
    return render_template('home.html',
        total_records=overview['total_records'],
        total_fields=overview['total_fields'],
        best_model=overview['best_model'],
        best_score=f"{overview['best_score']:.4f}",
        date_range=overview['date_range'],
        stress_rates=overview['stress_rates'],
        prettify=prettify
    )

@app.route('/dashboard')
def dashboard():
    df = get_dataset()
    report = get_report()
    
    # Get field options if available
    field_options = []
    if "field_id" in df.columns:
        field_options = ["All Fields"] + sorted(df["field_id"].dropna().unique().tolist())
    
    # Build model performance data
    model_data = []
    for model_name, details in report["models"].items():
        model_data.append({
            "name": details["label"],
            "type": details.get("type", "traditional").upper(),
            "f1": f"{details['average_f1_score']:.4f}",
            "accuracy": f"{details.get('average_accuracy', 0):.4f}"
        })
    
    # Build target metrics
    stress_rates = {target: float(df[target].mean() * 100) for target in TARGET_COLUMNS}
    
    return render_template('dashboard.html',
        field_options=field_options,
        model_data=model_data,
        stress_rates=stress_rates,
        targets=TARGET_COLUMNS,
        prettify=prettify
    )

@app.route('/predictor')
def predictor():
    df = get_dataset()
    report = get_report()
    
    selected_features = report["selected_features"]
    feature_ranges = get_feature_ranges(df, selected_features)
    
    # Prepare slider data
    sliders = {}
    for feature in selected_features:
        details = feature_ranges[feature]
        minimum = float(round(details["min"], 2))
        maximum = float(round(details["max"], 2))
        if minimum == maximum:
            maximum = minimum + 0.01
        default = min(max(float(round(details["mean"], 2)), minimum), maximum)
        sliders[feature] = {
            "min": minimum,
            "max": maximum,
            "default": default,
            "label": prettify(feature),
            "is_soil": feature in SOIL_PARAMETER_COLUMNS
        }
    
    best_model = report.get("best_model", {}).get("name", PRIMARY_MODEL)
    if best_model not in MODEL_LABELS:
        best_model = PRIMARY_MODEL

    return render_template('predictor.html',
        models=MODEL_LABELS,
        best_model=best_model,
        sliders=sliders,
        soil_params=report["soil_parameters"],
        prettify=prettify
    )

@app.route('/model-lab')
def model_lab():
    report = get_report()
    
    # Build comparison data
    models = []
    for model_name, details in report["models"].items():
        models.append({
            "name": details["label"],
            "type": details.get("type", "traditional").upper(),
            "f1": f"{details['average_f1_score']:.4f}",
            "accuracy": f"{details.get('average_accuracy', 0):.4f}"
        })
    
    return render_template('model_lab.html',
        models=models,
        best_model=report["best_model"]["label"],
        best_score=f"{report['best_model']['average_f1_score']:.4f}",
        selected_features=report["selected_features"],
        prettify=prettify
    )

@app.route('/analysis')
def analysis():
    report = get_report()
    df = get_dataset()
    
    # Build comparison data
    model_comparison = pd.DataFrame([
        {"Model": details["label"], "F1-Score": details["average_f1_score"]}
        for details in report["models"].values()
    ]).sort_values("F1-Score", ascending=False)
    
    return render_template('analysis.html',
        report=report,
        model_comparison=model_comparison.to_html(index=False),
        prettify=prettify
    )

@app.route('/project-info')
def project_info():
    report = get_report()
    
    models_info = []
    for model_name, details in report["models"].items():
        models_info.append({
            "name": details["label"],
            "type": details.get("type", "traditional").upper()
        })
    
    return render_template('project_info.html',
        models=models_info,
        feature_count=len(report["selected_features"]),
        soil_params_count=len(report["soil_parameters"]),
        target_count=len(TARGET_COLUMNS)
    )

# API Endpoints
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json or {}
        model_name = data.get("model_name") or PRIMARY_MODEL
        if model_name not in MODEL_LABELS:
            return jsonify({"success": False, "error": "Invalid model selected."}), 400
        
        # Extract payload (convert string keys to proper format)
        payload = {}
        for key, value in data.items():
            if key == "model_name":
                continue
            if value is None or value == "":
                continue
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": f"Invalid value for {key}."}), 400
        
        report = get_report()
        required_features = report.get("selected_features", [])
        missing_features = [feature for feature in required_features if feature not in payload]
        if missing_features:
            missing_preview = ", ".join(missing_features[:5])
            suffix = "..." if len(missing_features) > 5 else ""
            return jsonify({
                "success": False,
                "error": f"Missing inputs: {missing_preview}{suffix}"
            }), 400

        predictions = predict_stress(model_name, payload)
        return jsonify({"success": True, "predictions": predictions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/dashboard-data', methods=['GET'])
def api_dashboard_data():
    try:
        df = get_dataset()
        report = get_report()
        
        field = request.args.get('field', 'All Fields')
        if field != 'All Fields' and "field_id" in df.columns:
            filtered_df = df[df["field_id"] == field]
        else:
            filtered_df = df
        
        # Build chart data
        model_scores = []
        for model_name, details in report["models"].items():
            model_scores.append({
                "name": details["label"],
                "f1": float(details["average_f1_score"])
            })
        
        stress_distribution = []
        for target in TARGET_COLUMNS:
            stress_distribution.append({
                "name": prettify(target),
                "rate": float(filtered_df[target].mean() * 100)
            })
        
        return jsonify({
            "success": True,
            "model_scores": model_scores,
            "stress_distribution": stress_distribution
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    print("🌾 Crop Stress Prediction Platform")
    print("="*50)
    print("Starting Flask server on http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    print("="*50)
    app.run(host='localhost', port=8080, debug=True)
