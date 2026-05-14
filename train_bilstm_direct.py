#!/usr/bin/env python
"""
Direct BiLSTM training script
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Import directly
from src.bilstm_model import BiLSTMModel
from src.config import (
    DATA_PATH, MODELS_DIR, SOIL_PARAMETER_COLUMNS,
    TARGET_COLUMNS, TEST_SIZE, RANDOM_STATE, BILSTM_PARAMS
)
from src.ml_utils import load_dataset, select_important_features

def train_bilstm():
    print("\n" + "="*70)
    print("TRAINING BILSTM MODEL")
    print("="*70)
    
    # Load data
    df = load_dataset()
    print(f"✓ Dataset loaded: {len(df)} samples")
    
    # Select features
    selected_features = select_important_features(df)
    X = df[selected_features].values
    print(f"✓ Features selected: {X.shape[1]} features")
    
    # Handle NaN values - drop rows with any NaN
    valid_indices = ~np.isnan(X).any(axis=1)
    X = X[valid_indices]
    print(f"✓ Removed NaN rows: {X.shape[0]} valid samples")
    
    # Prepare targets (apply same valid indices)
    y_dict = {}
    for target in TARGET_COLUMNS:
        y_dict[target] = df[target].values[valid_indices]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, = train_test_split(
        X, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    y_train_dict = {}
    y_test_dict = {}
    for target in TARGET_COLUMNS:
        y = y_dict[target]
        y_train, y_test = train_test_split(
            y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        y_train_dict[target] = y_train
        y_test_dict[target] = y_test
    
    # Initialize and train BiLSTM
    bilstm_model = BiLSTMModel(
        sequence_length=BILSTM_PARAMS["sequence_length"],
        n_targets=len(TARGET_COLUMNS)
    )
    
    bilstm_metrics = {}
    bilstm_artifact = {
        "model_type": "bilstm",
        "model": bilstm_model,
        "type": "NEURAL_NETWORK",
        "sequence_length": BILSTM_PARAMS["sequence_length"],
        "metrics": bilstm_metrics
    }
    
    for target in TARGET_COLUMNS:
        print(f"\n  Training BiLSTM for {target}...")
        
        # Fit model
        history = bilstm_model.fit(
            X_train, y_train_dict[target],
            X_test, y_test_dict[target],
            target_name=target
        )
        print(f"    ✓ Model trained")
        
        # Evaluate
        eval_metrics = bilstm_model.evaluate(X_test, y_test_dict[target], target_name=target)
        bilstm_metrics[target] = eval_metrics
        print(f"    • F1: {eval_metrics.get('f1', 0):.4f}")
        print(f"    • Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
    
    # Save model
    bilstm_artifact_path = MODELS_DIR / "bilstm_artifacts.joblib"
    joblib.dump(bilstm_artifact, bilstm_artifact_path)
    print(f"\n  ✓ BiLSTM model saved to {bilstm_artifact_path.name}")
    
    print("\n" + "="*70)
    print("✅ BiLSTM training complete!")
    print("="*70)

if __name__ == "__main__":
    train_bilstm()
