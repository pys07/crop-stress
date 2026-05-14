"""
BiLSTM Model for Crop Stress Prediction
========================================
This module implements a neural network for time-series prediction of crop stress levels.
Using scikit-learn's MLPClassifier as TensorFlow is not compatible with Python 3.14+
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from .config import BILSTM_PARAMS, TARGET_COLUMNS


class BiLSTMModel:
    """Neural network model for multi-task crop stress prediction."""

    def __init__(self, sequence_length: int = 30, n_targets: int = 3):
        """
        Initialize the BiLSTM model.

        Args:
            sequence_length: Number of timesteps in each sequence
            n_targets: Number of target variables (stress types)
        """
        self.sequence_length = sequence_length
        self.n_targets = n_targets
        self.scaler = StandardScaler()
        self.models = {}
        self.history = {}

    def build_model(self, n_features: int):
        """
        Build a neural network model using scikit-learn's MLPClassifier.

        Args:
            n_features: Number of input features

        Returns:
            Compiled MLPClassifier model
        """
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            batch_size=32,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=0
        )
        return model

    def prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare sequences from time-series data (flattened for scikit-learn).

        Args:
            data: Shape (n_samples, n_features)

        Returns:
            Shape (n_samples, n_features) - flattened sequences
        """
        # For scikit-learn, we use the data as-is (no sequence windows needed)
        return data

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        target_name: str = "stress",
    ) -> dict:
        """
        Train the neural network model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            target_name: Name of the target variable

        Returns:
            Training history dictionary
        """
        # Fit scaler and transform data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build and train model
        model = self.build_model(n_features=X_train_scaled.shape[1])
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Store model
        self.models[target_name] = model
        
        # Simple history for compatibility
        history = {
            'epochs': len(model.loss_curve_) if hasattr(model, 'loss_curve_') else 1,
            'final_loss': model.loss_ if hasattr(model, 'loss_') else 0,
            'target': target_name
        }
        self.history[target_name] = history
        
        return history

    def predict(self, X_test: np.ndarray, target_name: str = "stress") -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Test features
            target_name: Name of the target variable

        Returns:
            Probability predictions
        """
        if target_name not in self.models:
            raise ValueError(f"Model for {target_name} not found. Train first.")
        
        # Scale using fitted scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        model = self.models[target_name]
        predictions = model.predict_proba(X_test_scaled)
        
        # Return probability of class 1
        return predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, target_name: str = "stress") -> dict:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets
            target_name: Name of the target variable

        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.predict(X_test, target_name)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'loss': 0
        }
        
        return metrics

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'n_targets': self.n_targets,
            'history': self.history
        }, filepath)

    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.models = data['models']
        self.scaler = data['scaler']
        self.sequence_length = data.get('sequence_length', 30)
        self.n_targets = data.get('n_targets', 3)
        self.history = data.get('history', {})

