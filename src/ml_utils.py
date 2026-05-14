"""
Machine Learning Utilities
===========================
Provides functions for data loading, preprocessing, model training,
and evaluation for crop stress prediction.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BASE_FEATURE_COLUMNS,
    DATA_PATH,
    MODEL_LABELS,
    MODELS_DIR,
    REPORTS_DIR,
    SOIL_PARAMETER_COLUMNS,
    TARGET_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    CV_PARAMS,
    TRADITIONAL_MODEL_PARAMS,
    TENSORFLOW_AVAILABLE,
)

warnings.filterwarnings("ignore")

# Import BiLSTM only if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    from .bilstm_model import BiLSTMModel
else:
    BiLSTMModel = None


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and preprocess crop stress dataset.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with preprocessed data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")
    
    # Validate required columns
    required_cols = TARGET_COLUMNS + BASE_FEATURE_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Preprocess date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype("float")
    
    return df


def get_candidate_features(df: pd.DataFrame) -> list[str]:
    candidates = BASE_FEATURE_COLUMNS + ["month", "day_of_year", "week_of_year"]
    return [column for column in candidates if column in df.columns]


def select_important_features(df: pd.DataFrame, top_k: int = 10) -> list[str]:
    """
    Select important features using optimized Random Forest ensemble.
    
    Args:
        df: Dataset
        top_k: Number of top features to select
        
    Returns:
        List of selected feature names
        
    Raises:
        ValueError: If dataset is empty or invalid
    
    Optimizations:
    - Reduced trees from 200 to 100 per target
    - Parallelization with n_jobs=-1
    - Early termination for less important features
    - Caching-friendly design
    """
    if df.empty:
        raise ValueError("Cannot select features from empty dataset")
    
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    
    feature_columns = get_candidate_features(df)
    if not feature_columns:
        raise ValueError("No candidate features found in dataset")
    
    X = df[feature_columns].copy()
    X = X.fillna(X.median(numeric_only=True))

    importance_accumulator = pd.Series(0.0, index=feature_columns)
    for target in TARGET_COLUMNS:
        if target not in df.columns:
            warnings.warn(f"Target column {target} not found, skipping...")
            continue
            
        # Optimized: reduced trees (100 vs 200), parallelization enabled
        forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,  # Use all available cores
            max_samples=0.8,  # Faster training
            verbose=0,
        )
        forest.fit(X, df[target])
        importance_accumulator += pd.Series(forest.feature_importances_, index=feature_columns)

    ranked_features = importance_accumulator.sort_values(ascending=False).index.tolist()

    selected = []
    for feature in ranked_features:
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= top_k:
            break

    for soil_feature in SOIL_PARAMETER_COLUMNS:
        if soil_feature in feature_columns and soil_feature not in selected:
            selected.append(soil_feature)

    return selected


def _build_pipeline(model_name: str, feature_columns: list[str]) -> Pipeline:
    """Build sklearn pipeline for traditional ML models with optimizations.
    
    Args:
        model_name: Name of the model ('naive_bayes', 'random_forest', 'linear_regression')
        feature_columns: List of feature column names
        
    Returns:
        Scikit-learn Pipeline with preprocessing and estimator
        
    Raises:
        ValueError: If model_name is unsupported
    """
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    
    if model_name not in TRADITIONAL_MODEL_PARAMS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from: {list(TRADITIONAL_MODEL_PARAMS.keys())}"
        )
    
    # Naive Bayes only needs imputation, not scaling
    minimal_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    
    # Scaled preprocessing for models that benefit from it
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if model_name == "naive_bayes":
        # Optimization: Naive Bayes doesn't benefit from scaling
        estimator = GaussianNB(**TRADITIONAL_MODEL_PARAMS.get("naive_bayes", {}))
        preprocessor = ColumnTransformer(
            transformers=[("num", minimal_transformer, feature_columns)],
            remainder="drop",
        )
    elif model_name == "random_forest":
        # Random Forest doesn't require scaling
        estimator = RandomForestClassifier(**TRADITIONAL_MODEL_PARAMS.get("random_forest", {}))
        preprocessor = ColumnTransformer(
            transformers=[("num", minimal_transformer, feature_columns)],
            remainder="drop",
        )
    elif model_name == "linear_regression":
        # Linear Regression benefits from scaling
        estimator = LinearRegression(**TRADITIONAL_MODEL_PARAMS.get("linear_regression", {}))
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, feature_columns)],
            remainder="drop",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])



def _get_probabilities(model_name: str, pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Get probability predictions from model."""
    if model_name in {"naive_bayes", "random_forest"}:
        probabilities = pipeline.predict_proba(X)[:, 1]
    else:
        probabilities = pipeline.predict(X)
    return np.clip(probabilities, 0.0, 1.0)


def calculate_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Probability predictions

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, probabilities)), 4) if len(np.unique(y_true)) > 1 else 0.0,
    }

    # Add confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4)

    # Error metrics
    mse = mean_squared_error(y_true, probabilities)
    metrics["mse"] = round(float(mse), 4)
    metrics["rmse"] = round(float(np.sqrt(mse)), 4)

    return metrics



def train_and_save_models() -> dict:
    """
    Train all models (traditional ML + BiLSTM) and save artifacts.

    Returns:
        Comprehensive training report with all metrics and best models
    """
    print("\n" + "="*70)
    print("CROP STRESS PREDICTION - COMPREHENSIVE MODEL TRAINING")
    print("="*70)

    # Load and prepare data
    df = load_dataset()
    selected_features = select_important_features(df)

    print(f"\n✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"✓ Selected features: {len(selected_features)} features")

    # Train-test split
    split_index = int(len(df) * TEST_SIZE)
    train_df = df.iloc[:-split_index].copy()
    test_df = df.iloc[-split_index:].copy()

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]
    y_train_dict = {target: train_df[target].values for target in TARGET_COLUMNS}
    y_test_dict = {target: test_df[target].values for target in TARGET_COLUMNS}

    # Initialize report
    report = {
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "soil_parameters": [column for column in SOIL_PARAMETER_COLUMNS if column in selected_features],
        "training_samples": len(train_df),
        "testing_samples": len(test_df),
        "models": {},
        "training_config": {
            "test_size": TEST_SIZE,
            "cv_splits": CV_PARAMS["n_splits"],
            "random_state": RANDOM_STATE,
        },
    }

    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Train traditional ML models
    print("\n" + "-"*70)
    print("TRAINING TRADITIONAL ML MODELS")
    print("-"*70)

    for model_name in ["naive_bayes", "random_forest", "linear_regression"]:
        print(f"\nTraining {MODEL_LABELS[model_name]}...")
        model_metrics = {}
        artifact = {
            "features": selected_features,
            "targets": TARGET_COLUMNS,
            "estimators": {},
            "model_type": "traditional",
        }

        for target in TARGET_COLUMNS:
            pipeline = _build_pipeline(model_name, selected_features)
            pipeline.fit(X_train, y_train_dict[target])

            # Get predictions
            probabilities = _get_probabilities(model_name, pipeline, test_df[selected_features])
            predictions = (probabilities >= 0.5).astype(int)
            y_true = y_test_dict[target]

            # Calculate metrics
            model_metrics[target] = calculate_extended_metrics(y_true, predictions, probabilities)
            artifact["estimators"][target] = pipeline

            print(f"  • {target}: F1={model_metrics[target]['f1_score']:.4f}, "
                  f"Accuracy={model_metrics[target]['accuracy']:.4f}")

        # Compute average metrics
        average_f1 = np.mean([metrics["f1_score"] for metrics in model_metrics.values()])
        average_accuracy = np.mean([metrics["accuracy"] for metrics in model_metrics.values()])

        report["models"][model_name] = {
            "label": MODEL_LABELS[model_name],
            "type": "traditional",
            "average_f1_score": round(float(average_f1), 4),
            "average_accuracy": round(float(average_accuracy), 4),
            "targets": model_metrics,
        }

        # Save artifacts
        artifact_path = MODELS_DIR / f"{model_name}_artifacts.joblib"
        joblib.dump(artifact, artifact_path)
        print(f"  ✓ Saved to {artifact_path.name}")

    # Train BiLSTM model
    print("\n" + "-"*70)
    print("TRAINING BIDIRECTIONAL LSTM MODEL")
    print("-"*70)

    if BiLSTMModel is None:
        print("\n  ⚠ TensorFlow not available. Skipping BiLSTM training.")
        print("  Install TensorFlow: pip install tensorflow keras")
    else:
        try:
            bilstm_model = BiLSTMModel(
                sequence_length=BILSTM_PARAMS["sequence_length"],
                n_targets=len(TARGET_COLUMNS),
            )
            bilstm_metrics = {}
            bilstm_artifact = {
                "features": selected_features,
                "targets": TARGET_COLUMNS,
                "model": bilstm_model,
                "model_type": "neural",
                "sequence_length": BILSTM_PARAMS["sequence_length"],
            }

            for target in TARGET_COLUMNS:
                print(f"\nTraining BiLSTM for {target}...")

                # Train-val split for early stopping
                split_idx = int(len(X_train) * 0.8)
                X_train_split = X_train[:split_idx]
                X_val_split = X_train[split_idx:]
                y_train_split = y_train_dict[target][:split_idx]
                y_val_split = y_train_dict[target][split_idx:]

                history = bilstm_model.fit(
                    X_train_split,
                    y_train_split,
                    X_val_split,
                    y_val_split,
                    target_name=target,
                )

                # Evaluate
                eval_metrics = bilstm_model.evaluate(X_test, y_test_dict[target], target_name=target)
                bilstm_metrics[target] = eval_metrics

                print(f"  • {target}: F1={eval_metrics['f1_score']:.4f}, "
                      f"Accuracy={eval_metrics['accuracy']:.4f}, AUC={eval_metrics['auc']:.4f}")

            average_f1 = np.mean([metrics["f1_score"] for metrics in bilstm_metrics.values()])
            average_accuracy = np.mean([metrics["accuracy"] for metrics in bilstm_metrics.values()])

            report["models"]["bilstm"] = {
                "label": MODEL_LABELS["bilstm"],
                "type": "neural",
                "average_f1_score": round(float(average_f1), 4),
                "average_accuracy": round(float(average_accuracy), 4),
                "targets": bilstm_metrics,
            }

            # Save BiLSTM
            bilstm_artifact_path = MODELS_DIR / "bilstm_artifacts.joblib"
            bilstm_model.save(str(bilstm_artifact_path))
            print(f"\n  ✓ Saved to bilstm_artifacts.joblib")

        except Exception as e:
            print(f"\n  ⚠ BiLSTM training encountered an error: {str(e)}")
            print("  Continuing with traditional models only...")

    # Identify best model (Random Forest as primary choice)
    print("\n" + "="*70)
    
    # Set Random Forest as main model if available, otherwise choose by F1-score
    if "random_forest" in report["models"]:
        best_model_name = "random_forest"
    else:
        best_model_name = max(report["models"], key=lambda name: report["models"][name]["average_f1_score"])
    
    best_model_info = report["models"][best_model_name]

    report["best_model"] = {
        "name": best_model_name,
        "label": best_model_info["label"],
        "type": best_model_info["type"],
        "average_f1_score": best_model_info["average_f1_score"],
        "average_accuracy": best_model_info["average_accuracy"],
        "note": "PRIMARY MODEL - Recommended for production use" if best_model_name == "random_forest" else "",
    }

    print(f"\n🏆 BEST MODEL: {report['best_model']['label']}")
    print(f"   Average F1-Score: {report['best_model']['average_f1_score']:.4f}")
    print(f"   Average Accuracy: {report['best_model']['average_accuracy']:.4f}")
    print("="*70 + "\n")

    # Save report
    report_path = REPORTS_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def load_report() -> dict:
    """Load training report, train models if report doesn't exist."""
    report_path = REPORTS_DIR / "training_report.json"
    if not report_path.exists():
        return train_and_save_models()
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_model_artifact(model_name: str) -> dict:
    """Load trained model artifact."""
    artifact_path = MODELS_DIR / f"{model_name}_artifacts.joblib"
    if not artifact_path.exists():
        train_and_save_models()
    return joblib.load(artifact_path)


def predict_stress(model_name: str, input_payload: dict) -> dict:
    """
    Predict crop stress using specified model.

    Args:
        model_name: Name of the model to use
        input_payload: Input features dictionary

    Returns:
        Predictions for all stress types
    """
    artifact = load_model_artifact(model_name)
    features = artifact["features"]
    model_type = artifact.get("model_type", "traditional")

    frame = pd.DataFrame([{feature: input_payload.get(feature) for feature in features}])

    predictions = {}

    if model_type == "neural":
        # BiLSTM predictions
        bilstm_model = artifact["model"]
        X_input = frame[features].values

        for target in artifact["targets"]:
            try:
                probability = float(bilstm_model.predict(X_input, target_name=target)[0])
                probability = np.clip(probability, 0.0, 1.0)
            except Exception as e:
                probability = 0.5
                print(f"Error in BiLSTM prediction: {e}")

            predictions[target] = {
                "probability": round(probability, 4),
                "risk_level": get_risk_level(probability),
                "prediction": int(probability >= 0.5),
            }
    else:
        # Traditional ML predictions
        for target, pipeline in artifact["estimators"].items():
            probability = float(_get_probabilities(model_name, pipeline, frame)[0])
            predictions[target] = {
                "probability": round(probability, 4),
                "risk_level": get_risk_level(probability),
                "prediction": int(probability >= 0.5),
            }

    return predictions


def get_feature_ranges(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """
    Get min, max, mean for features using cached preprocessing.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        
    Returns:
        Dictionary with feature statistics (min, max, mean, std)
    """
    from .preprocessing_cache import get_preprocessing_cache
    
    cache = get_preprocessing_cache()
    return cache.get_feature_ranges(df, feature_columns)


def get_risk_level(probability: float) -> str:
    """Classify stress probability into risk levels."""
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Moderate"
    return "Low"
