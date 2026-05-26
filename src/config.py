from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "crop_stress_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ==============================================================================
# TARGET & FEATURE COLUMNS
# ==============================================================================
TARGET_COLUMNS = [
    "temperature_stress",
    "water_stress",
    "waterlogging_stress",
]

BASE_FEATURE_COLUMNS = [
    "air_temperature_max",
    "air_temperature_min",
    "humidity_avg",
    "precipitation",
    "solar_radiation",
    "VPD",
    "soil_moisture_10cm",
    "soil_moisture_30cm",
    "soil_temperature",
    "soil_pH",
    "precipitation_lag3_mean",
    "precipitation_lag7_mean",
    "air_temperature_max_lag3_mean",
    "air_temperature_max_lag7_mean",
    "VPD_lag3_mean",
    "VPD_lag7_mean",
]

SOIL_PARAMETER_COLUMNS = [
    "soil_moisture_10cm",
    "soil_moisture_30cm",
    "soil_temperature",
    "soil_pH",
]

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================
MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "linear_regression": "Linear Regression",
    "bilstm": "Bidirectional LSTM",
}

# Primary/main model for production use
PRIMARY_MODEL = "random_forest"

# Hyperparameters for traditional ML models (tuned for best performance)
TRADITIONAL_MODEL_PARAMS = {
    "naive_bayes": {
        "var_smoothing": 1e-8,  # Slight increase for better numerical stability
    },
    "random_forest": {
        "n_estimators": 500,  # Increased for better ensemble diversity
        "max_depth": 15,  # Increased for more complex patterns
        "min_samples_split": 5,  # Optimized
        "min_samples_leaf": 2,  # Optimized
        "max_features": "sqrt",  # Better feature sampling
        "random_state": 42,
        "class_weight": "balanced_subsample",  # Better for imbalanced data
        "n_jobs": -1,  # Use all available cores
        "warm_start": False,
    },
    "linear_regression": {
        "fit_intercept": True,
        "copy_X": True,
        "n_jobs": -1,  # Use parallelization where available
    },
}

# BiLSTM model parameters (if TensorFlow is available) - Optimized
BILSTM_PARAMS = {
    "sequence_length": 30,  # 30-day sequence for temporal patterns
    "units": 128,  # LSTM units
    "dropout": 0.4,  # Increased regularization
    "recurrent_dropout": 0.3,  # Temporal dropout
    "epochs": 150,  # Extended training
    "batch_size": 16,  # Smaller batches for better convergence
    "validation_split": 0.2,
    "early_stopping_patience": 20,  # Better patience
    "learning_rate": 0.0005,  # Lower for stability
    "optimizer": "adam",
    "loss": "binary_crossentropy",
}

# Cross-validation parameters
CV_PARAMS = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Parallel processing configuration
PARALLEL_PROCESSING = {
    "n_jobs": -1,  # Use all available cores
    "verbose": 0,  # Suppress parallel backend messages
}

# Model recommendation thresholds
MODEL_RECOMMENDATIONS = {
    "high_performance": 0.85,  # F1-score >= 0.85
    "medium_performance": 0.70,  # F1-score >= 0.70
    "good_speed": 0.05,  # Inference time < 50ms
}

# Model characteristics for recommendations
MODEL_CHARACTERISTICS = {
    "random_forest": {
        "speed": "Fast",
        "accuracy": "Very High",
        "interpretability": "Medium",
        "use_case": "Best overall, production ready",
    },
    "bilstm": {
        "speed": "Slow",
        "accuracy": "Very High",
        "interpretability": "Low",
        "use_case": "Complex temporal patterns, advanced analysis",
    },
    "naive_bayes": {
        "speed": "Very Fast",
        "accuracy": "Good",
        "interpretability": "High",
        "use_case": "Quick predictions, baseline model, real-time IoT",
    },
    "linear_regression": {
        "speed": "Very Fast",
        "accuracy": "Good",
        "interpretability": "Very High",
        "use_case": "Linear relationships, maximum interpretability",
    },
}

# ==============================================================================
# THRESHOLDS & CONSTANTS
# ==============================================================================
STRESS_THRESHOLDS = {
    "low": 0.4,
    "moderate": 0.7,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==============================================================================
# CHECK FOR TENSORFLOW AVAILABILITY
# ==============================================================================
try:
    import tensorflow
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
