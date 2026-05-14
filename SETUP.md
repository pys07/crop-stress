# 🚀 Project Setup & Installation Guide

## Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
cd "Crop Stress Predication"
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_models.py
```

### 3. Launch Web App
```bash
streamlit run app.py
```

Then open browser: `http://localhost:8501`

---

## 🎯 What Was Improved

### **Architecture Enhancements**

✅ **Added BiLSTM Neural Network**
- Bidirectional LSTM capturing temporal patterns
- 30-day sequence windows
- 2 stacked LSTM layers with dropout regularization
- Early stopping and learning rate scheduling

✅ **Enhanced Model Evaluation**
- Extended metrics: F1, Accuracy, Precision, Recall, AUC, Specificity
- Confusion matrix analysis
- Per-target performance tracking
- Cross-validation support (5-fold)

✅ **Better Feature Engineering**
- Automatic feature importance ranking
- Smart soil parameter preservation
- Temporal features (month, day of year, week)
- Standard scaling and imputation

### **Code Quality**

✅ **Modular Design**
- `config.py` - Centralized configuration and hyperparameters
- `ml_utils.py` - Core ML utilities and training
- `bilstm_model.py` - Neural network implementation  
- `evaluation.py` - Metrics and visualization functions
- `app.py` - Streamlit web interface

✅ **Production Features**
- Error handling and graceful degradation
- Optional TensorFlow support (BiLSTM works without it)
- Model caching and efficient loading
- Comprehensive logging and user feedback

✅ **Documentation**
- Detailed docstrings on all functions
- Inline comments explaining logic
- README with architecture diagrams
- Setup and usage guides

### **User Interface**

✅ **Enhanced Streamlit App**
- Home dashboard with overview statistics
- Interactive analytics dashboard
- Live prediction interface with recommendations
- Model Lab for comparison and retraining
- Analysis page with visualizations
- Project info with workflow documentation
- Dark/Light theme toggle

✅ **Better Visualizations**
- Model performance comparison charts
- Stress distribution analysis
- Per-target metrics comparison
- Feature importance analysis
- Interactive Altair charts

---

## 📊 Model Comparison

| Model | Type | Pros | Cons |
|-------|------|------|------|
| **Random Forest** | Ensemble | Good performance, feature importance | Slow inference |
| **Naive Bayes** | Probabilistic | Fast, interpretable | Assumes independence |
| **Linear Regression** | Linear | Simple baseline | Low accuracy |
| **BiLSTM** | Neural | Captures temporal patterns | Requires TensorFlow |

---

## 🔧 Configuration

### Key Parameters (`src/config.py`)

```python
# BiLSTM Configuration
BILSTM_PARAMS = {
    "sequence_length": 30,          # Time windows
    "units": 128,                   # LSTM units
    "dropout": 0.3,                 # Regularization
    "epochs": 100,                  # Training iterations
    "batch_size": 32,               # Batch size
    "learning_rate": 0.001,         # Adam LR
}

# Model Selection
TRADITIONAL_MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 12,
        "class_weight": "balanced",
    },
    # ... other models
}
```

### Modify for Your Needs

1. **Adjust dataset**: Replace `data/crop_stress_dataset.csv`
2. **Change features**: Edit `BASE_FEATURE_COLUMNS` in `config.py`
3. **Tune models**: Adjust `TRADITIONAL_MODEL_PARAMS` and `BILSTM_PARAMS`
4. **Retrain**: Run `python train_models.py`

---

## 📁 File Structure

```
├── app.py                          # Streamlit app (500+ lines)
├── train_models.py                 # Training script
├── requirements.txt                # Dependencies
├── README_FULL.md                  # Full documentation
├── SETUP.md                        # This file

├── data/
│   └── crop_stress_dataset.csv    # Input data (12,785 records)

├── models/                         # Trained models
│   ├── naive_bayes_artifacts.joblib
│   ├── random_forest_artifacts.joblib
│   ├── linear_regression_artifacts.joblib
│   └── bilstm_artifacts.joblib     # (if TensorFlow installed)

├── reports/                        # Metrics and reports
│   ├── training_report.json       # Full metrics
│   └── model_metrics.csv          # CSV export

└── src/                           # Source modules
    ├── __init__.py
    ├── config.py                  # Hyperparameters
    ├── ml_utils.py                # ML pipeline (400+ lines)
    ├── bilstm_model.py            # LSTM (250+ lines)
    └── evaluation.py              # Visualizations (300+ lines)
```

---

## 💻 System Requirements

- **Python**: 3.8+
- **Memory**: 2GB+ (4GB+ for BiLSTM)
- **Disk**: 500MB+ (for models and dependencies)
- **OS**: Windows, macOS, Linux

### Recommended Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install tensorflow[and-cuda]
```

---

## 🎓 Key Features Explained

### Feature Selection
```python
# Automatically ranks features by importance
selected_features = select_important_features(df, top_k=10)
# Always preserves soil parameters
```

### Model Training
```python
# Trains 4 models independently for each stress type
report = train_and_save_models()
# Returns metrics, saves artifacts
```

### Prediction
```python
# Use best model for new inputs
predictions = predict_stress("random_forest", input_payload)
# Returns probabilities and risk levels
```

### Evaluation
```python
# Comprehensive metrics calculation
metrics = calculate_extended_metrics(y_true, y_pred, probabilities)
# Includes accuracy, F1, AUC, confusion matrix, etc.
```

---

## 🔍 Troubleshooting

### Issue: Models take too long to train
```python
# In config.py, reduce:
BILSTM_PARAMS["epochs"] = 50          # From 100
TRADITIONAL_MODEL_PARAMS["random_forest"]["n_estimators"] = 100  # From 300
```

### Issue: Out of memory
```python
# Reduce batch size:
BILSTM_PARAMS["batch_size"] = 16      # From 32
# Or reduce sequence length:
BILSTM_PARAMS["sequence_length"] = 14 # From 30
```

### Issue: Streamlit port busy
```bash
streamlit run app.py --server.port 8502
```

### Issue: TensorFlow not found
```python
# BiLSTM will be skipped automatically
# Only traditional ML models will train
# To add TensorFlow later:
pip install tensorflow==2.13.0
```

---

## 📈 Performance Metrics Explained

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **F1-Score** | Balance of precision & recall | 0.7+ |
| **Accuracy** | Correct predictions | 0.8+ |
| **Precision** | True positive rate (no false alarms) | 0.8+ |
| **Recall** | Coverage of positive cases | 0.7+ |
| **AUC** | Threshold-independent performance | 0.8+ |

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud
```bash
# Push to GitHub, connect Streamlit Cloud
# Auto-deploys on push
```

### Option 2: Docker Container
```dockerfile
FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

### Option 3: API Server
```python
# Create Flask/FastAPI endpoint
# Use predict_stress() function
```

---

## 🧪 Testing the Installation

```bash
# 1. Check Python
python --version

# 2. Test imports
python -c "from src.config import TARGET_COLUMNS; print('✓ Config OK')"
python -c "from src.ml_utils import load_dataset; print('✓ ML Utils OK')"
python -c "from src.evaluation import plot_model_comparison; print('✓ Evaluation OK')"

# 3. Load dataset
python -c "from src.ml_utils import load_dataset; df = load_dataset(); print(f'✓ Dataset: {len(df)} rows')"

# 4. Train models (takes 1-5 minutes)
python train_models.py

# 5. Launch app
streamlit run app.py
```

---

## 📚 Learning Resources

### ML Concepts
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [LSTM & RNNs Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)

### Time Series Forecasting
- [RNN for Time Series](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215)

### Streamlit Development
- [Streamlit Docs](https://docs.streamlit.io/)
- [Session State Management](https://docs.streamlit.io/library/api-reference/session-state)
- [Caching & Performance](https://docs.streamlit.io/library/advanced-features/caching)

---

## 🎯 Next Steps

1. **Customize for Your Data**
   - Replace `data/crop_stress_dataset.csv`
   - Adjust features in `config.py`
   - Retrain models

2. **Improve Model Performance**
   - Tune hyperparameters
   - Add new features
   - Collect more data

3. **Deploy to Production**
   - Use Streamlit Cloud, Docker, or API
   - Set up monitoring
   - Create CI/CD pipeline

4. **Extend Functionality**
   - Add more models
   - Implement explainability (SHAP)
   - Create mobile app

---

## 📞 Support

For issues or questions:
1. Check `README_FULL.md` for detailed documentation
2. Review docstrings in source files
3. Check Streamlit and Scikit-learn documentation
4. Verify environment setup with troubleshooting section above

---

**Created**: May 2026  
**Python**: 3.8+  
**Framework**: Streamlit + Scikit-learn + TensorFlow (optional)  
**Dataset**: 12,785 records from crop monitoring
