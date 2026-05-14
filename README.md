# Crop Stress Prediction Project

This project builds a working crop stress prediction system from your dataset and presentation idea.

It predicts:

- `temperature_stress`
- `water_stress`
- `waterlogging_stress`

The implemented models are:

- Naive Bayes
- Random Forest
- Linear Regression

## Dataset columns used

The project automatically selects the most important features from the dataset, instead of using every column blindly.

Core candidate inputs:

- air temperature
- humidity
- precipitation
- solar radiation
- VPD
- soil moisture at 10 cm
- soil moisture at 30 cm
- soil temperature
- soil pH
- lagged precipitation, temperature, and VPD features
- derived calendar features from `date`

## Soil parameters provided in the app

The app explicitly asks for these soil parameters:

- `soil_moisture_10cm`
- `soil_moisture_30cm`
- `soil_temperature`
- `soil_pH`

## How to run

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Train models:

```powershell
python train_models.py
```

3. Start the app:

```powershell
streamlit run app.py
```

## Output files

- `models/` stores trained model artifacts
- `reports/training_report.json` stores selected features and evaluation metrics

## Project structure

- [app.py](c:\Users\Chris\OneDrive\Documents\Desktop\Crop Stress Predication\app.py)
- [train_models.py](c:\Users\Chris\OneDrive\Documents\Desktop\Crop Stress Predication\train_models.py)
- [src/ml_utils.py](c:\Users\Chris\OneDrive\Documents\Desktop\Crop Stress Predication\src\ml_utils.py)
- [data/crop_stress_dataset.csv](c:\Users\Chris\OneDrive\Documents\Desktop\Crop Stress Predication\data\crop_stress_dataset.csv)
