"""
Linear Regression Model Artifact
=================================
Python module for Linear Regression crop stress prediction model.
Converted from: linear_regression_artifacts.joblib
"""

from typing import List, Dict, Any


class LinearRegressionArtifact:
    """Linear Regression model artifact for crop stress prediction."""
    
    # Model features
    FEATURES: List[str] = [
        'soil_pH',
        'soil_moisture_10cm',
        'air_temperature_max',
        'air_temperature_min',
        'soil_temperature',
        'soil_moisture_30cm',
        'air_temperature_max_lag3_mean',
        'day_of_year',
        'air_temperature_max_lag7_mean',
        'solar_radiation'
    ]
    
    # Prediction targets
    TARGETS: List[str] = [
        'temperature_stress',
        'water_stress',
        'waterlogging_stress'
    ]
    
    # Soil parameters included
    SOIL_PARAMETERS: List[str] = [
        'soil_pH',
        'soil_moisture_10cm',
        'soil_temperature',
        'soil_moisture_30cm'
    ]
    
    def __init__(self):
        """Initialize Linear Regression artifact."""
        self.model_type = "traditional"
        self.estimators = {}
        self.metadata = {
            'model': 'Linear Regression',
            'type': 'continuous regressor',
            'algorithm': 'Ordinary Least Squares (OLS)',
            'fit_intercept': True,
            'normalize': False,
            'n_jobs': -1,
            'characteristics': [
                'Assumes linear relationship',
                'Outputs continuous stress probability',
                'Fast training and inference',
                'Interpretable coefficients'
            ],
        }
    
    def get_features(self) -> List[str]:
        """Get list of input features."""
        return self.FEATURES
    
    def get_targets(self) -> List[str]:
        """Get list of prediction targets."""
        return self.TARGETS
    
    def get_soil_parameters(self) -> List[str]:
        """Get soil parameters used."""
        return self.SOIL_PARAMETERS
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'model_type': self.model_type,
            'features': len(self.FEATURES),
            'targets': len(self.TARGETS),
            'soil_parameters': len(self.SOIL_PARAMETERS),
            'configuration': self.metadata,
        }
    
    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about each feature."""
        return {
            feature: {
                'name': feature,
                'is_soil_param': feature in self.SOIL_PARAMETERS,
                'is_temporal': 'lag' in feature or 'day_of' in feature,
            }
            for feature in self.FEATURES
        }
    
    def get_characteristics(self) -> List[str]:
        """Get model characteristics."""
        return self.metadata.get('characteristics', [])
    
    def predict_range(self) -> Dict[str, tuple]:
        """Get typical prediction range (0 to 1 for probability, clipped)."""
        return {
            'min': 0.0,
            'max': 1.0,
            'typical_range': (0.0, 1.0),
        }
    
    def __repr__(self) -> str:
        return (
            f"LinearRegressionArtifact("
            f"features={len(self.FEATURES)}, "
            f"targets={len(self.TARGETS)}, "
            f"estimators={len(self.estimators)})"
        )
    
    def __str__(self) -> str:
        return (
            f"Linear Regression Model\n"
            f"  Features: {len(self.FEATURES)}\n"
            f"  Targets: {len(self.TARGETS)}\n"
            f"  Model Type: {self.model_type}\n"
            f"  Algorithm: OLS\n"
            f"  Configuration: {self.metadata}"
        )


# Create global instance
linear_regression_model = LinearRegressionArtifact()


if __name__ == "__main__":
    # Example usage
    model = LinearRegressionArtifact()
    
    print("\n" + "="*60)
    print("Linear Regression Model Artifact")
    print("="*60)
    print(f"\n{model}\n")
    
    print("Features:")
    for i, feature in enumerate(model.get_features(), 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nTargets:")
    for i, target in enumerate(model.get_targets(), 1):
        print(f"  {i}. {target}")
    
    print(f"\nSoil Parameters:")
    for param in model.get_soil_parameters():
        print(f"  • {param}")
    
    print(f"\nModel Characteristics:")
    for char in model.get_characteristics():
        print(f"  • {char}")
    
    print(f"\nPrediction Range: {model.predict_range()}")
    print(f"\nMetadata: {model.get_metadata()}\n")
