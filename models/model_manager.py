"""
Python Model Manager
====================
Unified interface for accessing converted Python model artifacts.
"""

from typing import Dict, List, Any
from models.random_forest_model import RandomForestArtifact
from models.naive_bayes_model import NaiveBayesArtifact
from models.linear_regression_model import LinearRegressionArtifact


class ModelManager:
    """Manages all model artifacts."""
    
    def __init__(self):
        """Initialize model manager with all models."""
        self.models = {
            'random_forest': RandomForestArtifact(),
            'naive_bayes': NaiveBayesArtifact(),
            'linear_regression': LinearRegressionArtifact(),
        }
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific model artifact.
        
        Args:
            model_name: Name of model ('random_forest', 'naive_bayes', 'linear_regression')
            
        Returns:
            Model artifact instance
        """
        if model_name not in self.models:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.models.keys())}"
            )
        return self.models[model_name]
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all model artifacts."""
        return self.models
    
    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self.models.keys())
    
    def get_features(self) -> List[str]:
        """Get common features used by all models."""
        return self.models['random_forest'].get_features()
    
    def get_targets(self) -> List[str]:
        """Get prediction targets used by all models."""
        return self.models['random_forest'].get_targets()
    
    def get_soil_parameters(self) -> List[str]:
        """Get soil parameters used by all models."""
        return self.models['random_forest'].get_soil_parameters()
    
    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare all models."""
        comparison = {}
        for name, model in self.models.items():
            comparison[name] = {
                'model': name.replace('_', ' ').title(),
                'type': model.model_type,
                'features': len(model.get_features()),
                'targets': len(model.get_targets()),
                'description': str(model).split('\n')[0],
            }
        return comparison
    
    def summary(self) -> str:
        """Get summary of all models."""
        lines = [
            "\n" + "="*60,
            "Model Manager Summary",
            "="*60,
            f"\nTotal Models: {len(self.models)}",
            f"Common Features: {len(self.get_features())}",
            f"Prediction Targets: {len(self.get_targets())}",
            f"Soil Parameters: {len(self.get_soil_parameters())}",
            "\nAvailable Models:",
        ]
        
        for name, model in self.models.items():
            lines.append(f"  • {name.replace('_', ' ').title()}")
            metadata = model.get_metadata()
            lines.append(f"    - Type: {metadata.get('model_type')}")
            lines.append(f"    - Estimators: {metadata.get('estimators')}")
        
        lines.extend([
            "\nFeatures Used:",
            *[f"  • {feature}" for feature in self.get_features()],
            "\nPrediction Targets:",
            *[f"  • {target}" for target in self.get_targets()],
            "="*60 + "\n"
        ])
        
        return "\n".join(lines)


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_model(model_name: str) -> Any:
    """Convenience function to get a model."""
    return get_model_manager().get_model(model_name)


def list_models() -> List[str]:
    """Convenience function to list available models."""
    return get_model_manager().list_models()


if __name__ == "__main__":
    # Example usage
    manager = get_model_manager()
    
    print(manager.summary())
    
    # Access specific model
    print("\nAccessing Random Forest Model:")
    rf_model = manager.get_model('random_forest')
    print(f"  Features: {len(rf_model.get_features())}")
    print(f"  Targets: {len(rf_model.get_targets())}")
    print(f"  Model: {rf_model}")
    
    # Model comparison
    print("\n" + "="*60)
    print("Model Comparison:")
    print("="*60)
    comparison = manager.get_model_comparison()
    for model_name, info in comparison.items():
        print(f"\n{info['model']}:")
        for key, value in info.items():
            if key != 'model':
                print(f"  • {key}: {value}")
