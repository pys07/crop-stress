"""
Python Model Loaders
====================
Python-based model artifact loaders and utilities.
These replace joblib-based loading with pure Python implementations.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

from src.config import MODELS_DIR


class ModelArtifact:
    """Base class for model artifacts."""
    
    def __init__(self, artifact_dict: Dict[str, Any]):
        """
        Initialize artifact from dictionary.
        
        Args:
            artifact_dict: Dictionary containing artifact data
        """
        self.artifact = artifact_dict
        self.model_type = artifact_dict.get('model_type', 'traditional')
        self.features = artifact_dict.get('features', [])
        self.targets = artifact_dict.get('targets', [])
        self.soil_parameters = artifact_dict.get('soil_parameters', [])
        self.estimators = artifact_dict.get('estimators', {})
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get artifact metadata."""
        return {
            'model_type': self.model_type,
            'features': self.features,
            'targets': self.targets,
            'soil_parameters': self.soil_parameters,
            'estimator_count': len(self.estimators),
        }
    
    def get_estimators(self) -> Dict[str, Any]:
        """Get all estimators."""
        return self.estimators
    
    def get_estimator(self, target: str) -> Optional[Any]:
        """Get estimator for specific target."""
        return self.estimators.get(target)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.model_type}, "
            f"features={len(self.features)}, "
            f"targets={len(self.targets)})"
        )


class ModelLoader:
    """Load model artifacts from various formats."""
    
    def __init__(self, model_dir: Path = MODELS_DIR):
        """
        Initialize loader.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        self._cache: Dict[str, ModelArtifact] = {}
    
    def load_joblib(self, model_name: str, cache: bool = True) -> ModelArtifact:
        """
        Load model from joblib file.
        
        Args:
            model_name: Model name (e.g., 'random_forest')
            cache: Whether to cache loaded model
            
        Returns:
            ModelArtifact instance
        """
        import joblib
        
        if cache and model_name in self._cache:
            return self._cache[model_name]
        
        joblib_path = self.model_dir / f"{model_name}_artifacts.joblib"
        if not joblib_path.exists():
            raise FileNotFoundError(f"Model file not found: {joblib_path}")
        
        artifact_dict = joblib.load(joblib_path)
        artifact = ModelArtifact(artifact_dict)
        
        if cache:
            self._cache[model_name] = artifact
        
        return artifact
    
    def load_pickle(self, model_name: str, cache: bool = True) -> ModelArtifact:
        """
        Load model from pickle file.
        
        Args:
            model_name: Model name
            cache: Whether to cache loaded model
            
        Returns:
            ModelArtifact instance
        """
        if cache and model_name in self._cache:
            return self._cache[model_name]
        
        pickle_path = self.model_dir / f"{model_name}_artifacts.pkl"
        if not pickle_path.exists():
            raise FileNotFoundError(f"Model file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            artifact_dict = pickle.load(f)
        
        artifact = ModelArtifact(artifact_dict)
        
        if cache:
            self._cache[model_name] = artifact
        
        return artifact
    
    def load_json_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Load model metadata from JSON file.
        
        Args:
            model_name: Model name
            
        Returns:
            Dictionary with model metadata
        """
        json_path = self.model_dir / f"{model_name}_artifacts.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def load(self, model_name: str, format: str = 'joblib', cache: bool = True) -> ModelArtifact:
        """
        Load model artifact (auto-detect or specified format).
        
        Args:
            model_name: Model name
            format: Format ('joblib', 'pickle', 'auto')
            cache: Whether to cache loaded model
            
        Returns:
            ModelArtifact instance
        """
        if cache and model_name in self._cache:
            return self._cache[model_name]
        
        if format == 'auto':
            # Try joblib first, then pickle
            try:
                return self.load_joblib(model_name, cache=cache)
            except FileNotFoundError:
                return self.load_pickle(model_name, cache=cache)
        elif format == 'joblib':
            return self.load_joblib(model_name, cache=cache)
        elif format == 'pickle':
            return self.load_pickle(model_name, cache=cache)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def load_all(self, format: str = 'joblib') -> Dict[str, ModelArtifact]:
        """
        Load all model artifacts.
        
        Args:
            format: Format to load
            
        Returns:
            Dictionary mapping model names to artifacts
        """
        models = {}
        model_names = ['naive_bayes', 'random_forest', 'linear_regression']
        
        for model_name in model_names:
            try:
                models[model_name] = self.load(model_name, format=format, cache=True)
            except FileNotFoundError:
                warnings.warn(f"Could not load model: {model_name}")
        
        return models
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        self._cache.clear()
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List available model files by format.
        
        Returns:
            Dictionary mapping formats to list of available models
        """
        available = {
            'joblib': [f.stem.replace('_artifacts', '') for f in self.model_dir.glob('*_artifacts.joblib')],
            'pickle': [f.stem.replace('_artifacts', '') for f in self.model_dir.glob('*_artifacts.pkl')],
            'json': [f.stem.replace('_artifacts', '') for f in self.model_dir.glob('*_artifacts.json')],
        }
        return available


# Global model loader instance
_model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return _model_loader


def load_model(model_name: str, format: str = 'auto') -> ModelArtifact:
    """
    Convenient function to load a model.
    
    Args:
        model_name: Model name (e.g., 'random_forest')
        format: Format ('joblib', 'pickle', 'auto')
        
    Returns:
        ModelArtifact instance
    """
    return _model_loader.load(model_name, format=format)


def load_all_models(format: str = 'auto') -> Dict[str, ModelArtifact]:
    """
    Convenient function to load all models.
    
    Args:
        format: Format to load
        
    Returns:
        Dictionary of loaded models
    """
    return _model_loader.load_all(format=format)
