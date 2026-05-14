"""
Data Preprocessing Cache
========================
Optimized caching for data preprocessing operations.
Reduces redundant calculations for scaling, imputation, and feature analysis.
"""

from typing import Dict, Any, Optional
import hashlib
import pandas as pd
import numpy as np
from functools import wraps


class PreprocessingCache:
    """Cache for expensive preprocessing operations."""
    
    def __init__(self):
        """Initialize preprocessing cache."""
        self._cache: Dict[str, Any] = {}
        self._hash_cache: Dict[str, str] = {}
    
    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """
        Get a hash of the DataFrame for cache invalidation.
        Uses shape and first row as quick hash.
        """
        key = f"{df.shape}_{hash(tuple(df.iloc[0].values)) if len(df) > 0 else 0}"
        return hashlib.md5(key.encode()).hexdigest()[:8]
    
    def get_feature_ranges(
        self, 
        df: pd.DataFrame, 
        feature_columns: list[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get cached feature ranges (min, max, mean).
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            
        Returns:
            Dictionary of feature statistics
        """
        cache_key = f"feature_ranges_{self._get_dataframe_hash(df)}_{hash(tuple(feature_columns))}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        feature_ranges = {}
        for feature in feature_columns:
            if feature in df.columns:
                feature_ranges[feature] = {
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "median": float(df[feature].median()),
                }
        
        self._cache[cache_key] = feature_ranges
        return feature_ranges
    
    def get_missing_value_stats(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get cached missing value statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with missing value percentages
        """
        cache_key = f"missing_stats_{self._get_dataframe_hash(df)}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        missing_stats = {
            col: float(df[col].isnull().sum() / len(df) * 100)
            for col in df.columns
        }
        
        self._cache[cache_key] = missing_stats
        return missing_stats
    
    def get_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Get cached data quality score (0-100).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Quality score as percentage
        """
        cache_key = f"quality_score_{self._get_dataframe_hash(df)}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate missing value percentage
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        quality_score = max(0, 100 - missing_pct)
        
        self._cache[cache_key] = quality_score
        return quality_score
    
    def clear_cache(self) -> None:
        """Clear all cached preprocessing data."""
        self._cache.clear()
        self._hash_cache.clear()


# Global preprocessing cache instance
_preprocessing_cache = PreprocessingCache()


def get_preprocessing_cache() -> PreprocessingCache:
    """Get the global preprocessing cache instance."""
    return _preprocessing_cache


def cached_preprocessing(func):
    """Decorator for caching preprocessing operations."""
    cache = get_preprocessing_cache()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # For now, just call the function directly
        # Can be extended with more sophisticated caching
        return func(*args, **kwargs)
    
    return wrapper
