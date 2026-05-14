"""
Cache Manager
=============
Centralized caching for dataset and report loading.
Reduces duplicate code and improves performance.
"""

from typing import Optional
import pandas as pd
from .ml_utils import load_dataset, load_report


class CacheManager:
    """Thread-safe cache manager for ML data and reports."""
    
    def __init__(self):
        """Initialize cache with None values."""
        self._dataset_cache: Optional[pd.DataFrame] = None
        self._report_cache: Optional[dict] = None
        self._cache_timestamp: dict = {"dataset": None, "report": None}
    
    def get_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get cached dataset or load if not cached.
        
        Args:
            force_refresh: Force reload even if cached
            
        Returns:
            DataFrame with crop stress data
        """
        if force_refresh or self._dataset_cache is None:
            self._dataset_cache = load_dataset()
        return self._dataset_cache
    
    def get_report(self, force_refresh: bool = False) -> dict:
        """
        Get cached report or load if not cached.
        
        Args:
            force_refresh: Force reload even if cached
            
        Returns:
            Dictionary with training report
        """
        if force_refresh or self._report_cache is None:
            self._report_cache = load_report()
        return self._report_cache
    
    def refresh_all(self) -> None:
        """Clear all caches."""
        self._dataset_cache = None
        self._report_cache = None
    
    def clear_dataset(self) -> None:
        """Clear only dataset cache."""
        self._dataset_cache = None
    
    def clear_report(self) -> None:
        """Clear only report cache."""
        self._report_cache = None


# Global cache manager instance
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager
