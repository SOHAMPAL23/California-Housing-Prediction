"""Base Model Class for All Components"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import json
import pickle

class BaseModel(ABC):
    """Abstract base class for all models in the system"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_trained = False
        self.metrics = {}
        self.metadata = {}
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def save(self, filepath: str) -> None:
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def log_metric(self, name: str, value: float) -> None:
        """Log a metric"""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all logged metrics"""
        return self.metrics.copy()
    
    def log_metadata(self, key: str, value: Any) -> None:
        """Log metadata"""
        self.metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata"""
        return self.metadata.copy()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()