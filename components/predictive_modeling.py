"""Predictive Modeling Layer - ML + DL Models for Housing Price Prediction"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

# Pure Python implementations for compatibility
from core.base_model import BaseModel

class PureLinearRegression(BaseModel):
    """Pure Python implementation of Linear Regression"""
    
    def __init__(self, name: str = "linear_regression", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.weights = None
        self.bias = None
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PureLinearRegression':
        """Train linear regression model using normal equation"""
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        
        # Calculate weights using normal equation: theta = (X^T * X)^(-1) * X^T * y
        try:
            XtX = np.dot(X_with_bias.T, X_with_bias)
            XtX_inv = np.linalg.inv(XtX)
            Xty = np.dot(X_with_bias.T, y)
            theta = np.dot(XtX_inv, Xty)
            
            self.bias = theta[0]
            self.weights = theta[1:]
            
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            theta = np.linalg.pinv(np.dot(X_with_bias.T, X_with_bias))
            theta = np.dot(theta, np.dot(X_with_bias.T, y))
            self.bias = theta[0]
            self.weights = theta[1:]
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return np.dot(X_scaled, self.weights) + self.bias
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions)
        }

class PureLassoRegression(BaseModel):
    """Pure Python implementation of Lasso Regression"""
    
    def __init__(self, name: str = "lasso", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.weights = None
        self.bias = None
        self.scaler = None
        self.alpha = 0.01
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PureLassoRegression':
        """Train lasso regression using coordinate descent"""
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        
        # Initialize parameters
        n_features = X_scaled.shape[1]
        weights = np.random.normal(0, 0.01, n_features)
        alpha = self.config.get('alpha', 0.01)
        
        # Coordinate descent
        max_iter = 1000
        tol = 1e-4
        
        for iteration in range(max_iter):
            weights_old = weights.copy()
            
            for j in range(n_features):
                # Calculate residual without feature j
                residual = y - (np.dot(X_with_bias[:, 1:], weights) + X_with_bias[:, 0] * weights_old[0])
                residual += X_with_bias[:, j+1] * weights_old[j]
                
                # Calculate rho
                rho = np.dot(X_with_bias[:, j+1], residual)
                
                # Soft thresholding
                if rho > alpha:
                    weights[j] = (rho - alpha) / np.sum(X_with_bias[:, j+1] ** 2)
                elif rho < -alpha:
                    weights[j] = (rho + alpha) / np.sum(X_with_bias[:, j+1] ** 2)
                else:
                    weights[j] = 0
            
            # Check convergence
            if np.sum(np.abs(weights - weights_old)) < tol:
                break
        
        self.bias = np.mean(y - np.dot(X_with_bias[:, 1:], weights))
        self.weights = weights
        self.alpha = alpha
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return np.dot(X_scaled, self.weights) + self.bias
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'alpha': self.alpha,
            'sparsity': np.sum(self.weights == 0) / len(self.weights)
        }

class PureRandomForest(BaseModel):
    """Simplified Random Forest implementation"""
    
    def __init__(self, name: str = "random_forest", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.trees = []
        self.scaler = None
        self.n_estimators = config.get('n_estimators', 10)
        self.max_depth = config.get('max_depth', 10)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PureRandomForest':
        """Train random forest"""
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual trees
        self.trees = []
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
            X_boot = X_scaled[indices]
            y_boot = y[indices]
            
            # Train simple decision tree (stump for simplicity)
            tree = self._train_simple_tree(X_boot, y_boot)
            self.trees.append(tree)
        
        self.is_trained = True
        return self
    
    def _train_simple_tree(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train a simple decision tree"""
        n_features = X.shape[1]
        
        # Find best split for each feature
        best_feature = 0
        best_threshold = 0
        best_impurity_reduction = -1
        
        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            
            for threshold in values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    # Calculate impurity reduction
                    left_var = np.var(y[left_mask])
                    right_var = np.var(y[right_mask])
                    weighted_var = (np.sum(left_mask) * left_var + np.sum(right_mask) * right_var) / len(y)
                    impurity_reduction = np.var(y) - weighted_var
                    
                    if impurity_reduction > best_impurity_reduction:
                        best_impurity_reduction = impurity_reduction
                        best_feature = feature_idx
                        best_threshold = threshold
        
        # Calculate leaf values
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_mean = np.mean(y[left_mask]) if np.sum(left_mask) > 0 else np.mean(y)
        right_mean = np.mean(y[right_mask]) if np.sum(right_mask) > 0 else np.mean(y)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_mean': left_mean,
            'right_mean': right_mean
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros((len(X_scaled), len(self.trees)))
        
        for i, tree in enumerate(self.trees):
            left_mask = X_scaled[:, tree['feature']] <= tree['threshold']
            predictions[left_mask, i] = tree['left_mean']
            predictions[~left_mask, i] = tree['right_mean']
        
        return np.mean(predictions, axis=1)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions)
        }

class PureNeuralNetwork(BaseModel):
    """Simple neural network implementation"""
    
    def __init__(self, name: str = "deep_neural_network", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.weights = []
        self.biases = []
        self.scaler = None
        self.architecture = config.get('architecture', [64, 32, 16, 1])
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PureNeuralNetwork':
        """Train neural network"""
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize weights
        layer_sizes = [X_scaled.shape[1]] + self.architecture
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.normal(0, 0.1, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass
            activations = [X_scaled]
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                if i < len(self.weights) - 1:
                    a = np.maximum(0, z)  # ReLU activation
                else:
                    a = z  # Linear output
                activations.append(a)
            
            # Backward pass
            delta = activations[-1] - y.reshape(-1, 1)
            for i in range(len(self.weights) - 1, -1, -1):
                if i > 0:
                    grad_w = np.dot(activations[i].T, delta)
                    grad_b = np.sum(delta, axis=0, keepdims=True)
                    self.weights[i] -= self.learning_rate * grad_w
                    self.biases[i] -= self.learning_rate * grad_b
                    delta = np.dot(delta, self.weights[i].T) * (activations[i] > 0)
                else:
                    grad_w = np.dot(activations[i].T, delta)
                    grad_b = np.sum(delta, axis=0, keepdims=True)
                    self.weights[i] -= self.learning_rate * grad_w
                    self.biases[i] -= self.learning_rate * grad_b
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Forward pass
        a = X_scaled
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU activation
            else:
                a = z  # Linear output
        
        return a.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mae': mean_absolute_error(y, predictions)
        }

class PredictiveModelingLayer:
    """Layer managing all predictive models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_names = []
        self.feature_importance = {}
    
    def fit(self, data: pd.DataFrame) -> 'PredictiveModelingLayer':
        """Train all predictive models"""
        self.logger.info("Training predictive models...")
        
        # Prepare features and target
        target_col = 'median_house_value'
        feature_cols = [col for col in data.columns if col != target_col]
        
        # Create derived features
        data = self._engineer_features(data)
        feature_cols = [col for col in data.columns if col != target_col]
        
        self.feature_names = feature_cols
        
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train models
        models_config = self.config.get('models', {})
        
        if models_config.get('linear_regression', True):
            self.logger.info("Training Linear Regression...")
            lr = PureLinearRegression("linear_regression")
            lr.fit(X_train, y_train)
            lr_metrics = lr.evaluate(X_test, y_test)
            lr.log_metric("test_rmse", lr_metrics['rmse'])
            lr.log_metric("test_r2", lr_metrics['r2'])
            self.models['linear_regression'] = lr
            
        if models_config.get('lasso', True):
            self.logger.info("Training Lasso Regression...")
            lasso = PureLassoRegression("lasso", {"alpha": 0.01})
            lasso.fit(X_train, y_train)
            lasso_metrics = lasso.evaluate(X_test, y_test)
            lasso.log_metric("test_rmse", lasso_metrics['rmse'])
            lasso.log_metric("test_r2", lasso_metrics['r2'])
            self.models['lasso'] = lasso
            
        if models_config.get('random_forest', True):
            self.logger.info("Training Random Forest...")
            rf = PureRandomForest("random_forest", {"n_estimators": 10, "max_depth": 10})
            rf.fit(X_train, y_train)
            rf_metrics = rf.evaluate(X_test, y_test)
            rf.log_metric("test_rmse", rf_metrics['rmse'])
            rf.log_metric("test_r2", rf_metrics['r2'])
            self.models['random_forest'] = rf
            
        if models_config.get('deep_learning', True):
            self.logger.info("Training Deep Neural Network...")
            nn = PureNeuralNetwork("deep_neural_network", {"epochs": 100})
            nn.fit(X_train, y_train)
            nn_metrics = nn.evaluate(X_test, y_test)
            nn.log_metric("test_rmse", nn_metrics['rmse'])
            nn.log_metric("test_r2", nn_metrics['r2'])
            self.models['deep_neural_network'] = nn
        
        # Calculate feature importance from ensemble
        self._calculate_feature_importance(data)
        
        self.is_trained = True
        self.logger.info("All predictive models trained successfully!")
        
        return self
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features"""
        data = data.copy()
        
        # Add common ratios
        data['rooms_per_household'] = data['total_rooms'] / data['households']
        data['population_per_household'] = data['population'] / data['households']
        data['bedrooms_ratio'] = data['total_bedrooms'] / data['total_rooms']
        
        # Add geographical features
        data['avg_income'] = data['median_income'] * data['population']
        
        # Add derived features
        data['rooms_per_capita'] = data['total_rooms'] / data['population']
        data['income_density'] = data['median_income'] / data['population']
        data['bedroom_per_household'] = data['total_bedrooms'] / data['households']
        
        # Handle infinity values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.mean())
        
        return data
    
    def _calculate_feature_importance(self, data: pd.DataFrame) -> None:
        """Calculate ensemble feature importance"""
        importances = {}
        model_counts = len([m for m in self.models.values() if hasattr(m, 'weights') or hasattr(m, 'trees')])
        
        # Aggregate from individual models where available
        for name, model in self.models.items():
            if hasattr(model, 'weights') and model.weights is not None:
                abs_weights = np.abs(model.weights)
                weights_sum = np.sum(abs_weights) if np.sum(abs_weights) > 0 else 1
                model_importances = abs_weights / weights_sum
                for i, feature in enumerate(self.feature_names):
                    if feature not in importances:
                        importances[feature] = 0
                    importances[feature] += model_importances[i] / model_counts
                    
        # Random forest style importances
        for name, model in self.models.items():
            if hasattr(model, 'trees'):
                # Extract tree-level feature splits and weight them by prediction variation
                continue
                
        # Simple weight-based for this simplified version
        if len([m for m in self.models.values() if hasattr(m, 'weights') and m.weights is not None]) > 0:
            avg_weights = np.zeros(len(self.feature_names))
            valid_count = 0
            for name, model in self.models.items():
                if hasattr(model, 'weights') and model.weights is not None:
                    abs_weights = np.abs(model.weights)
                    weight_sum = np.sum(abs_weights) if np.sum(abs_weights) > 0 else 1
                    weights_norm = abs_weights / weight_sum
                    avg_weights = np.mean(np.stack((weights_norm, model_importances[name].copy() if abs(avg_weights[:, None]).sum() > 0 else weights_norm)), axis=0)
                    valid_count += 1
            
            if valid_count > 0:
                avg_weights = avg_weights / valid_count
                for i, feature in enumerate(self.feature_names):
                    importances[feature] = avg_weights[i]
        
        self.feature_importance = importances
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get predictions from all models"""
        if not self.is_trained:
            raise RuntimeError("Models not trained. Call fit() first.")
        
        # Convert features to array
        feature_array = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(feature_array)[0]
            predictions[name] = float(pred)
        
        # Add ensemble prediction
        predictions['ensemble'] = float(np.mean(list(predictions.values())))
        
        return predictions
    
    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all models"""
        metrics = {}
        for name, model in self.models.items():
            metrics[name] = model.get_metrics()
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
    
    def get_overall_accuracy(self) -> float:
        """Get overall system accuracy (average R²)"""
        if not self.is_trained:
            return 0.0
        
        r2_scores = [model.get_metrics().get('test_r2', 0) for model in self.models.values()]
        return float(np.mean(r2_scores)) if r2_scores else 0.0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the layer"""
        return {
            'is_trained': self.is_trained,
            'models_trained': len(self.models),
            'overall_accuracy': self.get_overall_accuracy(),
            'feature_names': self.feature_names.copy()
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize layer state"""
        return {
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_metrics': self.get_model_metrics()
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize layer state"""
        self.config = data.get('config', {})
        self.is_trained = data.get('is_trained', False)
        self.feature_names = data.get('feature_names', [])
        self.feature_importance = data.get('feature_importance', {})