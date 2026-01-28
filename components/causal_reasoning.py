"""Causal & Counterfactual Reasoning Layer - Structural Causal Modeling and What-if Analysis"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy.stats import pearsonr
import json

class CausalGraph:
    """Simple causal graph representation"""
    
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # node -> set of children
        self.reverse_edges = {}  # node -> set of parents
    
    def add_node(self, node: str) -> None:
        """Add a node to the graph"""
        self.nodes.add(node)
        if node not in self.edges:
            self.edges[node] = set()
        if node not in self.reverse_edges:
            self.reverse_edges[node] = set()
    
    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge from parent to child"""
        self.add_node(parent)
        self.add_node(child)
        self.edges[parent].add(child)
        self.reverse_edges[child].add(parent)
    
    def get_parents(self, node: str) -> set:
        """Get all parents of a node"""
        return self.reverse_edges.get(node, set())
    
    def get_children(self, node: str) -> set:
        """Get all children of a node"""
        return self.edges.get(node, set())
    
    def get_ancestors(self, node: str) -> set:
        """Get all ancestors of a node"""
        ancestors = set()
        queue = list(self.get_parents(node))
        while queue:
            parent = queue.pop(0)
            if parent not in ancestors:
                ancestors.add(parent)
                queue.extend(self.get_parents(parent))
        return ancestors
    
    def get_descendants(self, node: str) -> set:
        """Get all descendants of a node"""
        descendants = set()
        queue = list(self.get_children(node))
        while queue:
            child = queue.pop(0)
            if child not in descendants:
                descendants.add(child)
                queue.extend(self.get_children(child))
        return descendants

class CausalModel:
    """Simple structural causal model"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.graph = CausalGraph()
        self.conditional_models = {}  # node -> model for P(node | parents)
        self.is_trained = False
    
    def add_causal_relationship(self, cause: str, effect: str) -> None:
        """Add a causal relationship to the model"""
        self.graph.add_edge(cause, effect)
    
    def fit_conditional_model(self, node: str, data: pd.DataFrame) -> None:
        """Fit a conditional model for a node given its parents"""
        parents = self.graph.get_parents(node)
        if not parents:
            # Root node - fit marginal distribution
            self.conditional_models[node] = {
                'type': 'marginal',
                'mean': data[node].mean(),
                'std': data[node].std()
            }
        else:
            # Fit conditional model P(node | parents)
            # Simple linear regression for demonstration
            X = data[list(parents)].values
            y = data[node].values
            
            # Calculate coefficients using normal equation
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            try:
                XtX = np.dot(X_with_bias.T, X_with_bias)
                XtX_inv = np.linalg.inv(XtX)
                Xty = np.dot(X_with_bias.T, y)
                theta = np.dot(XtX_inv, Xty)
                
                self.conditional_models[node] = {
                    'type': 'linear',
                    'coefficients': theta[1:],
                    'bias': theta[0],
                    'parents': list(parents)
                }
            except np.linalg.LinAlgError:
                # Use mean if matrix is singular
                self.conditional_models[node] = {
                    'type': 'mean',
                    'mean': y.mean()
                }
    
    def predict(self, node: str, data: pd.DataFrame) -> np.ndarray:
        """Predict values for a node"""
        if node not in self.conditional_models:
            raise ValueError(f"No model for node {node}")
        
        model = self.conditional_models[node]
        if model['type'] == 'marginal':
            return np.full(len(data), model['mean'])
        elif model['type'] == 'mean':
            return np.full(len(data), model['mean'])
        elif model['type'] == 'linear':
            parents = model['parents']
            X = data[parents].values
            return np.dot(X, model['coefficients']) + model['bias']
        else:
            raise ValueError(f"Unknown model type: {model['type']}")
    
    def intervene(self, node: str, value: float) -> 'CausalModel':
        """Perform intervention: set node to value"""
        intervened_model = CausalModel(f"{self.name}_intervened")
        intervened_model.graph = self.graph
        intervened_model.conditional_models = self.conditional_models.copy()
        
        # Remove all incoming edges to the intervened node
        intervened_model.graph.reverse_edges[node] = set()
        for parent in self.graph.reverse_edges[node]:
            intervened_model.graph.edges[parent].remove(node)
        
        # Set the intervened node to a constant value
        intervened_model.conditional_models[node] = {
            'type': 'constant',
            'value': value
        }
        
        return intervened_model

class CausalReasoningLayer:
    """Layer for causal reasoning and counterfactual analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.causal_models = {}
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_names = []
        self.causal_graph = None
    
    def fit(self, data: pd.DataFrame, predictive_layer: Any) -> 'CausalReasoningLayer':
        """Fit causal models using the data and predictive models"""
        self.logger.info("Building causal models...")
        
        # Get feature names from predictive layer
        self.feature_names = predictive_layer.feature_names.copy()
        target_col = 'median_house_value'
        
        # Create causal graph based on domain knowledge and correlations
        self.causal_graph = self._build_causal_graph(data)
        
        # Build causal models for key relationships
        self._build_causal_models(data, predictive_layer)
        
        self.is_trained = True
        self.logger.info("Causal reasoning layer trained successfully!")
        
        return self
    
    def _build_causal_graph(self, data: pd.DataFrame) -> CausalGraph:
        """Build causal graph based on domain knowledge and data analysis"""
        graph = CausalGraph()
        
        # Add nodes
        for col in data.columns:
            graph.add_node(col)
        
        # Add edges based on domain knowledge and correlations
        # Income affects house prices
        graph.add_edge('median_income', 'median_house_value')
        
        # Location affects house prices
        graph.add_edge('longitude', 'median_house_value')
        graph.add_edge('latitude', 'median_house_value')
        
        # Housing characteristics affect prices
        graph.add_edge('housing_median_age', 'median_house_value')
        graph.add_edge('total_rooms', 'median_house_value')
        graph.add_edge('total_bedrooms', 'median_house_value')
        
        # Population characteristics affect prices
        graph.add_edge('population', 'median_house_value')
        graph.add_edge('households', 'median_house_value')
        
        # Derived features
        if 'rooms_per_household' in data.columns:
            graph.add_edge('total_rooms', 'rooms_per_household')
            graph.add_edge('households', 'rooms_per_household')
            graph.add_edge('rooms_per_household', 'median_house_value')
        
        if 'population_per_household' in data.columns:
            graph.add_edge('population', 'population_per_household')
            graph.add_edge('households', 'population_per_household')
            graph.add_edge('population_per_household', 'median_house_value')
        
        return graph
    
    def _build_causal_models(self, data: pd.DataFrame, predictive_layer: Any) -> None:
        """Build individual causal models"""
        # Build model for target variable
        target_model = CausalModel("house_price_causal")
        target_model.graph = self.causal_graph
        
        # Add causal relationships
        for parent in self.causal_graph.get_parents('median_house_value'):
            target_model.add_causal_relationship(parent, 'median_house_value')
        
        # Fit conditional models for key variables
        key_variables = ['median_house_value', 'median_income', 'total_rooms', 'population']
        for var in key_variables:
            if var in data.columns:
                target_model.fit_conditional_model(var, data)
        
        self.causal_models['target_causal'] = target_model
        
        # Build additional models for counterfactual analysis
        self._build_counterfactual_models(data)
    
    def _build_counterfactual_models(self, data: pd.DataFrame) -> None:
        """Build models for counterfactual reasoning"""
        # Income intervention model
        income_model = CausalModel("income_intervention")
        income_model.graph = self.causal_graph
        income_model.add_causal_relationship('median_income', 'median_house_value')
        income_model.fit_conditional_model('median_house_value', data)
        self.causal_models['income_intervention'] = income_model
        
        # Location intervention model
        location_model = CausalModel("location_intervention")
        location_model.graph = self.causal_graph
        location_model.add_causal_relationship('longitude', 'median_house_value')
        location_model.add_causal_relationship('latitude', 'median_house_value')
        location_model.fit_conditional_model('median_house_value', data)
        self.causal_models['location_intervention'] = location_model
    
    def analyze_causal_impact(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causal impact of features on house prices"""
        if not self.is_trained:
            raise RuntimeError("Causal models not trained. Call fit() first.")
        
        results = {}
        
        # Create feature dataframe
        feature_df = pd.DataFrame([features])
        
        # Get baseline prediction
        baseline_model = self.causal_models['target_causal']
        baseline_prediction = baseline_model.predict('median_house_value', feature_df)[0]
        
        # Analyze impact of each feature
        causal_impacts = {}
        for feature in self.feature_names:
            if feature in features:
                # Create intervention
                intervened_model = baseline_model.intervene(feature, features[feature])
                
                # Calculate counterfactual prediction
                try:
                    counterfactual = intervened_model.predict('median_house_value', feature_df)[0]
                    impact = counterfactual - baseline_prediction
                    causal_impacts[feature] = {
                        'baseline_value': features[feature],
                        'counterfactual_prediction': float(counterfactual),
                        'causal_impact': float(impact),
                        'relative_impact': float(impact / baseline_prediction * 100) if baseline_prediction != 0 else 0
                    }
                except:
                    causal_impacts[feature] = {
                        'baseline_value': features[feature],
                        'counterfactual_prediction': float(baseline_prediction),
                        'causal_impact': 0.0,
                        'relative_impact': 0.0
                    }
        
        results = {
            'baseline_prediction': float(baseline_prediction),
            'causal_impacts': causal_impacts,
            'top_drivers': self._get_top_causal_drivers(causal_impacts)
        }
        
        return results
    
    def explain_causal_impact(self, baseline_features: Dict[str, Any], 
                            scenario_features: Dict[str, Any]) -> Dict[str, Any]:
        """Explain the causal impact between two scenarios"""
        if not self.is_trained:
            raise RuntimeError("Causal models not trained. Call fit() first.")
        
        # Create dataframes
        baseline_df = pd.DataFrame([baseline_features])
        scenario_df = pd.DataFrame([scenario_features])
        
        # Get predictions
        baseline_model = self.causal_models['target_causal']
        baseline_pred = baseline_model.predict('median_house_value', baseline_df)[0]
        scenario_pred = baseline_model.predict('median_house_value', scenario_df)[0]
        
        # Calculate overall impact
        price_impact = scenario_pred - baseline_pred
        percentage_change = (price_impact / baseline_pred) * 100 if baseline_pred != 0 else 0
        
        # Analyze feature changes
        feature_changes = {}
        for feature in self.feature_names:
            baseline_val = baseline_features.get(feature, 0)
            scenario_val = scenario_features.get(feature, 0)
            change = scenario_val - baseline_val
            relative_change = (change / baseline_val * 100) if baseline_val != 0 else 0
            
            if abs(change) > 1e-6:  # Significant change
                feature_changes[feature] = {
                    'baseline_value': baseline_val,
                    'scenario_value': scenario_val,
                    'absolute_change': change,
                    'relative_change': relative_change
                }
        
        # Identify key drivers of change
        key_drivers = []
        for feature, change_info in feature_changes.items():
            # Estimate contribution based on feature importance and change magnitude
            importance = abs(change_info['relative_change'])  # Simplified importance
            key_drivers.append({
                'feature': feature,
                'importance': importance,
                'change_info': change_info
            })
        
        key_drivers.sort(key=lambda x: x['importance'], reverse=True)
        
        explanation = {
            'baseline_prediction': float(baseline_pred),
            'scenario_prediction': float(scenario_pred),
            'price_impact': float(price_impact),
            'percentage_change': float(percentage_change),
            'feature_changes': feature_changes,
            'key_drivers': key_drivers[:5],  # Top 5 drivers
            'causal_explanation': self._generate_causal_explanation(
                baseline_pred, scenario_pred, key_drivers[:3]
            )
        }
        
        return explanation
    
    def _get_top_causal_drivers(self, causal_impacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get top causal drivers based on impact magnitude"""
        drivers = []
        for feature, impact_info in causal_impacts.items():
            drivers.append({
                'feature': feature,
                'impact': abs(impact_info['causal_impact']),
                'relative_impact': abs(impact_info['relative_impact']),
                'direction': 'positive' if impact_info['causal_impact'] > 0 else 'negative'
            })
        
        drivers.sort(key=lambda x: x['impact'], reverse=True)
        return drivers[:5]
    
    def _generate_causal_explanation(self, baseline: float, scenario: float, 
                                   key_drivers: List[Dict[str, Any]]) -> str:
        """Generate natural language explanation of causal impact"""
        change = scenario - baseline
        percentage = (change / baseline) * 100 if baseline != 0 else 0
        
        explanation = f"The predicted house price changes from ${baseline:.2f} to ${scenario:.2f} "
        explanation += f"({percentage:+.1f}%). "
        
        if key_drivers:
            explanation += "The main drivers of this change are: "
            for i, driver in enumerate(key_drivers):
                feature = driver['feature']
                change_info = driver['change_info']
                rel_change = change_info['relative_change']
                explanation += f"{feature} ({rel_change:+.1f}%), "
            explanation = explanation.rstrip(', ') + "."
        
        return explanation
    
    def get_model_quality(self) -> float:
        """Get quality score for causal models"""
        if not self.is_trained:
            return 0.0
        
        # Simple quality score based on number of causal relationships
        relationships = len(self.causal_graph.edges) if self.causal_graph else 0
        return min(1.0, relationships / 20.0)  # Normalize to 0-1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the layer"""
        return {
            'is_trained': self.is_trained,
            'models_trained': len(self.causal_models),
            'causal_relationships': len(self.causal_graph.edges) if self.causal_graph else 0,
            'model_quality': self.get_model_quality()
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize layer state"""
        return {
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'model_quality': self.get_model_quality()
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize layer state"""
        self.config = data.get('config', {})
        self.is_trained = data.get('is_trained', False)
        self.feature_names = data.get('feature_names', [])