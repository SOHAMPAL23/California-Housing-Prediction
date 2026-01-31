"""Main Architecture Class for GenAI Housing Price Intelligence System"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime

# Import all components
from core.base_model import BaseModel
from components.predictive_modeling import PredictiveModelingLayer
from components.causal_reasoning import CausalReasoningLayer
from components.reinforcement_learning import RLDecisionLayer
from components.rag_system import RAGLayer
from components.explanation_layer import ExplanationLayer

class Architecture:
    """Main system architecture that orchestrates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.layers = self._initialize_layers()
        self.is_initialized = False
        self.system_metrics = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'system': {
                'name': 'GenAI_Housing_Intelligence',
                'version': '1.0.0',
                'description': 'Enterprise-grade GenAI housing price intelligence system'
            },
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'validation_size': 0.1
            },
            'models': {
                'predictive': {
                    'linear_regression': True,
                    'lasso': True,
                    'random_forest': True,
                    'deep_learning': True
                },
                'rl': {
                    'learning_rate': 0.1,
                    'discount_factor': 0.95,
                    'exploration_rate': 0.1,
                    'episodes': 1000
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def _initialize_layers(self) -> Dict[str, Any]:
        """Initialize all system layers"""
        return {
            'predictive_modeling': PredictiveModelingLayer(self.config.get('models', {}).get('predictive', {})),
            'causal_reasoning': CausalReasoningLayer(),
            'rl_decision': RLDecisionLayer(self.config.get('models', {}).get('rl', {})),
            'rag_system': RAGLayer(),
            'explanation_layer': ExplanationLayer()
        }
    
    def initialize_system(self, data: pd.DataFrame) -> None:
        """Initialize the entire system with data"""
        self.logger.info("Initializing GenAI Housing Intelligence System...")
        
        # Initialize predictive modeling layer
        self.logger.info("Training predictive models...")
        self.layers['predictive_modeling'].fit(data)
        
        # Initialize causal reasoning layer
        self.logger.info("Building causal models...")
        self.layers['causal_reasoning'].fit(data, self.layers['predictive_modeling'])
        
        # Initialize RL decision layer
        self.logger.info("Training RL pricing agents...")
        self.layers['rl_decision'].fit(data, self.layers['predictive_modeling'])
        
        # Initialize RAG system
        self.logger.info("Building knowledge base...")
        self.layers['rag_system'].fit(data, self.layers['predictive_modeling'], self.layers['causal_reasoning'])
        
        # Initialize explanation layer
        self.logger.info("Setting up explanation generation...")
        self.layers['explanation_layer'].fit(self.layers['predictive_modeling'])
        
        self.is_initialized = True
        self.logger.info("System initialization completed successfully!")
        
        # Log system metrics
        self._update_system_metrics()
    
    def predict_price(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict housing price using all models"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        self.logger.info("Generating price prediction...")
        
        # Get predictions from all models
        predictions = self.layers['predictive_modeling'].predict(features)
        
        # Generate explanations
        explanations = self.layers['explanation_layer'].explain_prediction(features, predictions)
        
        # Get causal analysis
        causal_analysis = self.layers['causal_reasoning'].analyze_causal_impact(features)
        
        # Get RL-based pricing recommendation
        rl_recommendation = self.layers['rl_decision'].get_pricing_strategy(features)
        
        # Get RAG-based context
        rag_context = self.layers['rag_system'].retrieve_context(features)
        
        result = {
            'predictions': predictions,
            'explanations': explanations,
            'causal_analysis': causal_analysis,
            'rl_recommendation': rl_recommendation,
            'rag_context': rag_context,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def simulate_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate counterfactual scenarios"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        self.logger.info("Running counterfactual simulation...")
        
        # Get baseline prediction
        baseline_features = scenario.get('baseline_features', {})
        baseline_prediction = self.layers['predictive_modeling'].predict(baseline_features)
        
        # Get scenario prediction
        scenario_features = scenario.get('scenario_features', {})
        scenario_prediction = self.layers['predictive_modeling'].predict(scenario_features)
        
        # Calculate impact
        price_impact = scenario_prediction['ensemble'] - baseline_prediction['ensemble']
        percentage_change = (price_impact / baseline_prediction['ensemble']) * 100
        
        # Get causal explanation
        causal_explanation = self.layers['causal_reasoning'].explain_causal_impact(
            baseline_features, scenario_features
        )
        
        result = {
            'baseline_prediction': baseline_prediction,
            'scenario_prediction': scenario_prediction,
            'price_impact': price_impact,
            'percentage_change': percentage_change,
            'causal_explanation': causal_explanation,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Answer natural language questions using RAG and GenAI"""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        self.logger.info(f"Processing question: {question}")
        
        # Retrieve relevant context using RAG
        context = self.layers['rag_system'].retrieve_context_from_query(question)
        
        # Generate answer using explanation layer
        answer = self.layers['explanation_layer'].answer_question(question, context)
        
        result = {
            'question': question,
            'answer': answer,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'healthy',
            'is_initialized': self.is_initialized,
            'system_metrics': self.system_metrics,
            'layer_health': {
                'predictive_modeling': self.layers['predictive_modeling'].get_health_status(),
                'causal_reasoning': self.layers['causal_reasoning'].get_health_status(),
                'rl_decision': self.layers['rl_decision'].get_health_status(),
                'rag_system': self.layers['rag_system'].get_health_status(),
                'explanation_layer': self.layers['explanation_layer'].get_health_status()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        self.system_metrics = {
            'total_models_trained': sum(1 for layer in self.layers.values() 
                                      if hasattr(layer, 'models') and layer.models),
            'prediction_accuracy': self.layers['predictive_modeling'].get_overall_accuracy(),
            'causal_model_quality': self.layers['causal_reasoning'].get_model_quality(),
            'rl_policy_convergence': self.layers['rl_decision'].get_policy_convergence(),
            'rag_knowledge_coverage': self.layers['rag_system'].get_knowledge_coverage(),
            'explanation_quality': self.layers['explanation_layer'].get_explanation_quality()
        }
    
    def save_system(self, filepath: str) -> None:
        """Save the entire system state"""
        system_state = {
            'config': self.config,
            'is_initialized': self.is_initialized,
            'system_metrics': self.system_metrics,
            'layers': {name: layer.serialize() for name, layer in self.layers.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        self.logger.info(f"System saved to {filepath}")
    
    def load_system(self, filepath: str) -> None:
        """Load system state from file"""
        with open(filepath, 'r') as f:
            system_state = json.load(f)
        
        self.config = system_state['config']
        self.is_initialized = system_state['is_initialized']
        self.system_metrics = system_state['system_metrics']
        
        # Load each layer
        for name, layer_data in system_state['layers'].items():
            if name in self.layers:
                self.layers[name].deserialize(layer_data)
        
        self.logger.info(f"System loaded from {filepath}")
    
    def __str__(self) -> str:
        return f"GenAI Housing Intelligence System (v{self.config['system']['version']})"
    
    def __repr__(self) -> str:
        return self.__str__()