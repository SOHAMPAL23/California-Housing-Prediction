"""Generative Explanation Layer - LLM-based Narrative Generation for Human-Readable Insights"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime
import re

class ExplanationGenerator:
    """Simple explanation generator for housing price predictions"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates"""
        return {
            'prediction_summary': """
            Based on the {model_count} models analyzed, the predicted house price is ${price:,.2f}. 
            This prediction takes into account {feature_count} key features including {top_features}.
            """,
            
            'model_comparison': """
            Among the models evaluated:
            - Linear Regression shows {lr_r2:.3f} accuracy
            - Lasso Regression achieves {lasso_r2:.3f} accuracy with {sparsity:.1f}% feature sparsity
            - Random Forest demonstrates {rf_r2:.3f} accuracy through ensemble learning
            - Deep Neural Network provides {nn_r2:.3f} accuracy with complex pattern recognition
            """,
            
            'feature_analysis': """
            The most influential features for this prediction are:
            {feature_list}
            These features collectively explain {importance_sum:.1f}% of the price variation.
            """,
            
            'causal_insight': """
            Causal analysis reveals that {feature} has a {direction} impact on price, 
            with a counterfactual effect of {impact_value:+.2f} ({impact_percent:+.1f}%).
            """,
            
            'pricing_strategy': """
            The recommended pricing strategy suggests a {adjustment_direction} adjustment of {adjustment_percent:+.1f}%.
            This would set the price at ${recommended_price:,.2f}, with an expected confidence of {confidence:.1f}%.
            """,
            
            'risk_assessment': """
            Key considerations for this prediction:
            - Model uncertainty: {uncertainty_level}
            - Market conditions: {market_conditions}
            - Feature reliability: {feature_reliability}
            """
        }
    
    def generate_prediction_explanation(self, predictions: Dict[str, float], 
                                      feature_importance: Dict[str, float],
                                      model_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate explanation for prediction results"""
        # Calculate ensemble statistics
        ensemble_pred = predictions.get('ensemble', np.mean(list(predictions.values())))
        model_count = len([k for k in predictions.keys() if k != 'ensemble'])
        
        # Get top features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        top_feature_names = [f"{name} ({importance:.3f})" for name, importance in top_features]
        
        # Generate model comparison
        model_comparison = ""
        if model_metrics:
            lr_r2 = model_metrics.get('linear_regression', {}).get('test_r2', 0)
            lasso_r2 = model_metrics.get('lasso', {}).get('test_r2', 0)
            lasso_sparsity = model_metrics.get('lasso', {}).get('sparsity', 0) * 100
            rf_r2 = model_metrics.get('random_forest', {}).get('test_r2', 0)
            nn_r2 = model_metrics.get('deep_neural_network', {}).get('test_r2', 0)
            
            model_comparison = self.templates['model_comparison'].format(
                lr_r2=lr_r2, lasso_r2=lasso_r2, sparsity=lasso_sparsity,
                rf_r2=rf_r2, nn_r2=nn_r2
            )
        
        # Generate feature analysis
        importance_sum = min(100, sum(list(feature_importance.values())[:5]) * 100)
        feature_list = "\n".join([f"• {name}: {importance:.3f}" for name, importance in top_features])
        
        feature_analysis = self.templates['feature_analysis'].format(
            feature_list=feature_list, importance_sum=importance_sum
        )
        
        # Combine explanations
        explanation = self.templates['prediction_summary'].format(
            model_count=model_count,
            price=ensemble_pred * 100000,  # Convert back to actual dollars
            feature_count=len(feature_importance),
            top_features=", ".join([name for name, _ in top_features[:2]])
        )
        
        explanation += model_comparison
        explanation += feature_analysis
        
        return explanation.strip()
    
    def generate_causal_explanation(self, causal_analysis: Dict[str, Any]) -> str:
        """Generate explanation for causal analysis"""
        if not causal_analysis.get('causal_impacts'):
            return "No causal relationships detected."
        
        # Get top causal drivers
        top_drivers = causal_analysis.get('top_drivers', [])[:3]
        
        causal_insights = []
        for driver in top_drivers:
            feature = driver['feature']
            impact = driver['causal_impact']
            rel_impact = driver['relative_impact']
            direction = "positive" if impact > 0 else "negative"
            
            insight = self.templates['causal_insight'].format(
                feature=feature,
                direction=direction,
                impact_value=impact,
                impact_percent=rel_impact
            )
            causal_insights.append(insight)
        
        return "".join(causal_insights).strip()
    
    def generate_pricing_strategy_explanation(self, strategy: Dict[str, Any]) -> str:
        """Generate explanation for pricing strategy"""
        adjustment = strategy.get('price_adjustment_percentage', 0)
        direction = "positive" if adjustment > 0 else "negative" if adjustment < 0 else "no"
        
        explanation = self.templates['pricing_strategy'].format(
            adjustment_direction=direction,
            adjustment_percent=adjustment,
            recommended_price=strategy.get('recommended_price', 0) * 100000,
            confidence=strategy.get('confidence', 0) * 100
        )
        
        return explanation
    
    def generate_scenario_explanation(self, scenario_results: Dict[str, Any]) -> str:
        """Generate explanation for counterfactual scenario"""
        baseline = scenario_results.get('baseline_prediction', {}).get('ensemble', 0)
        scenario = scenario_results.get('scenario_prediction', {}).get('ensemble', 0)
        impact = scenario_results.get('price_impact', 0)
        percentage = scenario_results.get('percentage_change', 0)
        
        direction = "increase" if impact > 0 else "decrease" if impact < 0 else "no change"
        
        explanation = f"""
        Under the given scenario conditions:
        - Baseline predicted price: ${baseline * 100000:,.2f}
        - Scenario predicted price: ${scenario * 100000:,.2f}
        - Absolute price change: {direction} of ${abs(impact) * 100000:,.2f}
        - Percentage change: {percentage:+.1f}%
        
        The main factors driving this change were:
        """
        
        # Add key driver insights
        key_drivers = scenario_results.get('causal_explanation', {}).get('key_drivers', [])[:3]
        for i, driver in enumerate(key_drivers):
            feature = driver['feature']
            rel_change = driver['change_info'].get('relative_change', 0)
            explanation += f"\n{feature}: {rel_change:+.1f}% change"
        
        return explanation.strip()
    
    def generate_question_answer(self, question: str, context: Dict[str, Any]) -> str:
        """Generate answer to natural language question"""
        # Extract relevant information from context
        summary = context.get('summary', '')
        insights = context.get('insights', [])
        
        # Simple question classification and response generation
        question_lower = question.lower()
        
        if 'price' in question_lower and 'predict' in question_lower:
            return f"Based on our analysis, housing prices are predicted using multiple models that consider factors like income, location, and housing characteristics. {summary}"
        
        elif 'feature' in question_lower and 'important' in question_lower:
            important_features = [insight for insight in insights if insight['type'] == 'feature_context']
            if important_features:
                feature_info = important_features[0]
                return f"Key features affecting house prices include {feature_info.get('feature', 'various factors')}. {summary}"
        
        elif 'model' in question_lower and 'compare' in question_lower:
            model_info = [insight for insight in insights if insight['type'] == 'model_context']
            if model_info:
                return f"Different models show varying performance: {summary}"
        
        elif 'causal' in question_lower or 'why' in question_lower:
            causal_info = [insight for insight in insights if insight['type'] == 'domain_context']
            if causal_info:
                return f"Causal relationships in housing prices are influenced by: {summary}"
        
        else:
            # Default response
            return f"Based on the available information: {summary}"

class ExplanationLayer:
    """Layer for generating human-readable explanations and insights"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.generator = ExplanationGenerator()
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_metrics = {}
        self.feature_importance = {}
    
    def fit(self, predictive_layer: Any) -> 'ExplanationLayer':
        """Initialize explanation layer with predictive models"""
        self.logger.info("Setting up explanation generation...")
        
        # Get model metrics and feature importance
        self.model_metrics = predictive_layer.get_model_metrics()
        self.feature_importance = predictive_layer.get_feature_importance()
        
        self.is_trained = True
        self.logger.info("Explanation layer ready!")
        
        return self
    
    def explain_prediction(self, features: Dict[str, Any], 
                          predictions: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive explanation for prediction"""
        if not self.is_trained:
            raise RuntimeError("Explanation layer not initialized. Call fit() first.")
        
        # Generate different types of explanations
        prediction_explanation = self.generator.generate_prediction_explanation(
            predictions, self.feature_importance, self.model_metrics
        )
        
        # Create structured explanation
        explanation = {
            'prediction_summary': prediction_explanation,
            'model_comparison': self._get_model_comparison_summary(),
            'feature_analysis': self._get_feature_analysis_summary(),
            'confidence_assessment': self._get_confidence_assessment(predictions),
            'recommendations': self._get_recommendations(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        return explanation
    
    def explain_causal_impact(self, baseline_features: Dict[str, Any],
                            scenario_features: Dict[str, Any],
                            causal_analysis: Dict[str, Any]) -> str:
        """Generate explanation for causal impact analysis"""
        if not self.is_trained:
            return "Causal analysis not available."
        
        return self.generator.generate_causal_explanation(causal_analysis)
    
    def explain_pricing_strategy(self, strategy: Dict[str, Any]) -> str:
        """Generate explanation for pricing strategy"""
        if not self.is_trained:
            return "Pricing strategy explanation not available."
        
        return self.generator.generate_pricing_strategy_explanation(strategy)
    
    def explain_scenario(self, scenario_results: Dict[str, Any]) -> str:
        """Generate explanation for counterfactual scenario"""
        if not self.is_trained:
            return "Scenario analysis not available."
        
        return self.generator.generate_scenario_explanation(scenario_results)
    
    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """Answer natural language question using RAG context"""
        return self.generator.generate_question_answer(question, context)
    
    def _get_model_comparison_summary(self) -> str:
        """Generate model comparison summary"""
        if not self.model_metrics:
            return "No model performance data available."
        
        summary = "Model Performance Comparison:\n"
        for model_name, metrics in self.model_metrics.items():
            r2 = metrics.get('test_r2', 0)
            rmse = metrics.get('test_rmse', 0)
            summary += f"• {model_name.replace('_', ' ').title()}: R² = {r2:.3f}, RMSE = {rmse:.4f}\n"
        
        return summary.strip()
    
    def _get_feature_analysis_summary(self) -> str:
        """Generate feature importance summary"""
        if not self.feature_importance:
            return "No feature importance data available."
        
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        summary = "Top Feature Importances:\n"
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            summary += f"{i+1}. {feature}: {importance:.4f}\n"
        
        return summary.strip()
    
    def _get_confidence_assessment(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Assess prediction confidence"""
        # Simple confidence calculation based on model agreement
        model_predictions = [v for k, v in predictions.items() if k != 'ensemble']
        if len(model_predictions) < 2:
            return {'level': 'low', 'score': 0.5, 'explanation': 'Insufficient models for confidence assessment'}
        
        # Calculate variance among model predictions
        pred_variance = np.var(model_predictions)
        pred_std = np.std(model_predictions)
        pred_range = max(model_predictions) - min(model_predictions)
        
        # Convert to confidence score (0-1)
        max_expected_range = 1.0  # Arbitrary maximum expected range
        confidence_score = max(0, 1 - (pred_range / max_expected_range))
        
        if confidence_score > 0.8:
            confidence_level = 'high'
            explanation = 'Strong model agreement indicates high confidence'
        elif confidence_score > 0.6:
            confidence_level = 'medium'
            explanation = 'Moderate model agreement suggests reasonable confidence'
        else:
            confidence_level = 'low'
            explanation = 'Model disagreement indicates uncertainty in prediction'
        
        return {
            'level': confidence_level,
            'score': float(confidence_score),
            'explanation': explanation,
            'model_variance': float(pred_variance),
            'model_std': float(pred_std),
            'model_range': float(pred_range)
        }
    
    def _get_recommendations(self, predictions: Dict[str, float]) -> List[str]:
        """Generate recommendations based on prediction"""
        ensemble_pred = predictions.get('ensemble', 0)
        recommendations = []
        
        # Price-based recommendations
        if ensemble_pred < 2.0:  # Below $200k
            recommendations.append("Consider market conditions in lower-price segments")
        elif ensemble_pred > 4.0:  # Above $400k
            recommendations.append("High-value properties may require additional market analysis")
        
        # Model-based recommendations
        if len([k for k in predictions.keys() if k != 'ensemble']) >= 3:
            recommendations.append("Multiple model consensus provides robust prediction")
        else:
            recommendations.append("Consider additional modeling approaches for validation")
        
        # Feature-based recommendations
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:2]
            for feature, importance in top_features:
                if importance > 0.3:
                    recommendations.append(f"Pay special attention to {feature} as it strongly influences price")
        
        return recommendations
    
    def get_explanation_quality(self) -> float:
        """Get quality score for explanation generation"""
        if not self.is_trained:
            return 0.0
        
        # Simple quality score based on available information
        quality_components = [
            len(self.model_metrics) > 0,
            len(self.feature_importance) > 0,
            self.generator is not None
        ]
        
        return sum(quality_components) / len(quality_components)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the layer"""
        return {
            'is_trained': self.is_trained,
            'model_metrics_available': len(self.model_metrics) > 0,
            'feature_importance_available': len(self.feature_importance) > 0,
            'explanation_quality': self.get_explanation_quality()
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize layer state"""
        return {
            'config': self.config,
            'is_trained': self.is_trained,
            'model_metrics_count': len(self.model_metrics),
            'feature_importance_count': len(self.feature_importance),
            'explanation_quality': self.get_explanation_quality()
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize layer state"""
        self.config = data.get('config', {})
        self.is_trained = data.get('is_trained', False)