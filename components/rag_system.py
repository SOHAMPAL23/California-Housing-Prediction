"""RAG (Retrieval-Augmented Generation) Layer - Knowledge Base and Context Retrieval"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from collections import defaultdict
import re
from datetime import datetime

class SimpleEmbeddingModel:
    """Simple embedding model for demonstration purposes"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.word_vectors = {}
        self._initialize_vocab()
    
    def _initialize_vocab(self) -> None:
        """Initialize vocabulary with random vectors"""
        # Common words for housing domain
        common_words = [
            'house', 'price', 'value', 'income', 'location', 'rooms', 'bedrooms',
            'population', 'households', 'age', 'latitude', 'longitude', 'feature',
            'model', 'prediction', 'accuracy', 'importance', 'causal', 'effect',
            'impact', 'strategy', 'pricing', 'market', 'trend', 'analysis'
        ]
        
        for i, word in enumerate(common_words):
            self.vocab[word] = i
            self.word_vectors[word] = np.random.normal(0, 1, self.embedding_dim)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_embeddings = []
        
        for word in words:
            if word in self.word_vectors:
                word_embeddings.append(self.word_vectors[word])
            else:
                # Generate random embedding for unknown words
                word_embeddings.append(np.random.normal(0, 1, self.embedding_dim))
        
        if not word_embeddings:
            return np.zeros(self.embedding_dim)
        
        # Average word embeddings
        return np.mean(word_embeddings, axis=0)
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

class KnowledgeBase:
    """Knowledge base for storing and retrieving information"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.embedding_model = SimpleEmbeddingModel()
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a document to the knowledge base"""
        embedding = self.embedding_model.encode(content)
        
        self.documents.append(content)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add multiple documents to the knowledge base"""
        for doc in documents:
            self.add_document(doc['content'], doc.get('metadata', {}))
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.encode(query)
        similarities = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.embedding_model.similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, similarity in top_k:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': similarity
            })
        
        return results

class RAGLayer:
    """Retrieval-Augmented Generation Layer for contextual information retrieval"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.knowledge_base = KnowledgeBase()
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.system_insights = {}
    
    def fit(self, data: pd.DataFrame, predictive_layer: Any, causal_layer: Any) -> 'RAGLayer':
        """Build knowledge base from system components"""
        self.logger.info("Building RAG knowledge base...")
        
        # Add dataset statistics
        self._add_dataset_knowledge(data)
        
        # Add model performance insights
        self._add_model_knowledge(predictive_layer)
        
        # Add feature importance insights
        self._add_feature_knowledge(predictive_layer)
        
        # Add causal analysis insights
        self._add_causal_knowledge(causal_layer)
        
        # Add general housing market knowledge
        self._add_domain_knowledge()
        
        self.is_trained = True
        self.logger.info("RAG knowledge base built successfully!")
        
        return self
    
    def _add_dataset_knowledge(self, data: pd.DataFrame) -> None:
        """Add dataset statistics to knowledge base"""
        # Basic dataset information
        dataset_info = f"""
        Dataset contains {len(data)} housing records with {len(data.columns)} features.
        Target variable is median_house_value (house prices in hundreds of thousands).
        Features include: {', '.join(data.columns[:-1])}.
        """
        self.knowledge_base.add_document(dataset_info, {'type': 'dataset_info'})
        
        # Statistical summaries
        for column in data.columns:
            stats = data[column].describe()
            stat_summary = f"""
            Feature '{column}' statistics:
            Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}
            Min: {stats['min']:.2f}, Max: {stats['max']:.2f}
            25%: {stats['25%']:.2f}, 50%: {stats['50%']:.2f}, 75%: {stats['75%']:.2f}
            """
            self.knowledge_base.add_document(stat_summary, {'type': 'feature_stats', 'feature': column})
    
    def _add_model_knowledge(self, predictive_layer: Any) -> None:
        """Add model performance information"""
        metrics = predictive_layer.get_model_metrics()
        for model_name, model_metrics in metrics.items():
            performance_summary = f"""
            {model_name.replace('_', ' ').title()} Performance:
            Test RMSE: {model_metrics.get('test_rmse', 0):.4f}
            Test R² Score: {model_metrics.get('test_r2', 0):.4f}
            """
            self.knowledge_base.add_document(performance_summary, 
                                           {'type': 'model_performance', 'model': model_name})
    
    def _add_feature_knowledge(self, predictive_layer: Any) -> None:
        """Add feature importance information"""
        importance = predictive_layer.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        importance_summary = "Feature Importance Rankings:\n"
        for i, (feature, score) in enumerate(sorted_features[:10]):
            importance_summary += f"{i+1}. {feature}: {score:.4f}\n"
        
        self.knowledge_base.add_document(importance_summary, {'type': 'feature_importance'})
    
    def _add_causal_knowledge(self, causal_layer: Any) -> None:
        """Add causal analysis insights"""
        if hasattr(causal_layer, 'causal_graph') and causal_layer.causal_graph:
            graph = causal_layer.causal_graph
            causal_summary = "Causal Relationships Identified:\n"
            
            for node in graph.nodes:
                parents = graph.get_parents(node)
                if parents:
                    causal_summary += f"{node} is influenced by: {', '.join(parents)}\n"
            
            self.knowledge_base.add_document(causal_summary, {'type': 'causal_relationships'})
    
    def _add_domain_knowledge(self) -> None:
        """Add general housing market domain knowledge"""
        domain_knowledge = [
            {
                'content': """
                Housing prices are primarily driven by location, income levels, and housing characteristics.
                Median income is typically the strongest predictor of house prices across markets.
                Location features (latitude, longitude) capture geographical desirability and market conditions.
                Housing age can have complex effects - newer homes often command premiums, but very new construction may indicate developing areas.
                """,
                'metadata': {'type': 'domain_knowledge'}
            },
            {
                'content': """
                Feature engineering in housing data often involves creating ratios and derived metrics.
                Common transformations include: rooms per household, population density, income per capita.
                These derived features often capture economic relationships that raw features miss.
                Geographic clustering and distance calculations can reveal neighborhood effects.
                """,
                'metadata': {'type': 'feature_engineering'}
            },
            {
                'content': """
                Model interpretability is crucial in housing price prediction for regulatory and business reasons.
                Linear models provide coefficient-level interpretability but may miss non-linear relationships.
                Tree-based models capture feature interactions but offer less direct interpretation.
                Ensemble methods can balance accuracy and interpretability through feature importance rankings.
                """,
                'metadata': {'type': 'model_interpretability'}
            }
        ]
        
        self.knowledge_base.add_documents(domain_knowledge)
    
    def retrieve_context(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context for given features"""
        if not self.is_trained:
            return {'context': [], 'summary': 'System not initialized'}
        
        # Create query from features
        query_parts = []
        for feature, value in features.items():
            query_parts.append(f"{feature} is {value}")
        query = " ".join(query_parts)
        
        # Retrieve relevant documents
        results = self.knowledge_base.search(query, k=5)
        
        # Extract key insights
        insights = self._extract_insights(results, features)
        
        return {
            'query': query,
            'retrieved_documents': results,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def retrieve_context_from_query(self, query: str) -> Dict[str, Any]:
        """Retrieve context based on natural language query"""
        if not self.is_trained:
            return {'context': [], 'summary': 'System not initialized'}
        
        # Retrieve relevant documents
        results = self.knowledge_base.search(query, k=5)
        
        # Generate summary
        summary = self._generate_summary(results, query)
        
        return {
            'query': query,
            'retrieved_documents': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_insights(self, documents: List[Dict[str, Any]], 
                         features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key insights from retrieved documents"""
        insights = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            similarity = doc['similarity']
            
            # Extract relevant information based on document type
            insight_type = metadata.get('type', 'general')
            
            if insight_type == 'feature_stats':
                feature = metadata.get('feature', 'unknown')
                if feature in features:
                    actual_value = features[feature]
                    insights.append({
                        'type': 'feature_context',
                        'feature': feature,
                        'actual_value': actual_value,
                        'context': content,
                        'relevance': similarity
                    })
            
            elif insight_type == 'model_performance':
                insights.append({
                    'type': 'model_context',
                    'model': metadata.get('model', 'unknown'),
                    'context': content,
                    'relevance': similarity
                })
            
            elif insight_type in ['domain_knowledge', 'feature_engineering', 'model_interpretability']:
                insights.append({
                    'type': 'domain_context',
                    'category': insight_type,
                    'context': content,
                    'relevance': similarity
                })
        
        return insights
    
    def _generate_summary(self, documents: List[Dict[str, Any]], query: str) -> str:
        """Generate a summary from retrieved documents"""
        if not documents:
            return "No relevant information found for the query."
        
        # Simple summary generation
        summary_parts = []
        for doc in documents[:3]:  # Use top 3 most relevant
            content = doc['content'].strip()
            if len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(content)
        
        return " ".join(summary_parts)
    
    def get_knowledge_coverage(self) -> float:
        """Get knowledge base coverage score"""
        if not self.is_trained:
            return 0.0
        
        # Simple coverage metric based on number of documents
        doc_count = len(self.knowledge_base.documents)
        return min(1.0, doc_count / 50.0)  # Normalize to 0-1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the layer"""
        return {
            'is_trained': self.is_trained,
            'documents_stored': len(self.knowledge_base.documents),
            'knowledge_coverage': self.get_knowledge_coverage(),
            'embedding_dimension': self.knowledge_base.embedding_model.embedding_dim
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize layer state"""
        return {
            'config': self.config,
            'is_trained': self.is_trained,
            'documents_count': len(self.knowledge_base.documents),
            'knowledge_coverage': self.get_knowledge_coverage()
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize layer state"""
        self.config = data.get('config', {})
        self.is_trained = data.get('is_trained', False)