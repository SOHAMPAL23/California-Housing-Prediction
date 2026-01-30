"""Reinforcement Learning Decision Layer - Q-Learning for Optimal Pricing Strategy"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import defaultdict
import random
import json

class PricingEnvironment:
    """Environment for housing price optimization"""
    
    def __init__(self, data: pd.DataFrame, predictive_model: Any):
        self.data = data
        self.predictive_model = predictive_model
        self.feature_names = predictive_model.feature_names
        self.current_step = 0
        self.max_steps = len(data)
        self.state_history = []
        self.action_history = []
        self.reward_history = []
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state"""
        self.current_step = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        return self._get_current_state()
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return {}
        
        # Get current house features
        row = self.data.iloc[self.current_step]
        state = {}
        
        # Basic features
        for feature in self.feature_names:
            if feature in row:
                state[feature] = float(row[feature])
        
        # Add market context features
        if self.current_step > 0:
            recent_prices = [self.data.iloc[max(0, self.current_step - i)]['median_house_value'] 
                           for i in range(1, min(6, self.current_step + 1))]
            state['recent_avg_price'] = float(np.mean(recent_prices))
            state['recent_price_trend'] = float(np.mean(np.diff(recent_prices)) if len(recent_prices) > 1 else 0)
        else:
            state['recent_avg_price'] = float(row['median_house_value'])
            state['recent_price_trend'] = 0.0
        
        # Add time-based features
        state['time_step'] = self.current_step / self.max_steps
        state['market_phase'] = self._get_market_phase()
        
        return state
    
    def _get_market_phase(self) -> str:
        """Determine current market phase"""
        if self.current_step < self.max_steps * 0.3:
            return 'early'
        elif self.current_step < self.max_steps * 0.7:
            return 'mid'
        else:
            return 'late'
    
    def step(self, action: float) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take action and return (next_state, reward, done, info)"""
        if self.current_step >= len(self.data):
            return {}, 0, True, {}
        
        # Get current state and true price
        current_state = self._get_current_state()
        true_price = self.data.iloc[self.current_step]['median_house_value']
        
        # Calculate adjusted price based on action
        adjusted_price = true_price * (1 + action)  # action is percentage adjustment
        
        # Calculate reward based on pricing strategy
        reward = self._calculate_reward(true_price, adjusted_price, current_state)
        
        # Store history
        self.state_history.append(current_state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = self._get_current_state() if not done else {}
        
        info = {
            'true_price': float(true_price),
            'adjusted_price': float(adjusted_price),
            'action': float(action)
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, true_price: float, adjusted_price: float, 
                         state: Dict[str, Any]) -> float:
        """Calculate reward for pricing decision"""
        # Base reward: negative of price deviation (want to be close to true price)
        price_deviation = abs(adjusted_price - true_price) / true_price
        price_reward = -price_deviation * 10  # Scale the reward
        
        # Profit reward: positive for higher prices (up to a point)
        profit_margin = (adjusted_price - true_price) / true_price
        profit_reward = max(0, min(profit_margin * 5, 2))  # Cap profit reward
        
        # Market timing reward: consider market phase
        market_phase = state.get('market_phase', 'mid')
        timing_bonus = 0
        if market_phase == 'early' and profit_margin > 0:
            timing_bonus = 0.5
        elif market_phase == 'late' and profit_margin < 0:
            timing_bonus = 0.3  # Less penalty for conservative pricing in late market
        
        # Trend following reward: align with recent trends
        trend = state.get('recent_price_trend', 0)
        trend_alignment = 0
        if (trend > 0 and profit_margin > 0) or (trend < 0 and profit_margin < 0):
            trend_alignment = 0.2
        
        total_reward = price_reward + profit_reward + timing_bonus + trend_alignment
        return float(total_reward)

class QLearningAgent:
    """Q-Learning agent for pricing optimization"""
    
    def __init__(self, state_features: List[str], config: Dict[str, Any] = None):
        self.config = config or {}
        self.state_features = state_features
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        self.exploration_decay = self.config.get('exploration_decay', 0.995)
        self.min_exploration = self.config.get('min_exploration', 0.01)
        
        # Discretize action space: percentage adjustments (-20% to +20% in 5% increments)
        self.actions = [i * 0.05 for i in range(-4, 5)]  # [-20%, -15%, ..., +20%]
        
        # Q-table: state_hash -> action_index -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        
        # State discretization parameters
        self.state_bins = self._create_state_bins()
    
    def _create_state_bins(self) -> Dict[str, np.ndarray]:
        """Create bins for state discretization"""
        bins = {}
        # Create bins for numerical features (simplified approach)
        for feature in self.state_features:
            if 'price' in feature.lower():
                bins[feature] = np.array([0, 1, 2, 3, 4, 5])  # Price ranges
            elif 'trend' in feature.lower():
                bins[feature] = np.array([-1, -0.5, 0, 0.5, 1])  # Trend ranges
            elif 'time' in feature.lower():
                bins[feature] = np.array([0, 0.33, 0.67, 1])  # Time phases
            else:
                bins[feature] = np.array([0, 1, 2, 3])  # General ranges
        return bins
    
    def _discretize_state(self, state: Dict[str, Any]) -> Tuple:
        """Convert continuous state to discrete representation"""
        discretized = []
        for feature in self.state_features:
            if feature in state:
                value = state[feature]
                bins = self.state_bins.get(feature, np.array([0, 1, 2, 3]))
                # Find which bin the value falls into
                bin_idx = np.digitize(value, bins) - 1
                bin_idx = max(0, min(bin_idx, len(bins) - 1))
                discretized.append(bin_idx)
            else:
                discretized.append(0)
        return tuple(discretized)
    
    def get_action(self, state: Dict[str, Any]) -> int:
        """Get action index using epsilon-greedy policy"""
        state_hash = self._discretize_state(state)
        
        if random.random() < self.exploration_rate:
            # Explore: random action
            return random.randint(0, len(self.actions) - 1)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state_hash])
    
    def update_q_value(self, state: Dict[str, Any], action: int, 
                      reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """Update Q-value using Q-learning update rule"""
        state_hash = self._discretize_state(state)
        next_state_hash = self._discretize_state(next_state)
        
        current_q = self.q_table[state_hash][action]
        
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_hash])
            target_q = reward + self.discount_factor * next_max_q
        
        # Q-learning update
        self.q_table[state_hash][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_exploration(self) -> None:
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration, 
                                  self.exploration_rate * self.exploration_decay)

class RLDecisionLayer:
    """Reinforcement Learning Decision Layer for pricing optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent = None
        self.environment = None
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.training_history = []
        self.policy_performance = {}
    
    def fit(self, data: pd.DataFrame, predictive_layer: Any) -> 'RLDecisionLayer':
        """Train RL agent for pricing optimization"""
        self.logger.info("Training RL pricing agent...")
        
        # Create environment
        self.environment = PricingEnvironment(data, predictive_layer)
        
        # Create agent
        feature_names = predictive_layer.feature_names + ['recent_avg_price', 'recent_price_trend', 'time_step']
        self.agent = QLearningAgent(feature_names, self.config)
        
        # Training parameters
        episodes = self.config.get('episodes', 1000)
        max_steps_per_episode = min(100, len(data))  # Limit steps for efficiency
        
        # Training loop
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps_per_episode:
                # Get action from agent
                action_idx = self.agent.get_action(state)
                action = self.agent.actions[action_idx]
                
                # Take action in environment
                next_state, reward, done, info = self.environment.step(action)
                
                # Update Q-value
                self.agent.update_q_value(state, action_idx, reward, next_state, done)
                
                # Update state and tracking
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Decay exploration
            self.agent.decay_exploration()
            
            # Log training progress
            if episode % 100 == 0:
                avg_reward = total_reward / steps if steps > 0 else 0
                self.logger.info(f"Episode {episode}: Average reward = {avg_reward:.4f}, "
                               f"Epsilon = {self.agent.exploration_rate:.4f}")
            
            # Store training history
            self.training_history.append({
                'episode': episode,
                'total_reward': total_reward,
                'steps': steps,
                'avg_reward': total_reward / steps if steps > 0 else 0,
                'epsilon': self.agent.exploration_rate
            })
        
        # Evaluate final policy
        self._evaluate_policy()
        
        self.is_trained = True
        self.logger.info("RL pricing agent trained successfully!")
        
        return self
    
    def _evaluate_policy(self) -> None:
        """Evaluate the learned policy"""
        if not self.agent or not self.environment:
            return
        
        self.logger.info("Evaluating learned policy...")
        
        # Run evaluation episodes
        eval_episodes = 100
        total_rewards = []
        price_accuracies = []
        profit_margins = []
        
        for episode in range(eval_episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_prices = []
            episode_actions = []
            
            while True:
                # Use greedy policy (no exploration)
                action_idx = np.argmax(self.agent.q_table[self.agent._discretize_state(state)])
                action = self.agent.actions[action_idx]
                
                next_state, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                if 'true_price' in info and 'adjusted_price' in info:
                    episode_prices.append(info['true_price'])
                    episode_actions.append(action)
                
                state = next_state
                if done:
                    break
            
            total_rewards.append(episode_reward)
            
            # Calculate price accuracy and profit metrics
            if episode_prices and episode_actions:
                price_errors = [abs((true - (true * (1 + act))) / true) 
                              for true, act in zip(episode_prices, episode_actions)]
                avg_price_error = np.mean(price_errors)
                price_accuracies.append(1 - avg_price_error)
                
                profit_margins.extend([act for act in episode_actions])
        
        # Store evaluation results
        self.policy_performance = {
            'avg_reward': float(np.mean(total_rewards)),
            'reward_std': float(np.std(total_rewards)),
            'price_accuracy': float(np.mean(price_accuracies)) if price_accuracies else 0,
            'avg_profit_margin': float(np.mean(profit_margins)) if profit_margins else 0,
            'profit_margin_std': float(np.std(profit_margins)) if profit_margins else 0
        }
        
        self.logger.info(f"Policy evaluation complete: "
                        f"Avg reward = {self.policy_performance['avg_reward']:.4f}, "
                        f"Price accuracy = {self.policy_performance['price_accuracy']:.4f}")
    
    def get_pricing_strategy(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal pricing strategy for given features"""
        if not self.is_trained:
            raise RuntimeError("RL agent not trained. Call fit() first.")
        
        # Create state representation
        state = features.copy()
        
        # Add market context (simplified)
        state['recent_avg_price'] = features.get('median_house_value', 0)
        state['recent_price_trend'] = 0  # Default assumption
        state['time_step'] = 0.5  # Mid-market assumption
        
        # Get optimal action
        action_idx = np.argmax(self.agent.q_table[self.agent._discretize_state(state)])
        optimal_action = self.agent.actions[action_idx]
        
        # Calculate recommended price
        base_price = features.get('median_house_value', 0)
        recommended_price = base_price * (1 + optimal_action)
        
        # Get action probabilities (for uncertainty quantification)
        state_hash = self.agent._discretize_state(state)
        q_values = self.agent.q_table[state_hash]
        action_probs = np.exp(q_values) / np.sum(np.exp(q_values))
        
        strategy = {
            'base_price': float(base_price),
            'recommended_price': float(recommended_price),
            'price_adjustment': float(optimal_action),
            'price_adjustment_percentage': float(optimal_action * 100),
            'confidence': float(np.max(action_probs)),
            'alternative_strategies': self._get_alternative_strategies(state, q_values),
            'market_context': {
                'phase': 'mid',  # Simplified
                'trend': 'neutral'  # Simplified
            }
        }
        
        return strategy
    
    def _get_alternative_strategies(self, state: Dict[str, Any], 
                                  q_values: np.ndarray) -> List[Dict[str, Any]]:
        """Get alternative pricing strategies with their Q-values"""
        alternatives = []
        state_hash = self.agent._discretize_state(state)
        
        # Get top 3 actions
        top_actions = np.argsort(q_values)[-3:][::-1]
        
        for i, action_idx in enumerate(top_actions):
            action = self.agent.actions[action_idx]
            q_value = q_values[action_idx]
            
            alternatives.append({
                'rank': i + 1,
                'price_adjustment': float(action),
                'price_adjustment_percentage': float(action * 100),
                'q_value': float(q_value),
                'expected_reward': float(q_value)
            })
        
        return alternatives
    
    def get_policy_convergence(self) -> float:
        """Get policy convergence metric"""
        if not self.training_history:
            return 0.0
        
        # Simple convergence metric: stability of recent rewards
        recent_rewards = [ep['avg_reward'] for ep in self.training_history[-50:]]
        if len(recent_rewards) < 2:
            return 0.0
        
        reward_variance = np.var(recent_rewards)
        max_variance = 10.0  # Arbitrary maximum
        convergence = max(0.0, 1.0 - (reward_variance / max_variance))
        
        return float(convergence)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the layer"""
        return {
            'is_trained': self.is_trained,
            'episodes_trained': len(self.training_history),
            'policy_performance': self.policy_performance,
            'policy_convergence': self.get_policy_convergence()
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize layer state"""
        return {
            'config': self.config,
            'is_trained': self.is_trained,
            'policy_performance': self.policy_performance,
            'training_history_length': len(self.training_history)
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize layer state"""
        self.config = data.get('config', {})
        self.is_trained = data.get('is_trained', False)
        self.policy_performance = data.get('policy_performance', {})