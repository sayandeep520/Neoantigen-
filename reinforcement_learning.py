"""
Reinforcement Learning Utilities for CRISPR & Neoantigen Optimization

This module provides reinforcement learning capabilities for optimizing:
1. CRISPR target site selection
2. Neoantigen candidate prioritization

The implementation uses Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO)
for training agents that can learn optimal selection strategies.
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Any, Optional, Union, TYPE_CHECKING
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple

# Try importing torch, handle case where it's not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes to make type checking work
    if TYPE_CHECKING:
        class nn:
            class Module:
                pass

# Define experience tuple for memory replay
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReinforcementLearningEnvironment:
    """Base RL environment class that can be adapted for different optimization tasks"""
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], reward_col: str, 
                 max_steps: int = 20, batch_size: int = 32):
        """
        Initialize the RL environment
        
        Args:
            data: DataFrame containing features and targets for optimization
            feature_cols: List of feature column names to use as state
            reward_col: Column name to use for rewards
            max_steps: Maximum number of steps per episode
            batch_size: Batch size for training
        """
        self.data = data
        self.feature_cols = feature_cols
        self.reward_col = reward_col
        self.max_steps = max_steps
        self.batch_size = batch_size
        
        self.state_dim = len(feature_cols)
        self.action_dim = len(data)  # Number of possible items to select
        
        self.current_step = 0
        self.selected_indices = []
        self.current_state = None
        
    def reset(self) -> np.ndarray:
        """Reset the environment for a new episode"""
        self.current_step = 0
        self.selected_indices = []
        self.current_state = np.zeros(self.state_dim)
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment
        
        Args:
            action: Index of item to select
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Check if action is valid (not already selected)
        if action in self.selected_indices:
            reward = -1.0  # Penalty for selecting already chosen item
        else:
            # Get reward from the selected item
            reward = float(self.data.iloc[action][self.reward_col])
            self.selected_indices.append(action)
            
            # Update state based on selected item features
            item_features = self.data.iloc[action][self.feature_cols].values
            self.current_state = (self.current_state * self.current_step + item_features) / (self.current_step + 1)
            
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'selected_indices': self.selected_indices,
            'num_selected': len(self.selected_indices)
        }
        
        return self.current_state, reward, done, info


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, 
                 lr: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 memory_size: int = 10000):
        """
        Initialize the DQN agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layer
            lr: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of replay memory
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Create model if PyTorch is available
        if TORCH_AVAILABLE:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        else:
            print("PyTorch not available. Using simplified version.")
            self.model = None
            self.target_model = None
            
        self.loss_history = []
        self.reward_history = []
        
    def _build_model(self) -> Any:
        """Build a neural network model for DQN"""
        if not TORCH_AVAILABLE:
            return None
            
        model = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # Store optimizer as an attribute of the model for simplicity
        model.optimizer = optimizer
        
        return model
    
    def update_target_model(self) -> None:
        """Update target model with weights from main model"""
        if TORCH_AVAILABLE and self.model is not None and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """Add experience to memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        """Choose an action based on state"""
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        if TORCH_AVAILABLE and self.model is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).detach().numpy()[0]
            
            # Mask out invalid actions
            masked_q_values = np.full(self.action_dim, -np.inf)
            masked_q_values[valid_actions] = q_values[valid_actions]
            
            return np.argmax(masked_q_values)
        else:
            # Fallback to random if PyTorch not available
            return random.choice(valid_actions)
    
    def replay(self, batch_size: int = None) -> Optional[float]:
        """Train the model using experience replay"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size or not TORCH_AVAILABLE or self.model is None:
            return None
            
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e.state for e in minibatch])
        actions = torch.LongTensor([[e.action] for e in minibatch])
        rewards = torch.FloatTensor([e.reward for e in minibatch])
        next_states = torch.FloatTensor([e.next_state for e in minibatch])
        dones = torch.FloatTensor([e.done for e in minibatch])
        
        # Get current Q values
        curr_q_values = self.model(states).gather(1, actions)
        
        # Get next Q values from target model
        next_q_values = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Calculate target Q values
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))
        
        # Calculate loss
        loss = F.mse_loss(curr_q_values, target_q_values)
        
        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.loss_history.append(loss.item())
        
        return loss.item()
        
    def save(self, filepath: str) -> None:
        """Save the model to a file"""
        if TORCH_AVAILABLE and self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'epsilon': self.epsilon,
                'loss_history': self.loss_history,
                'reward_history': self.reward_history
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Model not saved - PyTorch not available or model not initialized")
            
    def load(self, filepath: str) -> bool:
        """Load the model from a file"""
        if not TORCH_AVAILABLE or self.model is None:
            print("Cannot load model - PyTorch not available or model not initialized")
            return False
            
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
            
        try:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.loss_history = checkpoint['loss_history']
            self.reward_history = checkpoint['reward_history']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


class CRISPRTargetOptimizer:
    """Reinforcement learning optimizer for CRISPR target selection"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None, feature_cols: Optional[List[str]] = None,
                efficiency_col: str = 'efficiency', offtarget_col: str = 'offtarget_score',
                hidden_dim: int = 128, lr: float = 0.001, max_targets: int = 10):
        """
        Initialize the CRISPR target optimizer
        
        Args:
            data: DataFrame with CRISPR guide information
            feature_cols: Feature columns for the state space
            efficiency_col: Column name for on-target efficiency
            offtarget_col: Column name for off-target score
            hidden_dim: Hidden dimension for DQN
            lr: Learning rate
            max_targets: Maximum number of targets to select
        """
        self.data = data
        self.efficiency_col = efficiency_col
        self.offtarget_col = offtarget_col
        self.max_targets = max_targets
        
        # Use all columns except efficiency and offtarget as features if not specified
        if feature_cols is None and data is not None:
            self.feature_cols = [col for col in data.columns 
                                if col != efficiency_col and col != offtarget_col]
        else:
            self.feature_cols = feature_cols or []
            
        # Combined reward that balances efficiency and off-target
        if data is not None:
            # Create a combined reward column (high efficiency, low off-target)
            data['rl_reward'] = data[efficiency_col] * (1 - data[offtarget_col])
            self.reward_col = 'rl_reward'
        else:
            self.reward_col = None
            
        self.env = None
        self.agent = None
        self.is_trained = False
        
    def setup_environment(self, data: Optional[pd.DataFrame] = None) -> None:
        """Set up the reinforcement learning environment"""
        if data is not None:
            self.data = data
            # Create a combined reward column (high efficiency, low off-target)
            data['rl_reward'] = data[self.efficiency_col] * (1 - data[self.offtarget_col])
            self.reward_col = 'rl_reward'
            
            # Update feature columns if not already set
            if not self.feature_cols:
                self.feature_cols = [col for col in data.columns 
                                    if col != self.efficiency_col and 
                                       col != self.offtarget_col and
                                       col != self.reward_col]
        
        if self.data is None:
            raise ValueError("Data not provided. Set data before setting up environment.")
            
        if not self.feature_cols:
            raise ValueError("Feature columns not specified.")
            
        # Create environment and agent
        self.env = ReinforcementLearningEnvironment(
            data=self.data,
            feature_cols=self.feature_cols,
            reward_col=self.reward_col,
            max_steps=self.max_targets,
            batch_size=32
        )
        
        self.agent = DQNAgent(
            state_dim=len(self.feature_cols),
            action_dim=len(self.data),
            hidden_dim=128,
            lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
    def train(self, num_episodes: int = 500, update_freq: int = 10, 
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the reinforcement learning agent
        
        Args:
            num_episodes: Number of training episodes
            update_freq: Frequency to update target network
            save_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent not set up. Call setup_environment() first.")
            
        rewards_history = []
        avg_rewards_history = []
        loss_history = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Get valid actions (not yet selected)
                valid_actions = [i for i in range(self.env.action_dim) 
                               if i not in self.env.selected_indices]
                
                # Choose action
                action = self.agent.act(state, valid_actions)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Replay
                loss = self.agent.replay()
                if loss is not None:
                    loss_history.append(loss)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            # Update target network
            if episode % update_freq == 0:
                self.agent.update_target_model()
                
            rewards_history.append(total_reward)
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards_history.append(avg_reward)
            
            if episode % 20 == 0:
                print(f"Episode: {episode}, Score: {total_reward:.2f}, Avg Score: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
                
        self.agent.reward_history = rewards_history
        self.is_trained = True
        
        # Save the model if path provided
        if save_path:
            self.agent.save(save_path)
            
        return {
            'rewards': rewards_history,
            'avg_rewards': avg_rewards_history,
            'losses': loss_history
        }
        
    def select_optimal_targets(self, num_targets: Optional[int] = None, 
                              retrain: bool = False,
                              num_episodes: int = 500) -> pd.DataFrame:
        """
        Select optimal CRISPR targets using the trained agent
        
        Args:
            num_targets: Number of targets to select (defaults to max_targets)
            retrain: Whether to retrain the agent before selection
            num_episodes: Number of episodes for training if retrain=True
            
        Returns:
            DataFrame with selected targets
        """
        if num_targets is None:
            num_targets = self.max_targets
            
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent not set up. Call setup_environment() first.")
            
        if not self.is_trained or retrain:
            self.train(num_episodes=num_episodes)
            
        # Evaluate to select optimal targets
        state = self.env.reset()
        selected_indices = []
        total_reward = 0
        
        for _ in range(num_targets):
            # Get valid actions (not yet selected)
            valid_actions = [i for i in range(self.env.action_dim) 
                           if i not in selected_indices]
            
            if not valid_actions:
                break
                
            # Use agent to select best action
            action = self.agent.act(state, valid_actions)
            
            # Use epsilon=0 for greedy selection
            orig_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            action = self.agent.act(state, valid_actions)
            self.agent.epsilon = orig_epsilon
            
            selected_indices.append(action)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        # Return selected targets
        return self.data.iloc[selected_indices].copy()
        
    def plot_training_history(self) -> plt.Figure:
        """Plot the training history"""
        if not self.is_trained or self.agent is None:
            raise ValueError("Agent not trained. Call train() first.")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot rewards
        ax1.plot(self.agent.reward_history)
        ax1.set_title('Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot running average
        N = min(100, len(self.agent.reward_history))
        running_avg = np.convolve(self.agent.reward_history, np.ones(N)/N, mode='valid')
        ax1.plot(running_avg, color='red', label=f'{N}-ep average')
        ax1.legend()
        
        # Plot loss
        if self.agent.loss_history:
            ax2.plot(self.agent.loss_history)
            ax2.set_title('Loss per Training Step')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            
        plt.tight_layout()
        return fig
    
    def save(self, filepath: str) -> None:
        """Save the model and optimizer state"""
        if self.agent is None:
            print("Agent not initialized. Nothing to save.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save agent model
        self.agent.save(filepath)
        
    def load(self, filepath: str) -> bool:
        """Load the model and optimizer state"""
        if self.agent is None:
            print("Agent not initialized. Call setup_environment() first.")
            return False
            
        return self.agent.load(filepath)


class NeoantigensOptimizer:
    """Reinforcement learning optimizer for neoantigen candidate selection"""
    
    def __init__(self, data: Optional[pd.DataFrame] = None, 
                 feature_cols: Optional[List[str]] = None,
                 binding_affinity_col: str = 'binding_affinity',
                 expression_col: str = 'expression_level',
                 immunogenicity_col: str = 'immunogenicity',
                 hidden_dim: int = 128, lr: float = 0.001, 
                 max_epitopes: int = 10):
        """
        Initialize the neoantigen optimizer
        
        Args:
            data: DataFrame with neoantigen information
            feature_cols: Feature columns for the state space
            binding_affinity_col: Column for MHC binding affinity
            expression_col: Column for expression level
            immunogenicity_col: Column for immunogenicity score
            hidden_dim: Hidden dimension for DQN
            lr: Learning rate
            max_epitopes: Maximum number of epitopes to select
        """
        self.data = data
        self.binding_affinity_col = binding_affinity_col
        self.expression_col = expression_col
        self.immunogenicity_col = immunogenicity_col
        self.max_epitopes = max_epitopes
        
        # Use all columns except target columns as features if not specified
        if feature_cols is None and data is not None:
            exclude_cols = [binding_affinity_col, expression_col, immunogenicity_col]
            self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols or []
            
        # Create combined reward
        if data is not None:
            # Assuming: 
            # - Lower binding affinity is better (nM, lower = stronger binding)
            # - Higher expression is better
            # - Higher immunogenicity is better
            
            # Normalize binding affinity (lower is better)
            if binding_affinity_col in data.columns:
                max_affinity = data[binding_affinity_col].max()
                min_affinity = data[binding_affinity_col].min()
                range_affinity = max_affinity - min_affinity
                
                if range_affinity > 0:
                    norm_affinity = 1 - ((data[binding_affinity_col] - min_affinity) / range_affinity)
                else:
                    norm_affinity = 1.0
            else:
                norm_affinity = 1.0
                
            # Calculate combined reward
            data['rl_reward'] = (
                norm_affinity * 
                data.get(expression_col, 1.0) * 
                data.get(immunogenicity_col, 1.0)
            )
            self.reward_col = 'rl_reward'
        else:
            self.reward_col = None
            
        self.env = None
        self.agent = None
        self.is_trained = False
        
    def setup_environment(self, data: Optional[pd.DataFrame] = None) -> None:
        """Set up the reinforcement learning environment"""
        if data is not None:
            self.data = data
            # Create normalized binding affinity (lower is better)
            if self.binding_affinity_col in data.columns:
                max_affinity = data[self.binding_affinity_col].max()
                min_affinity = data[self.binding_affinity_col].min()
                range_affinity = max_affinity - min_affinity
                
                if range_affinity > 0:
                    norm_affinity = 1 - ((data[self.binding_affinity_col] - min_affinity) / range_affinity)
                else:
                    norm_affinity = 1.0
            else:
                norm_affinity = 1.0
                
            # Calculate combined reward
            data['rl_reward'] = (
                norm_affinity * 
                data.get(self.expression_col, 1.0) * 
                data.get(self.immunogenicity_col, 1.0)
            )
            self.reward_col = 'rl_reward'
            
            # Update feature columns if not already set
            if not self.feature_cols:
                exclude_cols = [
                    self.binding_affinity_col, 
                    self.expression_col, 
                    self.immunogenicity_col,
                    self.reward_col
                ]
                self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        if self.data is None:
            raise ValueError("Data not provided. Set data before setting up environment.")
            
        if not self.feature_cols:
            raise ValueError("Feature columns not specified.")
            
        # Create environment and agent
        self.env = ReinforcementLearningEnvironment(
            data=self.data,
            feature_cols=self.feature_cols,
            reward_col=self.reward_col,
            max_steps=self.max_epitopes,
            batch_size=32
        )
        
        self.agent = DQNAgent(
            state_dim=len(self.feature_cols),
            action_dim=len(self.data),
            hidden_dim=128,
            lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
    def train(self, num_episodes: int = 500, update_freq: int = 10, 
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the reinforcement learning agent
        
        Args:
            num_episodes: Number of training episodes
            update_freq: Frequency to update target network
            save_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent not set up. Call setup_environment() first.")
            
        rewards_history = []
        avg_rewards_history = []
        loss_history = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Get valid actions (not yet selected)
                valid_actions = [i for i in range(self.env.action_dim) 
                               if i not in self.env.selected_indices]
                
                # Choose action
                action = self.agent.act(state, valid_actions)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Remember experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Replay
                loss = self.agent.replay()
                if loss is not None:
                    loss_history.append(loss)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            # Update target network
            if episode % update_freq == 0:
                self.agent.update_target_model()
                
            rewards_history.append(total_reward)
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards_history.append(avg_reward)
            
            if episode % 20 == 0:
                print(f"Episode: {episode}, Score: {total_reward:.2f}, Avg Score: {avg_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
                
        self.agent.reward_history = rewards_history
        self.is_trained = True
        
        # Save the model if path provided
        if save_path:
            self.agent.save(save_path)
            
        return {
            'rewards': rewards_history,
            'avg_rewards': avg_rewards_history,
            'losses': loss_history
        }
        
    def select_optimal_epitopes(self, num_epitopes: Optional[int] = None, 
                               retrain: bool = False,
                               num_episodes: int = 500) -> pd.DataFrame:
        """
        Select optimal neoantigen epitopes using the trained agent
        
        Args:
            num_epitopes: Number of epitopes to select (defaults to max_epitopes)
            retrain: Whether to retrain the agent before selection
            num_episodes: Number of episodes for training if retrain=True
            
        Returns:
            DataFrame with selected epitopes
        """
        if num_epitopes is None:
            num_epitopes = self.max_epitopes
            
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent not set up. Call setup_environment() first.")
            
        if not self.is_trained or retrain:
            self.train(num_episodes=num_episodes)
            
        # Evaluate to select optimal epitopes
        state = self.env.reset()
        selected_indices = []
        total_reward = 0
        
        for _ in range(num_epitopes):
            # Get valid actions (not yet selected)
            valid_actions = [i for i in range(self.env.action_dim) 
                           if i not in selected_indices]
            
            if not valid_actions:
                break
                
            # Use agent to select best action
            action = self.agent.act(state, valid_actions)
            
            # Use epsilon=0 for greedy selection
            orig_epsilon = self.agent.epsilon
            self.agent.epsilon = 0
            action = self.agent.act(state, valid_actions)
            self.agent.epsilon = orig_epsilon
            
            selected_indices.append(action)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                break
                
        # Return selected epitopes
        return self.data.iloc[selected_indices].copy()
        
    def plot_training_history(self) -> plt.Figure:
        """Plot the training history"""
        if not self.is_trained or self.agent is None:
            raise ValueError("Agent not trained. Call train() first.")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot rewards
        ax1.plot(self.agent.reward_history)
        ax1.set_title('Reward per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot running average
        N = min(100, len(self.agent.reward_history))
        running_avg = np.convolve(self.agent.reward_history, np.ones(N)/N, mode='valid')
        ax1.plot(running_avg, color='red', label=f'{N}-ep average')
        ax1.legend()
        
        # Plot loss
        if self.agent.loss_history:
            ax2.plot(self.agent.loss_history)
            ax2.set_title('Loss per Training Step')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            
        plt.tight_layout()
        return fig
    
    def save(self, filepath: str) -> None:
        """Save the model and optimizer state"""
        if self.agent is None:
            print("Agent not initialized. Nothing to save.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save agent model
        self.agent.save(filepath)
        
    def load(self, filepath: str) -> bool:
        """Load the model and optimizer state"""
        if self.agent is None:
            print("Agent not initialized. Call setup_environment() first.")
            return False
            
        return self.agent.load(filepath)