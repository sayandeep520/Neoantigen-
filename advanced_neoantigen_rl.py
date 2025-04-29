import numpy as np
import pandas as pd
import random
import json
import os
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime

class StateFeatureExtractor:
    """
    Extract features from mutation state for reinforcement learning
    """
    def __init__(self, feature_size: int = 10):
        """
        Initialize the feature extractor
        
        Args:
            feature_size: Size of output feature vector
        """
        self.feature_size = feature_size
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.mutation_types = ["missense", "nonsense", "frameshift", "silent", "insertion", "deletion"]
        
        # Define feature names for visualization
        self.feature_names = [
            "Amino Acid Content", 
            "Mutation Position", 
            "Mutation Type", 
            "Gene Length", 
            "Protein Length",
            "Hydrophobicity", 
            "Charge",
            "Size Factor", 
            "Polarity", 
            "Evolutionary Conservation",
            "MHC Binding Potential", 
            "TCR Recognition", 
            "Proteasomal Cleavage", 
            "TAP Transport",
            "Sequence Context"
        ]
        
        # Ensure we have enough feature names for the requested feature size
        while len(self.feature_names) < feature_size:
            self.feature_names.append(f"Feature {len(self.feature_names) + 1}")
    
    def extract_features(self, mutation: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features from a mutation
        
        Args:
            mutation: Mutation dictionary with gene, protein_change, etc.
            
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        # Gene name features
        gene = mutation.get('gene', '')
        features.append(len(gene) / 20.0)  # Normalized gene length
        
        # Protein change features
        protein_change = mutation.get('protein_change', '')
        
        # Extract position if available (e.g., from G12D, extract 12)
        position = 0
        try:
            if protein_change and any(c.isdigit() for c in protein_change):
                # Extract numeric portion
                num_part = ''.join(c for c in protein_change if c.isdigit())
                if num_part:
                    position = int(num_part)
        except:
            position = 0
        
        # Normalize position
        features.append(min(position / 1000.0, 1.0))
        
        # One-hot encoding of mutation type
        mutation_type = mutation.get('mutation_type', 'unknown').lower()
        for mt in self.mutation_types:
            features.append(1.0 if mutation_type == mt else 0.0)
        
        # Amino acid content if available
        aa_features = [0.0] * len(self.amino_acids)
        if protein_change:
            for i, aa in enumerate(self.amino_acids):
                aa_features[i] = protein_change.count(aa) / max(len(protein_change), 1)
        features.extend(aa_features)
        
        # Pad or truncate to match feature_size
        if len(features) < self.feature_size:
            features.extend([0.0] * (self.feature_size - len(features)))
        elif len(features) > self.feature_size:
            features = features[:self.feature_size]
        
        return np.array(features, dtype=np.float32)

class AdvancedQLearningAgent:
    """
    Advanced Q-Learning Agent for neoantigen prediction using NumPy
    """
    def __init__(
        self, 
        state_dim: int = 30, 
        action_dim: int = 5, 
        learning_rate: float = 0.01, 
        discount_factor: float = 0.99,
        double_q: bool = True
    ):
        """
        Initialize the advanced Q-Learning agent
        
        Args:
            state_dim: Dimension of state representation
            action_dim: Number of possible actions
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            double_q: Whether to use double Q-learning
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.double_q = double_q
        
        # Initialize Q-tables
        self.q_table = {}  # Main Q-table
        if double_q:
            self.target_q_table = {}  # Target Q-table for double Q-learning
        
        # Initialize feature extractor
        self.feature_extractor = StateFeatureExtractor(feature_size=state_dim)
        
        # Track advanced metrics
        self.visit_counts = {}  # State-action visit counts
        self.uncertainty_estimates = {}  # Uncertainty estimates
        self.reward_history = []  # History of rewards
        
        # Advanced exploration parameters
        self.ucb_constant = 2.0  # Constant for UCB exploration
        self.temperature = 1.0   # Temperature for softmax exploration
        self.min_temperature = 0.1
        self.temperature_decay = 0.995
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.1, 
                   exploration_strategy: str = "epsilon_greedy") -> int:
        """
        Select an action using advanced exploration strategy
        
        Args:
            state: Current state vector
            epsilon: Exploration rate for epsilon-greedy
            exploration_strategy: Strategy to use ("epsilon_greedy", "ucb", "softmax")
            
        Returns:
            Selected action
        """
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self._initialize_state(state_key)
        
        # Update visit counts for UCB
        if state_key not in self.visit_counts:
            self.visit_counts[state_key] = np.ones(self.action_dim)
        
        # Different exploration strategies
        if exploration_strategy == "epsilon_greedy":
            # Epsilon-greedy: random action with probability epsilon
            if random.random() < epsilon:
                return random.randint(0, self.action_dim - 1)
            else:
                return np.argmax(self.q_table[state_key])
                
        elif exploration_strategy == "ucb":
            # Upper Confidence Bound exploration
            total_visits = sum(self.visit_counts[state_key])
            ucb_values = self.q_table[state_key] + self.ucb_constant * np.sqrt(
                np.log(total_visits + 1) / (self.visit_counts[state_key] + 1e-5)
            )
            return np.argmax(ucb_values)
            
        elif exploration_strategy == "softmax":
            # Softmax/Boltzmann exploration with temperature
            # Lower temperature means more exploitation
            q_values = self.q_table[state_key]
            scaled_values = q_values / max(self.temperature, 1e-5)
            exp_values = np.exp(scaled_values - np.max(scaled_values))
            probabilities = exp_values / np.sum(exp_values)
            
            # Decay temperature (cooling)
            self.temperature = max(self.min_temperature, 
                                 self.temperature * self.temperature_decay)
            
            # Choose action based on probabilities
            return np.random.choice(self.action_dim, p=probabilities)
        
        else:
            # Default to epsilon-greedy
            if random.random() < epsilon:
                return random.randint(0, self.action_dim - 1)
            else:
                return np.argmax(self.q_table[state_key])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update Q-values based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values for states if they don't exist
        if state_key not in self.q_table:
            self._initialize_state(state_key)
        
        if next_state_key not in self.q_table:
            self._initialize_state(next_state_key)
        
        # Update visit counts
        if state_key not in self.visit_counts:
            self.visit_counts[state_key] = np.ones(self.action_dim)
        self.visit_counts[state_key][action] += 1
        
        # Record reward for history
        self.reward_history.append(reward)
        
        # Q-value update
        if self.double_q:
            # Double Q-learning
            if next_state_key not in self.target_q_table:
                self.target_q_table[next_state_key] = np.zeros(self.action_dim)
            
            # Use main network to select action, target network to evaluate
            next_action = np.argmax(self.q_table[next_state_key])
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * self.target_q_table[next_state_key][next_action]
                
            # Update main Q-table
            self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
            
            # Periodically update target network (soft update)
            tau = 0.01  # Soft update parameter
            if state_key in self.target_q_table:
                self.target_q_table[state_key] = (
                    (1 - tau) * self.target_q_table[state_key] + 
                    tau * self.q_table[state_key]
                )
            else:
                self.target_q_table[state_key] = self.q_table[state_key].copy()
        else:
            # Standard Q-learning
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(self.q_table[next_state_key])
            
            # Update Q-value with learning rate
            self.q_table[state_key][action] += self.learning_rate * (target - self.q_table[state_key][action])
        
        # Update uncertainty estimates using variance of rewards
        if len(self.reward_history) > 10:
            recent_rewards = self.reward_history[-10:]
            variance = np.var(recent_rewards)
            self.uncertainty_estimates[state_key] = variance
    
    def _get_state_key(self, state: np.ndarray) -> tuple:
        """
        Convert state array to hashable key
        
        Args:
            state: State array
            
        Returns:
            Hashable key for the state
        """
        # Reduce precision for better generalization
        quantized_state = np.round(state, 2)
        return tuple(quantized_state)
    
    def _initialize_state(self, state_key: tuple):
        """
        Initialize Q-values for a new state
        
        Args:
            state_key: State key
        """
        # Initialize with small random values for better exploration
        self.q_table[state_key] = np.random.uniform(
            0, 0.1, self.action_dim
        ).astype(np.float32)
        
        # For double Q-learning, initialize target network too
        if self.double_q:
            self.target_q_table[state_key] = self.q_table[state_key].copy()
    
    def save_model(self, filepath: str):
        """
        Save Q-table to file
        
        Args:
            filepath: Path to save file
        """
        # Convert state tuples to strings for JSON serialization
        serializable_q_table = {str(k): v.tolist() for k, v in self.q_table.items()}
        
        # Save metadata and model parameters
        model_data = {
            'q_table': serializable_q_table,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'double_q': self.double_q,
            'temperature': self.temperature,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """
        Load Q-table from file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Set parameters from loaded data
        self.state_dim = data.get('state_dim', self.state_dim)
        self.action_dim = data.get('action_dim', self.action_dim)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.discount_factor = data.get('discount_factor', self.discount_factor)
        self.double_q = data.get('double_q', self.double_q)
        self.temperature = data.get('temperature', self.temperature)
        
        # Convert string keys back to tuples
        self.q_table = {}
        serialized_q_table = data.get('q_table', {})
        for k, v in serialized_q_table.items():
            # Parse string tuple back to actual tuple
            key_tuple = tuple(float(x) for x in k.strip('()').split(',') if x)
            self.q_table[key_tuple] = np.array(v, dtype=np.float32)
        
        # Initialize target network for double Q-learning
        if self.double_q:
            self.target_q_table = {k: v.copy() for k, v in self.q_table.items()}

class AdvancedNeoantigenEnvironment:
    """
    Advanced simulated environment for neoantigen prediction reinforcement learning
    """
    def __init__(self, initial_mutations, feature_size=30, max_steps=100):
        """
        Initialize the environment with initial mutation data
        
        Args:
            initial_mutations: Initial set of mutations to explore
            feature_size: Size of state feature vector
            max_steps: Maximum steps per episode
        """
        self.mutations = initial_mutations
        self.current_state = None
        self.current_mutation = None
        self.episode_steps = 0
        self.max_steps = max_steps
        self.feature_size = feature_size
        
        # Feature extractor
        self.feature_extractor = StateFeatureExtractor(feature_size=feature_size)
        
        # Action space definition
        self.actions = {
            0: self._minimal_mutation_change,   # Small change to protein
            1: self._moderate_mutation_change,  # Moderate change
            2: self._generate_similar_gene,     # Change gene
            3: self._mutate_mutation_type,      # Change mutation type
            4: self._preserve_mutation          # Keep current mutation (useful for evaluation)
        }
        
        # Reward configuration with weights
        self.reward_weights = {
            'mhc_binding': 0.4,      # MHC binding affinity
            'immunogenicity': 0.3,   # Immunogenicity
            'tcr_recognition': 0.2,  # T-cell receptor recognition
            'conservation': 0.1      # Evolutionary conservation
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.best_mutations = []
    
    def reset(self):
        """
        Reset the environment to initial state
        
        Returns:
            Initial state representation
        """
        self.episode_steps = 0
        # Randomly select an initial mutation
        self.current_mutation = random.choice(self.mutations)
        self.current_state = self.feature_extractor.extract_features(self.current_mutation)
        return self.current_state
    
    def step(self, action):
        """
        Take a step in the environment based on the agent's action
        
        Args:
            action: Action taken by the agent
        
        Returns:
            next_state, reward, done, info
        """
        self.episode_steps += 1
        
        # Apply action to modify mutation
        action_fn = self.actions.get(action, self._preserve_mutation)
        modified_mutation = action_fn(self.current_mutation)
        
        # Extract features for next state
        next_state = self.feature_extractor.extract_features(modified_mutation)
        
        # Calculate reward
        reward, reward_components = self._calculate_reward(modified_mutation)
        
        # Update current state and mutation
        self.current_state = next_state
        self.current_mutation = modified_mutation
        
        # Check if episode is done
        done = (self.episode_steps >= self.max_steps)
        
        # Additional info for debugging and visualization
        info = {
            'mutation': modified_mutation,
            'reward_components': reward_components,
            'steps': self.episode_steps
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, mutation):
        """
        Calculate reward based on mutation characteristics with multiple components
        
        Args:
            mutation: Mutation to evaluate
        
        Returns:
            total_reward, reward_components
        """
        # Simulate different reward components
        # In a real system, these would be calculated by actual prediction models
        
        # MHC binding (lower is better, so we invert)
        mhc_binding_score = np.random.beta(2, 5)  # Tends toward lower values (better binding)
        
        # Immunogenicity (higher is better)
        immunogenicity_score = np.random.beta(5, 2)  # Tends toward higher values
        
        # TCR recognition (higher is better)
        tcr_recognition_score = np.random.beta(3, 3)  # More uniform distribution
        
        # Conservation score (lower is better for mutations targeting less conserved regions)
        conservation_score = np.random.beta(2, 3)
        
        # Reward components
        reward_components = {
            'mhc_binding': mhc_binding_score,
            'immunogenicity': immunogenicity_score,
            'tcr_recognition': tcr_recognition_score,
            'conservation': 1.0 - conservation_score  # Invert so higher is better
        }
        
        # Calculate weighted reward
        total_reward = sum(
            value * self.reward_weights[component] 
            for component, value in reward_components.items()
        )
        
        return total_reward, reward_components
    
    def _minimal_mutation_change(self, mutation):
        """
        Make a small change to the protein modification
        
        Args:
            mutation: Original mutation
        
        Returns:
            Modified mutation
        """
        modified = mutation.copy()
        protein_change = mutation.get('protein_change', '')
        
        if protein_change:
            # Select a random position to modify
            if len(protein_change) > 2:
                pos = random.randint(0, len(protein_change) - 1)
                
                # If position has a letter, replace with a random amino acid
                if protein_change[pos].isalpha():
                    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                    new_aa = random.choice(amino_acids)
                    modified['protein_change'] = (
                        protein_change[:pos] + new_aa + protein_change[pos+1:]
                    )
        
        return modified
    
    def _moderate_mutation_change(self, mutation):
        """
        Make a moderate change to the mutation
        
        Args:
            mutation: Original mutation
        
        Returns:
            Modified mutation
        """
        modified = mutation.copy()
        protein_change = mutation.get('protein_change', '')
        
        if protein_change:
            # Extract position if format is like "G12D"
            prefix = ""
            position = ""
            suffix = ""
            
            for char in protein_change:
                if char.isalpha() and not position:
                    prefix += char
                elif char.isdigit():
                    position += char
                else:
                    suffix += char
            
            # If we successfully parsed the format, modify it
            if prefix and position:
                # Change the position slightly
                try:
                    pos_int = int(position)
                    pos_change = random.choice([-1, 1])
                    new_pos = max(1, pos_int + pos_change)
                    modified['protein_change'] = f"{prefix}{new_pos}{suffix}"
                except:
                    # If parsing failed, make a minimal change instead
                    return self._minimal_mutation_change(mutation)
        
        return modified
    
    def _generate_similar_gene(self, mutation):
        """
        Generate a similar gene modification
        
        Args:
            mutation: Original mutation
        
        Returns:
            Modified mutation
        """
        modified = mutation.copy()
        gene = mutation.get('gene', '')
        
        if gene:
            # Add a suffix or prefix to gene name
            suffixes = ['L', 'R', 'A', 'B', 'P', '1', '2']
            if random.random() < 0.5 and len(gene) > 1:
                # Remove last character
                modified['gene'] = gene[:-1]
            else:
                # Add suffix
                modified['gene'] = gene + random.choice(suffixes)
        
        return modified
    
    def _mutate_mutation_type(self, mutation):
        """
        Change the mutation type
        
        Args:
            mutation: Original mutation
        
        Returns:
            Modified mutation
        """
        modified = mutation.copy()
        mutation_type = mutation.get('mutation_type', '').lower()
        
        mutation_types = ['missense', 'nonsense', 'frameshift', 'silent', 'insertion', 'deletion']
        available_types = [t for t in mutation_types if t != mutation_type]
        
        if available_types:
            modified['mutation_type'] = random.choice(available_types)
        
        return modified
    
    def _preserve_mutation(self, mutation):
        """
        Keep the mutation as is (useful for exploitation)
        
        Args:
            mutation: Original mutation
        
        Returns:
            Same mutation
        """
        return mutation.copy()

class AdvancedNeoantigenRLTrainer:
    """
    Advanced Reinforcement Learning Trainer for Neoantigen Prediction
    """
    def __init__(
        self, 
        initial_mutations,
        feature_size=30,
        action_dim=5,
        learning_rate=0.01,
        discount_factor=0.99,
        double_q=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        exploration_strategy="epsilon_greedy",
        max_steps_per_episode=100
    ):
        """
        Initialize the RL trainer
        
        Args:
            initial_mutations: Initial set of mutations
            feature_size: Size of state representation
            action_dim: Number of possible actions
            learning_rate: Learning rate for agent
            discount_factor: Discount factor for future rewards
            double_q: Whether to use double Q-learning
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration rate decay
            exploration_strategy: Exploration strategy ("epsilon_greedy", "ucb", "softmax")
            max_steps_per_episode: Maximum steps per episode
        """
        # Environment setup
        self.env = AdvancedNeoantigenEnvironment(
            initial_mutations=initial_mutations,
            feature_size=feature_size,
            max_steps=max_steps_per_episode
        )
        
        # Agent setup
        self.agent = AdvancedQLearningAgent(
            state_dim=feature_size,
            action_dim=action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            double_q=double_q
        )
        
        # For visualization compatibility
        if not hasattr(self.agent, 'visit_counts'):
            self.agent.visit_counts = {}
        if not hasattr(self.agent, 'uncertainty_estimates'):
            self.agent.uncertainty_estimates = {}
        
        # Training parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.exploration_strategy = exploration_strategy
        
        # Compatibility for visualizations
        if self.exploration_strategy == 'softmax' and (not hasattr(self.agent, 'temperature_decay') or not hasattr(self.agent, 'min_temperature')):
            self.agent.temperature_decay = 0.995
            self.agent.min_temperature = 0.1
        
        # Training metrics
        self.episode_rewards = []
        self.best_episodes = []
        self.episode_lengths = []
        self.training_history = {
            'rewards': [],
            'epsilons': [],
            'best_mutations': []
        }
    
    def train(self, num_episodes=1000, verbose=True, 
              save_best=True, save_interval=100, save_dir=None,
              progress_callback=None):
        """
        Train the agent using advanced Q-Learning
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress information
            save_best: Whether to save best mutations and episodes
            save_interval: How often to save model during training
            save_dir: Directory to save models
            progress_callback: Callback function for progress updates
        
        Returns:
            Training history
        """
        self.episode_rewards = []
        self.best_episodes = []
        best_reward = float('-inf')
        self.episode_lengths = []
        
        # Prepare save directory if needed
        if save_best and save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            episode_data = []
            
            while not done:
                # Select action using exploration strategy
                action = self.agent.get_action(
                    state, 
                    epsilon=self.epsilon,
                    exploration_strategy=self.exploration_strategy
                )
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store step data for tracking
                step_data = {
                    'step': steps,
                    'state': state.tolist(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.tolist(),
                    'mutation': info['mutation'],
                    'reward_components': info['reward_components']
                }
                episode_data.append(step_data)
                
                # Update agent
                self.agent.update(state, action, reward, next_state, done)
                
                # Update state and totals
                state = next_state
                total_reward += reward
                steps += 1
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Update training history
            self.training_history['rewards'].append(total_reward)
            self.training_history['epsilons'].append(self.epsilon)
            
            # Decay exploration rate
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Track best episodes
            if total_reward > best_reward:
                best_reward = total_reward
                if save_best:
                    self.best_episodes.append({
                        'episode': episode,
                        'reward': total_reward,
                        'steps': steps,
                        'data': episode_data,
                        'mutation': self.env.current_mutation
                    })
                    # Only keep top episodes
                    self.best_episodes = sorted(
                        self.best_episodes, 
                        key=lambda x: x['reward'], 
                        reverse=True
                    )[:5]
                    
                    # Save best mutation to history
                    self.training_history['best_mutations'].append({
                        'episode': episode,
                        'mutation': self.env.current_mutation,
                        'reward': total_reward
                    })
            
            # Save model at intervals if requested
            if save_best and save_dir and episode > 0 and episode % save_interval == 0:
                save_path = os.path.join(save_dir, f"model_episode_{episode}.json")
                self.agent.save_model(save_path)
            
            # Report progress
            if verbose and (episode % 10 == 0 or episode == num_episodes - 1):
                print(f"Episode {episode}/{num_episodes}: Reward = {total_reward:.4f}, Epsilon = {self.epsilon:.4f}")
            
            # Call progress callback if provided
            if progress_callback:
                progress = (episode + 1) / num_episodes
                progress_callback(progress, episode, total_reward, self.epsilon)
        
        # Save final model
        if save_best and save_dir:
            final_path = os.path.join(save_dir, "final_model.json")
            self.agent.save_model(final_path)
        
        return self.training_history
    
    def get_best_mutations(self, n=5):
        """
        Get the best mutations found during training
        
        Args:
            n: Number of best mutations to return
            
        Returns:
            List of best mutations
        """
        if not self.best_episodes:
            return []
        
        best_mutations = []
        for episode in self.best_episodes[:n]:
            mutation = episode['mutation'].copy()
            mutation['reward'] = episode['reward']
            mutation['episode'] = episode['episode']
            best_mutations.append(mutation)
        
        return best_mutations
    
    def generate_new_suggestions(self, num_suggestions=5, 
                                exploration_rate=0.1):
        """
        Generate new neoantigen suggestions using the trained policy
        
        Args:
            num_suggestions: Number of suggestions to generate
            exploration_rate: Exploration rate during generation
            
        Returns:
            List of suggested mutations
        """
        suggestions = []
        
        # Run episodes to generate suggestions
        for _ in range(max(20, num_suggestions * 2)):
            state = self.env.reset()
            done = False
            
            # Run the episode with limited exploration
            while not done:
                action = self.agent.get_action(
                    state, 
                    epsilon=exploration_rate,
                    exploration_strategy="epsilon_greedy"  # Use simple strategy for generation
                )
                next_state, reward, done, info = self.env.step(action)
                state = next_state
            
            # Add final mutation if good enough and not duplicate
            final_mutation = self.env.current_mutation.copy()
            final_mutation['reward'] = reward
            
            # Check if it's a duplicate
            is_duplicate = False
            for existing in suggestions:
                if (existing.get('gene') == final_mutation.get('gene') and
                    existing.get('protein_change') == final_mutation.get('protein_change')):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                suggestions.append(final_mutation)
                
                # Break if we have enough suggestions
                if len(suggestions) >= num_suggestions:
                    break
        
        # Sort by reward
        suggestions = sorted(suggestions, key=lambda x: x.get('reward', 0), reverse=True)
        
        # Add rank
        for i, suggestion in enumerate(suggestions):
            suggestion['rank'] = i + 1
        
        return suggestions[:num_suggestions]
    
    def save_training_results(self, output_dir):
        """
        Save training results to files
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save agent model
        self.agent.save_model(os.path.join(output_dir, 'agent_model.json'))
        
        # Save training metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'best_episodes': [{
                'episode': ep['episode'],
                'reward': ep['reward'],
                'steps': ep['steps'],
                'mutation': ep['mutation']
            } for ep in self.best_episodes],
            'training_history': self.training_history,
            'episode_lengths': self.episode_lengths,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = json.loads(
                json.dumps(metrics, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
            )
            json.dump(serializable_metrics, f, indent=2)
    
    def load_training_results(self, output_dir):
        """
        Load training results from files
        
        Args:
            output_dir: Directory to load results from
        """
        try:
            # Load agent model
            self.agent.load_model(os.path.join(output_dir, 'agent_model.json'))
            
            # Load training metrics
            with open(os.path.join(output_dir, 'training_metrics.json'), 'r') as f:
                metrics = json.load(f)
                self.episode_rewards = metrics.get('episode_rewards', [])
                self.best_episodes = metrics.get('best_episodes', [])
                self.training_history = metrics.get('training_history', {})
                self.episode_lengths = metrics.get('episode_lengths', [])
            
            return True
        except Exception as e:
            print(f"Error loading training results: {str(e)}")
            return False

def run_demonstration():
    """
    Run a demonstration of the advanced neoantigen RL system
    """
    # Sample initial mutations
    initial_mutations = [
        {
            'gene': 'KRAS',
            'protein_change': 'G12D',
            'mutation_type': 'missense'
        },
        {
            'gene': 'TP53',
            'protein_change': 'R273H',
            'mutation_type': 'missense'
        },
        {
            'gene': 'BRCA1',
            'protein_change': 'E1449K',
            'mutation_type': 'missense'
        },
        {
            'gene': 'EGFR',
            'protein_change': 'T790M',
            'mutation_type': 'missense'
        }
    ]
    
    # Create and train RL system
    trainer = AdvancedNeoantigenRLTrainer(
        initial_mutations=initial_mutations,
        feature_size=30,
        action_dim=5,
        learning_rate=0.01,
        discount_factor=0.99,
        double_q=True,
        epsilon_start=0.9,
        epsilon_decay=0.97
    )
    
    # Train for a small number of episodes
    print("Training RL agent...")
    history = trainer.train(num_episodes=200, verbose=True)
    
    # Get best mutations found during training
    best_mutations = trainer.get_best_mutations()
    
    print("\nBest mutations found:")
    for i, mutation in enumerate(best_mutations):
        print(f"{i+1}. Gene: {mutation['gene']}, "
              f"Protein Change: {mutation['protein_change']}, "
              f"Type: {mutation['mutation_type']}, "
              f"Reward: {mutation['reward']:.4f}")
    
    # Generate new suggestions
    print("\nGenerating new suggestions...")
    suggestions = trainer.generate_new_suggestions(num_suggestions=3)
    
    print("\nNew suggestions:")
    for i, suggestion in enumerate(suggestions):
        print(f"{i+1}. Gene: {suggestion['gene']}, "
              f"Protein Change: {suggestion['protein_change']}, "
              f"Type: {suggestion['mutation_type']}, "
              f"Reward: {suggestion['reward']:.4f}")
    
    return trainer, history

if __name__ == "__main__":
    run_demonstration()