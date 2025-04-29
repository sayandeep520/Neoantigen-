import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import random
from datetime import datetime
import streamlit as st
import io
import base64
from models.neoantigen_rl import NeoantigenRLTrainer, NeoantigenEnvironment, SimpleQLearningAgent
from models.advanced_neoantigen_rl import AdvancedNeoantigenRLTrainer, AdvancedNeoantigenEnvironment, AdvancedQLearningAgent

def get_initial_mutations_from_data(mutation_data, max_mutations=20):
    """
    Extract initial mutations from loaded mutation data
    
    Args:
        mutation_data: Mutation data as a DataFrame or list
        max_mutations: Maximum number of mutations to extract
        
    Returns:
        List of mutations in the format needed for RL
    """
    initial_mutations = []
    
    if isinstance(mutation_data, pd.DataFrame):
        # Extract from DataFrame
        if 'gene' in mutation_data.columns:
            gene_col = 'gene'
        elif 'Gene' in mutation_data.columns:
            gene_col = 'Gene'
        else:
            gene_col = None
            
        if 'protein_change' in mutation_data.columns:
            protein_col = 'protein_change'
        elif 'ProteinChange' in mutation_data.columns:
            protein_col = 'ProteinChange'
        elif 'protein_modification' in mutation_data.columns:
            protein_col = 'protein_modification'
        else:
            protein_col = None
            
        if 'mutation_type' in mutation_data.columns:
            type_col = 'mutation_type'
        elif 'MutationType' in mutation_data.columns:
            type_col = 'MutationType'
        elif 'mutation_subtype' in mutation_data.columns:
            type_col = 'mutation_subtype'
        else:
            type_col = None
        
        # If we have at least gene and protein change columns
        if gene_col and protein_col:
            # Select a subset of rows
            subset = mutation_data.sample(min(len(mutation_data), max_mutations)) if len(mutation_data) > max_mutations else mutation_data
            
            for _, row in subset.iterrows():
                mutation = {
                    'gene': row[gene_col],
                    'protein_change': row[protein_col],
                    'mutation_type': row[type_col] if type_col else 'unknown'
                }
                initial_mutations.append(mutation)
    
    elif isinstance(mutation_data, list):
        # Extract from list of dictionaries
        for item in mutation_data[:max_mutations]:
            if isinstance(item, dict):
                mutation = {
                    'gene': item.get('gene', ''),
                    'protein_change': item.get('protein_change', ''),
                    'mutation_type': item.get('mutation_type', 'unknown')
                }
                
                # Skip incomplete mutations
                if mutation['gene'] and mutation['protein_change']:
                    initial_mutations.append(mutation)
    
    # If no valid mutations were found, provide some defaults
    if not initial_mutations:
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
            }
        ]
    
    return initial_mutations

def prepare_training_configuration(initial_mutations, config=None):
    """
    Prepare configuration for RL training
    
    Args:
        initial_mutations: List of initial mutations
        config: Optional configuration dictionary
        
    Returns:
        Complete configuration dictionary
    """
    # Default configuration
    default_config = {
        'state_dim': 3,
        'action_dim': 3,
        'num_episodes': 200,
        'epsilon_start': 0.9,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.97,
        'save_dir': 'data/neoantigen_rl/models',
        'description': 'Neoantigen RL training run',
        'run_id': f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # Update with provided config
    if config:
        default_config.update(config)
    
    return default_config

def train_neoantigen_rl_model(initial_mutations, config, progress_callback=None, use_advanced_model=False):
    """
    Train a neoantigen RL model
    
    Args:
        initial_mutations: List of initial mutations
        config: Configuration dictionary
        progress_callback: Callback function for progress updates
        use_advanced_model: Whether to use the advanced RL model
        
    Returns:
        Trained RL trainer object and training results
    """
    if use_advanced_model:
        # Create advanced trainer
        rl_trainer = AdvancedNeoantigenRLTrainer(
            initial_mutations=initial_mutations,
            feature_size=config.get('feature_size', 30),
            action_dim=config.get('action_dim', 5),
            learning_rate=config.get('learning_rate', 0.01),
            discount_factor=config.get('discount_factor', 0.99),
            double_q=config.get('double_q', True),
            epsilon_start=config.get('epsilon_start', 0.9),
            epsilon_end=config.get('epsilon_end', 0.01),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            exploration_strategy=config.get('exploration_strategy', 'epsilon_greedy'),
            max_steps_per_episode=config.get('max_steps', 100)
        )
        
        # Train the model
        training_history = rl_trainer.train(
            num_episodes=config.get('num_episodes', 200), 
            verbose=True,
            save_best=True, 
            save_dir=config.get('save_dir'),
            progress_callback=progress_callback
        )
        
        # Return results
        return rl_trainer, {
            'rewards': training_history.get('rewards', []),
            'best_episodes': rl_trainer.best_episodes,
            'config': config,
            'training_history': training_history
        }
    
    else:
        # Create basic trainer
        rl_trainer = NeoantigenRLTrainer(
            config['state_dim'],
            config['action_dim'],
            initial_mutations,
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay']
        )
        
        # Run training with progress updates
        episode_rewards = []
        progress_interval = max(1, config['num_episodes'] // 20)  # Update approximately 20 times
        
        def training_loop():
            for episode in range(config['num_episodes']):
                state = rl_trainer.env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    # Select action
                    action = rl_trainer.agent.get_action(state, rl_trainer.epsilon)
                    
                    # Take action
                    next_state, reward, done, _ = rl_trainer.env.step(action)
                    
                    # Update agent
                    rl_trainer.agent.update(state, action, reward, next_state, done)
                    
                    # Update state and reward
                    state = next_state
                    total_reward += reward
                
                # Decay epsilon
                rl_trainer.epsilon = max(rl_trainer.epsilon_end, rl_trainer.epsilon * rl_trainer.epsilon_decay)
                
                # Record reward
                episode_rewards.append(total_reward)
                rl_trainer.episode_rewards.append(total_reward)
                
                # Check for best episode
                if not rl_trainer.best_episodes or total_reward > rl_trainer.best_episodes[0]['reward']:
                    rl_trainer.best_episodes.append({
                        'episode': episode,
                        'reward': total_reward,
                        'steps': rl_trainer.env.episode_steps,
                        'mutation': rl_trainer.env.current_state
                    })
                    rl_trainer.best_episodes = sorted(rl_trainer.best_episodes, key=lambda x: x['reward'], reverse=True)[:5]
                
                # Report progress
                if progress_callback and episode % progress_interval == 0:
                    progress = episode / config['num_episodes']
                    progress_callback(progress, episode, total_reward, rl_trainer.epsilon)
        
        # Run training loop
        training_loop()
        
        # Save model if directory specified
        if config.get('save_dir'):
            os.makedirs(config['save_dir'], exist_ok=True)
            model_path = os.path.join(config['save_dir'], f"{config['run_id']}")
            os.makedirs(model_path, exist_ok=True)
            rl_trainer.save_training_results(model_path)
        
        return rl_trainer, {
            'rewards': episode_rewards,
            'best_episodes': rl_trainer.best_episodes,
            'config': config
        }

def create_training_visualizations(results):
    """
    Create visualizations for RL training results
    
    Args:
        results: Training results dictionary
        
    Returns:
        Dictionary of matplotlib figures
    """
    visualizations = {}
    
    # Episode rewards plot
    if 'rewards' in results:
        rewards = results['rewards']
        
        # Learning curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rewards)
        # Add rolling average
        window_size = min(len(rewards) // 10, 10)
        if window_size > 0:
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax.plot(rolling_avg, 'r--', linewidth=2)
            ax.legend(['Episode Reward', f'{window_size}-Episode Rolling Average'])
        ax.set_title('Learning Curve')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(linestyle='--', alpha=0.7)
        visualizations['learning_curve'] = fig
        
        # Reward distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(rewards, kde=True, ax=ax)
        ax.set_title('Reward Distribution')
        ax.set_xlabel('Total Reward')
        ax.set_ylabel('Frequency')
        ax.grid(linestyle='--', alpha=0.7)
        visualizations['reward_distribution'] = fig
    
    # Best mutations visualization
    if 'best_episodes' in results and results['best_episodes']:
        # Top rewards comparison
        best_rewards = [episode['reward'] for episode in results['best_episodes']]
        best_episodes = [episode['episode'] for episode in results['best_episodes']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(best_rewards)), best_rewards)
        
        # Add episode numbers as labels
        for i, (bar, episode) in enumerate(zip(bars, best_episodes)):
            ax.text(i, bar.get_height() + 0.01, f"Ep {episode}", 
                    ha='center', va='bottom', rotation=0, fontsize=10)
        
        ax.set_title('Top Performing Episodes')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Total Reward')
        ax.set_xticks(range(len(best_rewards)))
        ax.set_xticklabels([f"#{i+1}" for i in range(len(best_rewards))])
        ax.grid(linestyle='--', alpha=0.7)
        visualizations['top_rewards'] = fig
    
    return visualizations

def format_mutation_for_display(mutation):
    """
    Format a mutation for display
    
    Args:
        mutation: Mutation dictionary
        
    Returns:
        Formatted mutation string
    """
    gene = mutation.get('gene', 'Unknown')
    protein_change = mutation.get('protein_change', 'Unknown')
    mutation_type = mutation.get('mutation_type', 'Unknown')
    
    return f"{gene} {protein_change} ({mutation_type})"

def get_neoantigen_suggestions(rl_trainer, num_suggestions=5):
    """
    Get neoantigen suggestions from trained RL model
    
    Args:
        rl_trainer: Trained RL trainer (basic or advanced)
        num_suggestions: Number of suggestions to generate
        
    Returns:
        List of suggested neoantigens
    """
    suggestions = []
    
    # Check if this is an advanced RL trainer
    if isinstance(rl_trainer, AdvancedNeoantigenRLTrainer):
        # Use advanced suggestion generation method
        return rl_trainer.generate_new_suggestions(num_suggestions=num_suggestions)
    
    # Basic RL trainer handling
    # If we have a trained model with best episodes
    if hasattr(rl_trainer, 'best_episodes') and rl_trainer.best_episodes:
        # Get mutations from best episodes
        for i, episode in enumerate(rl_trainer.best_episodes[:num_suggestions]):
            mutation = episode['mutation'].copy()
            mutation['reward'] = episode['reward']
            mutation['rank'] = i + 1
            suggestions.append(mutation)
    
    # If we need more suggestions, generate new ones using the model
    if len(suggestions) < num_suggestions and hasattr(rl_trainer, 'env'):
        env = rl_trainer.env
        agent = rl_trainer.agent
        
        # Run a few episodes and collect good mutations
        for _ in range(max(10, num_suggestions * 3)):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.get_action(state, epsilon=0.1)  # Low exploration
                next_state, reward, done, _ = env.step(action)
                state = next_state
            
            # Add final mutation if good enough
            if not suggestions or env.current_state not in [s['mutation'] for s in suggestions]:
                mutation = env.current_state.copy()
                mutation['reward'] = reward
                mutation['rank'] = len(suggestions) + 1
                suggestions.append(mutation)
                
                # Break if we have enough suggestions
                if len(suggestions) >= num_suggestions:
                    break
    
    # Sort by reward
    suggestions = sorted(suggestions, key=lambda x: x.get('reward', 0), reverse=True)
    
    return suggestions[:num_suggestions]

def mhc_binding_prediction(peptide, allele):
    """
    Simulate MHC binding prediction
    
    Args:
        peptide: Peptide sequence
        allele: MHC allele
        
    Returns:
        Predicted binding affinity (nM)
    """
    # This is a placeholder simulation
    # In production, this would call an actual prediction model or API
    binding_score = random.random() * 500  # Higher is worse (nM)
    return binding_score

def immunogenicity_prediction(peptide, mhc_binding):
    """
    Simulate immunogenicity prediction
    
    Args:
        peptide: Peptide sequence
        mhc_binding: MHC binding score
        
    Returns:
        Predicted immunogenicity score (0-1)
    """
    # This is a placeholder simulation
    # In production, this would call an actual prediction model or API
    # Invert binding (lower binding = higher score)
    binding_component = max(0, 1 - (mhc_binding / 500))
    
    # Random component for other factors
    random_component = random.random() * 0.5
    
    # Combined score
    immunogenicity = 0.5 * binding_component + 0.5 * random_component
    return immunogenicity

def plot_to_base64(fig):
    """
    Convert matplotlib figure to base64 string for HTML display
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str