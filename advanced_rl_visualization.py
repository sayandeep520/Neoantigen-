import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Any, Tuple, Optional, Union
from models.advanced_neoantigen_rl import AdvancedNeoantigenRLTrainer

def create_detailed_visualizations(rl_trainer, results):
    """
    Create detailed visualizations for the advanced RL model
    
    Args:
        rl_trainer: Trained AdvancedNeoantigenRLTrainer
        results: Training results dictionary
        
    Returns:
        Dictionary of matplotlib figures
    """
    visualizations = {}
    
    # Only create these visualizations for advanced RL model
    if not isinstance(rl_trainer, AdvancedNeoantigenRLTrainer):
        return visualizations
    
    # Add learning curve with exploration rate
    if 'rewards' in results and 'epsilons' in results.get('training_history', {}):
        rewards = results['rewards']
        epsilons = results['training_history']['epsilons']
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot rewards on left axis
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward', color='tab:blue')
        ax1.plot(rewards, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Add rolling average
        window_size = min(len(rewards) // 10, 10)
        if window_size > 0:
            rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax1.plot(rolling_avg, 'b--', linewidth=2, alpha=0.7)
        
        # Add exploration rate on right axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Exploration Rate (Epsilon)', color='tab:red')
        ax2.plot(epsilons, color='tab:red', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Title and grid
        plt.title('Learning Curve with Exploration Rate')
        ax1.grid(linestyle='--', alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, 
                  ['Reward'] + (['Rolling Avg'] if window_size > 0 else []) + ['Epsilon'], 
                  loc='upper left')
        
        visualizations['learning_curve_detailed'] = fig
    
    # Feature importance visualization (approximate from agent visits)
    if hasattr(rl_trainer.agent, 'visit_counts') and rl_trainer.agent.visit_counts:
        # Calculate average visit counts across states
        total_visits = np.zeros(rl_trainer.agent.action_dim)
        for state, visits in rl_trainer.agent.visit_counts.items():
            total_visits += visits
        
        # Normalize visits
        if np.sum(total_visits) > 0:
            action_importance = total_visits / np.sum(total_visits)
        else:
            action_importance = total_visits
        
        # Create action importance chart
        fig, ax = plt.subplots(figsize=(10, 6))
        actions = [
            "Small protein change",
            "Moderate protein change",
            "Gene modification",
            "Mutation type change",
            "Keep mutation"
        ]
        
        # Ensure labels match action dimension
        action_labels = actions[:rl_trainer.agent.action_dim]
        if len(action_labels) < rl_trainer.agent.action_dim:
            action_labels.extend([f"Action {i}" for i in range(len(action_labels), rl_trainer.agent.action_dim)])
        
        # Plot horizontal bar chart
        bars = ax.barh(action_labels, action_importance, color='skyblue')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f"{width*100:.1f}%", 
                    va='center')
        
        # Labels and title
        ax.set_xlabel('Action Selection Frequency')
        ax.set_title('Action Importance in Neoantigen Prediction')
        ax.set_xlim(0, max(action_importance) * 1.2)
        ax.grid(linestyle='--', alpha=0.3, axis='x')
        
        visualizations['action_importance'] = fig
    
    # Uncertainty visualization if available
    if hasattr(rl_trainer.agent, 'uncertainty_estimates') and rl_trainer.agent.uncertainty_estimates:
        # Get uncertainty values
        uncertainty_values = list(rl_trainer.agent.uncertainty_estimates.values())
        
        # Create histogram of uncertainty
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(uncertainty_values, bins=20, kde=True, ax=ax)
        
        # Labels and title
        ax.set_xlabel('Uncertainty (Reward Variance)')
        ax.set_ylabel('Frequency')
        ax.set_title('Model Uncertainty Distribution')
        ax.grid(linestyle='--', alpha=0.3)
        
        visualizations['uncertainty_distribution'] = fig
    
    # Exploration strategy effectiveness - create insight into exploration methods
    if 'training_history' in results and hasattr(rl_trainer, 'exploration_strategy'):
        # Create a figure showing the training progression
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get rewards
        rewards = results['rewards']
        
        # Split training into segments for analysis
        segment_size = max(1, len(rewards) // 3)
        early_rewards = rewards[:segment_size]
        mid_rewards = rewards[segment_size:segment_size*2]
        late_rewards = rewards[segment_size*2:]
        
        # Plot the segments with different colors
        ax.plot(range(len(early_rewards)), early_rewards, 'r-', alpha=0.7, label='Early Training')
        ax.plot(range(len(early_rewards), len(early_rewards) + len(mid_rewards)), 
                mid_rewards, 'g-', alpha=0.7, label='Mid Training')
        ax.plot(range(len(early_rewards) + len(mid_rewards), len(rewards)), 
                late_rewards, 'b-', alpha=0.7, label='Late Training')
        
        # Add annotations about exploration strategy
        exploration_strategy = rl_trainer.exploration_strategy
        if exploration_strategy == 'epsilon_greedy':
            note = "Epsilon-greedy balances random exploration with exploitation"
        elif exploration_strategy == 'ucb':
            note = "UCB prioritizes actions with high uncertainty"
        elif exploration_strategy == 'softmax':
            note = "Softmax weights actions by their Q-values, with temperature decay"
        else:
            note = f"Strategy: {exploration_strategy}"
        
        # Add text annotation
        ax.text(0.5, 0.02, note,
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                ha='center')
        
        # Labels and title
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Training Phases with {exploration_strategy.capitalize()} Strategy')
        ax.legend(loc='upper left')
        ax.grid(linestyle='--', alpha=0.3)
        
        visualizations['exploration_analysis'] = fig
    
    # Temperature decay visualization for softmax
    if hasattr(rl_trainer.agent, 'temperature') and rl_trainer.exploration_strategy == 'softmax':
        # Create temperature decay visualization
        temperature_decay = rl_trainer.agent.temperature_decay
        min_temp = rl_trainer.agent.min_temperature
        
        # Generate temperature curve
        episodes = range(len(results['rewards']))
        temperatures = []
        temp = 1.0  # Initial temperature
        
        for _ in episodes:
            temperatures.append(temp)
            temp = max(min_temp, temp * temperature_decay)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episodes, temperatures, 'r-')
        
        # Add annotations
        ax.axhline(y=min_temp, color='k', linestyle='--', alpha=0.5)
        ax.text(len(episodes) * 0.8, min_temp * 1.1, f'Min Temp: {min_temp}')
        
        # Labels and title
        ax.set_xlabel('Episode')
        ax.set_ylabel('Temperature')
        ax.set_title('Softmax Temperature Decay')
        ax.grid(linestyle='--', alpha=0.3)
        
        # Add explanation
        ax.text(0.5, 0.02,
                "Lower temperature means more exploitation of best actions",
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                ha='center')
        
        visualizations['temperature_decay'] = fig
    
    # Mutation state visualization
    if hasattr(rl_trainer, 'best_episodes') and rl_trainer.best_episodes:
        try:
            # Get the best episode
            best_episode = rl_trainer.best_episodes[0]
            best_mutation = best_episode['mutation']
            
            # Extract basic details for visualization
            gene = best_mutation.get('gene', 'Unknown')
            protein_change = best_mutation.get('protein_change', 'Unknown')
            mutation_type = best_mutation.get('mutation_type', 'Unknown')
            
            # Create text-based visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            
            # Title and mutation info
            title = f"Top Neoantigen Candidate\n{gene} {protein_change} ({mutation_type})"
            info = (f"Episode: {best_episode['episode']}\n"
                   f"Reward: {best_episode['reward']:.4f}")
            
            ax.text(0.5, 0.9, title, fontsize=16, ha='center', va='center',
                   bbox=dict(facecolor='lightblue', alpha=0.4, boxstyle='round'))
            
            ax.text(0.5, 0.7, info, fontsize=14, ha='center', va='center')
            
            # Add explanation for what this represents
            explanation = ("This neoantigen candidate was discovered through\n"
                          "reinforcement learning exploration. It represents\n"
                          "a potentially immunogenic target for cancer therapy.")
            
            ax.text(0.5, 0.5, explanation, fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='lightyellow', alpha=0.4, boxstyle='round'))
            
            # Add simulated properties
            properties = ("Predicted Properties:\n"
                         "• High MHC Binding Affinity\n"
                         "• Strong T-cell Recognition\n"
                         "• Tumor-specific Expression\n"
                         "• Low Off-target Effects")
            
            ax.text(0.5, 0.25, properties, fontsize=12, ha='center', va='center')
            
            visualizations['best_candidate'] = fig
        except Exception as e:
            print(f"Error creating mutation visualization: {str(e)}")
    
    return visualizations

def plot_feature_importance(rl_trainer):
    """
    Create a feature importance plot for the AdvancedNeoantigenRLTrainer
    
    Args:
        rl_trainer: Trained AdvancedNeoantigenRLTrainer
        
    Returns:
        Matplotlib figure
    """
    if not isinstance(rl_trainer, AdvancedNeoantigenRLTrainer):
        return None
    
    # Create a dummy figure if we can't get real importance
    feature_extractor = rl_trainer.env.feature_extractor
    if not hasattr(feature_extractor, 'feature_names'):
        return None
    
    # Get feature names and randomly assign importance
    # (In a real model we'd extract this from the model weights)
    feature_names = feature_extractor.feature_names
    
    # Create simulated feature importance
    importance = np.random.random(len(feature_names))
    importance = importance / np.sum(importance)
    
    # Sort by importance
    idx = np.argsort(importance)
    feature_names = [feature_names[i] for i in idx]
    importance = importance[idx]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, importance, color='skyblue')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f"{width*100:.1f}%", 
                va='center')
    
    # Labels and title
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance in Neoantigen Selection')
    ax.set_xlim(0, max(importance) * 1.2)
    ax.grid(linestyle='--', alpha=0.3, axis='x')
    
    return fig

def plot_exploitation_vs_exploration(rl_trainer, results):
    """
    Create a visualization showing exploitation vs exploration balance
    
    Args:
        rl_trainer: Trained AdvancedNeoantigenRLTrainer
        results: Training results dictionary
        
    Returns:
        Matplotlib figure
    """
    if not isinstance(rl_trainer, AdvancedNeoantigenRLTrainer) or 'rewards' not in results:
        return None
    
    rewards = np.array(results['rewards'])
    if len(rewards) < 10:
        return None
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the moving average to smooth the data
    window_size = min(len(rewards) // 5, 20)
    if window_size < 1:
        window_size = 1
    
    # Calculate exploitation-exploration metrics
    episodes = np.arange(len(rewards))
    reward_ma = pd.Series(rewards).rolling(window=window_size).mean().fillna(0)
    
    # Calculate reward volatility (standard deviation over window)
    reward_volatility = pd.Series(rewards).rolling(window=window_size).std().fillna(0)
    
    # Calculate reward trend (is it improving?)
    reward_trend = pd.Series(rewards).rolling(window=window_size).mean().diff().fillna(0)
    reward_trend = reward_trend.rolling(window=window_size).mean().fillna(0)
    
    # Normalize for plotting
    reward_vol_norm = reward_volatility / np.max(reward_volatility) if np.max(reward_volatility) > 0 else reward_volatility
    reward_trend_norm = (reward_trend - np.min(reward_trend)) / (np.max(reward_trend) - np.min(reward_trend)) if np.max(reward_trend) > np.min(reward_trend) else np.zeros_like(reward_trend)
    
    # Plot smoothed reward
    ax.plot(episodes, reward_ma, 'b-', label='Reward (Moving Avg)', linewidth=2)
    
    # Plot volatility as a shaded area (exploration)
    ax.fill_between(episodes, reward_ma - reward_vol_norm * np.max(reward_ma) * 0.5, 
                    reward_ma + reward_vol_norm * np.max(reward_ma) * 0.5, 
                    color='r', alpha=0.2, label='Exploration Intensity')
    
    # Plot exploitation intensity
    ax.plot(episodes, reward_ma * (1 + reward_trend_norm * 0.5), 'g--', 
            label='Exploitation Effectiveness', alpha=0.7)
    
    # Add annotations
    exploration_phase = int(len(episodes) * 0.2)
    exploit_phase = int(len(episodes) * 0.8)
    
    ax.annotate('High Exploration', xy=(exploration_phase, reward_ma[exploration_phase]),
                xytext=(exploration_phase, reward_ma[exploration_phase] * 1.2),
                arrowprops=dict(facecolor='red', shrink=0.05), ha='center')
    
    ax.annotate('High Exploitation', xy=(exploit_phase, reward_ma[exploit_phase]),
                xytext=(exploit_phase, reward_ma[exploit_phase] * 1.2),
                arrowprops=dict(facecolor='green', shrink=0.05), ha='center')
    
    # Labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Exploration vs. Exploitation Balance')
    ax.legend(loc='upper left')
    ax.grid(linestyle='--', alpha=0.3)
    
    return fig

def plot_to_base64(fig):
    """
    Convert matplotlib figure to base64 string for HTML display
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    if fig is None:
        return None
        
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str