"""
CRISPR Reinforcement Learning Optimization Page

This page provides an interface for using reinforcement learning
to optimize CRISPR target selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
import io
from datetime import datetime

# Import RL utilities
from utils.reinforcement_learning import CRISPRTargetOptimizer

# Set page config
st.set_page_config(
    page_title="CRISPR-RL Optimization",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state for the optimizer
if "crispr_rl_optimizer" not in st.session_state:
    st.session_state.crispr_rl_optimizer = None
    
if "crispr_rl_data" not in st.session_state:
    st.session_state.crispr_rl_data = None
    
if "crispr_rl_history" not in st.session_state:
    st.session_state.crispr_rl_history = None
    
if "crispr_rl_selected_targets" not in st.session_state:
    st.session_state.crispr_rl_selected_targets = None

# Path for saving models
MODELS_DIR = "models/reinforcement_learning"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    st.title("🧬 CRISPR Target Optimization - Reinforcement Learning")
    
    st.markdown("""
    ### Optimize CRISPR target selection using Deep Reinforcement Learning
    
    This page uses a Deep Q-Network (DQN) reinforcement learning agent to optimize the selection of CRISPR target sites
    based on both efficiency and off-target scores. The RL agent learns to balance these factors to find optimal combinations.
    
    #### How It Works:
    1. **Upload your CRISPR target data** - CSV with guide sequences and their properties
    2. **Configure the RL agent** - Set parameters for the reinforcement learning model
    3. **Train the agent** - Let the agent learn which targets are most promising
    4. **Select optimal targets** - Use the trained agent to prioritize target sites
    """)
    
    tabs = st.tabs(["Data Loading", "Agent Configuration", "Training", "Target Selection", "Models"])
    
    with tabs[0]:
        load_data_tab()
        
    with tabs[1]:
        configure_agent_tab()
        
    with tabs[2]:
        train_agent_tab()
        
    with tabs[3]:
        select_targets_tab()
        
    with tabs[4]:
        manage_models_tab()

def load_data_tab():
    st.header("1. Load CRISPR Target Data")
    
    # Option to use sample data
    use_sample = st.checkbox("Use sample data", value=False)
    
    if use_sample:
        st.info("Using synthetic sample data")
        # Generate synthetic data if checked
        data = generate_sample_data()
        st.session_state.crispr_rl_data = data
        st.dataframe(data)
        st.success("Sample data loaded successfully!")
    else:
        # File upload for user data
        st.subheader("Upload your CRISPR target data (CSV)")
        
        # Help text with required format
        with st.expander("Data Format Instructions"):
            st.markdown("""
            **Required CSV format:**
            
            Your data should contain at minimum:
            - A column for guide sequence
            - A column for on-target efficiency scores (0-1 scale preferred)
            - A column for off-target scores (0-1 scale, where 0 means no off-targets)
            
            Additional feature columns can be included to improve RL agent's performance.
            
            Example:
            ```
            sequence,gc_content,self_folding,efficiency,offtarget_score
            ATCGATCGATCGATCGATCGAGG,0.55,-3.2,0.85,0.12
            GGCATCGATCGTAGCTAGAACGG,0.60,-2.8,0.72,0.05
            ```
            """)
            
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.dataframe(data)
                
                # Check for minimum required columns
                st.subheader("Column Mapping")
                st.info("Please identify the columns in your data")
                
                sequence_col = st.selectbox("Guide Sequence Column", options=data.columns)
                efficiency_col = st.selectbox("Efficiency Score Column", options=data.columns)
                offtarget_col = st.selectbox("Off-target Score Column", options=data.columns)
                
                # Button to confirm dataset
                if st.button("Confirm Data"):
                    # Store the data in session state
                    st.session_state.crispr_rl_data = data
                    # Remember column mappings
                    st.session_state.crispr_rl_sequence_col = sequence_col
                    st.session_state.crispr_rl_efficiency_col = efficiency_col
                    st.session_state.crispr_rl_offtarget_col = offtarget_col
                    
                    st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Show data stats if available
    if st.session_state.crispr_rl_data is not None:
        data = st.session_state.crispr_rl_data
        st.subheader("Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Guides", len(data))
        
        if "crispr_rl_efficiency_col" in st.session_state:
            eff_col = st.session_state.crispr_rl_efficiency_col
            with col2:
                st.metric("Avg. Efficiency", f"{data[eff_col].mean():.3f}")
        
        if "crispr_rl_offtarget_col" in st.session_state:
            offt_col = st.session_state.crispr_rl_offtarget_col
            with col3:
                st.metric("Avg. Off-target", f"{data[offt_col].mean():.3f}")
        
        # Data visualization
        st.subheader("Data Visualization")
        if st.checkbox("Show Efficiency vs. Off-target Plot", value=True):
            if "crispr_rl_efficiency_col" in st.session_state and "crispr_rl_offtarget_col" in st.session_state:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(
                    data[st.session_state.crispr_rl_efficiency_col], 
                    data[st.session_state.crispr_rl_offtarget_col], 
                    alpha=0.6
                )
                ax.set_xlabel("Efficiency Score")
                ax.set_ylabel("Off-target Score")
                ax.set_title("Efficiency vs Off-target Scores")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add a trend line
                try:
                    z = np.polyfit(
                        data[st.session_state.crispr_rl_efficiency_col], 
                        data[st.session_state.crispr_rl_offtarget_col], 
                        1
                    )
                    p = np.poly1d(z)
                    ax.plot(
                        data[st.session_state.crispr_rl_efficiency_col],
                        p(data[st.session_state.crispr_rl_efficiency_col]),
                        "r--", alpha=0.8
                    )
                except:
                    pass
                
                st.pyplot(fig)
                
                # Correlation
                corr = data[st.session_state.crispr_rl_efficiency_col].corr(
                    data[st.session_state.crispr_rl_offtarget_col]
                )
                st.info(f"Correlation between efficiency and off-target: {corr:.3f}")

def configure_agent_tab():
    st.header("2. Configure Reinforcement Learning Agent")
    
    # Check if data is loaded
    if st.session_state.crispr_rl_data is None:
        st.warning("Please load data in the 'Data Loading' tab first")
        return
    
    st.markdown("""
    Configure the Deep Q-Network (DQN) agent parameters for CRISPR target optimization.
    These settings control how the agent learns to select optimal targets.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Learning Parameters")
        
        hidden_dim = st.slider("Hidden Layer Size", 32, 256, 128, 32)
        learning_rate = st.select_slider(
            "Learning Rate", 
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            value=0.001
        )
        gamma = st.slider("Discount Factor (Gamma)", 0.8, 0.999, 0.99, 0.01)
        max_targets = st.slider("Max Targets to Select", 1, 20, 10, 1)
    
    with col2:
        st.subheader("Exploration Parameters")
        
        epsilon = st.slider("Initial Exploration Rate (Epsilon)", 0.1, 1.0, 1.0, 0.1)
        epsilon_min = st.slider("Minimum Exploration Rate", 0.01, 0.5, 0.01, 0.01)
        epsilon_decay = st.slider("Exploration Decay Rate", 0.9, 0.999, 0.995, 0.001)
        
        st.subheader("Feature Selection")
        data = st.session_state.crispr_rl_data
        
        if "crispr_rl_efficiency_col" in st.session_state and "crispr_rl_offtarget_col" in st.session_state:
            efficiency_col = st.session_state.crispr_rl_efficiency_col
            offtarget_col = st.session_state.crispr_rl_offtarget_col
            
            # Exclude efficiency and offtarget columns
            candidate_feature_cols = [
                col for col in data.columns 
                if col != efficiency_col and col != offtarget_col
            ]
            
            # Multi-select for feature columns
            feature_cols = st.multiselect(
                "Feature Columns for State Space",
                options=candidate_feature_cols,
                default=candidate_feature_cols
            )
    
    # Initialize button
    if st.button("Initialize Reinforcement Learning Agent"):
        with st.spinner("Initializing agent..."):
            # Create the optimizer
            optimizer = CRISPRTargetOptimizer(
                data=data,
                feature_cols=feature_cols,
                efficiency_col=efficiency_col,
                offtarget_col=offtarget_col,
                hidden_dim=hidden_dim,
                lr=learning_rate,
                max_targets=max_targets
            )
            
            # Setup the environment
            try:
                optimizer.setup_environment()
                
                # Set exploration parameters
                if optimizer.agent:
                    optimizer.agent.epsilon = epsilon
                    optimizer.agent.epsilon_min = epsilon_min
                    optimizer.agent.epsilon_decay = epsilon_decay
                
                # Store in session state
                st.session_state.crispr_rl_optimizer = optimizer
                st.success("RL agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RL agent: {str(e)}")

def train_agent_tab():
    st.header("3. Train the RL Agent")
    
    # Check if optimizer is initialized
    if st.session_state.crispr_rl_optimizer is None:
        st.warning("Please initialize the RL agent in the 'Agent Configuration' tab first")
        return
    
    st.markdown("""
    Train the Deep Q-Network agent to learn optimal CRISPR target selection strategies.
    The agent will learn to balance on-target efficiency with off-target effects.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Parameters")
        
        num_episodes = st.slider("Number of Episodes", 50, 1000, 500, 50)
        update_freq = st.slider("Target Network Update Frequency", 1, 50, 10, 1)
        save_model = st.checkbox("Save Trained Model", value=True)
        
        if save_model:
            model_name = st.text_input(
                "Model Name", 
                value=f"crispr_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        # Training progress placeholder
        progress_placeholder = st.empty()
        training_metrics_placeholder = st.empty()
    
    with col2:
        st.subheader("Pre-trained Models")
        
        # List available models
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt') and f.startswith('crispr_rl_')]
        
        if model_files:
            selected_model = st.selectbox("Load Pre-trained Model", options=[''] + model_files)
            
            if selected_model and st.button("Load Model"):
                with st.spinner("Loading model..."):
                    try:
                        model_path = os.path.join(MODELS_DIR, selected_model)
                        loaded = st.session_state.crispr_rl_optimizer.load(model_path)
                        
                        if loaded:
                            st.success(f"Model {selected_model} loaded successfully!")
                            st.session_state.crispr_rl_optimizer.is_trained = True
                        else:
                            st.error("Failed to load model")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        else:
            st.info("No pre-trained models available")
    
    # Training button
    if st.button("Begin Training"):
        with st.spinner("Training RL agent..."):
            progress_bar = progress_placeholder.progress(0)
            
            # Define callback to update progress
            def progress_callback(episode, total_episodes, metrics):
                progress = int((episode + 1) / total_episodes * 100)
                progress_bar.progress(progress)
                
                # Update metrics display
                if episode % 20 == 0 or episode == total_episodes - 1:
                    reward = metrics.get('reward', 0)
                    avg_reward = metrics.get('avg_reward', 0)
                    epsilon = metrics.get('epsilon', 0)
                    
                    training_metrics_placeholder.info(
                        f"Episode: {episode+1}/{total_episodes}, "
                        f"Reward: {reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                        f"Epsilon: {epsilon:.3f}"
                    )
            
            try:
                # Mock callback usage - we'll update progress manually
                # In a real implementation, modify train() to accept callback
                optimizer = st.session_state.crispr_rl_optimizer
                
                # Train model and collect history
                history = optimizer.train(
                    num_episodes=num_episodes,
                    update_freq=update_freq,
                    save_path=os.path.join(MODELS_DIR, model_name) if save_model else None
                )
                
                # Store history in session state
                st.session_state.crispr_rl_history = history
                
                # Plot results
                st.subheader("Training Results")
                
                # Plot the training history
                try:
                    fig = optimizer.plot_training_history()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting training history: {str(e)}")
                
                st.success(f"Training completed successfully! Agent trained for {num_episodes} episodes.")
                
                if save_model:
                    st.info(f"Model saved as: {model_name}")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

def select_targets_tab():
    st.header("4. Select Optimal CRISPR Targets")
    
    # Check if optimizer is trained
    if st.session_state.crispr_rl_optimizer is None:
        st.warning("Please initialize and train the RL agent first")
        return
    
    optimizer = st.session_state.crispr_rl_optimizer
    
    if not optimizer.is_trained:
        st.warning("RL agent has not been trained. Please train the agent first.")
        return
    
    st.markdown("""
    Use the trained reinforcement learning agent to select the optimal CRISPR targets.
    The agent has learned to balance efficiency and off-target effects for optimal selection.
    """)
    
    # Selection parameters
    num_targets = st.slider(
        "Number of Targets to Select", 
        1, 
        optimizer.max_targets, 
        min(5, optimizer.max_targets)
    )
    
    use_greedy = st.checkbox("Use Greedy Selection (no exploration)", value=True)
    
    # Selection button
    if st.button("Select Optimal Targets"):
        with st.spinner("Selecting optimal targets..."):
            try:
                # Set epsilon to 0 for greedy selection if requested
                original_epsilon = None
                if use_greedy and optimizer.agent:
                    original_epsilon = optimizer.agent.epsilon
                    optimizer.agent.epsilon = 0
                
                # Select targets
                selected_targets = optimizer.select_optimal_targets(num_targets=num_targets)
                
                # Restore original epsilon
                if use_greedy and optimizer.agent and original_epsilon is not None:
                    optimizer.agent.epsilon = original_epsilon
                
                # Store in session state
                st.session_state.crispr_rl_selected_targets = selected_targets
                
                st.success(f"Successfully selected {len(selected_targets)} optimal targets!")
            except Exception as e:
                st.error(f"Error selecting targets: {str(e)}")
    
    # Display selected targets
    if st.session_state.crispr_rl_selected_targets is not None:
        st.subheader("Selected Optimal Targets")
        
        selected = st.session_state.crispr_rl_selected_targets
        
        # Display metrics
        if "crispr_rl_efficiency_col" in st.session_state and "crispr_rl_offtarget_col" in st.session_state:
            efficiency_col = st.session_state.crispr_rl_efficiency_col
            offtarget_col = st.session_state.crispr_rl_offtarget_col
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Number of Targets", len(selected))
            
            with col2:
                st.metric("Avg. Efficiency", f"{selected[efficiency_col].mean():.3f}")
                
                # Compare with dataset average
                data_avg = st.session_state.crispr_rl_data[efficiency_col].mean()
                improvement = (selected[efficiency_col].mean() - data_avg) / data_avg * 100
                st.metric("Efficiency Improvement", f"{improvement:.1f}%")
            
            with col3:
                st.metric("Avg. Off-target", f"{selected[offtarget_col].mean():.3f}")
                
                # Compare with dataset average (for off-target, lower is better)
                data_avg = st.session_state.crispr_rl_data[offtarget_col].mean()
                improvement = (data_avg - selected[offtarget_col].mean()) / data_avg * 100
                st.metric("Off-target Reduction", f"{improvement:.1f}%")
        
        # Display table
        st.dataframe(selected)
        
        # Download option
        csv = selected.to_csv(index=False)
        st.download_button(
            label="Download Selected Targets",
            data=csv,
            file_name=f"optimal_crispr_targets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.subheader("Visualization")
        
        if "crispr_rl_efficiency_col" in st.session_state and "crispr_rl_offtarget_col" in st.session_state:
            efficiency_col = st.session_state.crispr_rl_efficiency_col
            offtarget_col = st.session_state.crispr_rl_offtarget_col
            
            # Scatter plot showing selected targets vs. all targets
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all targets
            all_data = st.session_state.crispr_rl_data
            ax.scatter(
                all_data[efficiency_col], 
                all_data[offtarget_col], 
                alpha=0.3, 
                color='blue',
                label='All Targets'
            )
            
            # Plot selected targets
            ax.scatter(
                selected[efficiency_col], 
                selected[offtarget_col], 
                alpha=0.8, 
                color='red',
                s=80,
                label='Selected Targets'
            )
            
            # Add ideal zone (high efficiency, low off-target)
            ax.axvspan(0.7, 1.0, 0, 0.3, alpha=0.2, color='green', label='Ideal Zone')
            
            ax.set_xlabel("Efficiency Score")
            ax.set_ylabel("Off-target Score")
            ax.set_title("Selected Targets vs. All Targets")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)
            
            # Show reward distribution
            if 'rl_reward' in selected.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(all_data['rl_reward'], bins=30, alpha=0.5, label='All Targets', ax=ax)
                sns.histplot(selected['rl_reward'], bins=10, alpha=0.7, color='red', label='Selected Targets', ax=ax)
                ax.set_xlabel('Reward Score')
                ax.set_ylabel('Count')
                ax.set_title('Reward Distribution: Selected vs. All Targets')
                ax.legend()
                st.pyplot(fig)

def manage_models_tab():
    st.header("5. Model Management")
    
    st.markdown("""
    Manage your trained reinforcement learning models for CRISPR target optimization.
    You can download models for backup or upload previously trained models.
    """)
    
    # List available models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt') and f.startswith('crispr_rl_')]
    
    if model_files:
        st.subheader("Available Models")
        
        for model_file in model_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(model_file)
            
            with col2:
                # Download model
                model_path = os.path.join(MODELS_DIR, model_file)
                with open(model_path, 'rb') as f:
                    model_data = f.read()
                    
                st.download_button(
                    label="Download",
                    data=model_data,
                    file_name=model_file,
                    mime="application/octet-stream",
                    key=f"download_{model_file}"
                )
            
            with col3:
                # Delete model
                if st.button("Delete", key=f"delete_{model_file}"):
                    try:
                        os.remove(model_path)
                        st.success(f"Model {model_file} deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting model: {str(e)}")
    else:
        st.info("No saved models available")
    
    # Upload model
    st.subheader("Upload Model")
    
    uploaded_file = st.file_uploader("Upload a trained model (.pt file)", type=['pt'])
    
    if uploaded_file is not None:
        # Get filename or allow user to specify
        default_name = uploaded_file.name
        new_name = st.text_input("Save as (filename):", value=default_name)
        
        if st.button("Save Uploaded Model"):
            try:
                # Save the uploaded model
                model_path = os.path.join(MODELS_DIR, new_name)
                with open(model_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"Model saved as {new_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")

def generate_sample_data(n_samples=100):
    """Generate synthetic CRISPR guide data for demo purposes"""
    
    # Generate random guide sequences
    bases = ['A', 'T', 'G', 'C']
    guide_length = 20
    pam = 'NGG'
    
    sequences = []
    for _ in range(n_samples):
        guide = ''.join(np.random.choice(bases) for _ in range(guide_length))
        pam_seq = pam.replace('N', np.random.choice(bases))
        sequences.append(guide + pam_seq)
    
    # Calculate GC content
    gc_content = [
        (seq.count('G') + seq.count('C')) / len(seq.replace('NGG', '')) 
        for seq in sequences
    ]
    
    # Generate random folding energies
    self_folding = np.random.normal(-4, 2, n_samples)
    
    # Generate efficiency scores with some relationship to GC and folding
    efficiency_base = np.random.normal(0.6, 0.2, n_samples)
    efficiency = 0.7 * efficiency_base + 0.2 * np.array(gc_content) + 0.1 * (-self_folding / 10)
    efficiency = np.clip(efficiency, 0.1, 0.95)
    
    # Generate off-target scores with weak negative correlation to efficiency
    offtarget_base = np.random.normal(0.3, 0.15, n_samples)
    offtarget = 0.6 * offtarget_base + 0.4 * (1 - efficiency) + np.random.normal(0, 0.1, n_samples)
    offtarget = np.clip(offtarget, 0.01, 0.9)
    
    # Generate some additional features
    distance_to_pam = np.random.randint(5, 50, n_samples)
    chromatin_access = np.random.normal(0.5, 0.2, n_samples)
    chromatin_access = np.clip(chromatin_access, 0.1, 0.9)
    
    # Create dataset
    data = pd.DataFrame({
        'sequence': sequences,
        'gc_content': gc_content,
        'self_folding': self_folding,
        'distance_to_pam': distance_to_pam,
        'chromatin_access': chromatin_access,
        'efficiency': efficiency,
        'offtarget_score': offtarget
    })
    
    return data

if __name__ == "__main__":
    main()