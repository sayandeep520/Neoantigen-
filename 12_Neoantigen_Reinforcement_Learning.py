"""
Neoantigen Reinforcement Learning Optimization Page

This page provides an interface for using reinforcement learning
to optimize neoantigen candidate selection for personalized cancer immunotherapy.
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
from utils.reinforcement_learning import NeoantigensOptimizer

# Set page config
st.set_page_config(
    page_title="Neoantigen-RL Optimization",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state for the optimizer
if "neoantigen_rl_optimizer" not in st.session_state:
    st.session_state.neoantigen_rl_optimizer = None
    
if "neoantigen_rl_data" not in st.session_state:
    st.session_state.neoantigen_rl_data = None
    
if "neoantigen_rl_history" not in st.session_state:
    st.session_state.neoantigen_rl_history = None
    
if "neoantigen_rl_selected_epitopes" not in st.session_state:
    st.session_state.neoantigen_rl_selected_epitopes = None

# Path for saving models
MODELS_DIR = "models/reinforcement_learning"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    st.title("🧪 Neoantigen Candidate Optimization - Reinforcement Learning")
    
    st.markdown("""
    ### Optimize neoantigen selection using Deep Reinforcement Learning
    
    This module uses a Deep Q-Network (DQN) reinforcement learning agent to optimize the selection of neoantigen
    candidates for personalized cancer vaccines and immunotherapies. The agent learns to balance binding affinity,
    expression level, and immunogenicity to find optimal combinations.
    
    #### How It Works:
    1. **Upload your neoantigen data** - CSV with epitope sequences and their properties
    2. **Configure the RL agent** - Set parameters for the reinforcement learning model
    3. **Train the agent** - Let the agent learn which epitopes are most promising
    4. **Select optimal epitopes** - Use the trained agent to prioritize candidates
    """)
    
    tabs = st.tabs(["Data Loading", "Agent Configuration", "Training", "Epitope Selection", "Models"])
    
    with tabs[0]:
        load_data_tab()
        
    with tabs[1]:
        configure_agent_tab()
        
    with tabs[2]:
        train_agent_tab()
        
    with tabs[3]:
        select_epitopes_tab()
        
    with tabs[4]:
        manage_models_tab()

def load_data_tab():
    st.header("1. Load Neoantigen Data")
    
    # Option to use sample data
    use_sample = st.checkbox("Use sample data", value=False)
    
    if use_sample:
        st.info("Using synthetic sample data")
        # Generate synthetic data if checked
        data = generate_sample_data()
        st.session_state.neoantigen_rl_data = data
        st.dataframe(data)
        st.success("Sample data loaded successfully!")
    else:
        # File upload for user data
        st.subheader("Upload your neoantigen candidate data (CSV)")
        
        # Help text with required format
        with st.expander("Data Format Instructions"):
            st.markdown("""
            **Required CSV format:**
            
            Your data should contain at minimum:
            - A column for epitope sequence
            - A column for MHC binding affinity (lower is better, typically in nM)
            - A column for expression level (higher is better)
            - A column for immunogenicity score (higher is better)
            
            Additional feature columns can be included to improve RL agent's performance.
            
            Example:
            ```
            epitope_seq,mhc_allele,binding_affinity,expression_level,immunogenicity,mutation_type
            YLQDVFQKV,HLA-A*02:01,25.3,0.85,0.72,missense
            KMNERTLFL,HLA-A*02:01,125.7,0.65,0.88,frameshift
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
                
                sequence_col = st.selectbox("Epitope Sequence Column", options=data.columns)
                binding_col = st.selectbox("Binding Affinity Column (lower is better)", options=data.columns)
                expression_col = st.selectbox("Expression Level Column (higher is better)", options=data.columns)
                immunogenicity_col = st.selectbox("Immunogenicity Score Column (higher is better)", options=data.columns)
                
                # Button to confirm dataset
                if st.button("Confirm Data"):
                    # Store the data in session state
                    st.session_state.neoantigen_rl_data = data
                    # Remember column mappings
                    st.session_state.neoantigen_rl_sequence_col = sequence_col
                    st.session_state.neoantigen_rl_binding_col = binding_col
                    st.session_state.neoantigen_rl_expression_col = expression_col
                    st.session_state.neoantigen_rl_immunogenicity_col = immunogenicity_col
                    
                    st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Show data stats if available
    if st.session_state.neoantigen_rl_data is not None:
        data = st.session_state.neoantigen_rl_data
        st.subheader("Data Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Epitopes", len(data))
        
        if "neoantigen_rl_binding_col" in st.session_state:
            binding_col = st.session_state.neoantigen_rl_binding_col
            with col2:
                st.metric("Avg. Binding Affinity (nM)", f"{data[binding_col].mean():.1f}")
        
        if "neoantigen_rl_expression_col" in st.session_state:
            expr_col = st.session_state.neoantigen_rl_expression_col
            with col3:
                st.metric("Avg. Expression", f"{data[expr_col].mean():.3f}")
                
        if "neoantigen_rl_immunogenicity_col" in st.session_state:
            immuno_col = st.session_state.neoantigen_rl_immunogenicity_col
            with col4:
                st.metric("Avg. Immunogenicity", f"{data[immuno_col].mean():.3f}")
        
        # Data visualization
        st.subheader("Data Visualization")
        if st.checkbox("Show Binding vs. Immunogenicity Plot", value=True):
            if "neoantigen_rl_binding_col" in st.session_state and "neoantigen_rl_immunogenicity_col" in st.session_state:
                binding_col = st.session_state.neoantigen_rl_binding_col
                immuno_col = st.session_state.neoantigen_rl_immunogenicity_col
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(
                    data[binding_col], 
                    data[immuno_col], 
                    alpha=0.6
                )
                ax.set_xlabel("Binding Affinity (nM)")
                ax.set_ylabel("Immunogenicity Score")
                ax.set_title("Binding Affinity vs Immunogenicity")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Invert x-axis since lower binding affinity is better
                ax.invert_xaxis()
                
                # Add a trend line
                try:
                    z = np.polyfit(
                        data[binding_col], 
                        data[immuno_col], 
                        1
                    )
                    p = np.poly1d(z)
                    ax.plot(
                        data[binding_col],
                        p(data[binding_col]),
                        "r--", alpha=0.8
                    )
                except:
                    pass
                
                st.pyplot(fig)
                
                # Correlation
                corr = data[binding_col].corr(data[immuno_col])
                st.info(f"Correlation between binding affinity and immunogenicity: {corr:.3f}")

def configure_agent_tab():
    st.header("2. Configure Reinforcement Learning Agent")
    
    # Check if data is loaded
    if st.session_state.neoantigen_rl_data is None:
        st.warning("Please load data in the 'Data Loading' tab first")
        return
    
    st.markdown("""
    Configure the Deep Q-Network (DQN) agent parameters for neoantigen candidate optimization.
    These settings control how the agent learns to select optimal epitopes.
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
        max_epitopes = st.slider("Max Epitopes to Select", 1, 20, 10, 1)
    
    with col2:
        st.subheader("Exploration Parameters")
        
        epsilon = st.slider("Initial Exploration Rate (Epsilon)", 0.1, 1.0, 1.0, 0.1)
        epsilon_min = st.slider("Minimum Exploration Rate", 0.01, 0.5, 0.01, 0.01)
        epsilon_decay = st.slider("Exploration Decay Rate", 0.9, 0.999, 0.995, 0.001)
        
        st.subheader("Feature Selection")
        data = st.session_state.neoantigen_rl_data
        
        if all(col in st.session_state for col in [
            "neoantigen_rl_binding_col", 
            "neoantigen_rl_expression_col", 
            "neoantigen_rl_immunogenicity_col"
        ]):
            binding_col = st.session_state.neoantigen_rl_binding_col
            expr_col = st.session_state.neoantigen_rl_expression_col
            immuno_col = st.session_state.neoantigen_rl_immunogenicity_col
            
            # Exclude target columns
            exclude_cols = [binding_col, expr_col, immuno_col]
            candidate_feature_cols = [col for col in data.columns if col not in exclude_cols]
            
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
            optimizer = NeoantigensOptimizer(
                data=data,
                feature_cols=feature_cols,
                binding_affinity_col=binding_col,
                expression_col=expr_col,
                immunogenicity_col=immuno_col,
                hidden_dim=hidden_dim,
                lr=learning_rate,
                max_epitopes=max_epitopes
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
                st.session_state.neoantigen_rl_optimizer = optimizer
                st.success("RL agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RL agent: {str(e)}")

def train_agent_tab():
    st.header("3. Train the RL Agent")
    
    # Check if optimizer is initialized
    if st.session_state.neoantigen_rl_optimizer is None:
        st.warning("Please initialize the RL agent in the 'Agent Configuration' tab first")
        return
    
    st.markdown("""
    Train the Deep Q-Network agent to learn optimal neoantigen selection strategies.
    The agent will learn to balance binding affinity, expression level, and immunogenicity.
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
                value=f"neoantigen_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
        # Training progress placeholder
        progress_placeholder = st.empty()
        training_metrics_placeholder = st.empty()
    
    with col2:
        st.subheader("Pre-trained Models")
        
        # List available models
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt') and f.startswith('neoantigen_rl_')]
        
        if model_files:
            selected_model = st.selectbox("Load Pre-trained Model", options=[''] + model_files)
            
            if selected_model and st.button("Load Model"):
                with st.spinner("Loading model..."):
                    try:
                        model_path = os.path.join(MODELS_DIR, selected_model)
                        loaded = st.session_state.neoantigen_rl_optimizer.load(model_path)
                        
                        if loaded:
                            st.success(f"Model {selected_model} loaded successfully!")
                            st.session_state.neoantigen_rl_optimizer.is_trained = True
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
                optimizer = st.session_state.neoantigen_rl_optimizer
                
                # Train model and collect history
                history = optimizer.train(
                    num_episodes=num_episodes,
                    update_freq=update_freq,
                    save_path=os.path.join(MODELS_DIR, model_name) if save_model else None
                )
                
                # Store history in session state
                st.session_state.neoantigen_rl_history = history
                
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

def select_epitopes_tab():
    st.header("4. Select Optimal Neoantigen Epitopes")
    
    # Check if optimizer is trained
    if st.session_state.neoantigen_rl_optimizer is None:
        st.warning("Please initialize and train the RL agent first")
        return
    
    optimizer = st.session_state.neoantigen_rl_optimizer
    
    if not optimizer.is_trained:
        st.warning("RL agent has not been trained. Please train the agent first.")
        return
    
    st.markdown("""
    Use the trained reinforcement learning agent to select the optimal neoantigen epitopes.
    The agent has learned to balance binding affinity, expression level, and immunogenicity 
    for optimal immunotherapy candidate selection.
    """)
    
    # Selection parameters
    num_epitopes = st.slider(
        "Number of Epitopes to Select", 
        1, 
        optimizer.max_epitopes, 
        min(5, optimizer.max_epitopes)
    )
    
    use_greedy = st.checkbox("Use Greedy Selection (no exploration)", value=True)
    
    # Selection button
    if st.button("Select Optimal Epitopes"):
        with st.spinner("Selecting optimal epitopes..."):
            try:
                # Set epsilon to 0 for greedy selection if requested
                original_epsilon = None
                if use_greedy and optimizer.agent:
                    original_epsilon = optimizer.agent.epsilon
                    optimizer.agent.epsilon = 0
                
                # Select epitopes
                selected_epitopes = optimizer.select_optimal_epitopes(num_epitopes=num_epitopes)
                
                # Restore original epsilon
                if use_greedy and optimizer.agent and original_epsilon is not None:
                    optimizer.agent.epsilon = original_epsilon
                
                # Store in session state
                st.session_state.neoantigen_rl_selected_epitopes = selected_epitopes
                
                st.success(f"Successfully selected {len(selected_epitopes)} optimal epitopes!")
            except Exception as e:
                st.error(f"Error selecting epitopes: {str(e)}")
    
    # Display selected epitopes
    if st.session_state.neoantigen_rl_selected_epitopes is not None:
        st.subheader("Selected Optimal Epitopes")
        
        selected = st.session_state.neoantigen_rl_selected_epitopes
        
        # Display metrics
        if all(col in st.session_state for col in [
            "neoantigen_rl_binding_col", 
            "neoantigen_rl_expression_col", 
            "neoantigen_rl_immunogenicity_col"
        ]):
            binding_col = st.session_state.neoantigen_rl_binding_col
            expr_col = st.session_state.neoantigen_rl_expression_col
            immuno_col = st.session_state.neoantigen_rl_immunogenicity_col
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Number of Epitopes", len(selected))
            
            with col2:
                # For binding affinity, lower is better
                data_avg = st.session_state.neoantigen_rl_data[binding_col].mean()
                selected_avg = selected[binding_col].mean()
                improvement = (data_avg - selected_avg) / data_avg * 100
                
                st.metric("Avg. Binding Affinity", f"{selected_avg:.1f} nM")
                st.metric("Binding Improvement", f"{improvement:.1f}%")
            
            with col3:
                # For expression, higher is better
                data_avg = st.session_state.neoantigen_rl_data[expr_col].mean()
                selected_avg = selected[expr_col].mean()
                improvement = (selected_avg - data_avg) / data_avg * 100
                
                st.metric("Avg. Expression", f"{selected_avg:.3f}")
                st.metric("Expression Improvement", f"{improvement:.1f}%")
                
            with col4:
                # For immunogenicity, higher is better
                data_avg = st.session_state.neoantigen_rl_data[immuno_col].mean()
                selected_avg = selected[immuno_col].mean()
                improvement = (selected_avg - data_avg) / data_avg * 100
                
                st.metric("Avg. Immunogenicity", f"{selected_avg:.3f}")
                st.metric("Immunogenicity Improvement", f"{improvement:.1f}%")
        
        # Display table
        st.dataframe(selected)
        
        # Download option
        csv = selected.to_csv(index=False)
        st.download_button(
            label="Download Selected Epitopes",
            data=csv,
            file_name=f"optimal_neoantigen_epitopes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.subheader("Visualization")
        
        if all(col in st.session_state for col in [
            "neoantigen_rl_binding_col", 
            "neoantigen_rl_expression_col", 
            "neoantigen_rl_immunogenicity_col"
        ]):
            binding_col = st.session_state.neoantigen_rl_binding_col
            expr_col = st.session_state.neoantigen_rl_expression_col
            immuno_col = st.session_state.neoantigen_rl_immunogenicity_col
            
            # 3D visualization if available
            try:
                import plotly.express as px
                
                # Create figure for 3D visualization
                fig = px.scatter_3d(
                    st.session_state.neoantigen_rl_data,
                    x=binding_col,
                    y=expr_col, 
                    z=immuno_col,
                    opacity=0.5,
                    color_discrete_sequence=['blue'] * len(st.session_state.neoantigen_rl_data),
                    title="Neoantigen Properties - All vs Selected"
                )
                
                # Add selected epitopes with a different color
                fig.add_scatter3d(
                    x=selected[binding_col],
                    y=selected[expr_col],
                    z=selected[immuno_col],
                    mode='markers',
                    marker=dict(size=5, color='red'),
                    name='Selected Epitopes'
                )
                
                # Update layout
                fig.update_layout(
                    scene=dict(
                        xaxis_title='Binding Affinity (nM)',
                        yaxis_title='Expression Level',
                        zaxis_title='Immunogenicity',
                    ),
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Plotly is required for 3D visualization.")
                
                # Fallback to 2D plots
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: Binding vs Immunogenicity
                all_data = st.session_state.neoantigen_rl_data
                axes[0].scatter(
                    all_data[binding_col], 
                    all_data[immuno_col], 
                    alpha=0.3, 
                    color='blue',
                    label='All Epitopes'
                )
                
                axes[0].scatter(
                    selected[binding_col], 
                    selected[immuno_col], 
                    alpha=0.8, 
                    color='red',
                    s=80,
                    label='Selected Epitopes'
                )
                
                axes[0].set_xlabel("Binding Affinity (nM)")
                axes[0].set_ylabel("Immunogenicity")
                axes[0].invert_xaxis()  # Lower binding is better
                axes[0].grid(True, linestyle='--', alpha=0.7)
                axes[0].legend()
                
                # Plot 2: Expression vs Immunogenicity
                axes[1].scatter(
                    all_data[expr_col], 
                    all_data[immuno_col], 
                    alpha=0.3, 
                    color='blue',
                    label='All Epitopes'
                )
                
                axes[1].scatter(
                    selected[expr_col], 
                    selected[immuno_col], 
                    alpha=0.8, 
                    color='red',
                    s=80,
                    label='Selected Epitopes'
                )
                
                axes[1].set_xlabel("Expression Level")
                axes[1].set_ylabel("Immunogenicity")
                axes[1].grid(True, linestyle='--', alpha=0.7)
                axes[1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Show reward distribution
            if 'rl_reward' in selected.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(all_data['rl_reward'], bins=30, alpha=0.5, label='All Epitopes', ax=ax)
                sns.histplot(selected['rl_reward'], bins=10, alpha=0.7, color='red', label='Selected Epitopes', ax=ax)
                ax.set_xlabel('Reward Score')
                ax.set_ylabel('Count')
                ax.set_title('Reward Distribution: Selected vs. All Epitopes')
                ax.legend()
                st.pyplot(fig)

def manage_models_tab():
    st.header("5. Model Management")
    
    st.markdown("""
    Manage your trained reinforcement learning models for neoantigen epitope optimization.
    You can download models for backup or upload previously trained models.
    """)
    
    # List available models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt') and f.startswith('neoantigen_rl_')]
    
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
    """Generate synthetic neoantigen epitope data for demo purposes"""
    
    # Generate random epitope sequences
    aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    epitope_length = 9
    
    sequences = [''.join(np.random.choice(aa) for _ in range(epitope_length)) for _ in range(n_samples)]
    
    # Generate MHC alleles
    mhc_alleles = ['HLA-A*02:01', 'HLA-A*01:01', 'HLA-B*07:02', 'HLA-B*08:01', 'HLA-C*07:01']
    alleles = np.random.choice(mhc_alleles, n_samples)
    
    # Generate binding affinities (lower is better, in nM)
    # Good binders typically < 500 nM, strong binders < 50 nM
    binding_affinity = np.random.exponential(scale=200, size=n_samples)
    
    # Generate expression levels (higher is better, 0-1 scale)
    expression_level = np.random.beta(2, 2, n_samples)
    
    # Generate immunogenicity scores (higher is better, 0-1 scale)
    # Create some inverse correlation with binding affinity
    norm_binding = np.clip(1 - (binding_affinity / 1000), 0, 1)
    immunogenicity = 0.6 * np.random.beta(3, 2, n_samples) + 0.4 * norm_binding
    immunogenicity = np.clip(immunogenicity, 0, 1)
    
    # Generate additional features
    hydrophobicity = np.random.normal(0, 1, n_samples)
    stability_score = np.random.uniform(0, 1, n_samples)
    
    # Generate mutation types
    mutation_types = ['missense', 'frameshift', 'fusion', 'splicing']
    mutation_type = np.random.choice(mutation_types, n_samples)
    
    # Create dataset
    data = pd.DataFrame({
        'epitope_seq': sequences,
        'mhc_allele': alleles,
        'binding_affinity': binding_affinity,
        'expression_level': expression_level,
        'immunogenicity': immunogenicity,
        'hydrophobicity': hydrophobicity,
        'stability_score': stability_score,
        'mutation_type': mutation_type
    })
    
    return data

if __name__ == "__main__":
    main()