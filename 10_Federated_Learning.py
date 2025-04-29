import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import time
from datetime import datetime

# Import the federated learning module
from utils.federated_learning import (
    FederatedModelManager, 
    FederatedDataSimulator,
    run_federated_learning_simulation
)

# Import the new training utilities
from utils.training_utils import (
    TrainingResults,
    train_federated_model,
    display_training_results,
    benchmark_ai_model,
    get_benchmark_data
)

# Set page config
st.set_page_config(
    page_title="Federated Learning - Multi-Institutional Collaboration",
    page_icon="🔄",
    layout="wide"
)

# Page title
st.title("🔄 Federated Learning for Multi-Institutional Collaboration")

# Introduction
st.markdown("""
## Secure Multi-Institutional Collaboration

Federated learning enables multiple institutions to collaboratively train machine learning models without 
sharing raw patient data, addressing privacy concerns in medical research. This approach allows:

- **Privacy Preservation**: Patient data never leaves the local institution
- **Collaborative Learning**: Models benefit from diverse datasets across institutions
- **Data Sovereignty**: Each institution maintains control over their data
- **Regulatory Compliance**: Better alignment with HIPAA, GDPR, and other regulations
""")

# Sidebar options
with st.sidebar:
    st.header("Federated Learning Settings")
    
    fl_mode = st.radio(
        "Mode",
        ["Simulation", "Real Data Federation"],
        help="Simulation mode uses synthetic data to demonstrate federated learning principles"
    )

    st.subheader("⚙️ Model Configuration")
    
    model_type = st.selectbox(
        "Model Type",
        ["random_forest", "linear", "svm", "gradient_boosting"],
        help="Select the type of machine learning model to use"
    )

    use_differential_privacy = st.checkbox(
        "Enable Differential Privacy", 
        value=True,
        help="Add noise to model updates to provide formal privacy guarantees"
    )
    
    # Initialize privacy_epsilon with a default value
    privacy_epsilon = 1.0
    
    if use_differential_privacy:
        privacy_epsilon = st.slider(
            "Privacy Budget (ε)", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Lower values provide stronger privacy but may reduce model accuracy"
        )

    communication_rounds = st.slider(
        "Communication Rounds", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Number of rounds for institutions to exchange model updates"
    )
    
    institution_id = st.text_input(
        "Institution ID", 
        value="local_institution",
        help="Identifier for the current institution"
    )

# Data management
st.header("📊 Data Management")

tab1, tab2 = st.tabs(["Simulation", "Real Data"])

# Simulation mode
with tab1:
    st.subheader("Federated Learning Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_institutions = st.slider(
            "Number of Institutions", 
            min_value=2, 
            max_value=10, 
            value=3
        )
        
        num_samples = st.slider(
            "Samples per Institution", 
            min_value=100, 
            max_value=2000, 
            value=500, 
            step=100
        )
        
        num_features = st.slider(
            "Number of Features", 
            min_value=5, 
            max_value=50, 
            value=20
        )
    
    with col2:
        num_classes = st.slider(
            "Number of Classes", 
            min_value=2, 
            max_value=5, 
            value=2
        )
        
        feature_overlap = st.slider(
            "Feature Overlap", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            help="Fraction of features shared across institutions"
        )
        
        distribution_shift = st.slider(
            "Distribution Shift", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            help="Magnitude of distribution shift between institutions"
        )
    
    sim_output_dir = "./federated_results"
    
    if st.button("Run Simulation", key="run_simulation"):
        with st.spinner("Running federated learning simulation..."):
            simulation_results = run_federated_learning_simulation(
                num_institutions=num_institutions,
                num_samples=num_samples,
                num_features=num_features,
                num_classes=num_classes,
                feature_overlap=feature_overlap,
                distribution_shift=distribution_shift,
                communication_rounds=communication_rounds,
                use_differential_privacy=use_differential_privacy,
                model_type=model_type,
                output_dir=sim_output_dir
            )
            
            st.session_state["simulation_results"] = simulation_results
            st.success(f"Simulation completed with {num_institutions} institutions over {communication_rounds} rounds!")

# Real data mode
with tab2:
    st.subheader("Real Multi-Institutional Data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your institution's data (CSV)", 
        type=["csv"], 
        help="Upload a CSV file with features and target variable"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} columns")
            
            # Data preview
            st.dataframe(data.head())
            
            # Select target column
            target_col = st.selectbox("Select target column", data.columns)
            
            # Data splitting
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
            
            if st.button("Prepare Data for Federated Learning"):
                # Split data
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Save train/test sets
                os.makedirs(f"./federated_data/{institution_id}", exist_ok=True)
                
                train_df = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)
                
                train_df.to_csv(f"./federated_data/{institution_id}/train.csv", index=False)
                test_df.to_csv(f"./federated_data/{institution_id}/test.csv", index=False)
                
                # Save metadata
                metadata = {
                    "institution_id": institution_id,
                    "num_samples": len(data),
                    "num_features": len(X.columns),
                    "num_classes": len(y.unique()),
                    "features": list(X.columns),
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(f"./federated_data/{institution_id}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                st.success(f"Data prepared for federated learning and saved to ./federated_data/{institution_id}/")
                
                # Initialize federated manager
                st.session_state["federated_manager"] = FederatedModelManager(
                    model_type=model_type,
                    institution_id=institution_id,
                    model_dir=f"./federated_models/{institution_id}",
                    add_differential_privacy=use_differential_privacy,
                    privacy_epsilon=privacy_epsilon if use_differential_privacy else 1.0,
                    communication_rounds=communication_rounds
                )
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

# Federated training
st.header("🧠 Federated Model Training")

tab3, tab4, tab5 = st.tabs(["Train & Share", "Aggregate Models", "Quick Train & Benchmark"])

# Quick Train & Benchmark tab
with tab5:
    st.subheader("Quick Federated Learning & Model Benchmarking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Run Complete Federated Learning Pipeline")
        st.write("This will simulate a complete federated learning process with multiple institutions.")
        
        quick_institutions = st.slider("Number of Institutions", 2, 8, 3)
        quick_samples = st.slider("Samples per Institution", 100, 1000, 500, 100)
        quick_rounds = st.slider("Communication Rounds", 1, 10, 3)
        
        if st.button("Train Federated Model", key="quick_train"):
            # Use our cached and enhanced training utility
            training_results = train_federated_model(
                num_institutions=quick_institutions,
                num_samples=quick_samples,
                num_features=20,
                communication_rounds=quick_rounds,
                model_type=model_type,
                use_differential_privacy=True
            )
            
            if training_results:
                # Store in session state for further use
                st.session_state['training_results'] = training_results
                
                # Use the enhanced display function
                display_training_results(training_results)
            else:
                st.error("Failed to obtain training results. Please check logs and try again.")
    
    with col2:
        st.write("#### AI Model Benchmarking")
        st.write("Compare our federated learning model against other state-of-the-art CRISPR AI models.")
        
        if st.button("Run Benchmarking Tests", key="run_benchmarking"):
            # Use the cached benchmark function for better performance
            benchmark_ai_model()

# Train and share tab
with tab3:
    st.subheader("Train Local Model & Share Updates")
    
    # For simulation results
    if "simulation_results" in st.session_state:
        st.info("Simulation has already trained models for all simulated institutions")
        
        # Show available institutions
        if os.path.exists(sim_output_dir):
            institutions = []
            try:
                # Load metrics from simulation
                with open(os.path.join(sim_output_dir, "metrics.json"), 'r') as f:
                    metrics = json.load(f)
                institutions = list(metrics.keys())
            except:
                pass
            
            if institutions:
                selected_institution = st.selectbox("Select Institution", institutions)
                
                try:
                    # Load final comparison
                    with open(os.path.join(sim_output_dir, "final_comparison.json"), 'r') as f:
                        final_comparison = json.load(f)
                    
                    if selected_institution in final_comparison:
                        comparison = final_comparison[selected_institution]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Local-only Accuracy", 
                                f"{comparison['local_final']['accuracy']:.4f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Federated Accuracy", 
                                f"{comparison['federated_final']['accuracy']:.4f}",
                                f"{comparison['improvement']['accuracy']:.4f}"
                            )
                except Exception as e:
                    st.error(f"Error loading comparison data: {str(e)}")
    
    # For real data
    elif "federated_manager" in st.session_state:
        st.write("Train your local model and share updates with other institutions")
        
        if st.button("Train Local Model"):
            try:
                # Load training data
                train_file = f"./federated_data/{institution_id}/train.csv"
                
                if os.path.exists(train_file):
                    train_df = pd.read_csv(train_file)
                    
                    # Get target column (last column)
                    target_col = train_df.columns[-1]
                    
                    X_train = train_df.drop(columns=[target_col]).values
                    y_train = train_df[target_col].values
                    
                    # Train model
                    with st.spinner("Training local model..."):
                        update_info = st.session_state["federated_manager"].train_local_model(X_train, y_train)
                    
                    st.success(f"Local model trained! Update ID: {update_info['update_id']}")
                    
                    # Test on local test data
                    test_file = f"./federated_data/{institution_id}/test.csv"
                    if os.path.exists(test_file):
                        test_df = pd.read_csv(test_file)
                        
                        X_test = test_df.drop(columns=[target_col]).values
                        y_test = test_df[target_col].values
                        
                        metrics = st.session_state["federated_manager"].evaluate_model(X_test, y_test)
                        
                        st.write("Local model performance:")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                    
                    # Store update info
                    if "update_info" not in st.session_state:
                        st.session_state["update_info"] = []
                    st.session_state["update_info"].append(update_info)
                else:
                    st.error(f"Training data not found at {train_file}")
            
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    else:
        st.info("Please upload data or run a simulation to enable model training")

# Aggregate models tab
with tab4:
    st.subheader("Aggregate Model Updates")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Handle update IDs directly
        update_ids = st.text_area(
            "Enter Update IDs (one per line)", 
            help="Enter update IDs from other institutions"
        )
    
    with col2:
        # Option to upload update files
        uploaded_updates = st.file_uploader(
            "Upload Model Updates", 
            type=["json", "pkl"], 
            accept_multiple_files=True,
            help="Upload model update files from other institutions"
        )
    
    if update_ids or uploaded_updates:
        if st.button("Aggregate Models"):
            try:
                # Process update IDs
                update_id_list = []
                if update_ids:
                    update_id_list = [id.strip() for id in update_ids.split("\n") if id.strip()]
                
                # Process uploaded updates
                if uploaded_updates:
                    for uploaded_file in uploaded_updates:
                        # Save the file
                        file_path = os.path.join(f"./federated_models/{institution_id}", uploaded_file.name)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract update ID from filename
                        if "_update.json" in uploaded_file.name:
                            update_id = uploaded_file.name.replace("_update.json", "")
                            update_id_list.append(update_id)
                
                if not update_id_list:
                    st.warning("No valid update IDs found")
                    st.stop()
                
                # Ensure federated manager exists
                if "federated_manager" not in st.session_state:
                    st.session_state["federated_manager"] = FederatedModelManager(
                        model_type=model_type,
                        institution_id=institution_id,
                        model_dir=f"./federated_models/{institution_id}",
                        add_differential_privacy=use_differential_privacy,
                        privacy_epsilon=privacy_epsilon if use_differential_privacy else 1.0,
                        communication_rounds=communication_rounds
                    )
                
                # Load and aggregate updates
                with st.spinner("Aggregating model updates..."):
                    loaded_updates = st.session_state["federated_manager"].load_model_updates(update_id_list)
                    st.session_state["federated_manager"].aggregate_models(loaded_updates)
                
                st.success(f"Successfully aggregated {len(loaded_updates)} model updates!")
                
                # Evaluate aggregated model on local test data
                test_file = f"./federated_data/{institution_id}/test.csv"
                if os.path.exists(test_file):
                    test_df = pd.read_csv(test_file)
                    
                    # Get target column (last column)
                    target_col = test_df.columns[-1]
                    
                    X_test = test_df.drop(columns=[target_col]).values
                    y_test = test_df[target_col].values
                    
                    metrics = st.session_state["federated_manager"].evaluate_model(X_test, y_test)
                    
                    st.write("Federated model performance:")
                    metrics_df = pd.DataFrame([metrics])
                    st.dataframe(metrics_df)
            
            except Exception as e:
                st.error(f"Error aggregating models: {str(e)}")
    
    # Show available models
    st.subheader("Available Federated Models")
    
    # For simulation results
    if "simulation_results" in st.session_state:
        if os.path.exists(sim_output_dir):
            try:
                # Load metrics from simulation
                with open(os.path.join(sim_output_dir, "metrics.json"), 'r') as f:
                    metrics = json.load(f)
                
                # Create an enhanced line plot comparing local vs federated performance
                st.write("### Performance across Communication Rounds")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Define better color palettes
                local_palette = plt.cm.Blues(np.linspace(0.5, 0.8, 10))
                fed_palette = plt.cm.Reds(np.linspace(0.5, 0.8, 10))
                
                # Counter for color selection
                color_idx = 0
                
                for institution_id, metrics_list in metrics.items():
                    local_metrics = [m for m in metrics_list if m.get("local_only", False)]
                    federated_metrics = [m for m in metrics_list if not m.get("local_only", False)]
                    
                    if local_metrics:
                        local_rounds = [m["round"] for m in local_metrics]
                        local_accuracies = [m["accuracy"] for m in local_metrics]
                        ax.plot(local_rounds, local_accuracies, 'o--', 
                               color=local_palette[color_idx % len(local_palette)],
                               linewidth=2, markersize=8,
                               label=f"{institution_id} (Local)")
                    
                    if federated_metrics:
                        fed_rounds = [m["round"] for m in federated_metrics]
                        fed_accuracies = [m["accuracy"] for m in federated_metrics]
                        ax.plot(fed_rounds, fed_accuracies, 's-', 
                               color=fed_palette[color_idx % len(fed_palette)],
                               linewidth=2.5, markersize=8,
                               label=f"{institution_id} (Federated)")
                    
                    color_idx += 1
                
                # Add shaded region showing improvement
                institution_ids = list(metrics.keys())
                if institution_ids:
                    # Get metrics from the first institution to visualize improvement trend
                    first_id = institution_ids[0]
                    first_metrics = metrics[first_id]
                    
                    local_metrics = [m for m in first_metrics if m.get("local_only", False)]
                    federated_metrics = [m for m in first_metrics if not m.get("local_only", False)]
                    
                    if local_metrics and federated_metrics:
                        # Get values for the same rounds
                        common_rounds = []
                        local_accs = []
                        fed_accs = []
                        
                        for fed_m in federated_metrics:
                            r = fed_m["round"]
                            matching_local = [m for m in local_metrics if m["round"] == r]
                            if matching_local:
                                common_rounds.append(r)
                                fed_accs.append(fed_m["accuracy"])
                                local_accs.append(matching_local[0]["accuracy"])
                        
                        if common_rounds:
                            # Add shaded region between curves
                            ax.fill_between(common_rounds, local_accs, fed_accs, 
                                          color='green', alpha=0.1, 
                                          label="Accuracy Improvement")
                
                # Improve styling
                ax.set_xlabel("Communication Round", fontsize=12)
                ax.set_ylabel("Accuracy", fontsize=12)
                ax.set_title("Federated vs. Local-only Learning Performance", 
                           fontsize=14, fontweight='bold')
                
                # Place legend outside the plot to avoid overcrowding
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                        ncol=3, frameon=True, fontsize=10)
                
                # Add grid and styling
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add annotation for best performance
                best_acc = 0
                best_round = 0
                best_inst = ""
                
                for inst_id, m_list in metrics.items():
                    for m in m_list:
                        if not m.get("local_only", False) and m["accuracy"] > best_acc:
                            best_acc = m["accuracy"]
                            best_round = m["round"]
                            best_inst = inst_id
                
                if best_acc > 0:
                    ax.annotate(f"Best: {best_acc:.3f}",
                              xy=(best_round, best_acc),
                              xytext=(best_round + 0.5, best_acc + 0.02),
                              arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                              fontsize=10,
                              color="green",
                              fontweight="bold")
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error displaying simulation results: {str(e)}")
    
    # For real data
    elif "federated_manager" in st.session_state:
        try:
            available_models = st.session_state["federated_manager"].get_available_models()
            
            if available_models:
                models_df = pd.DataFrame(available_models)
                st.dataframe(models_df)
                
                selected_round = st.selectbox(
                    "Select model round to load", 
                    options=[m["round"] for m in available_models]
                )
                
                if st.button("Load Selected Model"):
                    success = st.session_state["federated_manager"].load_model(selected_round)
                    
                    if success:
                        st.success(f"Successfully loaded model from round {selected_round}")
                        
                        # Evaluate loaded model
                        test_file = f"./federated_data/{institution_id}/test.csv"
                        if os.path.exists(test_file):
                            test_df = pd.read_csv(test_file)
                            
                            # Get target column (last column)
                            target_col = test_df.columns[-1]
                            
                            X_test = test_df.drop(columns=[target_col]).values
                            y_test = test_df[target_col].values
                            
                            metrics = st.session_state["federated_manager"].evaluate_model(X_test, y_test)
                            
                            st.write("Loaded model performance:")
                            metrics_df = pd.DataFrame([metrics])
                            st.dataframe(metrics_df)
                    else:
                        st.error(f"Failed to load model from round {selected_round}")
            else:
                st.info("No federated models available yet")
        
        except Exception as e:
            st.error(f"Error loading available models: {str(e)}")
    
    else:
        st.info("Please upload data or run a simulation to view available models")

# Visualization and Analysis
st.header("📈 Results Visualization")

if "simulation_results" in st.session_state:
    st.subheader("Federated Learning Impact")
    
    try:
        # Load final comparison from simulation
        with open(os.path.join(sim_output_dir, "final_comparison.json"), 'r') as f:
            final_comparison = json.load(f)
        
        # Create comparison dataframe
        comparison_data = []
        
        for institution_id, comparison in final_comparison.items():
            row = {
                "Institution": institution_id,
                "Local Accuracy": comparison["local_final"]["accuracy"],
                "Federated Accuracy": comparison["federated_final"]["accuracy"],
                "Improvement": comparison["improvement"]["accuracy"]
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.dataframe(comparison_df)
        
        # Visualize comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(comparison_df))
            width = 0.35
            
            ax.bar(x - width/2, comparison_df["Local Accuracy"], width, label="Local-only")
            ax.bar(x + width/2, comparison_df["Federated Accuracy"], width, label="Federated")
            
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df["Institution"])
            ax.set_ylabel("Accuracy")
            ax.set_title("Local vs Federated Model Accuracy")
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            # Improvement visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.barplot(x="Institution", y="Improvement", data=comparison_df, ax=ax)
            
            ax.set_ylabel("Accuracy Improvement")
            ax.set_title("Improvement from Federated Learning")
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            for i, v in enumerate(comparison_df["Improvement"]):
                ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
            
            st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error visualizing results: {str(e)}")

else:
    st.info("Run a simulation or train with real data to see visualizations")

# Technical details
with st.expander("🔍 Technical Details"):
    st.markdown("""
    ### Federated Learning Implementation
    
    This implementation uses the following key components:
    
    1. **FederatedModelManager** - Handles the federated learning process for each institution
       - Trains local models on institutional data
       - Creates model updates with optional differential privacy
       - Aggregates model updates from multiple institutions
       
    2. **FederatedDataSimulator** - Simulates multi-institutional data for testing
       - Creates synthetic datasets with controllable parameters
       - Simulates distribution shifts between institutions
       - Allows testing of federated learning without real data
       
    3. **Differential Privacy** - Protects individual patient privacy
       - Adds calibrated noise to model updates
       - Provides mathematical privacy guarantees
       - Prevents reconstruction of training data
       
    4. **Model Aggregation** - Combines models from multiple institutions
       - Uses federated averaging to combine model parameters
       - Weights updates based on dataset size
       - Preserves model performance while enhancing privacy
    """)

# User guide
with st.expander("📖 User Guide"):
    st.markdown("""
    ### Getting Started with Federated Learning
    
    **Option 1: Simulation Mode**
    1. Configure simulation parameters in the sidebar
    2. Set the number of institutions, samples, and features
    3. Click "Run Simulation" to see how federated learning improves performance
    4. Analyze the results in the visualization section
    
    **Option 2: Real Data Mode**
    1. Upload your institution's dataset (CSV file)
    2. Select the target column and configure test split
    3. Click "Prepare Data for Federated Learning"
    4. Train your local model and obtain an update ID
    5. Share your update ID with collaborating institutions
    6. Collect update IDs from other institutions
    7. Enter update IDs in the "Aggregate Models" tab and click "Aggregate Models"
    8. Evaluate the performance of the federated model
    
    **Best Practices**
    - Use differential privacy to enhance data protection
    - Start with a small number of communication rounds and increase as needed
    - Validate model performance on local test data before and after federation
    - Consider data distribution and feature overlap when interpreting results
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 AI-Driven CRISPR Cancer Immunotherapy Platform | Federated Learning Module")