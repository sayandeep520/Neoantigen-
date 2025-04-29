import streamlit as st
import pandas as pd
import numpy as np
import time
from utils.data_processor import (
    normalize_data, 
    handle_missing_values, 
    feature_engineering, 
    detect_outliers,
    compute_tumor_mutation_burden,
    compute_immune_infiltration_score
)
from utils.visualization import plot_normalized_distribution, plot_feature_importance, plot_pca

# Page configuration
st.set_page_config(
    page_title="Data Preprocessing | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Session state check
if 'datasets' not in st.session_state or not st.session_state['datasets']:
    st.warning("No data available for preprocessing. Please collect data first.")
    st.button("Go to Data Collection", on_click=lambda: st.switch_page("pages/1_Data_Collection.py"))
    st.stop()

if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = {}

# Main header
st.title("🔍 Data Preprocessing & Feature Engineering")
st.markdown("""
This module handles the cleaning, normalization, and feature engineering of multi-omics data
to prepare it for AI model training. Proper preprocessing is critical for the accuracy of downstream models.
""")

# Data selection for preprocessing
available_datasets = list(st.session_state['datasets'].keys())
selected_datasets = st.multiselect(
    "Select datasets for preprocessing",
    options=available_datasets,
    default=available_datasets
)

if not selected_datasets:
    st.warning("Please select at least one dataset for preprocessing.")
    st.stop()

# Preprocessing options
st.header("Preprocessing Options")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Preprocessing")
    handle_missing = st.checkbox("Handle Missing Values", value=True)
    missing_strategy = st.selectbox(
        "Missing Value Strategy",
        options=["KNN Imputation", "Mean/Median Imputation", "Remove Rows"],
        index=0,
        disabled=not handle_missing
    )
    
    normalize = st.checkbox("Normalize Data", value=True)
    normalization_method = st.selectbox(
        "Normalization Method",
        options=["Min-Max Scaling", "Z-Score Normalization", "Robust Scaling"],
        index=0,
        disabled=not normalize
    )
    
    outlier_detection = st.checkbox("Detect & Handle Outliers", value=True)
    outlier_method = st.selectbox(
        "Outlier Handling Method",
        options=["IQR Method", "Z-Score Method", "Isolation Forest"],
        index=0,
        disabled=not outlier_detection
    )

with col2:
    st.subheader("Feature Engineering")
    compute_tmb = st.checkbox("Compute Tumor Mutation Burden (TMB)", value=True)
    compute_immune = st.checkbox("Calculate Immune Infiltration Score", value=True)
    compute_gene_interactions = st.checkbox("Generate Gene Interaction Networks", value=False)
    dimensionality_reduction = st.checkbox("Apply Dimensionality Reduction", value=True)
    dim_reduction_method = st.selectbox(
        "Dimensionality Reduction Method",
        options=["PCA", "t-SNE", "UMAP"],
        index=0,
        disabled=not dimensionality_reduction
    )

# Start preprocessing
if st.button("Start Preprocessing"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, dataset_name in enumerate(selected_datasets):
        status_text.text(f"Processing {dataset_name} dataset...")
        dataset = st.session_state['datasets'][dataset_name]
        
        # Step 1: Handle missing values
        if handle_missing:
            status_text.text(f"Handling missing values in {dataset_name} dataset...")
            dataset = handle_missing_values(dataset, strategy=missing_strategy.lower().split()[0])
        
        # Step 2: Detect and handle outliers
        if outlier_detection:
            status_text.text(f"Detecting outliers in {dataset_name} dataset...")
            dataset, outliers_count = detect_outliers(dataset, method=outlier_method.split()[0].lower())
            st.info(f"Detected {outliers_count} outliers in {dataset_name} dataset.")
        
        # Step 3: Normalize data
        if normalize:
            status_text.text(f"Normalizing {dataset_name} dataset...")
            dataset = normalize_data(dataset, method=normalization_method.split()[0].lower())
        
        # Step 4: Feature Engineering
        status_text.text(f"Engineering features for {dataset_name} dataset...")
        
        # 4.1: Compute TMB if applicable
        if compute_tmb and dataset_name in ['tcga', 'icgc']:
            dataset = compute_tumor_mutation_burden(dataset)
        
        # 4.2: Compute immune infiltration score if applicable
        if compute_immune and dataset_name in ['tcga', 'gtex']:
            dataset = compute_immune_infiltration_score(dataset)
        
        # 4.3: Apply other feature engineering techniques
        dataset = feature_engineering(dataset, dataset_type=dataset_name)
        
        # Store preprocessed data
        st.session_state['preprocessed_data'][dataset_name] = dataset
        
        # Update progress
        progress_bar.progress((idx + 1) / len(selected_datasets))
    
    status_text.text("✅ Preprocessing completed successfully!")

# Data Visualization after preprocessing
if st.session_state['preprocessed_data']:
    st.header("Preprocessed Data Visualization")
    
    # Select dataset to visualize
    preprocessed_dataset_names = list(st.session_state['preprocessed_data'].keys())
    selected_viz_dataset = st.selectbox(
        "Select dataset to visualize",
        options=preprocessed_dataset_names
    )
    
    if selected_viz_dataset:
        preprocessed_data = st.session_state['preprocessed_data'][selected_viz_dataset]
        
        # Tabs for different visualizations
        viz_tabs = st.tabs(["Data Preview", "Distribution", "Feature Importance", "Dimensionality Reduction"])
        
        with viz_tabs[0]:
            st.subheader(f"Preprocessed {selected_viz_dataset.upper()} Data Preview")
            st.dataframe(preprocessed_data.head(10))
            st.write(f"Shape: {preprocessed_data.shape}")
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(preprocessed_data.describe())
        
        with viz_tabs[1]:
            st.subheader("Data Distribution After Normalization")
            plot_normalized_distribution(preprocessed_data, dataset_type=selected_viz_dataset)
        
        with viz_tabs[2]:
            st.subheader("Feature Importance")
            plot_feature_importance(preprocessed_data, dataset_type=selected_viz_dataset)
        
        with viz_tabs[3]:
            st.subheader("Dimensionality Reduction Visualization")
            plot_pca(preprocessed_data, dataset_type=selected_viz_dataset)
    
    # Next Steps
    st.markdown("---")
    st.header("Next Steps")
    st.markdown("""
    Now that you have preprocessed the multi-omics data, you can proceed to:
    1. **CRISPR Target Optimization** - Train AI models to identify optimal CRISPR targets
    2. **Neoantigen Prediction** - Identify potential neoantigens for immunotherapy
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        crispr_button = st.button("Proceed to CRISPR Target Optimization")
        if crispr_button:
            st.switch_page("pages/3_CRISPR_Target_Optimization.py")
    
    with col2:
        neoantigen_button = st.button("Proceed to Neoantigen Prediction")
        if neoantigen_button:
            st.switch_page("pages/4_Neoantigen_Prediction.py")
