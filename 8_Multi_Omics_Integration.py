import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from utils.multi_omics_processor import data_collector, data_preprocessor, data_storage
from utils.file_upload import upload_multi_omics_data, list_available_datasets

# Page configuration
st.set_page_config(
    page_title="Multi-Omics Integration | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Function to create sample data
def create_example_dataset(rows=100, genomic_features=10, transcriptomic_features=8):
    """Generate a sample multi-omics dataset for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Create sample IDs
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(1, rows+1)]
    
    # Create genomic features (e.g., mutation data)
    genomic_data = np.random.choice([0, 1], size=(rows, genomic_features), p=[0.9, 0.1])
    genomic_cols = [f"Mutation_{i}" for i in range(1, genomic_features+1)]
    
    # Create transcriptomic features (e.g., gene expression)
    expression_data = np.random.normal(0, 1, size=(rows, transcriptomic_features))
    expression_cols = [f"Gene_{i}" for i in range(1, transcriptomic_features+1)]
    
    # Create clinical features
    age = np.random.normal(60, 15, size=rows).astype(int)
    age = np.clip(age, 18, 90)  # Clip to reasonable age range
    
    gender = np.random.choice(['M', 'F'], size=rows)
    tumor_stage = np.random.choice(['I', 'II', 'III', 'IV'], size=rows, p=[0.2, 0.3, 0.3, 0.2])
    
    # Add some missing values
    expression_data[np.random.choice([False, True], size=expression_data.shape, p=[0.95, 0.05])] = np.nan
    
    # Create response variable (for demonstration)
    # Simplified model: response influenced by age, stage, and some genomic and transcriptomic features
    response_prob = (
        0.3 * (age > 65).astype(int) +
        0.2 * (tumor_stage == 'III').astype(int) + 
        0.3 * (tumor_stage == 'IV').astype(int) +
        0.4 * genomic_data[:, 0] +
        0.3 * genomic_data[:, 2] -
        0.2 * (expression_data[:, 0] > 0).astype(int) +
        0.4 * (expression_data[:, 3] > 1).astype(int)
    ) / 2.0
    
    response = np.random.binomial(1, np.clip(response_prob, 0.1, 0.9))
    
    # Combine all data
    df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'Age': age,
        'Gender': gender,
        'Tumor_Stage': tumor_stage,
        'Treatment_Response': response
    })
    
    # Add genomic and transcriptomic data
    for i, col in enumerate(genomic_cols):
        df[col] = genomic_data[:, i]
        
    for i, col in enumerate(expression_cols):
        df[col] = expression_data[:, i]
    
    return df

def main():
    """Main function for Multi-Omics Integration page"""
    st.title("🔬 Multi-Omics Data Integration")
    st.markdown("""
    This module allows for integration and preprocessing of multi-omics data, including genomic,
    transcriptomic, and proteomic data sources. The integration of multiple data types enables
    more robust and comprehensive AI-driven cancer immunotherapy analysis.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Preprocessing", "Integration & Analysis"])
    
    # Session state for datasets
    if 'integrated_data' not in st.session_state:
        st.session_state['integrated_data'] = None
    
    with tab1:
        st.header("Multi-Omics Data Overview")
        
        # Add data upload region
        with st.expander("Upload New Data", expanded=True):
            # Call the upload function from utils.file_upload
            upload_success = upload_multi_omics_data()
            if upload_success:
                st.success("Data uploaded successfully! It will now appear in the Available Datasets section below.")
                st.rerun()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Available Datasets")
            
            # List datasets from storage
            datasets_df = data_storage.list_datasets()
            
            if not datasets_df.empty:
                # Show datasets
                st.dataframe(datasets_df[[
                    'id', 'name', 'data_type', 'sample_count', 
                    'feature_count', 'created_at'
                ]], height=300)
                
                # Select datasets for integration
                selected_datasets = st.multiselect(
                    "Select datasets for integration",
                    options=datasets_df['id'].tolist(),
                    format_func=lambda x: f"{datasets_df[datasets_df['id'] == x]['name'].values[0]} ({datasets_df[datasets_df['id'] == x]['data_type'].values[0]})"
                )
                
                if selected_datasets and st.button("Load Selected Datasets"):
                    multi_omics_data = {}
                    for dataset_id in selected_datasets:
                        dataset_info = datasets_df[datasets_df['id'] == dataset_id].iloc[0]
                        df = data_storage.load_dataset(dataset_id=dataset_id)
                        if not df.empty:
                            multi_omics_data[dataset_info['name']] = {
                                'data': df,
                                'type': dataset_info['data_type']
                            }
                            st.success(f"Loaded {dataset_info['name']} with {df.shape[0]} samples and {df.shape[1]} features")
                    
                    # Store in session state
                    st.session_state['multi_omics_data'] = multi_omics_data
                    
                    # Preview data
                    if multi_omics_data:
                        st.subheader("Data Preview")
                        data_tabs = st.tabs(list(multi_omics_data.keys()))
                        for i, (name, data_dict) in enumerate(multi_omics_data.items()):
                            with data_tabs[i]:
                                st.write(f"**Type:** {data_dict['type']}")
                                st.dataframe(data_dict['data'].head())
            else:
                st.info("No datasets available. Use the 'Create Example Dataset' button or upload data from the Data Collection page.")
        
        with col2:
            st.subheader("Create Example Dataset")
            st.write("Generate a sample multi-omics dataset for demonstration purposes.")
            
            # Parameters for sample data
            samples = st.slider("Number of samples", min_value=20, max_value=500, value=100, step=10)
            genomic_features = st.slider("Number of genomic features", min_value=5, max_value=50, value=10, step=5)
            transcriptomic_features = st.slider("Number of transcriptomic features", min_value=5, max_value=50, value=8, step=5)
            
            if st.button("Generate Example Dataset"):
                with st.spinner("Generating example dataset..."):
                    example_data = create_example_dataset(
                        rows=samples,
                        genomic_features=genomic_features,
                        transcriptomic_features=transcriptomic_features
                    )
                    
                    # Preview the data
                    st.success(f"Generated example dataset with {example_data.shape[0]} samples and {example_data.shape[1]} features")
                    st.dataframe(example_data.head())
                    
                    # Save to data storage
                    dataset_id = data_storage.save_dataset(
                        name="Example_MultiOmics",
                        data=example_data,
                        data_type="integrated",
                        description="Example multi-omics dataset with genomic and transcriptomic features",
                        format="csv"
                    )
                    
                    st.session_state['example_dataset_id'] = dataset_id
                    st.info(f"Dataset saved with ID: {dataset_id}. It can now be selected from the Available Datasets list.")
    
    with tab2:
        st.header("Data Preprocessing")
        
        # Check if we have data to work with
        if 'multi_omics_data' in st.session_state and st.session_state['multi_omics_data']:
            data_dict = st.session_state['multi_omics_data']
            dataset_names = list(data_dict.keys())
            
            selected_dataset = st.selectbox(
                "Select dataset to preprocess",
                options=dataset_names
            )
            
            if selected_dataset:
                df = data_dict[selected_dataset]['data']
                
                st.subheader(f"Preprocessing {selected_dataset}")
                
                # Missing value information
                missing_values = df.isnull().sum()
                cols_with_missing = missing_values[missing_values > 0]
                
                if not cols_with_missing.empty:
                    st.warning(f"Dataset contains {cols_with_missing.sum()} missing values in {len(cols_with_missing)} columns")
                    
                    # Display columns with missing values
                    st.write("Columns with missing values:")
                    missing_df = pd.DataFrame({
                        'Column': cols_with_missing.index,
                        'Missing Values': cols_with_missing.values,
                        'Percentage': (cols_with_missing.values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df)
                    
                    # Missing value handling options
                    st.subheader("Handle Missing Values")
                    imputation_method = st.selectbox(
                        "Select imputation method",
                        options=["knn", "mean", "median", "most_frequent"],
                        index=0
                    )
                    
                    if st.button("Impute Missing Values"):
                        with st.spinner("Imputing missing values..."):
                            imputed_df = data_preprocessor.handle_missing_values(df, strategy=imputation_method)
                            
                            # Update the dataset in session state
                            st.session_state['multi_omics_data'][selected_dataset]['data'] = imputed_df
                            
                            st.success("Missing values successfully imputed!")
                            
                            # Show before/after comparison
                            st.write("Data after imputation:")
                            st.dataframe(imputed_df.head())
                else:
                    st.success("Dataset has no missing values!")
                
                # Normalization options
                st.subheader("Data Normalization")
                st.write("Normalize numerical features to a common scale.")
                
                normalization_method = st.selectbox(
                    "Select normalization method",
                    options=["standard", "minmax", "robust"],
                    index=0
                )
                
                if st.button("Normalize Data"):
                    with st.spinner("Normalizing data..."):
                        # Get current data version (original or already imputed)
                        current_df = st.session_state['multi_omics_data'][selected_dataset]['data']
                        normalized_df = data_preprocessor.normalize_data(
                            current_df,
                            method=normalization_method
                        )
                        
                        # Update the dataset in session state
                        st.session_state['multi_omics_data'][selected_dataset]['data'] = normalized_df
                        
                        st.success("Data successfully normalized!")
                        
                        # Show before/after comparison for a few numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # First 5 numeric columns
                        if len(numeric_cols) > 0:
                            st.write("Comparison of first few numeric columns before/after normalization:")
                            
                            fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(10, 3*len(numeric_cols)))
                            if len(numeric_cols) == 1:
                                axes = axes.reshape(1, 2)
                                
                            for i, col in enumerate(numeric_cols):
                                # Before normalization
                                sns.histplot(df[col].dropna(), kde=True, ax=axes[i, 0])
                                axes[i, 0].set_title(f"{col} - Before")
                                
                                # After normalization
                                sns.histplot(normalized_df[col].dropna(), kde=True, ax=axes[i, 1])
                                axes[i, 1].set_title(f"{col} - After")
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                # Outlier detection
                st.subheader("Outlier Detection")
                st.write("Detect and optionally remove outliers from the dataset.")
                
                outlier_method = st.selectbox(
                    "Select outlier detection method",
                    options=["isolation_forest", "zscore"],
                    index=0
                )
                
                contamination = st.slider(
                    "Expected outlier proportion", 
                    min_value=0.01, 
                    max_value=0.2, 
                    value=0.05,
                    step=0.01
                )
                
                if st.button("Detect Outliers"):
                    with st.spinner("Detecting outliers..."):
                        # Get the current data version
                        current_df = st.session_state['multi_omics_data'][selected_dataset]['data']
                        
                        # Detect outliers
                        inlier_df = data_preprocessor.detect_outliers(
                            current_df,
                            method=outlier_method,
                            contamination=contamination
                        )
                        
                        outlier_count = len(current_df) - len(inlier_df)
                        
                        st.warning(f"Detected {outlier_count} outliers ({outlier_count/len(current_df)*100:.1f}% of the data)")
                        
                        # Option to remove outliers
                        if st.button("Remove Outliers", key="remove_outliers"):
                            st.session_state['multi_omics_data'][selected_dataset]['data'] = inlier_df
                            st.success(f"Removed {outlier_count} outliers. Dataset now has {len(inlier_df)} samples.")
                
                # Dimension reduction
                st.subheader("Dimension Reduction")
                st.write("Reduce the dimensionality of the dataset while preserving important information.")
                
                dimension_method = st.selectbox(
                    "Select dimension reduction method",
                    options=["pca", "selectk"],
                    index=0
                )
                
                n_components = st.slider(
                    "Number of components/features to keep",
                    min_value=2,
                    max_value=min(50, df.select_dtypes(include=[np.number]).shape[1]),
                    value=min(10, df.select_dtypes(include=[np.number]).shape[1])
                )
                
                if st.button("Reduce Dimensions"):
                    with st.spinner("Reducing dimensions..."):
                        # Get the current data version
                        current_df = st.session_state['multi_omics_data'][selected_dataset]['data']
                        
                        # Apply dimension reduction
                        reduced_df = data_preprocessor.reduce_dimensions(
                            current_df,
                            n_components=n_components,
                            method=dimension_method
                        )
                        
                        st.success(f"Dimension reduction complete. From {current_df.shape[1]} to {reduced_df.shape[1]} features.")
                        
                        # Update the dataset in session state with a new name to preserve the original
                        new_name = f"{selected_dataset}_reduced"
                        st.session_state['multi_omics_data'][new_name] = {
                            'data': reduced_df,
                            'type': f"{data_dict[selected_dataset]['type']}_reduced"
                        }
                        
                        # Show the reduced data
                        st.write("Reduced dataset preview:")
                        st.dataframe(reduced_df.head())
                        
                        # Visualization for PCA
                        if dimension_method == "pca" and "PC1" in reduced_df.columns and "PC2" in reduced_df.columns:
                            st.subheader("PCA Visualization")
                            
                            # See if we have a categorical target variable to color by
                            potential_targets = reduced_df.select_dtypes(include=['object', 'category']).columns.tolist()
                            potential_targets += ['Treatment_Response']  # Add our known target from example data
                            
                            color_by = None
                            for target in potential_targets:
                                if target in reduced_df.columns and reduced_df[target].nunique() <= 10:
                                    color_by = target
                                    break
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            if color_by:
                                # Plot with coloring by category
                                categories = reduced_df[color_by].unique()
                                for category in categories:
                                    subset = reduced_df[reduced_df[color_by] == category]
                                    ax.scatter(subset['PC1'], subset['PC2'], label=category, alpha=0.7)
                                ax.legend()
                                ax.set_title(f"PCA Colored by {color_by}")
                            else:
                                # Simple scatter plot
                                ax.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=0.7)
                                ax.set_title("PCA Visualization")
                                
                            ax.set_xlabel("PC1")
                            ax.set_ylabel("PC2")
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig)
                
                # Save processed data
                st.subheader("Save Processed Data")
                
                save_name = st.text_input("Dataset name for saving", f"{selected_dataset}_processed")
                save_description = st.text_area("Description", "Processed multi-omics dataset")
                
                if st.button("Save Processed Dataset"):
                    with st.spinner("Saving dataset..."):
                        # Get the current data version
                        current_df = st.session_state['multi_omics_data'][selected_dataset]['data']
                        
                        # Save to storage
                        dataset_id = data_storage.save_dataset(
                            name=save_name,
                            data=current_df,
                            data_type=f"{data_dict[selected_dataset]['type']}_processed",
                            description=save_description,
                            format="csv"
                        )
                        
                        st.success(f"Dataset saved with ID: {dataset_id}")
        else:
            st.info("Please load or create a dataset in the 'Data Overview' tab first.")
    
    with tab3:
        st.header("Multi-Omics Integration & Analysis")
        
        # Check if we have multiple datasets to integrate
        if 'multi_omics_data' in st.session_state and len(st.session_state['multi_omics_data']) >= 2:
            data_dict = st.session_state['multi_omics_data']
            dataset_names = list(data_dict.keys())
            
            st.subheader("Select Datasets to Integrate")
            
            selected_datasets = st.multiselect(
                "Select at least two datasets",
                options=dataset_names,
                default=dataset_names[:2] if len(dataset_names) >= 2 else []
            )
            
            if len(selected_datasets) >= 2:
                # Identify common sample identifiers across datasets
                st.subheader("Sample ID Mapping")
                
                id_column_options = {}
                for dataset in selected_datasets:
                    df = data_dict[dataset]['data']
                    # Guess potential ID columns based on name patterns
                    potential_ids = [col for col in df.columns if any(id_term in col.lower() for id_term in ['id', 'sample', 'patient', 'subject'])]
                    
                    if not potential_ids:
                        potential_ids = df.columns.tolist()
                        
                    id_column_options[dataset] = st.selectbox(
                        f"Select ID column for {dataset}",
                        options=potential_ids,
                        index=0 if potential_ids else None
                    )
                
                # Integration method
                st.subheader("Integration Method")
                
                integration_method = st.radio(
                    "Select integration method",
                    options=["Inner Join", "Outer Join"],
                    index=0,
                    help="Inner Join: Keep only samples present in all datasets. Outer Join: Keep all samples, fill missing values."
                )
                
                if st.button("Integrate Datasets"):
                    with st.spinner("Integrating datasets..."):
                        # Check if all ID columns are selected
                        if None in id_column_options.values():
                            st.error("Please select ID columns for all datasets.")
                        else:
                            # Start with the first dataset
                            first_dataset = selected_datasets[0]
                            integrated_df = data_dict[first_dataset]['data'].copy()
                            id_col_first = id_column_options[first_dataset]
                            
                            # Rename the ID column to a standard name for joining
                            integrated_df.rename(columns={id_col_first: 'Sample_ID'}, inplace=True)
                            
                            # Track the source of columns
                            column_sources = {col: first_dataset for col in integrated_df.columns if col != 'Sample_ID'}
                            
                            # Integrate remaining datasets
                            for dataset in selected_datasets[1:]:
                                df = data_dict[dataset]['data'].copy()
                                id_col = id_column_options[dataset]
                                
                                # Rename the ID column to match
                                df.rename(columns={id_col: 'Sample_ID'}, inplace=True)
                                
                                # Handle duplicate column names
                                for col in df.columns:
                                    if col in integrated_df.columns and col != 'Sample_ID':
                                        new_col = f"{col}_{dataset}"
                                        df.rename(columns={col: new_col}, inplace=True)
                                        # Update source tracking
                                        column_sources[new_col] = dataset
                                    elif col != 'Sample_ID':
                                        # Track source for non-duplicate columns
                                        column_sources[col] = dataset
                                
                                # Merge based on selected method
                                if integration_method == "Inner Join":
                                    integrated_df = pd.merge(
                                        integrated_df, df, on='Sample_ID', how='inner'
                                    )
                                else:  # Outer Join
                                    integrated_df = pd.merge(
                                        integrated_df, df, on='Sample_ID', how='outer'
                                    )
                            
                            # Analyze the integration results
                            n_samples = len(integrated_df)
                            n_features = integrated_df.shape[1] - 1  # Excluding Sample_ID
                            
                            st.success(f"Integration complete! Result has {n_samples} samples and {n_features} features.")
                            
                            # Save the integrated result in session state
                            st.session_state['integrated_data'] = integrated_df
                            st.session_state['column_sources'] = column_sources
                            
                            # Show preview of integrated data
                            st.subheader("Integrated Data Preview")
                            st.dataframe(integrated_df.head())
                            
                            # Save integrated dataset option
                            save_name = st.text_input("Name for integrated dataset", "Integrated_MultiOmics")
                            save_description = st.text_area("Description for integrated dataset", 
                                                          f"Integrated dataset from {', '.join(selected_datasets)}")
                            
                            if st.button("Save Integrated Dataset"):
                                dataset_id = data_storage.save_dataset(
                                    name=save_name,
                                    data=integrated_df,
                                    data_type="integrated",
                                    description=save_description,
                                    format="csv"
                                )
                                
                                st.success(f"Integrated dataset saved with ID: {dataset_id}")
                
                # Analysis section (only if integrated data exists)
                if 'integrated_data' in st.session_state and st.session_state['integrated_data'] is not None:
                    st.header("Analysis of Integrated Data")
                    
                    integrated_df = st.session_state['integrated_data']
                    column_sources = st.session_state.get('column_sources', {})
                    
                    # Data source distribution
                    if column_sources:
                        st.subheader("Feature Source Distribution")
                        source_counts = pd.Series(column_sources).value_counts()
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        source_counts.plot(kind='bar', ax=ax)
                        ax.set_title("Number of Features from Each Data Source")
                        ax.set_ylabel("Feature Count")
                        ax.set_xlabel("Data Source")
                        
                        st.pyplot(fig)
                    
                    # Correlation analysis (if we have numeric data)
                    numeric_cols = integrated_df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 1:
                        st.subheader("Feature Correlation Analysis")
                        
                        # Option to select specific columns
                        max_cols_for_corr = min(20, len(numeric_cols))  # Limit for readability
                        
                        if len(numeric_cols) > max_cols_for_corr:
                            selected_cols = st.multiselect(
                                "Select columns for correlation analysis (max 20 recommended)",
                                options=numeric_cols,
                                default=list(numeric_cols[:max_cols_for_corr])
                            )
                        else:
                            selected_cols = numeric_cols
                        
                        if selected_cols and len(selected_cols) > 1:
                            # Calculate correlation matrix
                            corr_matrix = integrated_df[selected_cols].corr()
                            
                            # Visualize correlation matrix
                            fig, ax = plt.subplots(figsize=(12, 10))
                            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
                            ax.set_title("Feature Correlation Matrix")
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Show highest correlations
                            st.subheader("Top Feature Correlations")
                            
                            # Get upper triangle of correlation matrix
                            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                            
                            # Find highest absolute correlations
                            highest_corrs = upper.abs().unstack().sort_values(ascending=False).head(10)
                            
                            if not highest_corrs.empty:
                                corr_df = pd.DataFrame(highest_corrs).reset_index()
                                corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
                                corr_df['Correlation'] = corr_df['Correlation'].round(3)
                                
                                st.dataframe(corr_df)
                    
                    # Predictive target selection (if example data with Treatment_Response)
                    potential_targets = integrated_df.columns[integrated_df.columns.str.contains('Response|Outcome|Survival|Target', case=False)]
                    
                    if not potential_targets.empty or 'Treatment_Response' in integrated_df.columns:
                        target_col = 'Treatment_Response' if 'Treatment_Response' in integrated_df.columns else potential_targets[0]
                        
                        st.subheader(f"Analysis of {target_col} Predictors")
                        
                        # For categorical target
                        if integrated_df[target_col].dtype == 'object' or integrated_df[target_col].nunique() < 5:
                            target_categories = integrated_df[target_col].value_counts()
                            
                            # Plot distribution
                            fig, ax = plt.subplots(figsize=(8, 6))
                            target_categories.plot(kind='bar', ax=ax)
                            ax.set_title(f"Distribution of {target_col}")
                            ax.set_ylabel("Count")
                            
                            st.pyplot(fig)
                            
                            # Feature importance analysis for each source
                            st.subheader("Feature Importance by Data Source")
                            st.write("This shows which data sources contain the most predictive features.")
                            
                            # Group feature importance by source
                            source_importance = {}
                            
                            # Calculate a simple correlation-based importance for numeric features
                            for col in numeric_cols:
                                if col != target_col:
                                    # Get source
                                    source = column_sources.get(col, "Unknown")
                                    
                                    # Calculate correlation with target
                                    if integrated_df[target_col].dtype == 'object':
                                        # For categorical target, use mean by category
                                        cat_means = integrated_df.groupby(target_col)[col].mean()
                                        # Calculate range of means as a measure of predictive power
                                        importance = abs(cat_means.max() - cat_means.min())
                                    else:
                                        # For numeric target, use correlation
                                        importance = abs(integrated_df[col].corr(integrated_df[target_col]))
                                    
                                    # Add to source importance
                                    if source not in source_importance:
                                        source_importance[source] = []
                                    source_importance[source].append(importance)
                            
                            # Calculate average importance by source
                            avg_importance = {source: np.mean(importances) for source, importances in source_importance.items()}
                            
                            # Create DataFrame for visualization
                            if avg_importance:
                                importance_df = pd.DataFrame({
                                    'Data Source': list(avg_importance.keys()),
                                    'Average Importance': list(avg_importance.values())
                                })
                                
                                # Sort by importance
                                importance_df.sort_values('Average Importance', ascending=False, inplace=True)
                                
                                # Visualize
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Data Source', y='Average Importance', data=importance_df, ax=ax)
                                ax.set_title(f"Average Feature Importance for Predicting {target_col} by Data Source")
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                
                                # Show top individual features
                                st.subheader("Top Individual Features")
                                
                                # Calculate importance for all features
                                feature_importance = {}
                                for col in numeric_cols:
                                    if col != target_col:
                                        if integrated_df[target_col].dtype == 'object':
                                            cat_means = integrated_df.groupby(target_col)[col].mean()
                                            feature_importance[col] = abs(cat_means.max() - cat_means.min())
                                        else:
                                            feature_importance[col] = abs(integrated_df[col].corr(integrated_df[target_col]))
                                
                                # Create DataFrame for top features
                                top_features = pd.DataFrame({
                                    'Feature': list(feature_importance.keys()),
                                    'Importance': list(feature_importance.values())
                                })
                                
                                # Sort and take top 15
                                top_features.sort_values('Importance', ascending=False, inplace=True)
                                top_features = top_features.head(15)
                                
                                # Add source information
                                top_features['Data Source'] = top_features['Feature'].map(column_sources)
                                
                                # Visualize
                                fig, ax = plt.subplots(figsize=(12, 8))
                                sns.barplot(x='Importance', y='Feature', hue='Data Source', data=top_features, ax=ax)
                                ax.set_title(f"Top Features for Predicting {target_col}")
                                plt.tight_layout()
                                
                                st.pyplot(fig)
            else:
                st.info("Please select at least two datasets to integrate.")
        elif 'multi_omics_data' in st.session_state and len(st.session_state['multi_omics_data']) == 1:
            st.info("Integration requires at least two datasets. Please load or create additional datasets.")
        else:
            st.info("Please load or create datasets in the 'Data Overview' tab first.")

if __name__ == "__main__":
    main()