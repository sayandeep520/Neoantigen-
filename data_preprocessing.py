import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Union, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


class DataPreprocessor:
    """
    Preprocess multi-omics data for AI model training.
    Handles data cleaning, normalization, feature selection, and integration.
    """
    
    def __init__(self):
        """Initialize the data preprocessor with necessary tools"""
        self.imputers = {
            'knn': KNNImputer(n_neighbors=5),
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
        }
        
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
        }
        
        self.encoders = {
            'onehot': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
        }
    
    def clean_data(self, df: pd.DataFrame, 
                   imputation_method: str = 'knn',
                   drop_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean data by handling missing values and removing low-quality features
        
        Args:
            df: Input DataFrame
            imputation_method: Method for imputing missing values ('knn', 'mean', 'median', 'most_frequent')
            drop_threshold: Threshold for dropping features with too many missing values
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            st.error("Empty DataFrame provided for cleaning")
            return df
        
        # Calculate missing value percentage for each column
        missing_percentage = df.isnull().mean()
        
        # Drop columns with too many missing values
        columns_to_drop = missing_percentage[missing_percentage > drop_threshold].index
        if len(columns_to_drop) > 0:
            st.warning(f"Dropping {len(columns_to_drop)} columns with more than {drop_threshold*100}% missing values")
            df = df.drop(columns=columns_to_drop)
        
        # Impute missing values
        if imputation_method in self.imputers:
            imputer = self.imputers[imputation_method]
            
            # Separate numeric and non-numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns
            
            # Impute numeric columns
            if len(numeric_cols) > 0:
                st.info(f"Imputing missing values in {len(numeric_cols)} numeric columns using {imputation_method}")
                df[numeric_cols] = pd.DataFrame(
                    imputer.fit_transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            
            # Impute non-numeric columns with most_frequent
            if len(non_numeric_cols) > 0:
                st.info(f"Imputing missing values in {len(non_numeric_cols)} non-numeric columns using most_frequent")
                non_numeric_imputer = self.imputers['most_frequent']
                df[non_numeric_cols] = pd.DataFrame(
                    non_numeric_imputer.fit_transform(df[non_numeric_cols]),
                    columns=non_numeric_cols,
                    index=df.index
                )
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize data using specified scaling method
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            st.error("Empty DataFrame provided for normalization")
            return df
        
        if method in self.scalers:
            scaler = self.scalers[method]
            
            # Only normalize numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                st.info(f"Normalizing {len(numeric_cols)} numeric columns using {method} scaling")
                df[numeric_cols] = pd.DataFrame(
                    scaler.fit_transform(df[numeric_cols]),
                    columns=numeric_cols,
                    index=df.index
                )
            else:
                st.warning("No numeric columns found for normalization")
        else:
            st.error(f"Unknown normalization method: {method}")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode (if None, all object/categorical columns are encoded)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        if df.empty:
            st.error("Empty DataFrame provided for categorical encoding")
            return df
        
        # If no columns specified, use all object/categorical columns
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(columns) == 0:
            st.warning("No categorical columns found for encoding")
            return df
        
        st.info(f"One-hot encoding {len(columns)} categorical columns")
        
        encoder = self.encoders['onehot']
        
        # Encode each column separately
        for col in columns:
            if col in df.columns:
                encoded_data = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                    index=df.index
                )
                
                # Replace original column with encoded columns
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                st.warning(f"Column '{col}' not found in DataFrame")
        
        return df
    
    def integrate_multi_omics(self, dfs: Dict[str, pd.DataFrame], on: str = 'sample_id') -> pd.DataFrame:
        """
        Integrate multiple omics datasets based on a common identifier
        
        Args:
            dfs: Dictionary of DataFrames, with keys as omics types
            on: Column to join on (e.g., 'sample_id', 'patient_id')
            
        Returns:
            Integrated DataFrame
        """
        if not dfs:
            st.error("No DataFrames provided for integration")
            return pd.DataFrame()
        
        omics_types = list(dfs.keys())
        st.info(f"Integrating {len(omics_types)} omics types: {', '.join(omics_types)}")
        
        # Check if all DataFrames have the join column
        for omics_type, df in dfs.items():
            if df.empty:
                st.warning(f"Empty DataFrame for {omics_type}")
                continue
                
            if on not in df.columns:
                st.error(f"Join column '{on}' not found in {omics_type} DataFrame")
                return pd.DataFrame()
        
        # Start with the first DataFrame
        first_omics = omics_types[0]
        integrated_df = dfs[first_omics]
        
        # Join with each remaining DataFrame
        for omics_type in omics_types[1:]:
            if dfs[omics_type].empty:
                continue
                
            omics_df = dfs[omics_type]
            
            # Rename columns to avoid conflicts
            omics_df = omics_df.rename(columns={col: f"{omics_type}_{col}" 
                                              for col in omics_df.columns 
                                              if col != on and col in integrated_df.columns})
            
            # Merge DataFrames
            integrated_df = pd.merge(integrated_df, omics_df, on=on, how='outer')
        
        # Check for successful integration
        if integrated_df.empty:
            st.error("Integration resulted in an empty DataFrame")
        else:
            st.success(f"Successfully integrated {len(omics_types)} omics types into a DataFrame with {integrated_df.shape[0]} rows and {integrated_df.shape[1]} columns")
        
        return integrated_df
    
    def feature_selection(self, df: pd.DataFrame, target: str, method: str = 'kbest', k: int = 50) -> pd.DataFrame:
        """
        Select top features based on statistical tests
        
        Args:
            df: Input DataFrame
            target: Target variable for supervised selection
            method: Feature selection method ('kbest', 'pca')
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if df.empty:
            st.error("Empty DataFrame provided for feature selection")
            return df
        
        if target not in df.columns:
            st.error(f"Target column '{target}' not found in DataFrame")
            return df
        
        # Extract features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Select only numeric columns for feature selection
        numeric_cols = X.select_dtypes(include=['number']).columns
        X_numeric = X[numeric_cols]
        
        if X_numeric.empty:
            st.error("No numeric features available for selection")
            return df
        
        st.info(f"Selecting top {k} features from {X_numeric.shape[1]} numeric features using {method}")
        
        if method == 'kbest':
            # Select top k features based on ANOVA F-value
            selector = SelectKBest(f_classif, k=min(k, X_numeric.shape[1]))
            X_selected = selector.fit_transform(X_numeric, y)
            
            # Get selected feature names
            selected_features = X_numeric.columns[selector.get_support()]
            
            # Create DataFrame with selected features and target
            result_df = pd.concat([
                X_numeric[selected_features],
                X[X.columns.difference(numeric_cols)],  # Keep non-numeric columns
                pd.Series(y, name=target, index=X.index)
            ], axis=1)
            
        elif method == 'pca':
            # Perform PCA for dimensionality reduction
            pca = PCA(n_components=min(k, X_numeric.shape[1]))
            X_pca = pca.fit_transform(X_numeric)
            
            # Create DataFrame with PCA components and target
            pca_df = pd.DataFrame(
                X_pca,
                columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
                index=X.index
            )
            
            result_df = pd.concat([
                pca_df,
                X[X.columns.difference(numeric_cols)],  # Keep non-numeric columns
                pd.Series(y, name=target, index=X.index)
            ], axis=1)
            
            # Display explained variance
            explained_var = pca.explained_variance_ratio_
            st.info(f"Top {k} PCA components explain {sum(explained_var)*100:.2f}% of variance")
            
        else:
            st.error(f"Unknown feature selection method: {method}")
            return df
        
        st.success(f"Feature selection complete. Reduced from {X.shape[1]} to {result_df.shape[1]-1} features")
        return result_df
    
    def compute_tmb(self, mutations_df: pd.DataFrame, sample_col: str = 'sample_id') -> pd.DataFrame:
        """
        Compute Tumor Mutation Burden (TMB) for each sample
        
        Args:
            mutations_df: DataFrame containing mutation data
            sample_col: Column containing sample identifiers
            
        Returns:
            DataFrame with TMB scores
        """
        if mutations_df.empty:
            st.error("Empty mutation DataFrame provided for TMB calculation")
            return pd.DataFrame()
        
        if sample_col not in mutations_df.columns:
            st.error(f"Sample column '{sample_col}' not found in mutations DataFrame")
            return pd.DataFrame()
        
        st.info("Computing Tumor Mutation Burden (TMB)")
        
        # Count mutations per sample
        tmb_df = mutations_df.groupby(sample_col).size().reset_index()
        tmb_df.columns = [sample_col, 'TMB']
        
        st.success(f"Computed TMB for {tmb_df.shape[0]} samples")
        return tmb_df
    
    def calculate_immune_infiltration(self, expression_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate immune cell infiltration scores from gene expression data
        
        Args:
            expression_df: Gene expression DataFrame (genes as rows, samples as columns)
            
        Returns:
            DataFrame with immune infiltration scores
        """
        if expression_df.empty:
            st.error("Empty expression DataFrame provided for immune infiltration calculation")
            return pd.DataFrame()
        
        st.info("Calculating immune cell infiltration scores")
        st.warning("This is a simplified implementation. Real implementation would use algorithms like CIBERSORT, xCell, or TIMER")
        
        # Define marker genes for major immune cell types
        immune_markers = {
            'T_cell': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B'],
            'B_cell': ['CD19', 'CD20', 'CD79A', 'CD79B', 'MS4A1'],
            'NK_cell': ['NCAM1', 'KLRD1', 'KLRK1', 'NCR1'],
            'Macrophage': ['CD68', 'CD163', 'CSF1R'],
            'Dendritic_cell': ['CD209', 'CD1C', 'CLEC9A'],
            'Neutrophil': ['CEACAM8', 'FCGR3B', 'CSF3R'],
        }
        
        # Initialize the results DataFrame
        infiltration_scores = pd.DataFrame(index=expression_df.columns)
        
        # Calculate scores for each immune cell type
        for cell_type, markers in immune_markers.items():
            # Check which markers are present in the expression data
            available_markers = [gene for gene in markers if gene in expression_df.index]
            
            if available_markers:
                # Calculate average expression of marker genes for each sample
                infiltration_scores[cell_type] = expression_df.loc[available_markers].mean()
            else:
                st.warning(f"No marker genes found for {cell_type}")
                infiltration_scores[cell_type] = np.nan
        
        # Reset index to have sample_id as a column
        infiltration_scores.reset_index(inplace=True)
        infiltration_scores.rename(columns={'index': 'sample_id'}, inplace=True)
        
        st.success(f"Calculated infiltration scores for {len(immune_markers)} immune cell types across {infiltration_scores.shape[0]} samples")
        return infiltration_scores
