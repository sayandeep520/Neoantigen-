import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer


class FeatureEngineer:
    """
    Feature engineering for multi-omics data.
    Handles feature extraction, selection, transformation and creation
    of derived features for AI model training.
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.transformers = {
            'standard_scaler': StandardScaler(),
            'minmax_scaler': MinMaxScaler(),
            'knn_imputer': KNNImputer(n_neighbors=5)
        }
        
        # Gene sets for signature features
        self.gene_sets = {
            'cell_cycle_genes': [
                'CCNA2', 'CCNB1', 'CCNB2', 'CCND1', 'CCNE1', 
                'CDC25A', 'CDC25B', 'CDC25C', 'CDK1', 'CDK2', 
                'CDK4', 'CDK6', 'E2F1', 'RB1', 'MCM2', 
                'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7'
            ],
            'dna_repair_genes': [
                'BRCA1', 'BRCA2', 'ATM', 'ATR', 'PARP1', 
                'XRCC1', 'XRCC2', 'XRCC3', 'XRCC4', 'XRCC5', 
                'RAD51', 'RAD52', 'RAD54', 'MLH1', 'MSH2', 
                'MSH6', 'PMS2', 'ERCC1', 'ERCC2', 'ERCC3'
            ],
            'immune_genes': [
                'CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 
                'CD8B', 'FOXP3', 'IL2RA', 'CTLA4', 'PDCD1', 
                'CD274', 'HAVCR2', 'LAG3', 'TIGIT', 'CD19', 
                'CD79A', 'CD79B', 'MS4A1', 'NCAM1', 'KLRD1'
            ],
            'ifn_gamma_genes': [
                'STAT1', 'IRF1', 'CXCL9', 'CXCL10', 'CXCL11', 
                'IDO1', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 
                'HLA-DRB1', 'PSMB9', 'PSMB10', 'TAP1', 'TAP2'
            ]
        }
    
    def calculate_gene_signature(self, 
                                expression_data: pd.DataFrame, 
                                gene_set: List[str]) -> pd.Series:
        """
        Calculate a gene signature score from expression data
        
        Args:
            expression_data: Gene expression data
            gene_set: List of genes in the signature
            
        Returns:
            Series with signature scores
        """
        # Check if any genes in the gene set are in the expression data
        genes_found = [gene for gene in gene_set if gene in expression_data.columns]
        
        if not genes_found:
            st.warning(f"None of the genes in the gene set were found in expression data")
            return pd.Series(index=expression_data.index, data=np.nan)
        
        # Calculate mean expression of genes in the signature
        signature_score = expression_data[genes_found].mean(axis=1)
        
        return signature_score
    
    def calculate_all_signatures(self, expression_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all predefined gene signatures
        
        Args:
            expression_data: Gene expression data
            
        Returns:
            DataFrame with signature scores for each sample
        """
        signatures = {}
        
        # Calculate each signature
        for signature_name, gene_set in self.gene_sets.items():
            st.info(f"Calculating {signature_name} signature")
            signatures[signature_name] = self.calculate_gene_signature(expression_data, gene_set)
        
        # Combine into a DataFrame
        signatures_df = pd.DataFrame(signatures)
        
        return signatures_df
    
    def compute_mutation_burden(self, mutation_data: pd.DataFrame, 
                              sample_column: str = 'sample_id',
                              gene_column: str = 'gene') -> pd.DataFrame:
        """
        Compute tumor mutation burden for each sample
        
        Args:
            mutation_data: Mutation data
            sample_column: Column containing sample identifiers
            gene_column: Column containing gene names
            
        Returns:
            DataFrame with mutation burden for each sample
        """
        if mutation_data.empty:
            st.warning("Empty mutation data provided")
            return pd.DataFrame()
        
        # Check if required columns exist
        if sample_column not in mutation_data.columns or gene_column not in mutation_data.columns:
            st.error(f"Required columns not found in mutation data: {sample_column}, {gene_column}")
            return pd.DataFrame()
        
        # Count mutations per sample
        mutation_counts = mutation_data.groupby(sample_column).size().reset_index()
        mutation_counts.columns = [sample_column, 'mutation_burden']
        
        # Calculate mutations per gene per sample
        gene_mutation_counts = mutation_data.groupby([sample_column, gene_column]).size().unstack(fill_value=0)
        
        # Combine mutation burden with gene-specific mutations
        result_df = pd.merge(mutation_counts, gene_mutation_counts, on=sample_column)
        
        st.success(f"Computed mutation burden for {len(result_df)} samples across {len(gene_mutation_counts.columns)} genes")
        return result_df
    
    def compute_gene_alterations(self, cna_data: pd.DataFrame,
                               sample_column: str = 'sample_id',
                               gene_column: str = 'gene',
                               alteration_column: str = 'alteration') -> pd.DataFrame:
        """
        Compute gene amplifications and deletions
        
        Args:
            cna_data: Copy number alteration data
            sample_column: Column containing sample identifiers
            gene_column: Column containing gene names
            alteration_column: Column containing alteration type
            
        Returns:
            DataFrame with counts of gene alterations by type
        """
        if cna_data.empty:
            st.warning("Empty CNA data provided")
            return pd.DataFrame()
        
        # Check if required columns exist
        required_columns = [sample_column, gene_column, alteration_column]
        if not all(col in cna_data.columns for col in required_columns):
            st.error(f"Required columns not found in CNA data: {required_columns}")
            return pd.DataFrame()
        
        # Count amplifications and deletions per sample
        amplifications = cna_data[cna_data[alteration_column].isin(['Amplification', 'Gain', '2', '1'])].groupby(sample_column).size()
        deletions = cna_data[cna_data[alteration_column].isin(['Deletion', 'Loss', '-1', '-2'])].groupby(sample_column).size()
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            sample_column: list(set(cna_data[sample_column].unique())),
            'gene_amplifications': 0,
            'gene_deletions': 0
        }).set_index(sample_column)
        
        # Fill in counts
        result_df.loc[amplifications.index, 'gene_amplifications'] = amplifications
        result_df.loc[deletions.index, 'gene_deletions'] = deletions
        
        # Reset index
        result_df = result_df.reset_index()
        
        st.success(f"Computed gene alterations for {len(result_df)} samples")
        return result_df
    
    def compute_immune_features(self, expression_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute immune-related features from expression data
        
        Args:
            expression_data: Gene expression data
            
        Returns:
            DataFrame with immune features
        """
        # Define marker genes for immune cell types
        immune_markers = {
            'cd8_infiltration': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'GZMK', 'PRF1'],
            'cd4_infiltration': ['CD4', 'CD40LG', 'IL7R', 'FOXP3', 'IL2RA'],
            'treg_infiltration': ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT'],
            'nk_cell_activity': ['NCAM1', 'KLRD1', 'KLRK1', 'NCR1', 'NCR2', 'NCR3'],
            'myeloid_score': ['CD14', 'CD68', 'CD163', 'CSF1R', 'ITGAM'],
            'pd1_expression': ['PDCD1'],
            'pdl1_expression': ['CD274'],
            'ctla4_expression': ['CTLA4']
        }
        
        immune_features = {}
        
        # For each immune feature, calculate mean expression of marker genes
        for feature, markers in immune_markers.items():
            # Find markers present in the data
            markers_present = [gene for gene in markers if gene in expression_data.columns]
            
            if markers_present:
                immune_features[feature] = expression_data[markers_present].mean(axis=1)
            else:
                st.warning(f"No markers found for {feature}")
                immune_features[feature] = pd.Series(0, index=expression_data.index)
        
        # Combine into DataFrame
        immune_df = pd.DataFrame(immune_features)
        
        st.success(f"Computed immune features for {len(immune_df)} samples")
        return immune_df
    
    def compute_stemness_score(self, expression_data: pd.DataFrame) -> pd.Series:
        """
        Compute cancer stemness score based on gene expression
        
        Args:
            expression_data: Gene expression data
            
        Returns:
            Series with stemness scores
        """
        # Define stemness-associated genes
        stemness_genes = [
            'POU5F1', 'SOX2', 'NANOG', 'KLF4', 'MYC', 'PROM1', 
            'ALDH1A1', 'ABCG2', 'CD44', 'THY1', 'EPCAM', 'BMI1'
        ]
        
        # Find stemness genes present in data
        genes_present = [gene for gene in stemness_genes if gene in expression_data.columns]
        
        if not genes_present:
            st.warning("No stemness genes found in expression data")
            return pd.Series(index=expression_data.index, data=0)
        
        # Calculate stemness score as mean expression of genes
        stemness_score = expression_data[genes_present].mean(axis=1)
        
        return stemness_score
    
    def select_features(self, data: pd.DataFrame, target_column: str, 
                      method: str = 'kbest', k: int = 20) -> pd.DataFrame:
        """
        Select most informative features for predicting a target
        
        Args:
            data: DataFrame with features and target
            target_column: Column name for target variable
            method: Feature selection method ('kbest', 'pca')
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if data.empty:
            st.warning("Empty data provided for feature selection")
            return pd.DataFrame()
        
        if target_column not in data.columns:
            st.error(f"Target column '{target_column}' not found in data")
            return pd.DataFrame()
        
        # Extract features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        X_numeric = X[numeric_cols]
        
        if X_numeric.empty:
            st.warning("No numeric features available for selection")
            return data
        
        # Apply feature selection
        selected_features = []
        
        if method == 'kbest':
            # Determine if classification or regression problem
            if y.dtype == 'object' or y.dtype == 'category' or len(y.unique()) < 10:
                # Classification
                selector = SelectKBest(f_classif, k=min(k, X_numeric.shape[1]))
                X_selected = selector.fit_transform(X_numeric, y)
                selected_features = X_numeric.columns[selector.get_support()]
            else:
                # Regression
                selector = SelectKBest(mutual_info_regression, k=min(k, X_numeric.shape[1]))
                X_selected = selector.fit_transform(X_numeric, y)
                selected_features = X_numeric.columns[selector.get_support()]
            
            # Create result DataFrame
            result_df = pd.concat([X[selected_features], data[target_column]], axis=1)
            
            st.success(f"Selected {len(selected_features)} features using {method}")
            
        elif method == 'pca':
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            
            # Apply PCA
            pca = PCA(n_components=min(k, X_numeric.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # Create DataFrame with PCA components
            pca_df = pd.DataFrame(
                X_pca,
                columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
                index=X.index
            )
            
            # Create result DataFrame
            result_df = pd.concat([pca_df, data[target_column]], axis=1)
            
            # Display explained variance
            explained_var = pca.explained_variance_ratio_
            st.success(f"Transformed data to {k} PCA components explaining {sum(explained_var)*100:.2f}% of variance")
            
        else:
            st.error(f"Unknown feature selection method: {method}")
            return data
        
        return result_df
    
    def impute_missing_values(self, data: pd.DataFrame, method: str = 'knn') -> pd.DataFrame:
        """
        Impute missing values in a DataFrame
        
        Args:
            data: DataFrame with missing values
            method: Imputation method ('knn', 'mean', 'median')
            
        Returns:
            DataFrame with imputed values
        """
        if data.empty:
            st.warning("Empty data provided for imputation")
            return data
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        
        if missing_count == 0:
            st.success("No missing values to impute")
            return data
        
        st.info(f"Imputing {missing_count} missing values using {method}")
        
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if numeric_cols.empty:
            st.warning("No numeric columns found for imputation")
            return data
        
        # Create a copy of the data
        result_df = data.copy()
        
        # Impute missing values
        if method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            result_df[numeric_cols] = imputer.fit_transform(result_df[numeric_cols])
        elif method == 'mean':
            result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].mean())
        elif method == 'median':
            result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
        else:
            st.error(f"Unknown imputation method: {method}")
            return data
        
        st.success("Imputation complete")
        return result_df
    
    def scale_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features in a DataFrame
        
        Args:
            data: DataFrame with features
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        if data.empty:
            st.warning("Empty data provided for scaling")
            return data
        
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if numeric_cols.empty:
            st.warning("No numeric columns found for scaling")
            return data
        
        st.info(f"Scaling {len(numeric_cols)} numeric features using {method}")
        
        # Create a copy of the data
        result_df = data.copy()
        
        # Scale features
        if method == 'standard':
            scaler = StandardScaler()
            result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])
        else:
            st.error(f"Unknown scaling method: {method}")
            return data
        
        st.success("Scaling complete")
        return result_df
    
    def create_interaction_features(self, data: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features by multiplying pairs of features
        
        Args:
            data: DataFrame with features
            feature_pairs: List of feature pairs to multiply
            
        Returns:
            DataFrame with added interaction features
        """
        if data.empty:
            st.warning("Empty data provided for interaction features")
            return data
        
        # Create a copy of the data
        result_df = data.copy()
        
        # Create interaction features
        st.info(f"Creating {len(feature_pairs)} interaction features")
        
        for feature1, feature2 in feature_pairs:
            if feature1 in data.columns and feature2 in data.columns:
                interaction_name = f"{feature1}_{feature2}_interaction"
                result_df[interaction_name] = data[feature1] * data[feature2]
            else:
                st.warning(f"Features not found: {feature1}, {feature2}")
        
        st.success("Created interaction features")
        return result_df
    
    def create_polynomial_features(self, data: pd.DataFrame, 
                                 features: List[str], 
                                 degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features by raising features to powers
        
        Args:
            data: DataFrame with features
            features: List of features to transform
            degree: Maximum polynomial degree
            
        Returns:
            DataFrame with added polynomial features
        """
        if data.empty:
            st.warning("Empty data provided for polynomial features")
            return data
        
        # Check if features exist
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            st.warning(f"Features not found: {missing_features}")
            features = [f for f in features if f in data.columns]
        
        if not features:
            st.error("No valid features provided for polynomial transformation")
            return data
        
        # Create a copy of the data
        result_df = data.copy()
        
        # Create polynomial features
        st.info(f"Creating polynomial features of degree {degree} for {len(features)} features")
        
        for feature in features:
            for d in range(2, degree + 1):
                poly_name = f"{feature}_degree_{d}"
                result_df[poly_name] = data[feature] ** d
        
        st.success("Created polynomial features")
        return result_df
