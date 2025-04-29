import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy import stats

def handle_missing_values(data, strategy="knn"):
    """
    Handle missing values in the dataset using various strategies.
    
    Args:
        data (pandas.DataFrame): Input dataset
        strategy (str): Strategy for handling missing values:
            - 'knn': K-nearest neighbors imputation
            - 'mean': Mean imputation
            - 'median': Median imputation
            - 'remove': Remove rows with missing values
    
    Returns:
        pandas.DataFrame: Data with missing values handled
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Get numeric columns that have missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols_with_na = [col for col in numeric_cols if df[col].isna().any()]
    
    # Get categorical columns that have missing values
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    categorical_cols_with_na = [col for col in categorical_cols if df[col].isna().any()]
    
    if strategy.lower() == "remove":
        # Remove rows with any missing values
        df = df.dropna()
        
    elif strategy.lower() == "knn":
        # KNN imputation for numeric columns
        if numeric_cols_with_na:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols_with_na] = imputer.fit_transform(df[numeric_cols_with_na])
        
        # Most frequent imputation for categorical columns
        if categorical_cols_with_na:
            for col in categorical_cols_with_na:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    elif strategy.lower() in ["mean", "median"]:
        # Mean or median imputation for numeric columns
        if numeric_cols_with_na:
            if strategy.lower() == "mean":
                imputer = SimpleImputer(strategy='mean')
            else:
                imputer = SimpleImputer(strategy='median')
                
            df[numeric_cols_with_na] = imputer.fit_transform(df[numeric_cols_with_na])
        
        # Most frequent imputation for categorical columns
        if categorical_cols_with_na:
            for col in categorical_cols_with_na:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def normalize_data(data, method="min-max"):
    """
    Normalize numeric features in the dataset.
    
    Args:
        data (pandas.DataFrame): Input dataset
        method (str): Normalization method:
            - 'min-max': Min-max scaling to [0,1]
            - 'z-score': Z-score normalization
            - 'robust': Robust scaling using quartiles
    
    Returns:
        pandas.DataFrame: Normalized dataset
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Skip columns we don't want to normalize
    skip_cols = ['sample_id', 'patient_id', 'case_id', 'donor_id', 'TMB', 'id']
    normalize_cols = [col for col in numeric_cols if not any(skip in col.lower() for skip in skip_cols)]
    
    if not normalize_cols:
        return df  # Nothing to normalize
    
    # Apply normalization
    if method.lower() == "min-max":
        scaler = MinMaxScaler()
    elif method.lower() == "z-score":
        scaler = StandardScaler()
    elif method.lower() == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
    
    return df

def detect_outliers(data, method="iqr"):
    """
    Detect and handle outliers in the dataset.
    
    Args:
        data (pandas.DataFrame): Input dataset
        method (str): Outlier detection method:
            - 'iqr': Interquartile range method
            - 'z-score': Z-score method (beyond 3 std devs)
            - 'isolation': Isolation forest algorithm
    
    Returns:
        tuple: (pandas.DataFrame, int) - Processed data and outlier count
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Skip columns we don't want to check for outliers
    skip_cols = ['sample_id', 'patient_id', 'case_id', 'donor_id', 'id']
    outlier_cols = [col for col in numeric_cols if not any(skip in col.lower() for skip in skip_cols)]
    
    if not outlier_cols:
        return df, 0  # No columns to check
    
    outliers_count = 0
    
    if method.lower() == "iqr":
        # IQR method
        for col in outlier_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count += outliers.sum()
            
            # Replace outliers with bounds
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            
    elif method.lower() == "z-score":
        # Z-score method
        for col in outlier_cols:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = z_scores > 3  # More than 3 standard deviations
            outliers_count += outliers.sum()
            
            # Replace outliers with mean
            df.loc[outliers, col] = df[col].mean()
            
    elif method.lower() == "isolation":
        # Isolation forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso_forest.fit_predict(df[outlier_cols]) == -1
        outliers_count = np.sum(outliers)
        
        # For outlier rows, replace with column medians
        for col in outlier_cols:
            median_val = df[col].median()
            df.loc[outliers, col] = median_val
    
    return df, outliers_count

def feature_engineering(data, dataset_type=None):
    """
    Apply feature engineering to the dataset based on dataset type.
    
    Args:
        data (pandas.DataFrame): Input dataset
        dataset_type (str): Type of dataset (e.g., 'tcga', 'gtex', 'depmap')
    
    Returns:
        pandas.DataFrame: Dataset with engineered features
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    if dataset_type == 'tcga' or dataset_type == 'icgc':
        # Genomic features for mutation data
        
        # Calculate TMB if not already present
        if 'TMB' not in df.columns and 'mutations_count' in df.columns:
            df['TMB'] = df['mutations_count'] / 3000  # Simplified TMB calculation
        
        # Create binary mutation flags for top genes
        if 'gene' in df.columns:
            top_genes = ['TP53', 'KRAS', 'CDKN2A', 'SMAD4', 'BRCA2']
            for gene in top_genes:
                if gene in df['gene'].values:
                    df[f'{gene}_mutated'] = df['gene'].apply(lambda x: 1 if x == gene else 0)
        
        # Create mutation type features
        if 'mutation_type' in df.columns:
            df = pd.concat([
                df, 
                pd.get_dummies(df['mutation_type'], prefix='mut_type')
            ], axis=1)
        
    elif dataset_type == 'gtex':
        # Expression features for transcriptomic data
        
        # Calculate expression variance if we have multiple samples per gene
        if 'expression' in df.columns and 'gene_symbol' in df.columns:
            variance_by_gene = df.groupby('gene_symbol')['expression'].var().reset_index()
            variance_by_gene.columns = ['gene_symbol', 'expression_variance']
            df = df.merge(variance_by_gene, on='gene_symbol', how='left')
        
        # Calculate Z-scores for each gene across samples
        if 'expression' in df.columns and 'gene_symbol' in df.columns and 'sample_id' in df.columns:
            # Group by gene and calculate z-score across samples
            df['expression_zscore'] = df.groupby('gene_symbol')['expression'].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
            )
        
    elif dataset_type == 'depmap':
        # CRISPR screen features
        
        # Convert dependency scores to essentiality scores if not already done
        if 'dependency_score' in df.columns and 'essentiality_score' not in df.columns:
            df['essentiality_score'] = -df['dependency_score']
        
        # Calculate rank of gene essentiality within each cell line
        if 'essentiality_score' in df.columns and 'cell_line' in df.columns:
            df['essentiality_rank'] = df.groupby('cell_line')['essentiality_score'].rank(ascending=False)
        
        # Create binary flags for significant dependencies
        if 'dependency_score' in df.columns:
            df['is_essential'] = (df['dependency_score'] < -0.5).astype(int)
        
    elif dataset_type == 'proteomic':
        # Protein features
        
        # Extract protein length as a numeric feature if available
        if 'length' in df.columns:
            # Bin protein length into categories
            df['length_category'] = pd.cut(
                df['length'], 
                bins=[0, 250, 500, 1000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )
        
        # Count number of PDB structures as a feature
        if 'pdb_structures' in df.columns:
            df['pdb_count'] = df['pdb_structures'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
    
    return df

def compute_tumor_mutation_burden(data):
    """
    Compute or refine Tumor Mutation Burden (TMB) from mutation data.
    
    Args:
        data (pandas.DataFrame): Input dataset with mutation information
        
    Returns:
        pandas.DataFrame: Dataset with TMB feature added
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Different calculation paths based on available columns
    if 'mutations_count' in df.columns:
        # If we already have mutations count, use it directly
        if 'TMB' not in df.columns:
            df['TMB'] = df['mutations_count'] / 3000  # Simplified TMB calculation
    
    elif 'case_id' in df.columns and 'gene' in df.columns:
        # Count mutations per case
        mutation_counts = df.groupby('case_id')['gene'].count().reset_index()
        mutation_counts.columns = ['case_id', 'mutations_count']
        
        # Merge back to original data
        df = df.merge(mutation_counts, on='case_id', how='left')
        
        # Calculate TMB
        df['TMB'] = df['mutations_count'] / 3000
    
    # Calculate TMB categories for easier interpretation
    if 'TMB' in df.columns:
        df['TMB_category'] = pd.cut(
            df['TMB'],
            bins=[0, 5, 10, 20, float('inf')],
            labels=['Low', 'Intermediate', 'High', 'Very High']
        )
    
    return df

def compute_immune_infiltration_score(data):
    """
    Compute or refine immune infiltration score from expression data.
    
    Args:
        data (pandas.DataFrame): Input dataset with expression information
        
    Returns:
        pandas.DataFrame: Dataset with immune infiltration features added
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Check if we already have immune infiltration score
    if 'immune_infiltration_score' in df.columns:
        return df
    
    # Create synthetic score if we have gene expression data
    # In a real application, this would implement a validated immune scoring algorithm
    
    # Define immune-related genes
    immune_genes = [
        'CD8A', 'CD8B', 'CD4', 'FOXP3', 'CD3D', 'CD3E', 'CD3G',
        'PTPRC', 'CD19', 'CD79A', 'CD79B', 'MZB1', 'CD14', 'CD68',
        'ITGAX', 'CD1C', 'CLEC9A', 'IL2RA'
    ]
    
    # Check if we have the necessary data structure
    if 'gene_symbol' in df.columns and 'expression' in df.columns and 'sample_id' in df.columns:
        # Filter for immune genes
        immune_df = df[df['gene_symbol'].isin(immune_genes)].copy()
        
        if not immune_df.empty:
            # Calculate mean expression of immune genes per sample
            immune_score = immune_df.groupby('sample_id')['expression'].mean().reset_index()
            immune_score.columns = ['sample_id', 'immune_infiltration_score']
            
            # Normalize to 0-1 range
            min_val = immune_score['immune_infiltration_score'].min()
            max_val = immune_score['immune_infiltration_score'].max()
            immune_score['immune_infiltration_score'] = (
                immune_score['immune_infiltration_score'] - min_val
            ) / (max_val - min_val)
            
            # Merge back to original data
            df = df.merge(immune_score, on='sample_id', how='left')
    
    # If we couldn't calculate a real score, create a synthetic one for demonstration
    if 'immune_infiltration_score' not in df.columns:
        if 'sample_id' in df.columns:
            # Generate random immune scores by sample
            unique_samples = df['sample_id'].unique()
            immune_scores = np.random.beta(2, 5, size=len(unique_samples))
            
            immune_score_df = pd.DataFrame({
                'sample_id': unique_samples,
                'immune_infiltration_score': immune_scores
            })
            
            # Merge back to original data
            df = df.merge(immune_score_df, on='sample_id', how='left')
        else:
            # No sample ID column, add a random score to each row
            df['immune_infiltration_score'] = np.random.beta(2, 5, size=len(df))
    
    # Add immune infiltration categories
    df['immune_category'] = pd.cut(
        df['immune_infiltration_score'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Cold', 'Low', 'Moderate', 'High']
    )
    
    return df
