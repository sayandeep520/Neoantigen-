"""
Multi-omics Data Processing Module

This module provides classes for collecting, preprocessing, and storing multi-omics data for the 
AI-Driven CRISPR Cancer Immunotherapy Platform. It offers optimized data handling capabilities
that integrate with the platform's file-based storage system.
"""

# Import essential libraries
import pandas as pd
import numpy as np
import requests
import os
import time
import json
from io import StringIO
from typing import Dict, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from Bio import Entrez
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from scipy import stats
import streamlit as st

# Set data directory paths
DATA_DIR = 'data'
DATA_CATALOG_FILE = os.path.join(DATA_DIR, 'data_catalog.json')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize data catalog if it doesn't exist
if not os.path.exists(DATA_CATALOG_FILE):
    with open(DATA_CATALOG_FILE, 'w') as f:
        json.dump([], f)


class DataCollector:
    """
    Optimized Data collector for multi-omics data
    """
    
    def __init__(self):
        self.api_endpoints = {
            "tcga": "https://api.gdc.cancer.gov/",
            "gtex": "https://gtexportal.org/rest/v1/",
            "icgc": "https://dcc.icgc.org/api/v1/",
            "depmap": "https://depmap.org/portal/api/",
            "uniprot": "https://www.ebi.ac.uk/proteins/api/"
        }
        self.session = requests.Session()

    def fetch_data(self, source: str, params: Dict) -> pd.DataFrame:
        """
        Fetch data from a specified source API
        
        Args:
            source: API source name (tcga, gtex, etc.)
            params: API request parameters
            
        Returns:
            DataFrame containing the fetched data
        """
        url = self.api_endpoints.get(source, "")
        if not url:
            raise ValueError(f"Invalid data source: {source}")
        
        try:
            response = self.session.get(url, params=params)
            if response.status_code == 200:
                return pd.DataFrame(response.json())
            else:
                st.error(f"API request failed with status code: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching data from {source}: {str(e)}")
            return pd.DataFrame()

    def fetch_multi_source(self, sources: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from multiple sources in parallel
        
        Args:
            sources: List of source configurations with 'name' and 'params' keys
            
        Returns:
            Dictionary mapping source names to their respective DataFrames
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(sources), 5)) as executor:
            future_to_source = {
                executor.submit(self.fetch_data, src['name'], src['params']): src['name'] 
                for src in sources
            }
            
            for future in future_to_source:
                source_name = future_to_source[future]
                try:
                    results[source_name] = future.result()
                except Exception as e:
                    st.error(f"Error fetching from {source_name}: {str(e)}")
                    results[source_name] = pd.DataFrame()
        
        return results


class DataPreprocessor:
    """
    Preprocess multi-omics data for AI model training
    """
    
    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing_values(self, data: pd.DataFrame, strategy: str = "knn") -> pd.DataFrame:
        """
        Impute missing values in the dataset
        
        Args:
            data: Input DataFrame with missing values
            strategy: Imputation strategy ('knn', 'mean', 'median', 'most_frequent')
            
        Returns:
            DataFrame with imputed values
        """
        # If no missing values, return the data as is
        if not data.isnull().any().any():
            return data
            
        if strategy == "knn":
            # Handle small datasets or high dimensionality
            if data.shape[0] < 5 or data.shape[1] > data.shape[0]:
                imputer = SimpleImputer(strategy="mean")
            else:
                imputer = KNNImputer(n_neighbors=min(5, data.shape[0]-1))
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        # Keep track of non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric data
        if len(numeric_cols) > 0:
            numeric_data = data[numeric_cols]
            imputed_data = pd.DataFrame(
                imputer.fit_transform(numeric_data), 
                columns=numeric_cols,
                index=data.index
            )
            
            # Handle non-numeric columns separately with mode imputation
            if len(non_numeric_cols) > 0:
                non_numeric_data = data[non_numeric_cols]
                mode_imputer = SimpleImputer(strategy="most_frequent")
                imputed_non_numeric = pd.DataFrame(
                    mode_imputer.fit_transform(non_numeric_data),
                    columns=non_numeric_cols,
                    index=data.index
                )
                # Combine numeric and non-numeric data
                imputed_data = pd.concat([imputed_data, imputed_non_numeric], axis=1)
            
            return imputed_data[data.columns]  # Preserve original column order
        else:
            # If no numeric columns, use mode imputation for all
            mode_imputer = SimpleImputer(strategy="most_frequent")
            return pd.DataFrame(
                mode_imputer.fit_transform(data),
                columns=data.columns,
                index=data.index
            )

    def normalize_data(self, data: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """
        Normalize numeric features in the dataset
        
        Args:
            data: Input DataFrame
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with normalized values
        """
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return data  # No numeric columns to scale
            
        # Select appropriate scaler
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:  # standard
            scaler = StandardScaler()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Apply scaling only to numeric columns
        result[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        return result

    def detect_outliers(self, data: pd.DataFrame, method: str = "isolation_forest", 
                       contamination: float = 0.05) -> pd.DataFrame:
        """
        Detect and filter outliers
        
        Args:
            data: Input DataFrame
            method: Outlier detection method ('isolation_forest', 'zscore')
            contamination: Expected proportion of outliers (for isolation_forest)
            
        Returns:
            DataFrame with outliers removed
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] == 0:
            return data  # No numeric columns for outlier detection
            
        if method == "isolation_forest":
            # Use Isolation Forest for outlier detection
            iso_forest = IsolationForest(contamination=contamination)
            outlier_labels = iso_forest.fit_predict(numeric_data)
            return data[outlier_labels == 1]  # Keep only inliers
            
        elif method == "zscore":
            # Use Z-score method for outlier detection
            z_scores = np.abs(stats.zscore(numeric_data))
            return data[(z_scores < 3).all(axis=1)]  # Keep rows where all columns have z-score < 3
            
        return data  # Default: return original data

    def reduce_dimensions(self, data: pd.DataFrame, n_components: int = 10, 
                         method: str = "pca") -> pd.DataFrame:
        """
        Reduce dimensionality of the dataset
        
        Args:
            data: Input DataFrame
            n_components: Number of dimensions to reduce to
            method: Dimension reduction method ('pca', 'selectk')
            
        Returns:
            DataFrame with reduced dimensions
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] == 0:
            return data  # No numeric columns for dimension reduction
            
        # Adjust n_components to not exceed data dimensions
        n_components = min(n_components, numeric_data.shape[1], numeric_data.shape[0])
        
        if method == "pca":
            # Use PCA for dimension reduction
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(numeric_data)
            # Create a new DataFrame with PCA components
            reduced_df = pd.DataFrame(
                reduced_data,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=data.index
            )
            
            # Add non-numeric columns back
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                return pd.concat([reduced_df, data[non_numeric_cols]], axis=1)
            return reduced_df
            
        elif method == "selectk":
            # For selectk method, we need a target variable
            # This is a placeholder - in practice, you would provide a target
            if "target" in data.columns:
                selector = SelectKBest(f_classif, k=n_components)
                reduced_data = selector.fit_transform(numeric_data, data["target"])
                selected_features = numeric_data.columns[selector.get_support()]
                
                # Keep selected features and non-numeric columns
                return data[list(selected_features) + list(data.select_dtypes(exclude=[np.number]).columns)]
            else:
                st.warning("SelectK method requires a target variable. Using PCA instead.")
                return self.reduce_dimensions(data, n_components, "pca")
                
        return data  # Default: return original data


class DataStorage:
    """
    File-based data storage and persistence
    """
    
    def __init__(self, data_dir: str = DATA_DIR, catalog_file: str = DATA_CATALOG_FILE):
        self.data_dir = data_dir
        self.catalog_file = catalog_file
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Initialize catalog file if it doesn't exist
        if not os.path.exists(self.catalog_file):
            with open(self.catalog_file, 'w') as f:
                json.dump([], f)
    
    def save_dataset(self, name: str, data: pd.DataFrame, data_type: str = "processed", 
                   description: str = "", format: str = "csv") -> str:
        """
        Save a dataset to file and update the catalog
        
        Args:
            name: Dataset name
            data: DataFrame to save
            data_type: Type of data (e.g., genomic, transcriptomic)
            description: Dataset description
            format: File format to save as (csv, json, pkl)
            
        Returns:
            ID of the saved dataset
        """
        # Create a subfolder based on data_type
        subfolder = os.path.join(self.data_dir, data_type.lower())
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
            
        # Generate a unique ID and filename
        dataset_id = str(int(time.time()))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"
        
        if format == "json":
            file_path = os.path.join(subfolder, f"{filename}.json")
            data.to_json(file_path)
        elif format == "pkl":
            file_path = os.path.join(subfolder, f"{filename}.pkl")
            data.to_pickle(file_path)
        else:  # default to csv
            file_path = os.path.join(subfolder, f"{filename}.csv")
            data.to_csv(file_path, index=False)
            
        # Update the catalog
        with open(self.catalog_file, 'r') as f:
            catalog = json.load(f)
            
        # Create a new entry
        entry = {
            'id': dataset_id,
            'name': name,
            'data_type': data_type,
            'filename': os.path.basename(file_path),
            'description': description,
            'file_format': format,
            'sample_count': data.shape[0],
            'feature_count': data.shape[1],
            'file_path': file_path,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to catalog
        catalog.append(entry)
        
        # Save updated catalog
        with open(self.catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        return dataset_id
    
    def load_dataset(self, dataset_id: str = None, name: str = None) -> pd.DataFrame:
        """
        Load a dataset by ID or name
        
        Args:
            dataset_id: Dataset ID (optional)
            name: Dataset name (optional, used if dataset_id not provided)
            
        Returns:
            DataFrame containing the loaded data
        """
        # Load the catalog
        with open(self.catalog_file, 'r') as f:
            catalog = json.load(f)
            
        # Find the dataset entry
        dataset_entry = None
        
        if dataset_id:
            for entry in catalog:
                if entry['id'] == dataset_id:
                    dataset_entry = entry
                    break
        elif name:
            for entry in catalog:
                if entry['name'] == name:
                    dataset_entry = entry
                    break
                    
        if not dataset_entry:
            st.error(f"Dataset not found with ID: {dataset_id} or name: {name}")
            return pd.DataFrame()
            
        # Load the file based on format
        file_path = dataset_entry['file_path']
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
            
        try:
            if file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.pkl'):
                return pd.read_pickle(file_path)
            else:  # default to csv
                return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()
            
    def list_datasets(self, data_type: str = None) -> pd.DataFrame:
        """
        List all available datasets, optionally filtered by type
        
        Args:
            data_type: Optional filter by data type
            
        Returns:
            DataFrame containing dataset information
        """
        # Load the catalog
        with open(self.catalog_file, 'r') as f:
            catalog = json.load(f)
            
        if not catalog:
            return pd.DataFrame()
            
        # Filter by data_type if provided
        if data_type:
            catalog = [entry for entry in catalog if entry['data_type'].lower() == data_type.lower()]
            
        # Convert to DataFrame
        return pd.DataFrame(catalog)
        
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset by ID
        
        Args:
            dataset_id: Dataset ID to delete
            
        Returns:
            Boolean indicating success
        """
        # Load the catalog
        with open(self.catalog_file, 'r') as f:
            catalog = json.load(f)
            
        # Find the dataset entry
        dataset_entry = None
        entry_index = -1
        
        for i, entry in enumerate(catalog):
            if entry['id'] == dataset_id:
                dataset_entry = entry
                entry_index = i
                break
                
        if not dataset_entry:
            st.error(f"Dataset not found with ID: {dataset_id}")
            return False
            
        # Delete the file
        file_path = dataset_entry['file_path']
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Error deleting file {file_path}: {str(e)}")
                return False
                
        # Remove from catalog
        catalog.pop(entry_index)
        
        # Save updated catalog
        with open(self.catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
            
        return True


# Create instance for import
data_collector = DataCollector()
data_preprocessor = DataPreprocessor()
data_storage = DataStorage()