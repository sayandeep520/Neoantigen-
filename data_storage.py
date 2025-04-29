import pandas as pd
import streamlit as st
import os
import pickle
from typing import Dict, List, Union, Tuple, Optional, Any
import json
import io


class DataStorage:
    """
    Manage data storage and persistence for the application.
    Handles caching, session state storage, and file I/O.
    """
    
    def __init__(self):
        """Initialize the data storage manager"""
        # Create a directory for storing data files if it doesn't exist
        os.makedirs('data', exist_ok=True)
    
    def save_dataframe(self, df: pd.DataFrame, name: str) -> bool:
        """
        Save a DataFrame to session state and optionally to file
        
        Args:
            df: DataFrame to save
            name: Name to associate with the DataFrame
            
        Returns:
            Boolean indicating success
        """
        if df is None or df.empty:
            st.error(f"Cannot save empty DataFrame: {name}")
            return False
        
        # Save to session state
        key = f"df_{name}"
        st.session_state[key] = df
        
        return True
    
    def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from session state
        
        Args:
            name: Name of the DataFrame to load
            
        Returns:
            DataFrame if found, None otherwise
        """
        key = f"df_{name}"
        
        # Try to load from session state
        if key in st.session_state:
            return st.session_state[key]
        
        st.warning(f"DataFrame '{name}' not found in session state")
        return None
    
    def list_available_dataframes(self) -> List[str]:
        """
        List all available DataFrames in session state
        
        Returns:
            List of DataFrame names
        """
        dataframes = []
        
        for key in st.session_state:
            if key.startswith("df_"):
                dataframes.append(key[3:])  # Remove 'df_' prefix
        
        return dataframes
    
    def save_to_file(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save a DataFrame to a CSV file
        
        Args:
            df: DataFrame to save
            filename: Name of the file (without extension)
            
        Returns:
            Path to the saved file
        """
        if df is None or df.empty:
            st.error(f"Cannot save empty DataFrame to file: {filename}")
            return ""
        
        # Ensure the filename has the correct extension
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        # Save to file
        filepath = os.path.join('data', filename)
        try:
            df.to_csv(filepath, index=False)
            st.success(f"DataFrame saved to {filepath}")
            return filepath
        except Exception as e:
            st.error(f"Error saving DataFrame to file: {str(e)}")
            return ""
    
    def load_from_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from a CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            # Check if the file exists
            if not os.path.exists(filepath):
                st.error(f"File not found: {filepath}")
                return None
            
            # Load the file
            df = pd.read_csv(filepath)
            st.success(f"Loaded DataFrame from {filepath} with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            st.error(f"Error loading DataFrame from file: {str(e)}")
            return None
    
    def save_model(self, model: Any, name: str) -> bool:
        """
        Save a model to session state and to a file
        
        Args:
            model: Model to save
            name: Name to associate with the model
            
        Returns:
            Boolean indicating success
        """
        if model is None:
            st.error(f"Cannot save None model: {name}")
            return False
        
        # Save to session state
        key = f"model_{name}"
        st.session_state[key] = model
        
        # Save to file
        filepath = os.path.join('data', f"{name}_model.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            st.success(f"Model saved to {filepath}")
            return True
        except Exception as e:
            st.error(f"Error saving model to file: {str(e)}")
            return False
    
    def load_model(self, name: str) -> Any:
        """
        Load a model from session state or from a file
        
        Args:
            name: Name of the model to load
            
        Returns:
            Model if found, None otherwise
        """
        key = f"model_{name}"
        
        # Try to load from session state
        if key in st.session_state:
            return st.session_state[key]
        
        # Try to load from file
        filepath = os.path.join('data', f"{name}_model.pkl")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                st.session_state[key] = model
                st.success(f"Loaded model from {filepath}")
                return model
            except Exception as e:
                st.error(f"Error loading model from file: {str(e)}")
        
        st.warning(f"Model '{name}' not found")
        return None
    
    def list_available_models(self) -> List[str]:
        """
        List all available models in session state
        
        Returns:
            List of model names
        """
        models = []
        
        for key in st.session_state:
            if key.startswith("model_"):
                models.append(key[6:])  # Remove 'model_' prefix
        
        return models
    
    def generate_download_link(self, df: pd.DataFrame, filename: str, link_text: str) -> str:
        """
        Generate a download link for a DataFrame
        
        Args:
            df: DataFrame to download
            filename: Name for the downloaded file
            link_text: Text to display for the download link
            
        Returns:
            HTML for the download link
        """
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        # Create a download link
        b64 = io.StringIO()
        b64.write(csv)
        b64_str = b64.getvalue()
        
        href = f'<a href="data:file/csv;base64,{b64_str}" download="{filename}">{link_text}</a>'
        return href
    
    def save_result(self, result: Dict[str, Any], name: str) -> bool:
        """
        Save a result dictionary to session state and to a file
        
        Args:
            result: Result dictionary to save
            name: Name to associate with the result
            
        Returns:
            Boolean indicating success
        """
        if not result:
            st.error(f"Cannot save empty result: {name}")
            return False
        
        # Save to session state
        key = f"result_{name}"
        st.session_state[key] = result
        
        # Save to file
        filepath = os.path.join('data', f"{name}_result.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            st.success(f"Result saved to {filepath}")
            return True
        except Exception as e:
            st.error(f"Error saving result to file: {str(e)}")
            return False
    
    def load_result(self, name: str) -> Dict[str, Any]:
        """
        Load a result dictionary from session state or from a file
        
        Args:
            name: Name of the result to load
            
        Returns:
            Result dictionary if found, empty dict otherwise
        """
        key = f"result_{name}"
        
        # Try to load from session state
        if key in st.session_state:
            return st.session_state[key]
        
        # Try to load from file
        filepath = os.path.join('data', f"{name}_result.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    result = json.load(f)
                st.session_state[key] = result
                st.success(f"Loaded result from {filepath}")
                return result
            except Exception as e:
                st.error(f"Error loading result from file: {str(e)}")
        
        st.warning(f"Result '{name}' not found")
        return {}


# Create a singleton instance
data_storage = DataStorage()
