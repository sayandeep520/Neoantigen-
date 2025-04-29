import os
import pandas as pd
import numpy as np
import streamlit as st
import time
import json
from datetime import datetime

# Create data directories if they don't exist
DATA_DIR = 'data'
DATA_CATALOG_FILE = os.path.join(DATA_DIR, 'data_catalog.json')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize data catalog if it doesn't exist
if not os.path.exists(DATA_CATALOG_FILE):
    with open(DATA_CATALOG_FILE, 'w') as f:
        json.dump([], f)

def save_uploaded_file(uploaded_file, subfolder=None):
    """
    Save an uploaded file to the data directory with an optional subfolder
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        subfolder: Optional subfolder name
        
    Returns:
        Path to the saved file
    """
    if subfolder:
        folder_path = os.path.join(DATA_DIR, subfolder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        folder_path = DATA_DIR
        
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(uploaded_file.name)[1]
    filename = f"{os.path.splitext(uploaded_file.name)[0]}_{timestamp}{file_extension}"
    file_path = os.path.join(folder_path, filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path


def load_genomic_data(file_path, file_format=None):
    """
    Load genomic data from a file into a pandas DataFrame
    
    Args:
        file_path: Path to the file
        file_format: Format of the file (csv, tsv, excel, etc.)
        
    Returns:
        DataFrame with the data or None on error
    """
    if file_format is None:
        # Try to infer file format from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.csv':
            file_format = 'csv'
        elif ext in ['.tsv', '.txt']:
            file_format = 'tsv'
        elif ext in ['.xlsx', '.xls']:
            file_format = 'excel'
        elif ext == '.json':
            file_format = 'json'
        else:
            st.error(f"Unsupported file format: {ext}")
            return None
    
    try:
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif file_format == 'excel':
            df = pd.read_excel(file_path)
        elif file_format == 'json':
            df = pd.read_json(file_path)
        else:
            st.error(f"Unsupported file format: {file_format}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def add_to_catalog(data_type, filename, description, file_format, file_path, df_shape):
    """
    Add dataset information to the catalog file
    
    Args:
        data_type: Type of data (genomic, transcriptomic, proteomic)
        filename: Original filename
        description: Data description
        file_format: File format
        file_path: Path to the saved file
        df_shape: Shape of the dataframe (rows, columns)
        
    Returns:
        Dataset ID (timestamp)
    """
    # Generate a unique ID based on timestamp
    dataset_id = str(int(time.time()))
    
    # Load existing catalog
    with open(DATA_CATALOG_FILE, 'r') as f:
        catalog = json.load(f)
    
    # Create new entry
    entry = {
        'id': dataset_id,
        'data_type': data_type,
        'filename': filename,
        'description': description,
        'file_format': file_format,
        'sample_count': df_shape[0],
        'feature_count': df_shape[1],
        'file_path': file_path,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add to catalog
    catalog.append(entry)
    
    # Save updated catalog
    with open(DATA_CATALOG_FILE, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    return dataset_id


def upload_multi_omics_data():
    """
    Display a form for uploading multi-omics data
    
    Returns:
        Boolean indicating success
    """
    st.subheader("Upload Custom Genomic, Transcriptomic, or Proteomic Data")
    
    with st.form(key="upload_form"):
        data_type = st.selectbox(
            "Data Type",
            ["Genomic", "Transcriptomic", "Proteomic", "Clinical", "Other"]
        )
        
        description = st.text_area(
            "Description",
            placeholder="Enter a description of the data"
        )
        
        uploaded_file = st.file_uploader(
            "Upload file (CSV, TSV, Excel, or JSON)",
            type=["csv", "tsv", "txt", "xlsx", "xls", "json"]
        )
        
        submitted = st.form_submit_button("Upload Data")
        
    if submitted and uploaded_file is not None:
        st.info("Processing upload... Please wait.")
        
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file, subfolder=data_type.lower())
        
        # Determine file format
        _, ext = os.path.splitext(uploaded_file.name)
        ext = ext.lower()
        
        if ext == '.csv':
            file_format = 'csv'
        elif ext in ['.tsv', '.txt']:
            file_format = 'tsv'
        elif ext in ['.xlsx', '.xls']:
            file_format = 'excel'
        elif ext == '.json':
            file_format = 'json'
        else:
            st.error(f"Unsupported file format: {ext}")
            return False
            
        # Load the data
        df = load_genomic_data(file_path, file_format)
        
        if df is not None:
            # Add to catalog
            dataset_id = add_to_catalog(
                data_type=data_type.lower(),
                filename=uploaded_file.name,
                description=description,
                file_format=file_format,
                file_path=file_path,
                df_shape=df.shape
            )
            
            st.success(f"Data uploaded successfully! Dataset ID: {dataset_id}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display data info
            st.subheader("Data Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
            
            with col2:
                st.write(f"**Data Type:** {data_type}")
                st.write(f"**File Format:** {file_format}")
            
            # Store the dataframe in session state for immediate use
            if 'datasets' not in st.session_state:
                st.session_state['datasets'] = {}
            
            st.session_state['datasets'][f'uploaded_{dataset_id}'] = df
            
            return True
        else:
            st.error("Failed to load data from file")
            return False
            
    return False


def list_available_datasets():
    """
    List all available datasets from the catalog file
    
    Returns:
        DataFrame with dataset information
    """
    try:
        # Load catalog
        with open(DATA_CATALOG_FILE, 'r') as f:
            catalog = json.load(f)
        
        if not catalog:
            return pd.DataFrame()
        
        # Convert to DataFrame
        return pd.DataFrame(catalog)
    except Exception as e:
        st.error(f"Error reading dataset catalog: {e}")
        return pd.DataFrame()
