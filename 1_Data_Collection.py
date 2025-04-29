import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import os
from utils.data_fetcher import (
    fetch_tcga_mutations, 
    fetch_icgc_data, 
    fetch_gtex_data, 
    fetch_depmap_data, 
    fetch_proteomic_data
)
from utils.visualization import plot_data_completeness, plot_sample_distribution
from utils.file_upload import upload_multi_omics_data, list_available_datasets

# Page configuration
st.set_page_config(
    page_title="Data Collection | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Session state initialization for data collection progress
if 'data_collection_progress' not in st.session_state:
    st.session_state['data_collection_progress'] = {
        'tcga': False,
        'icgc': False,
        'gtex': False,
        'depmap': False,
        'proteomic': False
    }

if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}

# Header
st.title("🔬 Multi-Omics Data Collection")
st.markdown("""
This module automates the collection of genomic, transcriptomic, proteomic, and CRISPR screening data 
from various public databases and custom uploads to support AI-driven cancer immunotherapy research.
""")

# Create tabs for public data vs. custom uploads
tab1, tab2, tab3 = st.tabs(["Public Data Sources", "Custom Data Upload", "Uploaded Datasets"])

with tab1:
    # Cancer type selection
    cancer_types = {
        "PAAD": "Pancreatic Adenocarcinoma",
        "BRCA": "Breast Cancer",
        "LUAD": "Lung Adenocarcinoma",
        "COAD": "Colorectal Adenocarcinoma",
        "GBM": "Glioblastoma Multiforme"
    }

    selected_cancer = st.selectbox(
        "Select Cancer Type for Data Collection",
        options=list(cancer_types.keys()),
        format_func=lambda x: f"{x} - {cancer_types[x]}"
    )

    # Data source selection
    st.header("Data Sources Selection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Genomic & Transcriptomic Data")
        fetch_tcga = st.checkbox("TCGA (Genomic Mutations & CNVs)", value=True)
        fetch_icgc = st.checkbox("ICGC (International Genomic Data)", value=True)
        fetch_gtex = st.checkbox("GTEx (RNA Expression)", value=True)

    with col2:
        st.subheader("Functional & CRISPR Data")
        fetch_depmap = st.checkbox("DepMap (CRISPR Screening)", value=True)
        fetch_proteomic = st.checkbox("PDB/UniProt (Protein Data)", value=True)

    # Data collection button
    if st.button("Start Data Collection", key="fetch_public_data"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_count = 0
        total_sources = sum([fetch_tcga, fetch_icgc, fetch_gtex, fetch_depmap, fetch_proteomic])
        
        if total_sources == 0:
            st.warning("Please select at least one data source.")
        else:
            # TCGA Data
            if fetch_tcga:
                status_text.text("Fetching TCGA mutation data...")
                try:
                    tcga_data = fetch_tcga_mutations(cancer_type=selected_cancer)
                    if tcga_data is not None:
                        st.session_state['datasets']['tcga'] = tcga_data
                        st.session_state['data_collection_progress']['tcga'] = True
                        progress_count += 1
                        progress_bar.progress(progress_count/total_sources)
                        status_text.text(f"TCGA data retrieved: {len(tcga_data)} mutation records")
                    else:
                        st.error("Failed to retrieve TCGA data.")
                except Exception as e:
                    st.error(f"Error fetching TCGA data: {str(e)}")
            
            # ICGC Data
            if fetch_icgc:
                status_text.text("Fetching ICGC data...")
                try:
                    icgc_data = fetch_icgc_data(cancer_type=selected_cancer)
                    if icgc_data is not None:
                        st.session_state['datasets']['icgc'] = icgc_data
                        st.session_state['data_collection_progress']['icgc'] = True
                        progress_count += 1
                        progress_bar.progress(progress_count/total_sources)
                        status_text.text(f"ICGC data retrieved: {len(icgc_data)} records")
                    else:
                        st.error("Failed to retrieve ICGC data.")
                except Exception as e:
                    st.error(f"Error fetching ICGC data: {str(e)}")
            
            # GTEx Data
            if fetch_gtex:
                status_text.text("Fetching GTEx expression data...")
                try:
                    gtex_data = fetch_gtex_data(tissue_type="Pancreas" if selected_cancer == "PAAD" else "Mixed")
                    if gtex_data is not None:
                        st.session_state['datasets']['gtex'] = gtex_data
                        st.session_state['data_collection_progress']['gtex'] = True
                        progress_count += 1
                        progress_bar.progress(progress_count/total_sources)
                        status_text.text(f"GTEx data retrieved: {gtex_data.shape[0]} expression records")
                    else:
                        st.error("Failed to retrieve GTEx data.")
                except Exception as e:
                    st.error(f"Error fetching GTEx data: {str(e)}")
            
            # DepMap Data
            if fetch_depmap:
                status_text.text("Fetching DepMap CRISPR screening data...")
                try:
                    depmap_data = fetch_depmap_data(cancer_type=selected_cancer)
                    if depmap_data is not None:
                        st.session_state['datasets']['depmap'] = depmap_data
                        st.session_state['data_collection_progress']['depmap'] = True
                        progress_count += 1
                        progress_bar.progress(progress_count/total_sources)
                        status_text.text(f"DepMap data retrieved: {depmap_data.shape[0]} CRISPR screening records")
                    else:
                        st.error("Failed to retrieve DepMap data.")
                except Exception as e:
                    st.error(f"Error fetching DepMap data: {str(e)}")
            
            # Proteomic Data
            if fetch_proteomic:
                status_text.text("Fetching protein structural data...")
                try:
                    proteomic_data = fetch_proteomic_data(cancer_type=selected_cancer)
                    if proteomic_data is not None:
                        st.session_state['datasets']['proteomic'] = proteomic_data
                        st.session_state['data_collection_progress']['proteomic'] = True
                        progress_count += 1
                        progress_bar.progress(progress_count/total_sources)
                        status_text.text(f"Proteomic data retrieved: {proteomic_data.shape[0]} protein records")
                    else:
                        st.error("Failed to retrieve proteomic data.")
                except Exception as e:
                    st.error(f"Error fetching proteomic data: {str(e)}")
            
            status_text.text(f"{progress_count} out of {total_sources} data collection tasks completed.")
    
    # Show data collection status
    if any(st.session_state['data_collection_progress'].values()):
        st.subheader("Data Collection Status")
        
        # Count successful and total tasks
        completed_tasks = sum(1 for v in st.session_state['data_collection_progress'].values() if v)
        total_tasks = len(st.session_state['data_collection_progress'])
        
        st.info(f"✅ {completed_tasks} out of {total_tasks} data collection tasks completed.")
        
        # Show success/failure for each source
        for source, status in st.session_state['data_collection_progress'].items():
            if status:
                st.success(f"✓ {source.upper()} data retrieved successfully")
            elif source in [k.lower() for k, v in {'tcga': fetch_tcga, 'icgc': fetch_icgc, 'gtex': fetch_gtex, 'depmap': fetch_depmap, 'proteomic': fetch_proteomic}.items() if v]:
                st.error(f"✗ Failed to retrieve {source.upper()} data.")
        
        # Data Preview
        st.subheader("Collected Data Preview")
        
        # Display tabs for each dataset
        if st.session_state['datasets']:
            tabs = st.tabs(list(source.upper() for source in st.session_state['datasets'].keys()))
            
            for i, source in enumerate(st.session_state['datasets'].keys()):
                with tabs[i]:
                    if source in st.session_state['datasets']:
                        st.dataframe(st.session_state['datasets'][source].head())
                        st.write(f"Shape: {st.session_state['datasets'][source].shape}")
                        
                        # Display column information
                        st.write("Column Types:")
                        col_types = pd.DataFrame({
                            'Column': st.session_state['datasets'][source].columns,
                            'Data Type': st.session_state['datasets'][source].dtypes,
                            'Missing Values': st.session_state['datasets'][source].isnull().sum().values
                        })
                        st.dataframe(col_types)

with tab2:
    st.header("Upload Custom Multi-Omics Data")
    st.markdown("""
    Upload your own genomic, transcriptomic, or proteomic data files for analysis.
    Supported file formats: CSV, TSV, Excel (.xlsx), JSON
    """)
    
    # Call the upload function
    upload_multi_omics_data()

with tab3:
    st.header("Available Datasets")
    st.markdown("""
    View and manage all uploaded datasets.
    """)
    
    # Refresh button
    if st.button("Refresh Dataset List", key="refresh_datasets"):
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
    # Get datasets from file catalog
    datasets_df = list_available_datasets()
    
    if datasets_df is not None and not datasets_df.empty:
        # Display dataset information
        st.dataframe(datasets_df[[
            'id', 'data_type', 'filename', 'description', 
            'sample_count', 'feature_count', 'created_at'
        ]])
        
        # Allow selection of a dataset for preview
        selected_dataset_id = st.selectbox(
            "Select a dataset to preview",
            options=datasets_df['id'].tolist(),
            format_func=lambda x: f"ID: {x} - {datasets_df[datasets_df['id'] == x]['filename'].values[0]} ({datasets_df[datasets_df['id'] == x]['data_type'].values[0]})"
        )
        
        if selected_dataset_id:
            # Get the selected dataset
            selected_row = datasets_df[datasets_df['id'] == selected_dataset_id].iloc[0]
            file_path = selected_row['file_path']
            
            if os.path.exists(file_path):
                # Determine file format
                _, ext = os.path.splitext(file_path)
                ext = ext.lower()
                
                if ext == '.csv':
                    df = pd.read_csv(file_path)
                elif ext in ['.tsv', '.txt']:
                    df = pd.read_csv(file_path, sep='\t')
                elif ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif ext == '.json':
                    df = pd.read_json(file_path)
                else:
                    st.error(f"Unsupported file format: {ext}")
                    df = None
                
                if df is not None:
                    st.subheader(f"Preview of {selected_row['filename']}")
                    st.dataframe(df.head())
                    
                    # Display basic stats
                    st.subheader("Dataset Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
                    # Display column information
                    st.write("Column Information:")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes,
                        'Missing Values': df.isnull().sum().values,
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info)
            else:
                st.error(f"File not found: {file_path}")
    else:
        st.info("No datasets found. Upload data using the 'Custom Data Upload' tab.")
