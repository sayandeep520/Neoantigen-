import streamlit as st
import os
import pandas as pd
import numpy as np
import glob

# Page configuration
st.set_page_config(
    page_title="AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customize sidebar
# Hide default page names in sidebar
css = '''
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    .custom-sidebar {
        padding: 1rem;
    }
    .custom-sidebar-header {
        font-weight: bold;
        font-size: 1.1em;
        margin-top: 15px;
        margin-bottom: 5px;
        color: #0077b6;
        border-bottom: 1px solid #e6e6e6;
        padding-bottom: 5px;
    }
    .sidebar-link {
        padding: 5px 0;
        text-decoration: none;
        color: #333;
        display: block;
    }
    .sidebar-link:hover {
        background-color: #f0f2f6;
        border-radius: 4px;
    }
    .active-link {
        background-color: #e6f0f6;
        border-radius: 4px;
        font-weight: bold;
        color: #0077b6;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Use Streamlit's built-in sidebar navigation
with st.sidebar:
    st.title("AI-CRISPR Platform v2.5")
    
    # Show platform logo
    st.markdown("## 🧬 Navigation")
    st.markdown("---")
    
    # Data Management section
    st.markdown("### Data Management")
    data_pages = [
        ("📊 Data Collection", "/1_Data_Collection"),
        ("🔍 Data Preprocessing", "/2_Data_Preprocessing"),
        ("🧩 Multi-Omics Integration", "/8_Multi_Omics_Integration")
    ]
    for page_name, page_path in data_pages:
        if st.button(page_name, key=f"data_{page_path}"):
            st.switch_page(f"pages{page_path}.py")
    
    # CRISPR Tools section
    st.markdown("### CRISPR Tools")
    crispr_pages = [
        ("🎯 Target Optimization", "/3_CRISPR_Target_Optimization"),
        ("📡 Realtime Data", "/6_Realtime_CRISPR_Data"),
        ("🧪 Model Training", "/9_CRISPR_Model_Training"),
        ("🤖 RL Optimization", "/11_CRISPR_Reinforcement_Learning")
    ]
    for page_name, page_path in crispr_pages:
        if st.button(page_name, key=f"crispr_{page_path}"):
            st.switch_page(f"pages{page_path}.py")
    
    # Immunotherapy section
    st.markdown("### Immunotherapy")
    immuno_pages = [
        ("🔬 Neoantigen Prediction", "/4_Neoantigen_Prediction"),
        ("🧠 RL Neoantigen Selection", "/12_Neoantigen_Reinforcement_Learning"),
        ("📈 Response Prediction", "/9_Therapy_Response_Prediction")
    ]
    for page_name, page_path in immuno_pages:
        if st.button(page_name, key=f"immuno_{page_path}"):
            st.switch_page(f"pages{page_path}.py")
    
    # Advanced Tools section
    st.markdown("### Advanced Tools")
    advanced_pages = [
        ("🧫 Synthetic Biology", "/5_Synthetic_Biology_Simulation"),
        ("⚡ Circuit Modeling", "/7_Advanced_Circuit_Modeling"),
        ("🔄 Federated Learning", "/10_Federated_Learning")
    ]
    for page_name, page_path in advanced_pages:
        if st.button(page_name, key=f"advanced_{page_path}"):
            st.switch_page(f"pages{page_path}.py")
            
    st.markdown("---")

# Main title
st.title("🧬 AI-Driven CRISPR Cancer Immunotherapy Platform")

# Introduction
st.markdown("""
## Platform Overview
This comprehensive platform integrates multi-omics data analysis, CRISPR target optimization, 
neoantigen prediction, and therapy response prediction for cancer immunotherapy research.

### Key Features
- **Multi-omics Data Integration**: Automated collection from TCGA, ICGC, GTEx, and DepMap
- **AI-Driven CRISPR Target Prediction**: Identify optimal sgRNAs for tumor-specific gene editing
- **Real-time CRISPR Data Integration**: Connect to experimental platforms and analyze CRISPR results in real-time
- **Neoantigen Discovery & Immune Modeling**: Predict TCR-peptide binding affinity
- **Synthetic Biology Circuit Design**: Simulate bacterial circuits for controlled drug release
- **Therapy Response Prediction**: Multi-omics based patient outcome forecasting
""")

# Data availability section with static data (for faster loading)
st.header("📊 Data Resource Status")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Genomic Data Sources")
    # Use static data for initial load
    status_df = pd.DataFrame({
        "Database": ["TCGA", "ICGC", "GTEx"],
        "Status": [
            "✅ Available",
            "✅ Available",
            "✅ Available"
        ]
    })
    st.table(status_df)

with col2:
    st.subheader("Functional & CRISPR Data")
    # Use static data for initial load
    func_df = pd.DataFrame({
        "Database": ["DepMap", "PDB/UniProt", "CRISPR Screening"],
        "Status": [
            "✅ Available",
            "✅ Available",
            "✅ Available"
        ]
    })
    st.table(func_df)

# Dynamic check button for data sources (optional check)
if st.button("Check Data Sources Availability"):
    # Import only when button is clicked
    try:
        from utils.data_fetcher import check_dataset_availability
        
        with st.spinner("Checking data sources..."):
            tcga_status = check_dataset_availability("TCGA")
            icgc_status = check_dataset_availability("ICGC")
            gtex_status = check_dataset_availability("GTEx")
            depmap_status = check_dataset_availability("DepMap")
            proteomic_status = check_dataset_availability("PDB")
            crispr_status = check_dataset_availability("CRISPR")
            
            col1, col2 = st.columns(2)
            with col1:
                status_df = pd.DataFrame({
                    "Database": ["TCGA", "ICGC", "GTEx"],
                    "Status": [
                        "✅ Available" if tcga_status else "❌ Unavailable",
                        "✅ Available" if icgc_status else "❌ Unavailable",
                        "✅ Available" if gtex_status else "❌ Unavailable"
                    ]
                })
                st.table(status_df)
            
            with col2:
                func_df = pd.DataFrame({
                    "Database": ["DepMap", "PDB/UniProt", "CRISPR Screening"],
                    "Status": [
                        "✅ Available" if depmap_status else "❌ Unavailable",
                        "✅ Available" if proteomic_status else "❌ Unavailable",
                        "✅ Available" if crispr_status else "❌ Unavailable"
                    ]
                })
                st.table(func_df)
    except Exception as e:
        st.error(f"Error checking data sources: {str(e)}")

# Workflow Diagram 
st.header("🔄 Integrated AI Workflow")

# Include the project overview render function only when requested
if st.button("Show Workflow Diagram"):
    try:
        from utils.visualization import render_project_overview
        render_project_overview()
    except Exception as e:
        st.error(f"Error rendering workflow diagram: {str(e)}")
        # Fallback simple diagram
        st.info("AI-Driven CRISPR Workflow Diagram will be displayed here.")

# Getting Started
st.header("🚀 Getting Started")
st.markdown("""
1. Navigate to **Data Collection** to fetch multi-omics data
2. Process and normalize data in **Data Preprocessing**
3. Train AI models for **CRISPR Target Optimization**
4. Connect to experimental data in **Realtime CRISPR Data** for live integration
5. Predict potential **Neoantigens** for immunotherapy
6. Design synthetic biology circuits in the **Synthetic Biology Simulation** module
7. Predict therapy response using the **Therapy Response Prediction** module
8. Process and integrate datasets with the **Multi-Omics Integration** module
9. Train and evaluate models in the **CRISPR Model Training** module
""")

# Footer
st.markdown("---")
st.markdown("© 2023 AI-Driven CRISPR Cancer Immunotherapy Platform")
