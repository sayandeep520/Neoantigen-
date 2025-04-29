import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.neoantigen_utils import (
    predict_mhc_binding, 
    predict_tcr_affinity, 
    screen_neoantigens,
    rank_neoantigens
)
from models.neoantigen_model import NeoantiGenAI
from utils.visualization import plot_binding_affinity, plot_neoantigen_ranking

# Page configuration
st.set_page_config(
    page_title="Neoantigen Prediction | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Session state check
if 'preprocessed_data' not in st.session_state or not st.session_state['preprocessed_data']:
    st.warning("No preprocessed data available. Please complete data preprocessing first.")
    preprocess_button = st.button("Go to Data Preprocessing")
    if preprocess_button:
        st.switch_page("pages/2_Data_Preprocessing.py")
    st.stop()

# Initialize session states for this page
if 'neoantigens' not in st.session_state:
    st.session_state['neoantigens'] = None
if 'neoantigen_model' not in st.session_state:
    st.session_state['neoantigen_model'] = None

# Main header
st.title("🔬 AI-Driven Neoantigen Prediction")
st.markdown("""
This module uses deep learning to identify tumor-specific peptides (neoantigens) that can be 
targeted by the immune system. These neoantigens are crucial for developing personalized cancer immunotherapies.
""")

# Input data selection
st.header("Input Data Selection")

# Check for mutation data
mutation_data_available = False
if 'tcga' in st.session_state['preprocessed_data']:
    mutation_data_available = True
    mutation_data_source = 'tcga'
elif 'icgc' in st.session_state['preprocessed_data']:
    mutation_data_available = True
    mutation_data_source = 'icgc'

if not mutation_data_available:
    st.warning("No mutation data available. Neoantigen prediction requires genomic mutation data.")
    st.stop()

# HLA type selection
st.subheader("HLA Type Selection")
st.markdown("""
HLA (Human Leukocyte Antigen) types determine which peptides can be presented to T cells.
You can either use a predefined HLA type or enter custom HLA alleles.
""")

hla_selection_method = st.radio(
    "HLA Type Selection Method",
    options=["Common HLA Types", "Custom HLA Alleles"],
    index=0
)

if hla_selection_method == "Common HLA Types":
    common_hla_types = {
        "HLA-A*02:01 (Common in many populations)": ["HLA-A*02:01"],
        "HLA-A*01:01, HLA-B*08:01 (Northern European)": ["HLA-A*01:01", "HLA-B*08:01"],
        "HLA-A*24:02, HLA-B*35:01 (East Asian)": ["HLA-A*24:02", "HLA-B*35:01"],
        "HLA-A*03:01, HLA-B*07:02 (Caucasian)": ["HLA-A*03:01", "HLA-B*07:02"],
        "HLA Supertype Representatives": ["HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*24:02", "HLA-B*07:02", "HLA-B*44:03"]
    }
    
    selected_hla_preset = st.selectbox(
        "Select HLA Type",
        options=list(common_hla_types.keys())
    )
    
    hla_alleles = common_hla_types[selected_hla_preset]
else:
    custom_hla_input = st.text_input(
        "Enter HLA Alleles (comma separated)",
        value="HLA-A*02:01,HLA-B*07:02",
        help="Example: HLA-A*02:01,HLA-B*07:02"
    )
    
    hla_alleles = [allele.strip() for allele in custom_hla_input.split(",")]

# Prediction parameters
st.header("Neoantigen Prediction Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("MHC Binding Parameters")
    binding_threshold = st.slider(
        "MHC Binding Affinity Threshold (nM)",
        min_value=50,
        max_value=1000,
        value=500,
        step=50,
        help="Lower values = higher binding affinity"
    )
    
    peptide_lengths = st.multiselect(
        "Peptide Lengths",
        options=[8, 9, 10, 11, 12, 13, 14, 15],
        default=[9, 10, 11],
        help="Common peptide lengths for MHC-I binding"
    )

with col2:
    st.subheader("Filtering Parameters")
    expression_threshold = st.slider(
        "Gene Expression Threshold (Percentile)",
        min_value=0,
        max_value=100,
        value=50,
        help="Genes with expression above this percentile will be considered"
    )
    
    tcr_affinity_threshold = st.slider(
        "TCR Recognition Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum predicted probability of TCR recognition"
    )

# Advanced options
with st.expander("Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_algorithm = st.selectbox(
            "MHC Binding Prediction Algorithm",
            options=["NetMHCpan", "MHCflurry", "DeepHLApan"],
            index=0,
            help="Algorithm to predict MHC binding"
        )
        
        mutation_types = st.multiselect(
            "Mutation Types to Consider",
            options=["Missense", "Frameshift", "Indel", "Nonsense"],
            default=["Missense"],
            help="Types of mutations to consider for neoantigen prediction"
        )
    
    with col2:
        include_gene_expression = st.checkbox(
            "Include Gene Expression in Ranking",
            value=True,
            help="Consider gene expression levels when ranking neoantigens"
        )
        
        include_proteasomal_cleavage = st.checkbox(
            "Consider Proteasomal Cleavage",
            value=True,
            help="Predict proteasomal cleavage for neoantigen processing"
        )
        
        include_tap_transport = st.checkbox(
            "Consider TAP Transport",
            value=True,
            help="Predict TAP transport efficiency"
        )

# Run neoantigen prediction
if st.button("Run Neoantigen Prediction"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Prepare mutation data
    status_text.text("Preparing mutation data...")
    
    mutation_data = st.session_state['preprocessed_data'][mutation_data_source]
    
    # Progress update
    progress_bar.progress(0.1)
    
    # Step 2: Initialize neoantigen prediction model
    status_text.text("Initializing neoantigen prediction model...")
    
    neoantigen_model = NeoantiGenAI(
        algorithm=prediction_algorithm,
        include_gene_expression=include_gene_expression,
        include_proteasomal_cleavage=include_proteasomal_cleavage,
        include_tap_transport=include_tap_transport
    )
    
    st.session_state['neoantigen_model'] = neoantigen_model
    
    # Progress update
    progress_bar.progress(0.2)
    
    # Step 3: Predict MHC binding
    status_text.text("Predicting MHC binding for neoantigens...")
    
    mhc_binding_results = predict_mhc_binding(
        mutation_data=mutation_data,
        hla_alleles=hla_alleles,
        binding_threshold=binding_threshold,
        peptide_lengths=peptide_lengths,
        prediction_model=neoantigen_model
    )
    
    # Progress update
    progress_bar.progress(0.5)
    
    # Step 4: Predict TCR affinity
    status_text.text("Predicting TCR recognition for potential neoantigens...")
    
    tcr_affinity_results = predict_tcr_affinity(
        mhc_binding_results=mhc_binding_results,
        tcr_affinity_threshold=tcr_affinity_threshold,
        prediction_model=neoantigen_model
    )
    
    # Progress update
    progress_bar.progress(0.7)
    
    # Step 5: Screen neoantigens
    status_text.text("Screening neoantigens based on criteria...")
    
    screened_neoantigens = screen_neoantigens(
        tcr_affinity_results=tcr_affinity_results,
        expression_threshold=expression_threshold,
        mutation_types=mutation_types
    )
    
    # Progress update
    progress_bar.progress(0.9)
    
    # Step 6: Rank neoantigens
    status_text.text("Ranking neoantigens by predicted immunogenicity...")
    
    ranked_neoantigens = rank_neoantigens(
        screened_neoantigens=screened_neoantigens,
        include_gene_expression=include_gene_expression
    )
    
    # Store results
    st.session_state['neoantigens'] = ranked_neoantigens
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("✅ Neoantigen prediction completed successfully!")

# Display neoantigen results
if st.session_state['neoantigens'] is not None:
    st.header("Predicted Neoantigens")
    
    # Filter to top neoantigens
    top_n = st.slider("Show Top N Neoantigens", min_value=5, max_value=100, value=20)
    top_neoantigens = st.session_state['neoantigens'].head(top_n)
    
    # Display results
    st.dataframe(top_neoantigens)
    
    # Visualize results
    st.subheader("Neoantigen Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("MHC Binding Affinity Distribution")
        plot_binding_affinity(top_neoantigens)
    
    with col2:
        st.write("Neoantigen Ranking by Immunogenicity Score")
        plot_neoantigen_ranking(top_neoantigens)
    
    # Detailed view of selected neoantigen
    st.subheader("Detailed Neoantigen View")
    
    selected_neoantigen_idx = st.selectbox(
        "Select Neoantigen for Detailed View",
        options=range(len(top_neoantigens)),
        format_func=lambda x: f"Neoantigen {x+1}: {top_neoantigens.iloc[x]['peptide']} ({top_neoantigens.iloc[x]['gene']})"
    )
    
    selected_neoantigen = top_neoantigens.iloc[selected_neoantigen_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Neoantigen Details")
        details = {
            "Peptide": selected_neoantigen['peptide'],
            "Source Gene": selected_neoantigen['gene'],
            "Mutation": selected_neoantigen['mutation'],
            "HLA Allele": selected_neoantigen['hla_allele'],
            "MHC Binding Affinity": f"{selected_neoantigen['mhc_affinity']:.2f} nM",
            "TCR Recognition Probability": f"{selected_neoantigen['tcr_probability']:.4f}",
            "Immunogenicity Score": f"{selected_neoantigen['immunogenicity_score']:.4f}",
            "Overall Rank": selected_neoantigen_idx + 1
        }
        
        for k, v in details.items():
            st.write(f"**{k}:** {v}")
    
    with col2:
        st.write("Sequence Visualization")
        
        # Create a simple sequence visualization
        peptide = selected_neoantigen['peptide']
        wild_type = selected_neoantigen['wild_type'] if 'wild_type' in selected_neoantigen else "N/A"
        
        fig, ax = plt.subplots(figsize=(10, 3))
        
        # Amino acid colors
        aa_colors = {
            'A': '#77dd77', 'R': '#7777dd', 'N': '#dd77dd', 'D': '#dd7777',
            'C': '#dddd77', 'E': '#77dddd', 'Q': '#ff7777', 'G': '#77ff77',
            'H': '#7777ff', 'I': '#ffff77', 'L': '#77ffff', 'K': '#ff77ff',
            'M': '#ffff77', 'F': '#77ffff', 'P': '#ff77ff', 'S': '#77dd77',
            'T': '#7777dd', 'W': '#dd77dd', 'Y': '#dd7777', 'V': '#dddd77'
        }
        
        # Render mutated peptide
        for i, aa in enumerate(peptide):
            color = aa_colors.get(aa, '#aaaaaa')
            ax.add_patch(plt.Rectangle((i, 0.5), 0.9, 0.9, color=color))
            ax.text(i + 0.45, 0.95, aa, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Render wild type peptide if available
        if wild_type != "N/A":
            for i, aa in enumerate(wild_type):
                color = aa_colors.get(aa, '#aaaaaa')
                ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, color=color, alpha=0.5))
                ax.text(i + 0.45, 0.45, aa, ha='center', va='center', fontsize=14)
        
        # Show mutation position
        if 'mutation_position' in selected_neoantigen:
            pos = selected_neoantigen['mutation_position']
            ax.add_patch(plt.Rectangle((pos, 0.4), 0.9, 1.1, fill=False, edgecolor='red', linewidth=2))
        
        ax.set_xlim(-0.5, len(peptide) + 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0.25, 0.75])
        ax.set_yticklabels(['Wild Type', 'Mutated'])
        ax.set_xticks(range(len(peptide)))
        ax.set_xticklabels(range(1, len(peptide) + 1))
        ax.set_xlabel('Position')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Download All Neoantigens (CSV)",
            data=st.session_state['neoantigens'].to_csv(index=False),
            file_name="predicted_neoantigens.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            "Download Top Neoantigens (CSV)",
            data=top_neoantigens.to_csv(index=False),
            file_name="top_neoantigens.csv",
            mime="text/csv"
        )
    
    # Next steps
    st.markdown("---")
    st.header("Next Steps")
    st.markdown("""
    With neoantigens identified, you can now:
    1. **Synthetic Biology Simulation** - Design bacterial circuits for drug delivery
    2. **Therapy Response Prediction** - Predict patient responses to the designed therapy
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        synbio_button = st.button("Proceed to Synthetic Biology Simulation")
        if synbio_button:
            # Check if the page exists before switching
            if os.path.exists("./pages/5_Synthetic_Biology_Simulation.py"):
                st.switch_page("pages/5_Synthetic_Biology_Simulation.py")
            else:
                st.error("Synthetic Biology Simulation page not available.")
    
    with col2:
        response_button = st.button("Proceed to Therapy Response Prediction")
        if response_button:
            # Check if the page exists before switching
            if os.path.exists("./pages/9_Therapy_Response_Prediction.py"):
                st.switch_page("pages/9_Therapy_Response_Prediction.py")
            else:
                st.error("Therapy Response Prediction page not available.")
                st.info("Alternatively, you can proceed to Federated Learning.")
                if st.button("Go to Federated Learning"):
                    st.switch_page("pages/10_Federated_Learning.py")
