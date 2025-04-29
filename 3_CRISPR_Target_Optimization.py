import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.crispr_utils import (
    design_sgrna, 
    predict_off_target_effects, 
    evaluate_on_target_efficiency,
    rank_crispr_targets
)
from utils.model_trainer import train_crispr_model, evaluate_model
from models.crispr_model import CRISPRTargetModel
from utils.visualization import plot_crispr_efficiency, plot_off_target_distribution
from utils.gemini_integration import GeminiIntegration

# Page configuration
st.set_page_config(
    page_title="CRISPR Target Optimization | AI-Driven CRISPR Cancer Immunotherapy Platform",
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
if 'crispr_targets' not in st.session_state:
    st.session_state['crispr_targets'] = None
if 'crispr_model' not in st.session_state:
    st.session_state['crispr_model'] = None

# Main header
st.title("🎯 AI-Driven CRISPR Target Optimization")
st.markdown("""
This module uses machine learning to identify optimal CRISPR-Cas9 target sites for gene editing in cancer therapy.
The AI model balances on-target efficiency with minimizing off-target effects.
""")

# Target gene selection
st.header("Target Gene Selection")

# Get genes from preprocessed data (assuming TCGA or similar data is available)
available_genes = []
if 'tcga' in st.session_state['preprocessed_data']:
    # Extract gene names from the dataset
    if 'gene' in st.session_state['preprocessed_data']['tcga'].columns:
        available_genes = st.session_state['preprocessed_data']['tcga']['gene'].unique().tolist()
    elif 'hugo_symbol' in st.session_state['preprocessed_data']['tcga'].columns:
        available_genes = st.session_state['preprocessed_data']['tcga']['hugo_symbol'].unique().tolist()

# If no genes found, provide some common cancer genes
if not available_genes:
    available_genes = [
        "TP53", "KRAS", "CDKN2A", "SMAD4", "BRCA1", "BRCA2", "EGFR", 
        "ALK", "ROS1", "BRAF", "APC", "PIK3CA", "PTEN"
    ]

# Allow single or multiple gene selection
target_selection_mode = st.radio(
    "Target Selection Mode",
    options=["Single Gene", "Multiple Genes"],
    index=0
)

if target_selection_mode == "Single Gene":
    selected_gene = st.selectbox(
        "Select Target Gene",
        options=available_genes
    )
    target_genes = [selected_gene]
else:
    target_genes = st.multiselect(
        "Select Target Genes",
        options=available_genes,
        default=[available_genes[0]] if available_genes else []
    )

if not target_genes:
    st.warning("Please select at least one target gene.")
    st.stop()

# CRISPR design parameters
st.header("CRISPR Design Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("On-Target Parameters")
    gc_content_range = st.slider(
        "GC Content (%)",
        min_value=30, 
        max_value=80, 
        value=(40, 60),
        help="GC content range for optimal sgRNA performance"
    )
    
    efficiency_threshold = st.slider(
        "Minimum Efficiency Score",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        help="Minimum predicted efficiency score (0-1)"
    )

with col2:
    st.subheader("Off-Target Parameters")
    max_off_targets = st.slider(
        "Maximum Off-Target Count",
        min_value=0,
        max_value=10,
        value=3,
        help="Maximum number of potential off-target sites"
    )
    
    min_mismatch_distance = st.slider(
        "Minimum Mismatch Distance",
        min_value=1,
        max_value=5,
        value=3,
        help="Minimum number of mismatches for off-target sites"
    )

# Advanced options
with st.expander("Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        pam_sequence = st.text_input(
            "PAM Sequence",
            value="NGG",
            help="Protospacer Adjacent Motif (PAM) sequence"
        )
        
        guide_length = st.slider(
            "Guide RNA Length",
            min_value=18,
            max_value=25,
            value=20,
            help="Length of the guide RNA sequence"
        )
    
    with col2:
        prediction_model = st.selectbox(
            "Prediction Model",
            options=["DeepCRISPR", "Azimuth", "CRISPRscan"],
            index=0,
            help="Model to predict on-target efficiency"
        )
        
        off_target_algorithm = st.selectbox(
            "Off-Target Prediction Algorithm",
            options=["Cas-OFFinder", "CFD Score", "MIT Score"],
            index=0,
            help="Algorithm to predict off-target effects"
        )

# AI Model Training
st.header("AI Model for CRISPR Target Optimization")

train_model = st.checkbox("Train Custom CRISPR Target Model", value=True)

if train_model:
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%.4f",
            help="Learning rate for model training"
        )
        
        epochs = st.slider(
            "Training Epochs",
            min_value=10,
            max_value=500,
            value=100,
            help="Number of training epochs"
        )
    
    with col2:
        batch_size = st.slider(
            "Batch Size",
            min_value=16,
            max_value=256,
            value=64,
            step=16,
            help="Batch size for model training"
        )
        
        validation_split = st.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            help="Fraction of data to use for validation"
        )

# Run CRISPR target optimization
if st.button("Run CRISPR Target Optimization"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Select training data
    status_text.text("Preparing CRISPR training data...")
    
    # Use DepMap data if available, otherwise fallback
    if 'depmap' in st.session_state['preprocessed_data']:
        training_data = st.session_state['preprocessed_data']['depmap']
    else:
        st.warning("DepMap data not available, using synthetic CRISPR data for demonstration.")
        # Create some synthetic data for demonstration
        training_data = pd.DataFrame({
            'gene': available_genes * 10,
            'sequence': ['ACGT' * 5] * (len(available_genes) * 10),
            'efficiency_score': np.random.random(len(available_genes) * 10),
            'off_target_count': np.random.randint(0, 10, len(available_genes) * 10)
        })
    
    progress_bar.progress(0.1)
    
    # Step 2: Train CRISPR model if requested
    if train_model:
        status_text.text("Training CRISPR target optimization model...")
        model, training_history = train_crispr_model(
            training_data,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        st.session_state['crispr_model'] = model
        
        # Show training metrics
        st.subheader("Model Training Results")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                training_history, 
                y=['loss', 'val_loss'],
                labels={'value': 'Loss', 'variable': 'Dataset', 'index': 'Epoch'},
                title='Training & Validation Loss'
            )
            st.plotly_chart(fig)
        
        with col2:
            fig = px.line(
                training_history, 
                y=['accuracy', 'val_accuracy'], 
                labels={'value': 'Accuracy', 'variable': 'Dataset', 'index': 'Epoch'},
                title='Training & Validation Accuracy'
            )
            st.plotly_chart(fig)
    else:
        # Use pre-trained model
        status_text.text("Loading pre-trained CRISPR model...")
        model = CRISPRTargetModel()
    
    progress_bar.progress(0.4)
    
    # Step 3: Design sgRNAs for target genes
    all_sgrnas = []
    for idx, gene in enumerate(target_genes):
        status_text.text(f"Designing sgRNAs for {gene} ({idx+1}/{len(target_genes)})...")
        
        # Design candidate sgRNAs
        sgrnas = design_sgrna(
            gene,
            gc_content_range=gc_content_range,
            guide_length=guide_length,
            pam_sequence=pam_sequence
        )
        
        # Predict on-target efficiency
        sgrnas = evaluate_on_target_efficiency(
            sgrnas,
            model=model,
            prediction_method=prediction_model
        )
        
        # Filter by efficiency threshold
        sgrnas = sgrnas[sgrnas['efficiency_score'] >= efficiency_threshold]
        
        # Predict off-target effects
        sgrnas = predict_off_target_effects(
            sgrnas,
            algorithm=off_target_algorithm,
            min_mismatch_distance=min_mismatch_distance
        )
        
        # Filter by off-target count
        sgrnas = sgrnas[sgrnas['off_target_count'] <= max_off_targets]
        
        all_sgrnas.append(sgrnas)
        progress_bar.progress(0.4 + 0.5 * (idx + 1) / len(target_genes))
    
    # Combine all sgRNAs
    combined_sgrnas = pd.concat(all_sgrnas, ignore_index=True)
    
    # Rank CRISPR targets
    status_text.text("Ranking CRISPR targets...")
    ranked_targets = rank_crispr_targets(combined_sgrnas)
    
    # Store results
    st.session_state['crispr_targets'] = ranked_targets
    
    progress_bar.progress(1.0)
    status_text.text("✅ CRISPR target optimization completed successfully!")

# Display CRISPR target results
if st.session_state['crispr_targets'] is not None:
    st.header("Optimized CRISPR Targets")
    
    # Filter to top targets
    top_n = st.slider("Show Top N Targets", min_value=5, max_value=50, value=10)
    top_targets = st.session_state['crispr_targets'].head(top_n)
    
    # Display results
    st.dataframe(top_targets)
    
    # Visualize results
    st.subheader("Target Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("On-Target Efficiency Scores")
        plot_crispr_efficiency(top_targets)
    
    with col2:
        st.write("Off-Target Distribution")
        plot_off_target_distribution(top_targets)
    
    # Detailed view of selected target
    st.subheader("Detailed Target View")
    
    selected_target_idx = st.selectbox(
        "Select Target for Detailed View",
        options=range(len(top_targets)),
        format_func=lambda x: f"Target {x+1}: {top_targets.iloc[x]['gene']} - {top_targets.iloc[x]['sequence'][:10]}..."
    )
    
    selected_target = top_targets.iloc[selected_target_idx]
    
    tab1, tab2 = st.tabs(["Basic Details", "AI-Powered Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Target Details")
            details = {
                "Gene": selected_target['gene'],
                "sgRNA Sequence": selected_target['sequence'],
                "PAM": pam_sequence,
                "On-Target Efficiency": f"{selected_target['efficiency_score']:.4f}",
                "Off-Target Count": selected_target['off_target_count'],
                "GC Content": f"{selected_target['gc_content']:.2f}%",
                "Overall Rank": selected_target_idx + 1
            }
            
            for k, v in details.items():
                st.write(f"**{k}:** {v}")
        
        with col2:
            st.write("Sequence Visualization")
            
            # Create a simple sequence visualization
            sequence = selected_target['sequence']
            fig, ax = plt.subplots(figsize=(10, 2))
            for i, nucleotide in enumerate(sequence):
                color = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', 'U': 'red'}.get(nucleotide, 'gray')
                ax.text(i, 0, nucleotide, fontsize=12, ha='center', va='center', color='white',
                       bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.3'))
            ax.set_xlim(-1, len(sequence))
            ax.set_ylim(-1, 1)
            ax.axis('off')
            st.pyplot(fig)
    
    # AI-Powered Analysis tab with Gemini integration
    with tab2:
        st.subheader("AI-Powered CRISPR Target Analysis")
        
        # Check for Gemini API key
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        
        if not gemini_api_key:
            st.info("💡 **AI-powered analysis requires a Gemini API key.** This enables advanced analysis of CRISPR targets using Google's Gemini model.")
            
            # Option to add API key
            api_key_input = st.text_input(
                "Enter Gemini API Key",
                type="password",
                help="Your Gemini API key for accessing Google's AI models"
            )
            
            if api_key_input and st.button("Save API Key"):
                # In a real app, this would securely store the API key
                # For this demo, we'll just use it for the current session
                os.environ["GEMINI_API_KEY"] = api_key_input
                st.success("API key saved for this session.")
                st.rerun()  # Refresh to show AI analysis
        
        # Show AI analysis if we have an API key
        if gemini_api_key or os.environ.get("GEMINI_API_KEY"):
            # Initialize Gemini integration
            gemini = GeminiIntegration(api_key=os.environ.get("GEMINI_API_KEY"))
            
            if gemini.is_available():
                # Prepare context for analysis
                context = {
                    "gene": selected_target['gene'],
                    "efficiency_score": float(selected_target['efficiency_score']),
                    "off_target_count": int(selected_target['off_target_count']),
                    "gc_content": float(selected_target['gc_content'])
                }
                
                # Check if analysis is already in session state
                analysis_key = f"gemini_analysis_{selected_target['gene']}_{selected_target['sequence']}"
                
                if analysis_key not in st.session_state:
                    with st.spinner("Analyzing CRISPR target with Google's Gemini model..."):
                        # Run analysis
                        analysis_result = gemini.analyze_crispr_target(
                            selected_target['sequence'],
                            context=context
                        )
                        # Store in session state
                        st.session_state[analysis_key] = analysis_result
                else:
                    # Use cached analysis
                    analysis_result = st.session_state[analysis_key]
                
                # Display analysis results
                if "error" in analysis_result:
                    st.error(f"API Error: {analysis_result['message']}")
                else:
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ai_efficiency = analysis_result.get('efficiency_score', 0.0)
                        st.metric("AI Efficiency Score", f"{ai_efficiency:.2f}")
                    
                    with col2:
                        off_target_risk = analysis_result.get('off_target_risk', 'unknown')
                        st.metric("Off-Target Risk", off_target_risk.upper())
                    
                    with col3:
                        gc_content = analysis_result.get('gc_content', 0.0)
                        st.metric("GC Content", f"{gc_content:.1f}%")
                    
                    # Display sequence issues
                    if 'sequence_issues' in analysis_result and analysis_result['sequence_issues']:
                        st.subheader("Potential Sequence Issues")
                        for issue in analysis_result['sequence_issues']:
                            st.markdown(f"- {issue}")
                    else:
                        st.markdown("**No major sequence issues detected.**")
                    
                    # Display recommendations
                    if 'recommendations' in analysis_result and analysis_result['recommendations']:
                        st.subheader("AI Recommendations")
                        for rec in analysis_result['recommendations']:
                            st.markdown(f"- {rec}")
                    
                    # Option to analyze gene editing strategy
                    st.subheader("Full Gene Editing Strategy Analysis")
                    
                    if st.button("Analyze Complete Gene Editing Strategy"):
                        with st.spinner("Analyzing gene editing strategy..."):
                            # Get disease context
                            disease_context = "Cancer therapy targeting driver mutations"
                            edit_approach = f"CRISPR-Cas9 targeting {', '.join(target_genes)} with sgRNA optimization"
                            
                            # Run analysis
                            strategy_analysis = gemini.analyze_gene_editing_strategy(
                                target_genes,
                                disease_context, 
                                edit_approach
                            )
                            
                            if "error" in strategy_analysis:
                                st.error(f"API Error: {strategy_analysis['message']}")
                            else:
                                # Display efficacy rating
                                st.metric("Efficacy Rating", f"{strategy_analysis.get('efficacy_rating', 0.0) * 100:.1f}%")
                                
                                # Display safety concerns
                                if 'safety_concerns' in strategy_analysis and strategy_analysis['safety_concerns']:
                                    st.subheader("Safety Considerations")
                                    safety_df = pd.DataFrame(strategy_analysis['safety_concerns'])
                                    st.dataframe(safety_df)
                                
                                # Display delivery recommendations
                                if 'delivery_recommendations' in strategy_analysis and strategy_analysis['delivery_recommendations']:
                                    st.subheader("Delivery Recommendations")
                                    for rec in strategy_analysis['delivery_recommendations']:
                                        st.markdown(f"- {rec}")
                                
                                # Display alternative targets
                                if 'alternative_targets' in strategy_analysis and strategy_analysis['alternative_targets']:
                                    st.subheader("Alternative Targets to Consider")
                                    alt_df = pd.DataFrame(strategy_analysis['alternative_targets'])
                                    st.dataframe(alt_df)
                                
                                # Display overall assessment
                                if 'overall_assessment' in strategy_analysis:
                                    st.subheader("Overall Assessment")
                                    st.write(strategy_analysis['overall_assessment'])
            else:
                st.error("Failed to initialize Gemini integration. Please check your API key and try again.")
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Download All CRISPR Targets (CSV)",
            data=st.session_state['crispr_targets'].to_csv(index=False),
            file_name="crispr_targets.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            "Download Top CRISPR Targets (CSV)",
            data=top_targets.to_csv(index=False),
            file_name="top_crispr_targets.csv",
            mime="text/csv"
        )
    
    # Next steps
    st.markdown("---")
    st.header("Next Steps")
    st.markdown("""
    With optimized CRISPR targets identified, you can now:
    1. **Neoantigen Prediction** - Identify potential neoantigens for immunotherapy
    2. **Synthetic Biology Simulation** - Design bacterial circuits for drug delivery
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        neoantigen_button = st.button("Proceed to Neoantigen Prediction")
        if neoantigen_button:
            st.switch_page("pages/4_Neoantigen_Prediction.py")
    
    with col2:
        synbio_button = st.button("Proceed to Synthetic Biology Simulation")
        if synbio_button:
            st.switch_page("pages/5_Synthetic_Biology_Simulation.py")
