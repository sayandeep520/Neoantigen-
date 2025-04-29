import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from datetime import datetime
import io
import base64
from PIL import Image

# Import CRISPR utils
from utils.realtime_crispr_utils import RealtimeCRISPRMonitor
from models.crispr_model import CRISPRTargetModel

# Set page configuration
st.set_page_config(
    page_title="Real-time CRISPR Data Integration",
    page_icon="🧬",
    layout="wide"
)

def get_session_state_var(key, default_value=None):
    """Get variable from session state, initializing it if needed"""
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def download_dataframe_as_csv(df, filename):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    # Initialize session state variables
    if 'crispr_monitor' not in st.session_state:
        st.session_state.crispr_monitor = RealtimeCRISPRMonitor()
    
    if 'experiments' not in st.session_state:
        st.session_state.experiments = []
    
    if 'selected_experiment' not in st.session_state:
        st.session_state.selected_experiment = None
    
    if 'experiment_data' not in st.session_state:
        st.session_state.experiment_data = {}
    
    if 'comparison_experiments' not in st.session_state:
        st.session_state.comparison_experiments = []
    
    if 'integration_results' not in st.session_state:
        st.session_state.integration_results = {}
    
    # Page Title and Introduction
    st.title("Real-time CRISPR Experimental Data Integration")
    
    st.markdown("""
    This module enables integration with real-time CRISPR experimental data, allowing you to:
    
    - Connect to CRISPR experimental platforms and databases
    - Upload and analyze your own CRISPR experimental data
    - Monitor ongoing CRISPR experiments in real-time
    - Compare results across multiple experiments
    - Integrate experimental data with predictive models
    
    Use the tabs below to access different functionalities.
    """)
    
    # Create tabs for different functionalities
    tabs = st.tabs([
        "Data Source Configuration", 
        "Experiment Browser", 
        "Experiment Analysis",
        "Real-time Monitoring",
        "Experiment Comparison",
        "Model Integration"
    ])
    
    # ==========================================
    # Tab 1: Data Source Configuration
    # ==========================================
    with tabs[0]:
        st.header("Configure CRISPR Data Sources")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Select Data Source")
            data_source = st.selectbox(
                "Select primary CRISPR data source",
                options=["GenomeCRISPR", "CRISPResso", "DepMap", "custom"],
                help="Choose the primary source for CRISPR experimental data."
            )
            
            api_key = st.text_input(
                "API Key (if required)",
                type="password",
                help="Enter an API key if required by the selected data source."
            )
            
            # Local cache settings
            st.subheader("Local Cache Settings")
            cache_dir = st.text_input(
                "Local Cache Directory",
                value="data/crispr_cache",
                help="Directory to store locally cached CRISPR data."
            )
            
            if st.button("Create Cache Directory"):
                os.makedirs(cache_dir, exist_ok=True)
                st.success(f"Cache directory created at {cache_dir}")
            
            if st.button("Apply Configuration"):
                st.session_state.crispr_monitor = RealtimeCRISPRMonitor(
                    api_key=api_key,
                    data_source=data_source,
                    local_cache_dir=cache_dir
                )
                st.success("Configuration applied successfully!")
        
        with col2:
            st.subheader("Upload Experimental Data")
            
            # Form for metadata
            with st.form("experiment_metadata_form"):
                exp_name = st.text_input("Experiment Name", "My CRISPR Experiment")
                exp_desc = st.text_area("Description", "CRISPR experiment description")
                cell_line = st.text_input("Cell Line", "HEK293T")
                target_gene = st.text_input("Target Gene", "KRAS")
                cas_type = st.selectbox(
                    "Cas Type",
                    options=["Cas9", "Cas12a", "Cas13", "dCas9", "Cas9-nickase", "Base editor", "Prime editor"]
                )
                grna_sequence = st.text_input("gRNA Sequence", "")
                
                uploaded_file = st.file_uploader(
                    "Upload Experiment Data (CSV or Excel)",
                    type=["csv", "xlsx", "xls"]
                )
                
                submit_button = st.form_submit_button("Upload Data")
                
                if submit_button and uploaded_file is not None:
                    # Prepare metadata
                    metadata = {
                        'name': exp_name,
                        'description': exp_desc,
                        'cell_line': cell_line,
                        'target_gene': target_gene,
                        'cas_type': cas_type,
                        'gRNA_sequence': grna_sequence,
                        'source': 'custom',
                        'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Upload data
                    success = st.session_state.crispr_monitor.upload_experiment_data(
                        file_data=uploaded_file,
                        metadata=metadata
                    )
                    
                    if success:
                        st.success("Experiment data uploaded successfully!")
                    else:
                        st.error("Failed to upload experiment data.")
            
            st.subheader("Data Source Information")
            st.info("""
            **GenomeCRISPR**: A database of published genome-wide CRISPR screens.
            
            **CRISPResso**: Platform for analysis of CRISPR-Cas9 genome editing outcomes.
            
            **DepMap**: Cancer Dependency Map provides CRISPR screening data across cancer cell lines.
            
            **Custom**: Use your own local database of CRISPR experiments.
            """)
    
    # ==========================================
    # Tab 2: Experiment Browser
    # ==========================================
    with tabs[1]:
        st.header("Browse CRISPR Experiments")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Filter Options")
            
            # Filter inputs
            filter_cell_line = st.text_input("Filter by Cell Line")
            filter_target_gene = st.text_input("Filter by Target Gene")
            limit = st.slider("Maximum Results", 1, 100, 10)
            
            if st.button("Fetch Experiments"):
                with st.spinner("Fetching experiments..."):
                    experiments = st.session_state.crispr_monitor.fetch_recent_experiments(
                        cell_line=filter_cell_line if filter_cell_line else None,
                        target_gene=filter_target_gene if filter_target_gene else None,
                        limit=limit
                    )
                    
                    st.session_state.experiments = experiments.to_dict('records') if not experiments.empty else []
                    
                    if st.session_state.experiments:
                        st.success(f"Found {len(st.session_state.experiments)} experiments")
                    else:
                        st.info("No experiments found matching your criteria.")
        
        with col2:
            st.subheader("Experiment Results")
            
            if st.session_state.experiments:
                # Create DataFrame for display
                exp_df = pd.DataFrame(st.session_state.experiments)
                
                # Allow column selection
                default_columns = ["id", "name", "cell_line", "target_gene", "cas_type", "date", "status"]
                display_columns = st.multiselect(
                    "Select columns to display",
                    options=exp_df.columns.tolist(),
                    default=list(set(default_columns).intersection(set(exp_df.columns.tolist())))
                )
                
                if display_columns:
                    st.dataframe(exp_df[display_columns], height=400)
                    
                    # Export options
                    st.download_button(
                        label="Download as CSV",
                        data=exp_df.to_csv(index=False).encode('utf-8'),
                        file_name="crispr_experiments.csv",
                        mime="text/csv"
                    )
                    
                    # Select experiment for detailed analysis
                    selected_id = st.selectbox(
                        "Select experiment for detailed analysis",
                        options=[exp["id"] for exp in st.session_state.experiments],
                        format_func=lambda x: next((exp["name"] for exp in st.session_state.experiments if exp["id"] == x), x)
                    )
                    
                    if st.button("Analyze Selected Experiment"):
                        st.session_state.selected_experiment = selected_id
                        # Switch to the Analysis tab programmatically
                        st.experimental_set_query_params(tab="Experiment Analysis")
                        # Force rerun to update UI
                        st.rerun()
            else:
                st.info("Fetch experiments to see results.")
    
    # ==========================================
    # Tab 3: Experiment Analysis
    # ==========================================
    with tabs[2]:
        st.header("CRISPR Experiment Analysis")
        
        # Select experiment if not already selected
        if st.session_state.selected_experiment is None:
            experiment_options = [exp["id"] for exp in st.session_state.experiments] if st.session_state.experiments else []
            
            if experiment_options:
                selected_id = st.selectbox(
                    "Select experiment to analyze",
                    options=experiment_options,
                    format_func=lambda x: next((exp["name"] for exp in st.session_state.experiments if exp["id"] == x), x)
                )
                
                if st.button("Analyze Experiment"):
                    st.session_state.selected_experiment = selected_id
                    st.rerun()
            else:
                st.info("Fetch experiments from the 'Experiment Browser' tab first.")
                if st.button("Fetch Sample Experiments", key="fetch_sample_exp_analyzer"):
                    with st.spinner("Fetching sample experiments..."):
                        experiments = st.session_state.crispr_monitor.fetch_recent_experiments(limit=5)
                        st.session_state.experiments = experiments.to_dict('records') if not experiments.empty else []
                        st.rerun()
        
        # If experiment is selected, analyze it
        if st.session_state.selected_experiment is not None:
            with st.spinner("Analyzing experiment..."):
                # Get the experiment data
                experiment_data = st.session_state.crispr_monitor.analyze_experiment(
                    st.session_state.selected_experiment
                )
                
                st.session_state.experiment_data = experiment_data
            
            if experiment_data:
                # Display experiment information
                exp_info = experiment_data.get("experiment", {})
                
                st.subheader("Experiment Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Experiment ID", exp_info.get("id", "N/A"))
                    st.metric("Target Gene", exp_info.get("target_gene", "N/A"))
                
                with col2:
                    st.metric("Cell Line", exp_info.get("cell_line", "N/A"))
                    st.metric("Cas Type", exp_info.get("cas_type", "N/A"))
                
                with col3:
                    st.metric("Date", exp_info.get("date", "N/A"))
                    st.metric("Source", exp_info.get("source", "N/A"))
                
                st.markdown(f"**Description:** {exp_info.get('description', 'No description available.')}")
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                stats = experiment_data.get("summary_stats", {})
                
                if stats:
                    metrics_cols = st.columns(4)
                    
                    if "avg_efficiency" in stats:
                        metrics_cols[0].metric("Avg. Efficiency", f"{stats['avg_efficiency']:.2f}")
                    
                    if "avg_offtarget" in stats:
                        metrics_cols[1].metric("Avg. Off-target", f"{stats['avg_offtarget']:.2f}")
                    
                    if "avg_editing_efficiency" in stats:
                        metrics_cols[2].metric("Editing Efficiency", f"{stats['avg_editing_efficiency']:.2f}%")
                    
                    if "total_reads" in stats and "total_edited" in stats:
                        edit_percentage = (stats['total_edited'] / stats['total_reads']) * 100 if stats['total_reads'] > 0 else 0
                        metrics_cols[3].metric("Total Edit %", f"{edit_percentage:.2f}%")
                
                # Generate and display analysis plots
                st.subheader("Analysis Visualizations")
                plots = st.session_state.crispr_monitor.generate_analysis_plots(experiment_data)
                
                if plots:
                    plot_tabs = st.tabs([
                        "Efficiency Distribution", 
                        "Editing Efficiency", 
                        "Additional Analysis"
                    ])
                    
                    with plot_tabs[0]:
                        if "efficiency_distribution" in plots:
                            st.pyplot(plots["efficiency_distribution"])
                        
                        if "efficiency_vs_offtarget" in plots:
                            st.pyplot(plots["efficiency_vs_offtarget"])
                    
                    with plot_tabs[1]:
                        if "editing_distribution" in plots:
                            st.pyplot(plots["editing_distribution"])
                        
                        if "indel_distribution" in plots:
                            st.pyplot(plots["indel_distribution"])
                    
                    with plot_tabs[2]:
                        if "correlation_matrix" in plots:
                            st.pyplot(plots["correlation_matrix"])
                
                # Display raw data
                st.subheader("Experiment Data")
                data_tabs = st.tabs([
                    "Efficiency Data", 
                    "Sequencing Data", 
                    "Raw Data"
                ])
                
                with data_tabs[0]:
                    if experiment_data.get("efficiency_data"):
                        eff_df = pd.DataFrame(experiment_data["efficiency_data"])
                        st.dataframe(eff_df, height=300)
                        
                        st.download_button(
                            label="Download Efficiency Data",
                            data=eff_df.to_csv(index=False).encode('utf-8'),
                            file_name="efficiency_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No efficiency data available for this experiment.")
                
                with data_tabs[1]:
                    if experiment_data.get("sequencing_data"):
                        seq_df = pd.DataFrame(experiment_data["sequencing_data"])
                        st.dataframe(seq_df, height=300)
                        
                        st.download_button(
                            label="Download Sequencing Data",
                            data=seq_df.to_csv(index=False).encode('utf-8'),
                            file_name="sequencing_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No sequencing data available for this experiment.")
                
                with data_tabs[2]:
                    if experiment_data.get("raw_data"):
                        raw_df = pd.DataFrame(experiment_data["raw_data"])
                        st.dataframe(raw_df, height=300)
                        
                        st.download_button(
                            label="Download Raw Data",
                            data=raw_df.to_csv(index=False).encode('utf-8'),
                            file_name="raw_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No raw data available for this experiment.")
                
                # Add to comparison list
                if st.button("Add to Comparison List"):
                    if st.session_state.selected_experiment not in st.session_state.comparison_experiments:
                        st.session_state.comparison_experiments.append(st.session_state.selected_experiment)
                        st.success(f"Added experiment {st.session_state.selected_experiment} to comparison list.")
                    else:
                        st.info("This experiment is already in the comparison list.")
                
                # Integrate with model
                if st.button("Integrate with CRISPR Model"):
                    st.session_state.integration_results = st.session_state.crispr_monitor.integrate_with_crispr_target_model(
                        experiment_data
                    )
                    
                    # Switch to the Model Integration tab
                    st.experimental_set_query_params(tab="Model Integration")
                    st.rerun()
            else:
                st.error("Failed to analyze experiment. Please try again.")
    
    # ==========================================
    # Tab 4: Real-time Monitoring
    # ==========================================
    with tabs[3]:
        st.header("Real-time CRISPR Experiment Monitoring")
        
        st.markdown("""
        This module allows you to monitor CRISPR experiments in real-time. 
        Select an ongoing experiment or start a new monitoring session.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Monitor Options")
            
            # Select from existing experiments or create new
            monitor_type = st.radio(
                "Monitor Type",
                options=["Existing Experiment", "New Monitoring Session"]
            )
            
            if monitor_type == "Existing Experiment":
                experiment_options = [exp["id"] for exp in st.session_state.experiments] if st.session_state.experiments else []
                
                if experiment_options:
                    monitor_id = st.selectbox(
                        "Select experiment to monitor",
                        options=experiment_options,
                        format_func=lambda x: next((exp["name"] for exp in st.session_state.experiments if exp["id"] == x), x)
                    )
                    
                    update_interval = st.slider(
                        "Update Interval (seconds)",
                        min_value=1,
                        max_value=30,
                        value=5
                    )
                    
                    if st.button("Start Monitoring"):
                        with st.spinner("Initializing monitoring..."):
                            st.session_state.monitoring_results = st.session_state.crispr_monitor.monitor_experiment_progress(
                                monitor_id,
                                update_interval=update_interval
                            )
                else:
                    st.info("Fetch experiments from the 'Experiment Browser' tab first.")
                    if st.button("Fetch Sample Experiments", key="fetch_sample_exp_monitor"):
                        with st.spinner("Fetching sample experiments..."):
                            experiments = st.session_state.crispr_monitor.fetch_recent_experiments(limit=5)
                            st.session_state.experiments = experiments.to_dict('records') if not experiments.empty else []
                            st.rerun()
            
            else:  # New Monitoring Session
                with st.form("new_monitoring_form"):
                    session_name = st.text_input("Session Name", "New CRISPR Experiment")
                    cell_line = st.text_input("Cell Line", "HEK293T")
                    target_gene = st.text_input("Target Gene", "KRAS")
                    cas_type = st.selectbox(
                        "Cas Type",
                        options=["Cas9", "Cas12a", "Cas13", "dCas9", "Cas9-nickase", "Base editor", "Prime editor"]
                    )
                    grna_sequence = st.text_input("gRNA Sequence", "")
                    
                    update_interval = st.slider(
                        "Update Interval (seconds)",
                        min_value=1,
                        max_value=30,
                        value=5
                    )
                    
                    start_button = st.form_submit_button("Start New Monitoring Session")
                    
                    if start_button:
                        # Generate a new experiment ID
                        new_exp_id = f"EXP{int(time.time())}"
                        
                        # Create experiment metadata
                        metadata = {
                            'name': session_name,
                            'description': f"Real-time monitoring session for {target_gene} in {cell_line}",
                            'cell_line': cell_line,
                            'target_gene': target_gene,
                            'cas_type': cas_type,
                            'gRNA_sequence': grna_sequence,
                            'source': 'live_monitoring',
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Create a DataFrame with basic experiment info
                        exp_df = pd.DataFrame([metadata])
                        
                        # Upload to local database
                        st.session_state.crispr_monitor.upload_experiment_data(
                            file_data=io.StringIO(exp_df.to_csv(index=False)),
                            metadata=metadata
                        )
                        
                        # Start monitoring
                        with st.spinner("Initializing monitoring..."):
                            st.session_state.monitoring_results = st.session_state.crispr_monitor.monitor_experiment_progress(
                                new_exp_id,
                                update_interval=update_interval
                            )
        
        with col2:
            st.subheader("Monitoring Results")
            
            if 'monitoring_results' in st.session_state and st.session_state.monitoring_results:
                results = st.session_state.monitoring_results
                
                st.success("Monitoring session completed!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Experiment ID", results.get("experiment_id", "N/A"))
                with col2:
                    st.metric("Status", results.get("status", "N/A"))
                with col3:
                    st.metric("Completion Time", results.get("completion_time", "N/A"))
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Editing Efficiency", f"{results.get('editing_efficiency', 0):.2f}%")
                with metrics_col2:
                    st.metric("Reads Processed", f"{results.get('reads_processed', 0):,}")
                with metrics_col3:
                    st.metric("Time Elapsed", results.get("time_elapsed", "N/A"))
                
                st.markdown("## Next Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Analyze Results"):
                        st.session_state.selected_experiment = results.get("experiment_id")
                        # Switch to Analysis tab
                        st.experimental_set_query_params(tab="Experiment Analysis")
                        st.rerun()
                
                with col2:
                    if st.button("Start New Monitoring Session"):
                        st.session_state.pop('monitoring_results', None)
                        st.rerun()
            else:
                st.info("Select an experiment and start monitoring to see results.")
    
    # ==========================================
    # Tab 5: Experiment Comparison
    # ==========================================
    with tabs[4]:
        st.header("CRISPR Experiment Comparison")
        
        st.markdown("""
        Compare multiple CRISPR experiments to identify patterns and differences.
        Add experiments to the comparison list from the Experiment Analysis tab.
        """)
        
        # Display current comparison list
        if st.session_state.comparison_experiments:
            st.subheader("Experiments Selected for Comparison")
            
            # Create a list to display selected experiments
            comparison_info = []
            for exp_id in st.session_state.comparison_experiments:
                exp_info = next((exp for exp in st.session_state.experiments if exp["id"] == exp_id), None)
                if exp_info:
                    comparison_info.append({
                        "id": exp_id,
                        "name": exp_info.get("name", "Unknown"),
                        "cell_line": exp_info.get("cell_line", "Unknown"),
                        "target_gene": exp_info.get("target_gene", "Unknown"),
                        "cas_type": exp_info.get("cas_type", "Unknown")
                    })
            
            if comparison_info:
                st.dataframe(pd.DataFrame(comparison_info))
                
                # Option to remove from comparison
                to_remove = st.multiselect(
                    "Select experiments to remove from comparison",
                    options=st.session_state.comparison_experiments,
                    format_func=lambda x: next((exp["name"] for exp in comparison_info if exp["id"] == x), x)
                )
                
                if to_remove and st.button("Remove Selected"):
                    for exp_id in to_remove:
                        if exp_id in st.session_state.comparison_experiments:
                            st.session_state.comparison_experiments.remove(exp_id)
                    st.success("Selected experiments removed from comparison.")
                    st.rerun()
                
                # Generate comparison if we have multiple experiments
                if len(st.session_state.comparison_experiments) >= 2:
                    if st.button("Generate Comparison"):
                        with st.spinner("Generating comparison..."):
                            comparison_results = st.session_state.crispr_monitor.compare_experiments(
                                st.session_state.comparison_experiments
                            )
                            
                            if comparison_results:
                                # Display comparative visualizations
                                st.subheader("Comparative Analysis")
                                
                                # Display target genes and cell lines
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Target Genes:** " + ", ".join(comparison_results.get("target_genes", [])))
                                with col2:
                                    st.write("**Cell Lines:** " + ", ".join(comparison_results.get("cell_lines", [])))
                                
                                # Display efficiency comparison
                                st.subheader("Efficiency Comparison")
                                if "efficiency_comparison" in comparison_results and comparison_results["efficiency_comparison"]:
                                    st.dataframe(pd.DataFrame(comparison_results["efficiency_comparison"]))
                                    
                                    if "efficiency_boxplot" in comparison_results:
                                        st.pyplot(comparison_results["efficiency_boxplot"])
                                else:
                                    st.info("No efficiency data available for comparison.")
                                
                                # Display sequencing comparison
                                st.subheader("Editing Efficiency Comparison")
                                if "sequencing_comparison" in comparison_results and comparison_results["sequencing_comparison"]:
                                    st.dataframe(pd.DataFrame(comparison_results["sequencing_comparison"]))
                                    
                                    if "editing_barplot" in comparison_results:
                                        st.pyplot(comparison_results["editing_barplot"])
                                else:
                                    st.info("No sequencing data available for comparison.")
                                
                                # Experiment details
                                st.subheader("Experiment Details")
                                if "experiments" in comparison_results:
                                    st.dataframe(pd.DataFrame(comparison_results["experiments"]))
                                    
                                    st.download_button(
                                        label="Download Comparison Data",
                                        data=pd.DataFrame(comparison_results["experiments"]).to_csv(index=False).encode('utf-8'),
                                        file_name="experiment_comparison.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.error("Failed to generate comparison. Please try again.")
                else:
                    st.info("Add at least 2 experiments to generate a comparison.")
        else:
            st.info("""
            No experiments added to comparison yet. 
            
            Go to the 'Experiment Analysis' tab and use the 'Add to Comparison List' button to add experiments.
            """)
            
            if st.button("Fetch Sample Experiments for Comparison", key="fetch_sample_exp_comparison"):
                with st.spinner("Fetching sample experiments..."):
                    experiments = st.session_state.crispr_monitor.fetch_recent_experiments(limit=5)
                    if not experiments.empty:
                        st.session_state.experiments = experiments.to_dict('records')
                        # Add first two experiments to comparison list
                        if len(st.session_state.experiments) >= 2:
                            st.session_state.comparison_experiments = [
                                st.session_state.experiments[0]["id"],
                                st.session_state.experiments[1]["id"]
                            ]
                        st.rerun()
    
    # ==========================================
    # Tab 6: Model Integration
    # ==========================================
    with tabs[5]:
        st.header("Integration with CRISPR Target Models")
        
        st.markdown("""
        Integrate experimental CRISPR data with predictive models to improve target selection
        and optimize experimental design.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Integration Options")
            
            if not st.session_state.experiment_data:
                st.info("""
                No experiment data selected for integration. 
                
                Go to the 'Experiment Analysis' tab, select an experiment, and use the 
                'Integrate with CRISPR Model' button.
                """)
                
                if st.button("Use Sample Experiment for Integration", key="use_sample_exp_integration"):
                    with st.spinner("Fetching sample data..."):
                        # Fetch a sample experiment
                        experiments = st.session_state.crispr_monitor.fetch_recent_experiments(limit=1)
                        if not experiments.empty:
                            st.session_state.experiments = experiments.to_dict('records')
                            exp_id = st.session_state.experiments[0]["id"]
                            
                            # Analyze the experiment
                            experiment_data = st.session_state.crispr_monitor.analyze_experiment(exp_id)
                            st.session_state.experiment_data = experiment_data
                            
                            # Integrate with model
                            st.session_state.integration_results = st.session_state.crispr_monitor.integrate_with_crispr_target_model(
                                experiment_data
                            )
                            
                            st.rerun()
            else:
                # Display experiment info
                exp_info = st.session_state.experiment_data.get("experiment", {})
                st.write(f"**Experiment:** {exp_info.get('name', 'Unknown')}")
                st.write(f"**ID:** {exp_info.get('id', 'Unknown')}")
                st.write(f"**Target Gene:** {exp_info.get('target_gene', 'Unknown')}")
                st.write(f"**Cell Line:** {exp_info.get('cell_line', 'Unknown')}")
                
                # Model selection
                model_type = st.selectbox(
                    "Select Model Type",
                    options=["Default CRISPR Model", "Custom Model"]
                )
                
                if model_type == "Custom Model":
                    model_path = st.text_input("Path to Custom Model", "models/custom_crispr_model.pkl")
                    st.info("Custom model integration not implemented in this demo.")
                
                if st.button("Run Integration", key="run_integration"):
                    with st.spinner("Integrating with model..."):
                        st.session_state.integration_results = st.session_state.crispr_monitor.integrate_with_crispr_target_model(
                            st.session_state.experiment_data
                        )
                        
                        if st.session_state.integration_results:
                            st.success("Integration completed successfully!")
                            st.rerun()
                        else:
                            st.error("Integration failed. Please try again.")
        
        with col2:
            st.subheader("Integration Results")
            
            if st.session_state.integration_results:
                # Display summary
                st.markdown("### Summary")
                summary = st.session_state.integration_results.get("summary", {})
                
                if summary:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Sequences", summary.get("num_sequences", 0))
                    
                    with col2:
                        st.metric("Avg. Predicted Efficiency", f"{summary.get('avg_predicted_efficiency', 0):.3f}")
                    
                    with col3:
                        st.metric("Avg. Predicted Off-target", f"{summary.get('avg_predicted_offtarget', 0):.3f}")
                    
                    if "correlation" in summary:
                        st.metric("Model-Experiment Correlation", f"{summary.get('correlation', 0):.3f}")
                    
                    if "mean_abs_error" in summary:
                        st.metric("Mean Absolute Error", f"{summary.get('mean_abs_error', 0):.3f}")
                    
                    if "mean_percent_error" in summary:
                        st.metric("Mean Percent Error", f"{summary.get('mean_percent_error', 0):.1f}%")
                
                # Display visualizations
                st.markdown("### Visualizations")
                
                if "scatter_plot" in st.session_state.integration_results and st.session_state.integration_results["scatter_plot"]:
                    st.pyplot(st.session_state.integration_results["scatter_plot"])
                
                if "error_plot" in st.session_state.integration_results and st.session_state.integration_results["error_plot"]:
                    st.pyplot(st.session_state.integration_results["error_plot"])
                
                # Display predictions
                st.markdown("### Model Predictions")
                predictions = st.session_state.integration_results.get("predictions", [])
                
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    st.dataframe(pred_df)
                    
                    st.download_button(
                        label="Download Predictions",
                        data=pred_df.to_csv(index=False).encode('utf-8'),
                        file_name="model_predictions.csv",
                        mime="text/csv"
                    )
                
                # Display comparison
                st.markdown("### Experimental vs. Predicted Comparison")
                comparison = st.session_state.integration_results.get("comparison", [])
                
                if comparison:
                    comp_df = pd.DataFrame(comparison)
                    st.dataframe(comp_df)
                    
                    st.download_button(
                        label="Download Comparison",
                        data=comp_df.to_csv(index=False).encode('utf-8'),
                        file_name="experiment_model_comparison.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No experimental data available for comparison with model predictions.")
            else:
                st.info("Run integration to see results.")

if __name__ == "__main__":
    main()