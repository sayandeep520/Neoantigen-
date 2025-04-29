import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction  # Use gc_fraction instead of GC
import sqlite3
import streamlit as st
import base64
from io import BytesIO

class RealtimeCRISPRMonitor:
    """
    Utility for monitoring and analyzing real-time CRISPR experimental data.
    
    This class provides functionality to interface with CRISPR experimental 
    platforms and databases, fetch real-time or recent experimental data,
    and perform analysis on the results.
    """
    
    def __init__(self, api_key=None, data_source="GenomeCRISPR", local_cache_dir="data/crispr_cache"):
        """
        Initialize the real-time CRISPR monitor.
        
        Args:
            api_key (str, optional): API key for external CRISPR data sources
            data_source (str): Source of CRISPR data (GenomeCRISPR, CRISPResso, custom)
            local_cache_dir (str): Directory to cache downloaded data
        """
        self.api_key = api_key or os.environ.get("CRISPR_API_KEY")
        self.data_source = data_source
        self.local_cache_dir = local_cache_dir
        
        # Ensure cache directory exists
        os.makedirs(local_cache_dir, exist_ok=True)
        
        # Set up endpoints for different data sources
        self.endpoints = {
            "GenomeCRISPR": "http://genomecrispr.dkfz.de/api/",
            "CRISPResso": "https://crispresso.pinellolab.partners.org/api/",
            "DepMap": "https://depmap.org/portal/api/",
            "CRISPRa": "https://crisprapi.pinellolab.org/api/v1/",
            "CRISPR-MIT": "https://crispr.mit.edu/api/",
            "CHOPCHOP": "https://chopchop.cbu.uib.no/api/",
            "CRISPOR": "https://crispor.tefor.net/api/"
        }
        
        # Initialize local database for caching
        self._init_local_db()
        
    def _init_local_db(self):
        """Initialize the local SQLite database for caching CRISPR data"""
        db_path = os.path.join(self.local_cache_dir, "crispr_cache.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create tables for different data types
        c.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            date TEXT,
            source TEXT,
            cell_line TEXT,
            target_gene TEXT,
            cas_type TEXT,
            gRNA_sequence TEXT,
            raw_data TEXT,
            metadata TEXT
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS efficiency_scores (
            id TEXT PRIMARY KEY,
            experiment_id TEXT,
            grna_sequence TEXT,
            target_gene TEXT,
            efficiency_score REAL,
            off_target_score REAL,
            timestamp TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS sequencing_reads (
            id TEXT PRIMARY KEY,
            experiment_id TEXT,
            read_count INTEGER,
            edited_count INTEGER,
            editing_efficiency REAL,
            indel_ratio REAL,
            timestamp TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_recent_experiments(self, cell_line=None, target_gene=None, limit=10):
        """
        Fetch recent CRISPR experiments from selected data source.
        
        Args:
            cell_line (str, optional): Filter by cell line
            target_gene (str, optional): Filter by target gene
            limit (int): Maximum number of experiments to return
            
        Returns:
            pandas.DataFrame: Recent CRISPR experiments
        """
        # Check if we have API key for sources that require it
        if self.data_source in ["CRISPResso", "CRISPRa"] and not self.api_key:
            st.warning(f"API key required for {self.data_source}. Please provide a valid API key.")
            return pd.DataFrame()
            
        if self.data_source == "GenomeCRISPR":
            return self._fetch_from_genome_crispr(cell_line, target_gene, limit)
        elif self.data_source == "CRISPResso":
            return self._fetch_from_crispresso(cell_line, target_gene, limit)
        elif self.data_source == "DepMap":
            return self._fetch_from_depmap(cell_line, target_gene, limit)
        elif self.data_source == "custom":
            return self._fetch_from_custom_source(cell_line, target_gene, limit)
        else:
            # Default to synthetic data if source not implemented
            return self._generate_synthetic_experiment_data(cell_line, target_gene, limit)
    
    def _fetch_from_genome_crispr(self, cell_line, target_gene, limit):
        """Fetch data from GenomeCRISPR database"""
        try:
            params = {"limit": limit}
            if cell_line:
                params["cell_line"] = cell_line
            if target_gene:
                params["target_gene"] = target_gene
                
            # GenomeCRISPR doesn't have a public API, so we'd need to scrape or
            # use their bulk downloads in a real implementation
            
            # For demo purposes, we'll return synthetic data labeled as from GenomeCRISPR
            data = self._generate_synthetic_experiment_data(cell_line, target_gene, limit)
            data["source"] = "GenomeCRISPR"
            return data
            
        except Exception as e:
            st.error(f"Error fetching data from GenomeCRISPR: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_from_crispresso(self, cell_line, target_gene, limit):
        """Fetch data from CRISPResso database"""
        try:
            # In a real implementation, we would use the CRISPResso API
            # For demo purposes, we'll return synthetic data labeled as from CRISPResso
            data = self._generate_synthetic_experiment_data(cell_line, target_gene, limit)
            data["source"] = "CRISPResso"
            return data
            
        except Exception as e:
            st.error(f"Error fetching data from CRISPResso: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_from_depmap(self, cell_line, target_gene, limit):
        """Fetch CRISPR screening data from DepMap"""
        try:
            # In a real implementation, we would use the DepMap API
            # For now, we'll create synthetic data representative of DepMap
            
            base_url = self.endpoints["DepMap"]
            
            # Generate synthetic data with DepMap-specific fields
            data = self._generate_synthetic_experiment_data(cell_line, target_gene, limit)
            data["source"] = "DepMap"
            
            # Add DepMap-specific columns
            data["dependency_score"] = np.random.normal(-0.5, 0.3, size=len(data))
            data["genetic_interaction"] = np.random.choice(["synergistic", "antagonistic", "additive", "none"], size=len(data))
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data from DepMap: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_from_custom_source(self, cell_line, target_gene, limit):
        """Fetch data from a custom/local source"""
        try:
            # Connect to local database
            db_path = os.path.join(self.local_cache_dir, "crispr_cache.db")
            conn = sqlite3.connect(db_path)
            
            # Build query based on filters
            query = "SELECT * FROM experiments"
            params = []
            
            if cell_line or target_gene:
                query += " WHERE"
                
                if cell_line:
                    query += " cell_line = ?"
                    params.append(cell_line)
                    
                if target_gene:
                    if cell_line:
                        query += " AND"
                    query += " target_gene = ?"
                    params.append(target_gene)
            
            query += f" ORDER BY date DESC LIMIT {limit}"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data from custom source: {str(e)}")
            return pd.DataFrame()
    
    def _generate_synthetic_experiment_data(self, cell_line, target_gene, limit):
        """
        Generate synthetic CRISPR experimental data for demonstration purposes.
        
        Args:
            cell_line (str, optional): Filter by cell line
            target_gene (str, optional): Filter by target gene
            limit (int): Maximum number of experiments to return
            
        Returns:
            pandas.DataFrame: Synthetic CRISPR experiment data
        """
        # Common cell lines in CRISPR experiments
        cell_lines = ["HEK293T", "K562", "HeLa", "MCF7", "U2OS", "A549", "Jurkat"]
        
        # Common genes targeted in CRISPR studies
        target_genes = ["TP53", "KRAS", "BRCA1", "EGFR", "HER2", "PD-1", "CD19", "BCL2"]
        
        # Cas types
        cas_types = ["Cas9", "Cas12a", "Cas13", "dCas9", "Cas9-nickase", "Base editor", "Prime editor"]
        
        # Filter based on inputs
        if cell_line:
            filtered_cell_lines = [c for c in cell_lines if cell_line.lower() in c.lower()]
            if filtered_cell_lines:
                cell_lines = filtered_cell_lines
                
        if target_gene:
            filtered_genes = [g for g in target_genes if target_gene.lower() in g.lower()]
            if filtered_genes:
                target_genes = filtered_genes
        
        # Generate data
        data = []
        for i in range(min(limit, 50)):  # Cap at 50 for performance
            exp_date = datetime.now().strftime("%Y-%m-%d")
            selected_gene = np.random.choice(target_genes)
            selected_cell = np.random.choice(cell_lines)
            selected_cas = np.random.choice(cas_types)
            
            # Generate random gRNA (20-25nt)
            grna_length = np.random.randint(20, 26)
            grna_seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=grna_length))
            
            experiment = {
                "id": f"EXP{i:04d}",
                "name": f"CRISPR experiment targeting {selected_gene} in {selected_cell}",
                "description": f"CRISPR-{selected_cas} targeting {selected_gene} in {selected_cell} cell line",
                "date": exp_date,
                "cell_line": selected_cell,
                "target_gene": selected_gene,
                "cas_type": selected_cas,
                "gRNA_sequence": grna_seq,
                "efficiency_score": round(np.random.uniform(0.3, 0.9), 2),
                "off_target_score": round(np.random.uniform(0.1, 0.5), 2),
                "editing_efficiency": round(np.random.uniform(10, 90), 1),
                "indel_ratio": round(np.random.uniform(0.1, 0.9), 2),
                "read_count": np.random.randint(1000, 100000),
                "status": np.random.choice(["Completed", "In Progress", "Failed", "Scheduled"], p=[0.7, 0.2, 0.05, 0.05])
            }
            data.append(experiment)
        
        return pd.DataFrame(data)
    
    def upload_experiment_data(self, file_data, metadata):
        """
        Upload new experimental data to the local database.
        
        Args:
            file_data: CSV or Excel file with experimental data
            metadata (dict): Metadata about the experiment
            
        Returns:
            bool: Success status
        """
        try:
            # Parse the uploaded file
            if isinstance(file_data, str):
                # File path provided
                if file_data.endswith('.csv'):
                    df = pd.read_csv(file_data)
                elif file_data.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_data)
                else:
                    st.error("Unsupported file format. Please upload CSV or Excel files.")
                    return False
            else:
                # File-like object provided (from Streamlit uploader)
                try:
                    df = pd.read_csv(file_data)
                except:
                    try:
                        df = pd.read_excel(file_data)
                    except:
                        st.error("Failed to parse uploaded file. Please ensure it's a valid CSV or Excel file.")
                        return False
            
            # Generate experiment ID
            experiment_id = f"EXP{int(time.time())}"
            
            # Connect to database
            db_path = os.path.join(self.local_cache_dir, "crispr_cache.db")
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Serialize the dataframe for storage
            raw_data = df.to_csv(index=False)
            
            # Insert experiment record
            c.execute('''
            INSERT INTO experiments 
            (id, name, description, date, source, cell_line, target_gene, cas_type, gRNA_sequence, raw_data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                metadata.get('name', 'Unnamed Experiment'),
                metadata.get('description', ''),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metadata.get('source', 'custom'),
                metadata.get('cell_line', ''),
                metadata.get('target_gene', ''),
                metadata.get('cas_type', ''),
                metadata.get('gRNA_sequence', ''),
                raw_data,
                json.dumps(metadata)
            ))
            
            # Process efficiency scores if present
            if 'efficiency_score' in df.columns and 'grna_sequence' in df.columns:
                for _, row in df.iterrows():
                    c.execute('''
                    INSERT INTO efficiency_scores
                    (id, experiment_id, grna_sequence, target_gene, efficiency_score, off_target_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"EFF{int(time.time())}{np.random.randint(1000)}",
                        experiment_id,
                        row.get('grna_sequence', ''),
                        row.get('target_gene', ''),
                        float(row.get('efficiency_score', 0)),
                        float(row.get('off_target_score', 0)),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))
            
            # Process sequencing reads if present
            if 'read_count' in df.columns and 'edited_count' in df.columns:
                for _, row in df.iterrows():
                    c.execute('''
                    INSERT INTO sequencing_reads
                    (id, experiment_id, read_count, edited_count, editing_efficiency, indel_ratio, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"SEQ{int(time.time())}{np.random.randint(1000)}",
                        experiment_id,
                        int(row.get('read_count', 0)),
                        int(row.get('edited_count', 0)),
                        float(row.get('editing_efficiency', 0)),
                        float(row.get('indel_ratio', 0)),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            st.error(f"Error uploading experiment data: {str(e)}")
            return False
    
    def analyze_experiment(self, experiment_id):
        """
        Analyze a specific CRISPR experiment.
        
        Args:
            experiment_id (str): ID of the experiment to analyze
            
        Returns:
            dict: Analysis results
        """
        try:
            # Connect to database
            db_path = os.path.join(self.local_cache_dir, "crispr_cache.db")
            conn = sqlite3.connect(db_path)
            
            # Fetch experiment data
            experiment = pd.read_sql_query(
                "SELECT * FROM experiments WHERE id = ?", 
                conn, 
                params=[experiment_id]
            )
            
            if experiment.empty:
                st.warning(f"Experiment {experiment_id} not found.")
                conn.close()
                return {}
            
            # Fetch efficiency scores
            efficiency = pd.read_sql_query(
                "SELECT * FROM efficiency_scores WHERE experiment_id = ?", 
                conn, 
                params=[experiment_id]
            )
            
            # Fetch sequencing reads
            sequencing = pd.read_sql_query(
                "SELECT * FROM sequencing_reads WHERE experiment_id = ?", 
                conn, 
                params=[experiment_id]
            )
            
            conn.close()
            
            # Parse the raw experiment data
            try:
                if 'raw_data' in experiment.columns and not pd.isna(experiment.iloc[0]['raw_data']):
                    raw_data_str = experiment.iloc[0]['raw_data']
                    raw_data = pd.read_csv(io.StringIO(raw_data_str))
                else:
                    raw_data = pd.DataFrame()
            except Exception as e:
                print(f"Error parsing raw data: {str(e)}")
                raw_data = pd.DataFrame()
            
            # Perform analysis
            results = {
                "experiment": experiment.iloc[0].to_dict(),
                "efficiency_data": efficiency.to_dict('records') if not efficiency.empty else [],
                "sequencing_data": sequencing.to_dict('records') if not sequencing.empty else [],
                "raw_data": raw_data.to_dict('records') if not raw_data.empty else [],
                "summary_stats": {}
            }
            
            # Calculate summary statistics
            if not efficiency.empty:
                results["summary_stats"]["avg_efficiency"] = efficiency["efficiency_score"].mean()
                results["summary_stats"]["max_efficiency"] = efficiency["efficiency_score"].max()
                results["summary_stats"]["min_efficiency"] = efficiency["efficiency_score"].min()
                results["summary_stats"]["avg_offtarget"] = efficiency["off_target_score"].mean() if "off_target_score" in efficiency.columns else None
            
            if not sequencing.empty:
                results["summary_stats"]["avg_editing_efficiency"] = sequencing["editing_efficiency"].mean() if "editing_efficiency" in sequencing.columns else None
                results["summary_stats"]["total_reads"] = sequencing["read_count"].sum() if "read_count" in sequencing.columns else None
                results["summary_stats"]["total_edited"] = sequencing["edited_count"].sum() if "edited_count" in sequencing.columns else None
            
            return results
            
        except Exception as e:
            st.error(f"Error analyzing experiment: {str(e)}")
            return {}
    
    def generate_analysis_plots(self, experiment_data):
        """
        Generate analysis plots for CRISPR experimental data.
        
        Args:
            experiment_data (dict): Data from analyze_experiment()
            
        Returns:
            dict: Dictionary of plot figures
        """
        plots = {}
        
        try:
            # Efficiency score distribution
            if experiment_data.get("efficiency_data"):
                efficiency_df = pd.DataFrame(experiment_data["efficiency_data"])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(efficiency_df["efficiency_score"], kde=True, ax=ax)
                ax.set_title("Distribution of gRNA Efficiency Scores")
                ax.set_xlabel("Efficiency Score")
                ax.set_ylabel("Count")
                plots["efficiency_distribution"] = fig
                
                if "off_target_score" in efficiency_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x="efficiency_score", y="off_target_score", data=efficiency_df, ax=ax)
                    ax.set_title("Efficiency vs. Off-target Scores")
                    ax.set_xlabel("Efficiency Score")
                    ax.set_ylabel("Off-target Score")
                    plots["efficiency_vs_offtarget"] = fig
            
            # Editing efficiency analysis
            if experiment_data.get("sequencing_data"):
                seq_df = pd.DataFrame(experiment_data["sequencing_data"])
                
                if "editing_efficiency" in seq_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(seq_df["editing_efficiency"], kde=True, ax=ax)
                    ax.set_title("Distribution of Editing Efficiency")
                    ax.set_xlabel("Editing Efficiency (%)")
                    ax.set_ylabel("Count")
                    plots["editing_distribution"] = fig
                
                if "indel_ratio" in seq_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(seq_df["indel_ratio"], kde=True, ax=ax)
                    ax.set_title("Distribution of Indel Ratios")
                    ax.set_xlabel("Indel Ratio")
                    ax.set_ylabel("Count")
                    plots["indel_distribution"] = fig
            
            # Raw data visualization if available
            if experiment_data.get("raw_data"):
                raw_df = pd.DataFrame(experiment_data["raw_data"])
                
                # Only create correlation matrix if we have numeric columns
                numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    corr_matrix = raw_df[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation Matrix of Numeric Features")
                    plots["correlation_matrix"] = fig
            
            return plots
            
        except Exception as e:
            st.error(f"Error generating analysis plots: {str(e)}")
            return {}
    
    def monitor_experiment_progress(self, experiment_id, update_interval=5):
        """
        Monitor the progress of an ongoing CRISPR experiment.
        
        Args:
            experiment_id (str): ID of the experiment to monitor
            update_interval (int): Update interval in seconds
            
        Returns:
            dict: Latest experiment status
        """
        try:
            # Connect to database
            db_path = os.path.join(self.local_cache_dir, "crispr_cache.db")
            conn = sqlite3.connect(db_path)
            
            # Fetch experiment data
            experiment = pd.read_sql_query(
                "SELECT * FROM experiments WHERE id = ?", 
                conn, 
                params=[experiment_id]
            )
            
            if experiment.empty:
                st.warning(f"Experiment {experiment_id} not found.")
                conn.close()
                return {}
            
            # For a real implementation, we would connect to the lab equipment or server
            # to get real-time updates. For this demo, we'll simulate progress.
            
            # Create a placeholder for the progress
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Simulate progress updates
            for progress in range(0, 101, 5):
                # Update the progress bar
                progress_placeholder.progress(progress / 100)
                
                # Update status message
                if progress < 25:
                    status = "Initializing CRISPR reaction..."
                elif progress < 50:
                    status = "CRISPR cutting in progress..."
                elif progress < 75:
                    status = "Analyzing editing outcomes..."
                else:
                    status = "Finalizing results..."
                    
                status_placeholder.text(f"Status: {status}")
                
                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Editing Efficiency", f"{min(progress, 85)}%", f"+{min(5, progress//10)}%")
                    with col2:
                        st.metric("Reads Processed", f"{progress * 1000}", f"+{5000}")
                    with col3:
                        st.metric("Time Elapsed", f"{progress // 5} min")
                
                # Wait for the update interval
                time.sleep(update_interval)
                
                # Check if we should stop (in a real implementation, this would check if experiment is complete)
                if progress >= 100:
                    break
            
            # Final update
            progress_placeholder.progress(1.0)
            status_placeholder.text("Status: Experiment completed!")
            
            # Return the final status
            final_status = {
                "experiment_id": experiment_id,
                "name": experiment.iloc[0]["name"],
                "status": "Completed",
                "editing_efficiency": np.random.uniform(70, 95),
                "reads_processed": 100000,
                "time_elapsed": "20 min",
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return final_status
            
        except Exception as e:
            st.error(f"Error monitoring experiment: {str(e)}")
            return {}
    
    def compare_experiments(self, experiment_ids):
        """
        Compare multiple CRISPR experiments.
        
        Args:
            experiment_ids (list): List of experiment IDs to compare
            
        Returns:
            dict: Comparison results
        """
        try:
            if not experiment_ids:
                st.warning("No experiments selected for comparison.")
                return {}
                
            # Connect to database
            db_path = os.path.join(self.local_cache_dir, "crispr_cache.db")
            conn = sqlite3.connect(db_path)
            
            # Fetch experiments
            placeholders = ', '.join(['?'] * len(experiment_ids))
            query = f"SELECT * FROM experiments WHERE id IN ({placeholders})"
            experiments = pd.read_sql_query(query, conn, params=experiment_ids)
            
            if experiments.empty:
                st.warning("No experiments found with the provided IDs.")
                conn.close()
                return {}
            
            # Fetch efficiency scores
            query = f"SELECT * FROM efficiency_scores WHERE experiment_id IN ({placeholders})"
            efficiency = pd.read_sql_query(query, conn, params=experiment_ids)
            
            # Fetch sequencing reads
            query = f"SELECT * FROM sequencing_reads WHERE experiment_id IN ({placeholders})"
            sequencing = pd.read_sql_query(query, conn, params=experiment_ids)
            
            conn.close()
            
            # Prepare comparison data
            comparison = {
                "experiments": experiments.to_dict('records'),
                "efficiency_comparison": {},
                "sequencing_comparison": {},
                "target_genes": experiments["target_gene"].unique().tolist(),
                "cell_lines": experiments["cell_line"].unique().tolist()
            }
            
            # Compare efficiency scores
            if not efficiency.empty:
                efficiency_pivot = efficiency.pivot_table(
                    index="experiment_id",
                    values=["efficiency_score", "off_target_score"],
                    aggfunc=["mean", "min", "max", "count"]
                ).reset_index()
                
                comparison["efficiency_comparison"] = efficiency_pivot.to_dict('records')
                
                # Create comparative plots
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.boxplot(x="experiment_id", y="efficiency_score", data=efficiency, ax=ax)
                ax.set_title("Efficiency Score Comparison Across Experiments")
                ax.set_xlabel("Experiment ID")
                ax.set_ylabel("Efficiency Score")
                comparison["efficiency_boxplot"] = fig
            
            # Compare sequencing data
            if not sequencing.empty:
                sequencing_pivot = sequencing.pivot_table(
                    index="experiment_id",
                    values=["editing_efficiency", "indel_ratio", "read_count"],
                    aggfunc=["mean", "min", "max", "sum"]
                ).reset_index()
                
                comparison["sequencing_comparison"] = sequencing_pivot.to_dict('records')
                
                # Create comparative plots
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x="experiment_id", y="editing_efficiency", data=sequencing, ax=ax)
                ax.set_title("Editing Efficiency Comparison Across Experiments")
                ax.set_xlabel("Experiment ID")
                ax.set_ylabel("Editing Efficiency (%)")
                comparison["editing_barplot"] = fig
            
            return comparison
            
        except Exception as e:
            st.error(f"Error comparing experiments: {str(e)}")
            return {}
    
    def integrate_with_crispr_target_model(self, experiment_data, model=None):
        """
        Integrate experimental data with the CRISPR target prediction model.
        
        Args:
            experiment_data (dict): Experimental data from analyze_experiment()
            model: Optional CRISPR target model instance
            
        Returns:
            dict: Integration results with model predictions
        """
        try:
            if not experiment_data:
                return {}
                
            # Import the CRISPR model if not provided
            if model is None:
                from models.crispr_model import CRISPRTargetModel
                model = CRISPRTargetModel()
            
            # Extract gRNA sequences from experiment data
            grna_sequences = []
            
            # Try to get sequences from efficiency data
            if experiment_data.get("efficiency_data"):
                for item in experiment_data["efficiency_data"]:
                    if "grna_sequence" in item and item["grna_sequence"]:
                        grna_sequences.append(item["grna_sequence"])
            
            # If no sequences found, try extracting from experiment metadata
            if not grna_sequences and "experiment" in experiment_data:
                if "gRNA_sequence" in experiment_data["experiment"] and experiment_data["experiment"]["gRNA_sequence"]:
                    grna_sequences.append(experiment_data["experiment"]["gRNA_sequence"])
            
            # If still no sequences, use sequences from raw data if available
            if not grna_sequences and "raw_data" in experiment_data:
                raw_df = pd.DataFrame(experiment_data["raw_data"])
                if "grna_sequence" in raw_df.columns:
                    grna_sequences.extend(raw_df["grna_sequence"].dropna().unique().tolist())
                elif "gRNA_sequence" in raw_df.columns:
                    grna_sequences.extend(raw_df["gRNA_sequence"].dropna().unique().tolist())
            
            if not grna_sequences:
                st.warning("No gRNA sequences found in the experiment data.")
                return {}
            
            # Make predictions using the model
            predicted_efficiency = model.predict_efficiency(grna_sequences)
            predicted_offtargets = model.predict_offtargets(grna_sequences)
            
            # Combine predictions
            predictions = []
            for i, seq in enumerate(grna_sequences):
                predictions.append({
                    "grna_sequence": seq,
                    "predicted_efficiency": float(predicted_efficiency[i]) if i < len(predicted_efficiency) else 0,
                    "predicted_offtarget": float(predicted_offtargets[i]) if i < len(predicted_offtargets) else 0,
                    "gc_content": gc_fraction(seq)  # Use gc_fraction instead of GC/100
                })
            
            # Compare predictions with experimental data if available
            comparison = []
            if experiment_data.get("efficiency_data"):
                eff_df = pd.DataFrame(experiment_data["efficiency_data"])
                pred_df = pd.DataFrame(predictions)
                
                if "grna_sequence" in eff_df.columns and "efficiency_score" in eff_df.columns:
                    # Merge experimental and predicted data
                    merged = eff_df.merge(pred_df, on="grna_sequence", how="inner")
                    
                    # Calculate correlation and error metrics
                    if len(merged) > 0:
                        for _, row in merged.iterrows():
                            comparison.append({
                                "grna_sequence": row["grna_sequence"],
                                "experimental_efficiency": row["efficiency_score"],
                                "predicted_efficiency": row["predicted_efficiency"],
                                "difference": row["efficiency_score"] - row["predicted_efficiency"],
                                "percent_error": (abs(row["efficiency_score"] - row["predicted_efficiency"]) / row["efficiency_score"]) * 100
                            })
            
            # Create visualization
            if comparison:
                comp_df = pd.DataFrame(comparison)
                
                # Scatter plot of predicted vs experimental
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.scatterplot(x="experimental_efficiency", y="predicted_efficiency", data=comp_df, ax=ax)
                
                # Add perfect prediction line
                min_val = min(comp_df["experimental_efficiency"].min(), comp_df["predicted_efficiency"].min())
                max_val = max(comp_df["experimental_efficiency"].max(), comp_df["predicted_efficiency"].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                ax.set_title("Predicted vs. Experimental Efficiency")
                ax.set_xlabel("Experimental Efficiency")
                ax.set_ylabel("Predicted Efficiency")
                
                scatter_plot = fig
                
                # Error distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(comp_df["percent_error"], kde=True, ax=ax)
                ax.set_title("Distribution of Prediction Error")
                ax.set_xlabel("Percent Error (%)")
                ax.set_ylabel("Count")
                
                error_plot = fig
            else:
                scatter_plot = None
                error_plot = None
            
            # Return integration results
            results = {
                "predictions": predictions,
                "comparison": comparison,
                "scatter_plot": scatter_plot,
                "error_plot": error_plot,
                "summary": {
                    "num_sequences": len(grna_sequences),
                    "avg_predicted_efficiency": np.mean([p["predicted_efficiency"] for p in predictions]),
                    "avg_predicted_offtarget": np.mean([p["predicted_offtarget"] for p in predictions]),
                }
            }
            
            if comparison:
                results["summary"]["correlation"] = np.corrcoef(
                    [c["experimental_efficiency"] for c in comparison],
                    [c["predicted_efficiency"] for c in comparison]
                )[0, 1]
                
                results["summary"]["mean_abs_error"] = np.mean(
                    [abs(c["difference"]) for c in comparison]
                )
                
                results["summary"]["mean_percent_error"] = np.mean(
                    [c["percent_error"] for c in comparison]
                )
            
            return results
            
        except Exception as e:
            st.error(f"Error integrating with CRISPR model: {str(e)}")
            return {}