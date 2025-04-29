import pandas as pd
import numpy as np
import streamlit as st
import os
import re
import time
from typing import Dict, List, Union, Tuple, Optional, Any
import base64
import plotly.express as px
import plotly.graph_objects as go
from Bio.Seq import Seq


def display_info_box(title: str, content: str, box_type: str = "info"):
    """
    Display a formatted information box
    
    Args:
        title: Title of the box
        content: Content to display
        box_type: Type of box ('info', 'success', 'warning', 'error')
    """
    if box_type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif box_type == "success":
        st.success(f"**{title}**\n\n{content}")
    elif box_type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif box_type == "error":
        st.error(f"**{title}**\n\n{content}")
    else:
        st.write(f"**{title}**\n\n{content}")


def format_gene_sequence(sequence: str, highlight_position: int = None, highlight_length: int = 3) -> str:
    """
    Format a gene sequence for display with optional highlighting
    
    Args:
        sequence: DNA or protein sequence
        highlight_position: Position to highlight (0-based)
        highlight_length: Length of the region to highlight
        
    Returns:
        Formatted sequence
    """
    if not sequence:
        return ""
    
    # Add spaces every 10 characters for readability
    chunks = []
    for i in range(0, len(sequence), 10):
        chunks.append(sequence[i:i+10])
    
    formatted_sequence = ' '.join(chunks)
    
    # Add highlighting if requested
    if highlight_position is not None:
        start_pos = highlight_position
        end_pos = highlight_position + highlight_length
        
        # Adjust for spaces
        space_count = start_pos // 10
        highlight_start = start_pos + space_count
        
        space_count_end = end_pos // 10
        highlight_end = end_pos + space_count_end
        
        # Add highlighting markdown
        return f"`{formatted_sequence[:highlight_start]}` **`{formatted_sequence[highlight_start:highlight_end]}`** `{formatted_sequence[highlight_end:]}`"
    
    return f"`{formatted_sequence}`"


def is_valid_dna_sequence(sequence: str) -> bool:
    """
    Check if a string is a valid DNA sequence
    
    Args:
        sequence: String to check
        
    Returns:
        True if valid DNA sequence, False otherwise
    """
    if not sequence:
        return False
    
    # Check if all characters are valid DNA nucleotides
    valid_chars = set('ATCGatcg')
    return all(c in valid_chars for c in sequence)


def is_valid_protein_sequence(sequence: str) -> bool:
    """
    Check if a string is a valid protein sequence
    
    Args:
        sequence: String to check
        
    Returns:
        True if valid protein sequence, False otherwise
    """
    if not sequence:
        return False
    
    # Set of valid amino acid codes
    valid_chars = set('ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy')
    return all(c in valid_chars for c in sequence)


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence
    
    Args:
        sequence: DNA sequence
        
    Returns:
        GC content as a fraction (0-1)
    """
    if not sequence or not is_valid_dna_sequence(sequence):
        return 0.0
    
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def reverse_complement(sequence: str) -> str:
    """
    Generate the reverse complement of a DNA sequence
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Reverse complement sequence
    """
    if not sequence or not is_valid_dna_sequence(sequence):
        return ""
    
    return str(Seq(sequence).reverse_complement())


def translate_dna(sequence: str) -> str:
    """
    Translate a DNA sequence to protein
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Protein sequence
    """
    if not sequence or not is_valid_dna_sequence(sequence):
        return ""
    
    return str(Seq(sequence).translate())


def display_sequence_stats(sequence: str, is_dna: bool = True):
    """
    Display statistics for a sequence
    
    Args:
        sequence: Biological sequence
        is_dna: Flag indicating if sequence is DNA (True) or protein (False)
    """
    if not sequence:
        st.error("No sequence provided")
        return
    
    # Create a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sequence Length:**", len(sequence))
        
        if is_dna:
            gc_content = calculate_gc_content(sequence)
            st.write("**GC Content:**", f"{gc_content:.2%}")
            
            # Count nucleotides
            counts = {
                'A': sequence.upper().count('A'),
                'T': sequence.upper().count('T'),
                'G': sequence.upper().count('G'),
                'C': sequence.upper().count('C'),
                'Other': len(sequence) - (
                    sequence.upper().count('A') + 
                    sequence.upper().count('T') + 
                    sequence.upper().count('G') + 
                    sequence.upper().count('C')
                )
            }
            
            st.write("**Nucleotide Counts:**")
            for base, count in counts.items():
                if count > 0:
                    st.write(f"- {base}: {count} ({count/len(sequence):.2%})")
        else:
            # Count amino acids
            st.write("**Amino Acid Composition:**")
            aa_count = {}
            for aa in set(sequence.upper()):
                if aa in 'ACDEFGHIKLMNPQRSTVWY':
                    count = sequence.upper().count(aa)
                    aa_count[aa] = count
            
            # Display top amino acids
            for aa, count in sorted(aa_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"- {aa}: {count} ({count/len(sequence):.2%})")
    
    with col2:
        if is_dna:
            # Show a nucleotide composition pie chart
            labels = ['A', 'T', 'G', 'C']
            values = [sequence.upper().count(base) for base in labels]
            
            fig = px.pie(
                names=labels,
                values=values,
                title="Nucleotide Composition",
                color=labels,
                color_discrete_map={'A': '#6495ED', 'T': '#F08080', 'G': '#90EE90', 'C': '#FFFF99'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Group amino acids by property and show composition
            properties = {
                'Hydrophobic': 'AVILMFYW',
                'Polar': 'NQST',
                'Acidic': 'DE',
                'Basic': 'RHK',
                'Special': 'CGP'
            }
            
            property_counts = {}
            for prop, aas in properties.items():
                count = sum(sequence.upper().count(aa) for aa in aas)
                property_counts[prop] = count
            
            fig = px.pie(
                names=list(property_counts.keys()),
                values=list(property_counts.values()),
                title="Amino Acid Properties"
            )
            st.plotly_chart(fig, use_container_width=True)


def create_mutation_diagram(gene_name: str, protein_change: str, 
                           protein_length: int = 400, domain_info: Dict[str, Tuple[int, int]] = None) -> go.Figure:
    """
    Create a protein mutation diagram
    
    Args:
        gene_name: Name of the gene
        protein_change: Protein change in standard format (e.g., 'G12D')
        protein_length: Length of the protein
        domain_info: Dictionary mapping domain names to (start, end) positions
        
    Returns:
        Plotly figure with the mutation diagram
    """
    # Parse the protein change
    match = re.match(r'([A-Z])(\d+)([A-Z\*])', protein_change)
    if not match:
        st.error(f"Could not parse protein change format: {protein_change}")
        return None
    
    wild_aa, position, mutant_aa = match.groups()
    position = int(position)
    
    # Validate position
    if position > protein_length:
        st.warning(f"Position {position} is beyond the provided protein length ({protein_length})")
    
    # Create a protein diagram
    fig = go.Figure()
    
    # Add the protein backbone
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0.4,
        x1=protein_length,
        y1=0.6,
        fillcolor="lightgrey",
        line=dict(color="black"),
        layer="below"
    )
    
    # Add protein domains if provided
    if domain_info:
        domain_colors = px.colors.qualitative.Set3
        for i, (domain, (start, end)) in enumerate(domain_info.items()):
            color = domain_colors[i % len(domain_colors)]
            
            # Add domain shape
            fig.add_shape(
                type="rect",
                x0=start,
                y0=0.3,
                x1=end,
                y1=0.7,
                fillcolor=color,
                opacity=0.7,
                line=dict(color="black"),
                layer="below"
            )
            
            # Add domain label
            fig.add_annotation(
                x=(start + end) / 2,
                y=0.2,
                text=domain,
                showarrow=False,
                font=dict(size=10)
            )
    
    # Add mutation marker
    fig.add_shape(
        type="rect",
        x0=position - 0.5,
        y0=0.25,
        x1=position + 0.5,
        y1=0.75,
        fillcolor="red",
        opacity=0.7,
        line=dict(color="black"),
        layer="above"
    )
    
    # Add mutation label
    fig.add_annotation(
        x=position,
        y=0.8,
        text=f"{wild_aa}{position}{mutant_aa}",
        showarrow=True,
        arrowhead=2,
        font=dict(size=12, color="black")
    )
    
    # Add title and layout settings
    fig.update_layout(
        title=f"{gene_name} Protein with {protein_change} Mutation",
        xaxis=dict(
            title="Amino Acid Position",
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white"
    )
    
    return fig


def format_elapsed_time(seconds: float) -> str:
    """
    Format elapsed time in a readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def create_progress_tracker(steps: List[str], current_step: int):
    """
    Create a visual progress tracker for a multi-step process
    
    Args:
        steps: List of step names
        current_step: Index of the current step (0-based)
    """
    # Create a horizontal container
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        # Calculate completion status
        if i < current_step:
            status = "✅"  # Completed
            color = "green"
        elif i == current_step:
            status = "🔄"  # In progress
            color = "blue"
        else:
            status = "⬜"  # Not started
            color = "gray"
        
        # Display the step with proper styling
        col.markdown(f"<div style='text-align: center; color: {color};'>{status}<br>{step}</div>", unsafe_allow_html=True)


def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Check if a DataFrame has all required columns
    
    Args:
        df: DataFrame to check
        required_columns: List of required column names
        
    Returns:
        True if all required columns are present, False otherwise
    """
    if df is None or df.empty:
        st.error("DataFrame is empty or None")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    
    return True


def display_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Display information about a DataFrame
    
    Args:
        df: DataFrame to display information for
        name: Name of the DataFrame for display purposes
    """
    if df is None or df.empty:
        st.error(f"{name} is empty or None")
        return
    
    st.write(f"**{name} Information:**")
    st.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display column types
    col_types = df.dtypes.value_counts().to_dict()
    col_types_str = ", ".join([f"{count} {dtype}" for dtype, count in col_types.items()])
    st.write(f"- Column Types: {col_types_str}")
    
    # Display missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        st.write(f"- Missing Values: {missing_values} ({missing_values/(df.shape[0]*df.shape[1]):.2%} of all cells)")
    else:
        st.write("- Missing Values: None")
    
    # Display memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    if memory_usage < 1024**2:
        st.write(f"- Memory Usage: {memory_usage/1024:.2f} KB")
    else:
        st.write(f"- Memory Usage: {memory_usage/(1024**2):.2f} MB")


def display_model_info(model: Any, name: str = "Model"):
    """
    Display information about a machine learning model
    
    Args:
        model: Model to display information for
        name: Name of the model for display purposes
    """
    if model is None:
        st.error(f"{name} is None")
        return
    
    st.write(f"**{name} Information:**")
    
    # Display model type
    model_type = type(model).__name__
    st.write(f"- Type: {model_type}")
    
    # Display model parameters if available
    if hasattr(model, 'get_params'):
        params = model.get_params()
        st.write("- Parameters:")
        for param, value in params.items():
            # Limit display of complex parameters
            if isinstance(value, (str, int, float, bool)) or value is None:
                st.write(f"  - {param}: {value}")
    
    # Display feature importance if available
    if hasattr(model, 'feature_importances_'):
        st.write("- Feature Importance Available: Yes")
    else:
        st.write("- Feature Importance Available: No")
    
    # Display additional model-specific information
    if hasattr(model, 'n_features_in_'):
        st.write(f"- Number of Features: {model.n_features_in_}")
    
    if hasattr(model, 'classes_'):
        st.write(f"- Number of Classes: {len(model.classes_)}")
    
    if hasattr(model, 'n_iter_'):
        st.write(f"- Number of Iterations: {model.n_iter_}")


def format_pvalue(p_value: float) -> str:
    """
    Format a p-value for display
    
    Args:
        p_value: P-value to format
        
    Returns:
        Formatted p-value string
    """
    if p_value < 0.001:
        return "p < 0.001"
    elif p_value < 0.01:
        return f"p = {p_value:.3f}"
    elif p_value < 0.05:
        return f"p = {p_value:.3f}"
    else:
        return f"p = {p_value:.3f} (not significant)"


def get_data_preview(df: pd.DataFrame, max_rows: int = 5, max_cols: int = 10) -> pd.DataFrame:
    """
    Get a preview of a DataFrame with sensible limits
    
    Args:
        df: DataFrame to preview
        max_rows: Maximum number of rows to include
        max_cols: Maximum number of columns to include
        
    Returns:
        Preview DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Limit the number of rows
    preview_df = df.head(max_rows)
    
    # Limit the number of columns if needed
    if len(preview_df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
        st.info(f"Showing only the first {max_cols} of {len(df.columns)} columns. Use 'full view' to see all data.")
    
    return preview_df
