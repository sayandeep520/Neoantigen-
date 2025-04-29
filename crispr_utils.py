import numpy as np
import pandas as pd
from Bio.Seq import Seq
import random
import re

# Temporarily comment out to prevent errors
# from utils.deepseek_reasoning import DeepSeekReasoning

def design_sgrna(gene, gc_content_range=(40, 60), guide_length=20, pam_sequence="NGG"):
    """
    Design candidate sgRNAs for a target gene.

    Args:
        gene (str): Target gene symbol
        gc_content_range (tuple): Range of acceptable GC content percentages
        guide_length (int): Length of the guide RNA
        pam_sequence (str): PAM sequence to use

    Returns:
        pandas.DataFrame: Candidate sgRNAs with properties
    """
    # This function would normally fetch gene sequence from a database
    # For demonstration, we'll generate synthetic sequences

    # Create a synthetic DNA sequence for the gene
    gene_length = random.randint(1000, 3000)  # Random gene length
    nucleotides = ['A', 'C', 'G', 'T']

    # Bias GC content based on gene name (for demonstration variety)
    if gene in ['TP53', 'BRCA1', 'BRCA2', 'ATM']:
        weights = [0.25, 0.25, 0.25, 0.25]  # Balanced
    elif gene in ['KRAS', 'NRAS', 'BRAF']:
        weights = [0.2, 0.3, 0.3, 0.2]  # Higher GC
    else:
        weights = [0.3, 0.2, 0.2, 0.3]  # Lower GC

    gene_sequence = ''.join(np.random.choice(nucleotides, size=gene_length, p=weights))

    # Find all PAM sites
    # Replace N in PAM with regex pattern
    pam_pattern = pam_sequence.replace('N', '[ACGT]')

    # Find all matches
    pam_matches = []
    for match in re.finditer(f'(?=({pam_pattern}))', gene_sequence):
        position = match.start()
        if position >= guide_length:  # Ensure we have enough sequence upstream
            pam_matches.append(position)

    # Generate sgRNAs for each PAM site
    sgrna_data = []

    for position in pam_matches[:100]:  # Limit to 100 candidates
        # Get guide sequence
        guide_seq = gene_sequence[position - guide_length:position]

        # Calculate GC content
        gc_count = guide_seq.count('G') + guide_seq.count('C')
        gc_content = (gc_count / guide_length) * 100

        # Check if GC content is within desired range
        if gc_content_range[0] <= gc_content <= gc_content_range[1]:
            entry = {
                'gene': gene,
                'position': position,
                'sequence': guide_seq,
                'pam': gene_sequence[position:position+len(pam_sequence)],
                'gc_content': gc_content,
                'self_complementarity': calculate_self_complementarity(guide_seq),
                'efficiency_score': None,  # Will be filled later
                'off_target_count': None  # Will be filled later
            }
            sgrna_data.append(entry)

    # Create DataFrame
    sgrna_df = pd.DataFrame(sgrna_data)

    # If no candidates found, create some synthetic ones
    if len(sgrna_df) == 0:
        sgrna_data = []

        for i in range(20):
            # Generate guide with GC content in the desired range
            min_gc = int(guide_length * gc_content_range[0] / 100)
            max_gc = int(guide_length * gc_content_range[1] / 100)
            gc_count = random.randint(min_gc, max_gc)
            at_count = guide_length - gc_count

            guide_seq = ''.join(random.choices(['G', 'C'], k=gc_count) + random.choices(['A', 'T'], k=at_count))
            guide_seq = ''.join(random.sample(guide_seq, len(guide_seq)))  # Shuffle

            gc_content = (guide_seq.count('G') + guide_seq.count('C')) / guide_length * 100

            entry = {
                'gene': gene,
                'position': i * 50,
                'sequence': guide_seq,
                'pam': ''.join(random.choices(['A', 'C', 'G', 'T'], k=len(pam_sequence))),
                'gc_content': gc_content,
                'self_complementarity': calculate_self_complementarity(guide_seq),
                'efficiency_score': None,
                'off_target_count': None
            }
            sgrna_data.append(entry)

        sgrna_df = pd.DataFrame(sgrna_data)

    return sgrna_df

def calculate_self_complementarity(sequence):
    """
    Calculate the self-complementarity score of a sequence.

    Args:
        sequence (str): DNA sequence

    Returns:
        float: Self-complementarity score (0-1)
    """
    # Convert to Seq object
    seq = Seq(sequence)

    # Get reverse complement
    rev_comp = seq.reverse_complement()

    # Calculate complementary bases
    complementary_count = 0
    for i in range(len(sequence)):
        if i < len(sequence) - 1 and i + 1 < len(rev_comp):
            dinucleotide = sequence[i:i+2]
            rev_dinucleotide = str(rev_comp)[i:i+2]

            if dinucleotide == rev_dinucleotide:
                complementary_count += 1

    # Normalize by sequence length
    return complementary_count / max(1, len(sequence) - 1)

def predict_off_target_effects(sgrnas, algorithm="Cas-OFFinder", min_mismatch_distance=3):
    """
    Predict off-target effects for sgRNAs.

    Args:
        sgrnas (pandas.DataFrame): Candidate sgRNAs
        algorithm (str): Algorithm to use for prediction
        min_mismatch_distance (int): Minimum number of mismatches to consider

    Returns:
        pandas.DataFrame: sgRNAs with off-target predictions
    """
    # Make a copy to avoid modifying the original
    df = sgrnas.copy()

    # In a real implementation, this would use the specified algorithm
    # For demonstration, we'll generate synthetic off-target scores

    for idx in df.index:
        sequence = df.loc[idx, 'sequence']

        # Adjust off-target likelihood based on sequence properties
        # Higher GC content often correlates with more off-targets
        gc_factor = df.loc[idx, 'gc_content'] / 50  # Normalize to 1

        # Self-complementarity can affect off-target binding
        self_comp_factor = df.loc[idx, 'self_complementarity'] * 2

        # Base randomness factor
        random_factor = np.random.beta(2, 5)  # Gives values biased toward lower range

        # Calculate off-target count, biased by factors
        base_count = int(random_factor * 10 * gc_factor * (1 + self_comp_factor))

        # Add some variability
        off_target_count = max(0, base_count + np.random.randint(-2, 3))

        # Update DataFrame
        df.loc[idx, 'off_target_count'] = off_target_count

        # Add additional off-target metrics
        if algorithm == "CFD Score":
            df.loc[idx, 'cfd_score'] = np.random.beta(2, 5)
        elif algorithm == "MIT Score":
            df.loc[idx, 'mit_score'] = np.random.beta(2, 5)

    return df

def evaluate_on_target_efficiency(sgrnas, model=None, prediction_method="DeepCRISPR"):
    """
    Evaluate on-target efficiency for sgRNAs.
    
    Args:
        sgrnas (pandas.DataFrame): Candidate sgRNAs
        model: Optional pre-trained model for prediction (unused in this version)
        prediction_method (str): Method to use for prediction
        
    Returns:
        pandas.DataFrame: sgRNAs with updated efficiency scores
    """
    # Make a copy to avoid modifying the original
    df = sgrnas.copy()
    
    # For each sgRNA, predict efficiency
    for idx in df.index:
        sequence = df.loc[idx, 'sequence']
        
        # Base efficiency on sequence properties
        gc_content = df.loc[idx, 'gc_content']
        
        # Optimal GC content is around 50-55%
        gc_factor = 1 - (abs(gc_content - 52.5) / 50)
        
        # Self-complementarity reduces efficiency
        self_comp_factor = 1 - df.loc[idx, 'self_complementarity']
        
        # Method-specific adjustments
        if prediction_method == "DeepCRISPR":
            # Base score (random for demonstration)
            base_score = np.random.beta(5, 2)  # Bias toward higher values
            # Adjust by factors
            efficiency = base_score * gc_factor * self_comp_factor
        elif prediction_method == "Azimuth":
            base_score = np.random.beta(4, 2)
            efficiency = base_score * (gc_factor ** 1.2) * self_comp_factor
        elif prediction_method == "CRISPRscan":
            base_score = np.random.beta(6, 3)
            efficiency = base_score * gc_factor * (self_comp_factor ** 1.3)
        else:
            # Default method
            base_score = np.random.beta(5, 2)
            efficiency = base_score * gc_factor * self_comp_factor
        
        # Update DataFrame
        df.loc[idx, 'efficiency_score'] = min(1.0, max(0.0, efficiency))
        
        # Add reasoning (for compatibility but not used)
        df.loc[idx, 'efficiency_reasoning'] = f"GC content: {gc_content:.1f}%, Self-complementarity: {df.loc[idx, 'self_complementarity']:.2f}"
    
    return df

def rank_crispr_targets(sgrnas):
    """
    Rank CRISPR targets by combined efficiency and specificity.

    Args:
        sgrnas (pandas.DataFrame): sgRNAs with efficiency and off-target scores

    Returns:
        pandas.DataFrame: Ranked sgRNAs
    """
    # Make a copy to avoid modifying the original
    df = sgrnas.copy()

    # Ensure we have the necessary columns
    if 'efficiency_score' not in df.columns or 'off_target_count' not in df.columns:
        raise ValueError("sgRNA DataFrame must contain 'efficiency_score' and 'off_target_count' columns")

    # Calculate specificity score (inverse of off-target count)
    max_off_targets = df['off_target_count'].max() if df['off_target_count'].max() > 0 else 1
    df['specificity_score'] = 1 - (df['off_target_count'] / max_off_targets)

    # Calculate combined score
    # Weight efficiency vs. specificity (adjustable)
    efficiency_weight = 0.7
    specificity_weight = 0.3

    df['combined_score'] = (
        efficiency_weight * df['efficiency_score'] + 
        specificity_weight * df['specificity_score']
    )

    # Rank by combined score
    df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)

    return df