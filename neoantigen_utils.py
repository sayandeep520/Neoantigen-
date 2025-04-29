import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
import re

def predict_mhc_binding(mutation_data, hla_alleles, binding_threshold=500, peptide_lengths=[9, 10, 11], prediction_model=None):
    """
    Predict MHC binding for potential neoantigens.
    
    Args:
        mutation_data (pandas.DataFrame): Mutation data with gene and mutation information
        hla_alleles (list): List of HLA alleles to predict binding for
        binding_threshold (float): Binding affinity threshold in nM (lower is stronger)
        peptide_lengths (list): List of peptide lengths to consider
        prediction_model: Pre-trained model for prediction (optional)
        
    Returns:
        pandas.DataFrame: Predicted MHC binding results
    """
    # In a real implementation, this would extract mutated peptides from the mutations
    # and predict MHC binding using NetMHCpan or similar tools
    
    # Generate synthetic peptides for demonstration
    binding_results = []
    
    # Get unique mutations if available
    unique_genes = mutation_data['gene'].unique() if 'gene' in mutation_data.columns else []
    
    # If no genes found, use common cancer genes
    if len(unique_genes) == 0:
        unique_genes = ["TP53", "KRAS", "CDKN2A", "SMAD4", "BRCA2", "EGFR", "PIK3CA", "PTEN", "RB1", "APC"]
    
    # Generate peptides for each gene and HLA allele combination
    for gene in unique_genes:
        # Generate 2-5 peptides per gene
        num_peptides = random.randint(2, 5)
        
        for i in range(num_peptides):
            # Select random peptide length
            peptide_length = random.choice(peptide_lengths)
            
            # Generate random peptide
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            peptide = ''.join(random.choices(amino_acids, k=peptide_length))
            
            # Generate "wild type" peptide (1-3 mutations different)
            num_mutations = random.randint(1, min(3, peptide_length))
            mutation_positions = random.sample(range(peptide_length), num_mutations)
            
            wild_type = list(peptide)
            for pos in mutation_positions:
                # Ensure different amino acid
                current = wild_type[pos]
                alternatives = amino_acids.replace(current, '')
                wild_type[pos] = random.choice(alternatives)
            wild_type = ''.join(wild_type)
            
            # Set mutation position to the middle if not specified
            mutation_position = mutation_positions[0] if mutation_positions else peptide_length // 2
            
            # For each HLA allele
            for hla in hla_alleles:
                # In a real model, binding would be predicted based on peptide and HLA
                # Here we generate synthetic binding scores
                
                # Generate binding affinity (lower is stronger binding)
                # Use beta distribution for realistic values (most peptides bind poorly)
                raw_affinity = np.random.beta(1.2, 3.5)
                
                # Scale to nM range (1-5000)
                # Stronger binding (lower values) for certain genes like KRAS, TP53
                if gene in ["KRAS", "TP53", "BRAF"]:
                    affinity = 1 + raw_affinity * 500  # 1-500 nM (strong binding)
                else:
                    affinity = 100 + raw_affinity * 5000  # 100-5000 nM (mostly weak binding)
                
                # Consider peptide length (9-mers often bind better)
                if peptide_length == 9:
                    affinity *= 0.7  # Boost 9-mers
                
                # Only include results below threshold
                if affinity <= binding_threshold:
                    result = {
                        'peptide': peptide,
                        'wild_type': wild_type,
                        'gene': gene,
                        'mutation': f"{wild_type[mutation_position]}{mutation_position+1}{peptide[mutation_position]}",
                        'mutation_position': mutation_position,
                        'hla_allele': hla,
                        'mhc_affinity': affinity,
                        'binding_level': 'Strong' if affinity < 50 else 'Moderate' if affinity < 500 else 'Weak'
                    }
                    binding_results.append(result)
    
    # Create DataFrame
    binding_df = pd.DataFrame(binding_results)
    
    # If we have no results, create at least a few
    if len(binding_df) == 0:
        for i in range(10):
            gene = random.choice(unique_genes)
            peptide_length = random.choice(peptide_lengths)
            peptide = ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=peptide_length))
            hla = random.choice(hla_alleles)
            affinity = random.uniform(1, binding_threshold)
            
            result = {
                'peptide': peptide,
                'wild_type': ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=peptide_length)),
                'gene': gene,
                'mutation': f"X{random.randint(1, peptide_length)}X",
                'mutation_position': random.randint(0, peptide_length-1),
                'hla_allele': hla,
                'mhc_affinity': affinity,
                'binding_level': 'Strong' if affinity < 50 else 'Moderate' if affinity < 500 else 'Weak'
            }
            binding_results.append(result)
        
        binding_df = pd.DataFrame(binding_results)
    
    return binding_df

def predict_tcr_affinity(mhc_binding_results, tcr_affinity_threshold=0.5, prediction_model=None):
    """
    Predict TCR recognition probability for MHC-bound peptides.
    
    Args:
        mhc_binding_results (pandas.DataFrame): Results from MHC binding prediction
        tcr_affinity_threshold (float): Threshold for TCR recognition probability
        prediction_model: Pre-trained model for prediction (optional)
        
    Returns:
        pandas.DataFrame: Results with TCR affinity predictions
    """
    # Make a copy to avoid modifying the original
    df = mhc_binding_results.copy()
    
    # In a real implementation, this would use a model to predict TCR recognition
    # For demonstration, we'll generate synthetic recognition probabilities
    
    for idx in df.index:
        peptide = df.loc[idx, 'peptide']
        
        # Calculate TCR recognition probability
        # This would normally use features of the peptide-MHC complex
        
        # Factors that influence TCR recognition:
        # 1. MHC binding strength (stronger binding = better presentation)
        mhc_factor = 1 - (df.loc[idx, 'mhc_affinity'] / 5000)  # Normalize to 0-1
        
        # 2. Peptide characteristics
        try:
            # Analyze peptide
            analysis = ProteinAnalysis(peptide)
            
            # Hydrophobicity correlates with immunogenicity
            hydrophobicity = analysis.gravy()
            hydro_factor = (hydrophobicity + 2) / 4  # Normalize to ~0-1
            
            # Aromatic content correlates with immunogenicity
            aromatic_count = sum(1 for aa in peptide if aa in 'FWY')
            aromatic_factor = aromatic_count / len(peptide)
            
        except Exception:
            # Fallback if analysis fails
            hydro_factor = 0.5
            aromatic_factor = 0.3
        
        # 3. Base randomness
        random_factor = np.random.beta(2, 3)  # Gives values between 0-1, biased toward middle range
        
        # Calculate overall probability
        base_probability = (0.4 * mhc_factor + 0.2 * hydro_factor + 0.2 * aromatic_factor + 0.2 * random_factor)
        
        # Add some gene-specific effects
        if df.loc[idx, 'gene'] in ['KRAS', 'TP53', 'BRAF']:
            # Common oncogenes often have more immunogenic mutations
            base_probability *= 1.2
        
        # Ensure value is in 0-1 range
        tcr_probability = max(0, min(1, base_probability))
        
        # Update DataFrame
        df.loc[idx, 'tcr_probability'] = tcr_probability
    
    # Filter by threshold
    if tcr_affinity_threshold > 0:
        df = df[df['tcr_probability'] >= tcr_affinity_threshold]
    
    return df

def screen_neoantigens(tcr_affinity_results, expression_threshold=50, mutation_types=None):
    """
    Screen neoantigens based on additional criteria.
    
    Args:
        tcr_affinity_results (pandas.DataFrame): Results from TCR affinity prediction
        expression_threshold (float): Percentile threshold for gene expression
        mutation_types (list): Types of mutations to include
        
    Returns:
        pandas.DataFrame: Filtered neoantigen results
    """
    # Make a copy to avoid modifying the original
    df = tcr_affinity_results.copy()
    
    # Add gene expression data (simulated)
    genes = df['gene'].unique()
    
    # Create synthetic expression data
    expression_data = {}
    for gene in genes:
        # Generate expression value using log-normal distribution (realistic for expression data)
        expression = np.random.lognormal(mean=2, sigma=1)
        
        # Adjust expression for known cancer genes
        if gene in ['KRAS', 'MYC', 'EGFR']:
            expression *= 2  # Commonly overexpressed
        elif gene in ['TP53', 'CDKN2A', 'PTEN']:
            expression *= 0.5  # Commonly underexpressed
        
        expression_data[gene] = expression
    
    # Add expression to DataFrame
    df['gene_expression'] = df['gene'].map(expression_data)
    
    # Calculate expression percentile
    expression_values = list(expression_data.values())
    percentiles = {gene: 100 * np.searchsorted(np.sort(expression_values), expr) / len(expression_values)
                  for gene, expr in expression_data.items()}
    
    df['expression_percentile'] = df['gene'].map(percentiles)
    
    # Filter by expression threshold if specified
    if expression_threshold > 0:
        df = df[df['expression_percentile'] >= expression_threshold]
    
    # Filter by mutation type if specified
    if mutation_types:
        # Extract mutation type from mutation column (format: X123Y)
        def extract_mutation_type(mutation):
            if not mutation or len(mutation) < 3:
                return 'Unknown'
            
            if mutation[0] == mutation[-1]:
                return 'Silent'
            elif mutation[0] == 'X' or mutation[-1] == 'X':
                return 'Unknown'
            else:
                return 'Missense'
        
        df['mutation_type'] = df['mutation'].apply(extract_mutation_type)
        
        # Filter to specified types
        df = df[df['mutation_type'].isin(mutation_types)]
    
    return df

def rank_neoantigens(screened_neoantigens, include_gene_expression=True):
    """
    Rank neoantigens by predicted immunogenicity.
    
    Args:
        screened_neoantigens (pandas.DataFrame): Screened neoantigen results
        include_gene_expression (bool): Whether to include gene expression in ranking
        
    Returns:
        pandas.DataFrame: Ranked neoantigens
    """
    # Make a copy to avoid modifying the original
    df = screened_neoantigens.copy()
    
    # Calculate immunogenicity score
    # Factors:
    # 1. MHC binding affinity (lower is better)
    # 2. TCR recognition probability (higher is better)
    # 3. Gene expression (higher is better, if included)
    
    # Normalize MHC affinity (0-1, higher is better)
    max_affinity = 5000  # Set maximum value for normalization
    df['mhc_score'] = 1 - (df['mhc_affinity'] / max_affinity)
    df.loc[df['mhc_score'] < 0, 'mhc_score'] = 0  # Clip negative values
    
    # TCR probability is already 0-1
    
    # Normalize gene expression if included
    if include_gene_expression and 'expression_percentile' in df.columns:
        df['expression_score'] = df['expression_percentile'] / 100
    else:
        df['expression_score'] = 1  # Default if not using expression
    
    # Compute overall immunogenicity score
    # Weights can be adjusted based on importance
    mhc_weight = 0.4
    tcr_weight = 0.4
    expression_weight = 0.2 if include_gene_expression else 0
    
    df['immunogenicity_score'] = (
        mhc_weight * df['mhc_score'] +
        tcr_weight * df['tcr_probability'] +
        expression_weight * df['expression_score']
    )
    
    # Rank by immunogenicity score
    df = df.sort_values('immunogenicity_score', ascending=False).reset_index(drop=True)
    
    return df
