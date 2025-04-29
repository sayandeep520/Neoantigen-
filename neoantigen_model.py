import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import random
import re

class NeoantiGenAI:
    """
    Model for predicting neoantigens for cancer immunotherapy.
    """
    
    def __init__(self, algorithm="NetMHCpan", include_gene_expression=True, include_proteasomal_cleavage=True, include_tap_transport=True):
        """
        Initialize the neoantigen prediction model.
        
        Args:
            algorithm (str): Algorithm to use for MHC binding prediction
            include_gene_expression (bool): Whether to include gene expression in predictions
            include_proteasomal_cleavage (bool): Whether to consider proteasomal cleavage
            include_tap_transport (bool): Whether to consider TAP transport
        """
        self.algorithm = algorithm
        self.include_gene_expression = include_gene_expression
        self.include_proteasomal_cleavage = include_proteasomal_cleavage
        self.include_tap_transport = include_tap_transport
        
        # Initialize cached HLA-peptide binding data
        self.binding_cache = {}
        
        # Load amino acid properties for peptide analysis
        self.aa_properties = self._load_amino_acid_properties()
        
        # Load HLA allele frequencies
        self.hla_frequencies = self._load_hla_frequencies()
    
    def _load_amino_acid_properties(self):
        """
        Load amino acid physicochemical properties.
        
        Returns:
            dict: Dictionary of amino acid properties
        """
        # Basic hydrophobicity values (Kyte & Doolittle scale)
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # Charge
        charge = {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
            'G': 0, 'H': 0.5, 'I': 0, 'K': 1, 'L': 0,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # Size/bulkiness
        size = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 6,
            'G': 0, 'H': 4, 'I': 4, 'K': 5, 'L': 4,
            'M': 4, 'N': 2, 'P': 3, 'Q': 3, 'R': 6,
            'S': 1, 'T': 2, 'V': 3, 'W': 8, 'Y': 7
        }
        
        # Combine properties
        properties = {}
        for aa in hydrophobicity.keys():
            properties[aa] = {
                'hydrophobicity': hydrophobicity[aa],
                'charge': charge[aa],
                'size': size[aa]
            }
        
        return properties
    
    def _load_hla_frequencies(self):
        """
        Load HLA allele frequencies in the population.
        
        Returns:
            dict: Dictionary of HLA allele frequencies
        """
        # Population frequencies for common HLA alleles
        frequencies = {
            'HLA-A*01:01': 0.15,
            'HLA-A*02:01': 0.30,
            'HLA-A*03:01': 0.10,
            'HLA-A*24:02': 0.08,
            'HLA-B*07:02': 0.10,
            'HLA-B*08:01': 0.08,
            'HLA-B*15:01': 0.06,
            'HLA-B*35:01': 0.05,
            'HLA-B*44:03': 0.04,
            'HLA-C*07:01': 0.15
        }
        
        # Add frequency data for more alleles
        for allele in ['HLA-A*11:01', 'HLA-A*23:01', 'HLA-A*26:01', 'HLA-A*29:02', 
                      'HLA-A*30:01', 'HLA-A*31:01', 'HLA-A*32:01', 'HLA-A*33:01', 
                      'HLA-B*13:01', 'HLA-B*14:01', 'HLA-B*27:05', 'HLA-B*38:01',
                      'HLA-B*40:01', 'HLA-B*51:01', 'HLA-B*57:01']:
            # Assign a small random frequency for less common alleles
            frequencies[allele] = 0.01 + 0.03 * random.random()
        
        return frequencies
    
    def predict_mhc_binding(self, peptides, hla_alleles):
        """
        Predict MHC binding for a list of peptides and HLA alleles.
        
        Args:
            peptides (list): List of peptide sequences
            hla_alleles (list): List of HLA alleles
            
        Returns:
            pandas.DataFrame: Predicted binding results
        """
        # Initialize results list
        binding_results = []
        
        # For each peptide-HLA combination
        for peptide in peptides:
            for hla in hla_alleles:
                # Check if result is in cache
                cache_key = f"{peptide}_{hla}"
                if cache_key in self.binding_cache:
                    binding_score = self.binding_cache[cache_key]
                else:
                    # Predict binding based on selected algorithm
                    if self.algorithm == "NetMHCpan":
                        binding_score = self._predict_binding_netmhcpan(peptide, hla)
                    elif self.algorithm == "MHCflurry":
                        binding_score = self._predict_binding_mhcflurry(peptide, hla)
                    elif self.algorithm == "DeepHLApan":
                        binding_score = self._predict_binding_deephlpan(peptide, hla)
                    else:
                        binding_score = self._predict_binding_generic(peptide, hla)
                    
                    # Cache the result
                    self.binding_cache[cache_key] = binding_score
                
                # Create result entry
                result = {
                    'peptide': peptide,
                    'hla_allele': hla,
                    'mhc_affinity': binding_score,
                    'binding_level': 'Strong' if binding_score < 50 else 'Moderate' if binding_score < 500 else 'Weak'
                }
                
                # Add proteasomal cleavage score if enabled
                if self.include_proteasomal_cleavage:
                    result['proteasomal_score'] = self._predict_proteasomal_cleavage(peptide)
                
                # Add TAP transport score if enabled
                if self.include_tap_transport:
                    result['tap_score'] = self._predict_tap_transport(peptide)
                
                binding_results.append(result)
        
        # Convert to DataFrame
        return pd.DataFrame(binding_results)
    
    def predict_tcr_recognition(self, mhc_peptide_results):
        """
        Predict T-cell receptor recognition for MHC-bound peptides.
        
        Args:
            mhc_peptide_results (pandas.DataFrame): Results from MHC binding prediction
            
        Returns:
            pandas.DataFrame: Results with TCR recognition predictions
        """
        # Make a copy to avoid modifying the original
        results = mhc_peptide_results.copy()
        
        # Add TCR recognition probability
        results['tcr_probability'] = results.apply(
            lambda row: self._predict_tcr_probability(row['peptide'], row['mhc_affinity']), 
            axis=1
        )
        
        return results
    
    def calculate_immunogenicity(self, tcr_results):
        """
        Calculate overall immunogenicity scores.
        
        Args:
            tcr_results (pandas.DataFrame): Results from TCR recognition prediction
            
        Returns:
            pandas.DataFrame: Results with immunogenicity scores
        """
        # Make a copy to avoid modifying the original
        results = tcr_results.copy()
        
        # Normalize MHC binding affinity (lower is better)
        max_affinity = 5000  # Maximum considered value
        results['mhc_score'] = 1 - (results['mhc_affinity'] / max_affinity)
        results.loc[results['mhc_score'] < 0, 'mhc_score'] = 0  # Clip negative values
        
        # Combine factors to calculate immunogenicity
        # Weights can be adjusted based on importance
        mhc_weight = 0.4
        tcr_weight = 0.4
        other_weight = 0.2
        
        # Calculate other scores (e.g., proteasomal cleavage, TAP transport)
        other_score = 0.5  # Default value
        
        if self.include_proteasomal_cleavage and 'proteasomal_score' in results.columns:
            other_score = 0.5 * results['proteasomal_score']
            
        if self.include_tap_transport and 'tap_score' in results.columns:
            other_score = 0.5 * other_score + 0.5 * results['tap_score']
        
        # Calculate overall immunogenicity score
        results['immunogenicity_score'] = (
            mhc_weight * results['mhc_score'] +
            tcr_weight * results['tcr_probability'] +
            other_weight * other_score
        )
        
        return results
    
    def _predict_binding_netmhcpan(self, peptide, hla_allele):
        """
        Simulate NetMHCpan binding prediction.
        
        Args:
            peptide (str): Peptide sequence
            hla_allele (str): HLA allele
            
        Returns:
            float: Predicted binding affinity (nM)
        """
        # In a real implementation, this would call NetMHCpan
        # For demonstration, we'll generate a realistic prediction
        
        # Calculate binding factors based on peptide properties
        binding_factors = []
        
        # Factor 1: Peptide length (9-mers typically bind best)
        length_factor = 1.0
        if len(peptide) == 9:
            length_factor = 0.7  # Boost 9-mers
        elif len(peptide) == 10:
            length_factor = 0.9  # Slight boost for 10-mers
        elif len(peptide) == 11:
            length_factor = 1.2  # Slight penalty for 11-mers
        else:
            length_factor = 1.5  # Larger penalty for other lengths
        
        binding_factors.append(length_factor)
        
        # Factor 2: Anchor positions
        # For most HLA-A alleles, positions 2 and 9 (0-based: 1 and 8) are anchors
        anchor_factor = 1.0
        
        if 'HLA-A*02' in hla_allele:
            # A*02 prefers hydrophobic anchors
            if len(peptide) >= 9:
                if peptide[1] in 'LMIV' and peptide[8 if len(peptide) >= 9 else -1] in 'LIV':
                    anchor_factor = 0.4  # Strong binding
                elif peptide[1] in 'LMIVAFYWT' or peptide[8 if len(peptide) >= 9 else -1] in 'LIVFYWM':
                    anchor_factor = 0.7  # Moderate binding
        
        elif 'HLA-A*01' in hla_allele or 'HLA-A*03' in hla_allele:
            # A*01 and A*03 prefer charged/polar anchors
            if len(peptide) >= 9:
                if peptide[1] in 'TSEDY' and peptide[8 if len(peptide) >= 9 else -1] in 'KRY':
                    anchor_factor = 0.4  # Strong binding
                elif peptide[1] in 'TSEDYNQ' or peptide[8 if len(peptide) >= 9 else -1] in 'KRYH':
                    anchor_factor = 0.7  # Moderate binding
        
        elif 'HLA-B*07' in hla_allele or 'HLA-B*08' in hla_allele:
            # B*07 and B*08 have their own preferences
            if len(peptide) >= 9:
                if peptide[1] in 'P' and peptide[8 if len(peptide) >= 9 else -1] in 'L':
                    anchor_factor = 0.4  # Strong binding for B*07
                elif peptide[1] in 'RK' and peptide[8 if len(peptide) >= 9 else -1] in 'L':
                    anchor_factor = 0.4  # Strong binding for B*08
        
        binding_factors.append(anchor_factor)
        
        # Factor 3: HLA-specific factor
        # Different HLA alleles have different binding stringency
        hla_factor = 1.0
        
        # Adjust based on HLA frequency/promiscuity
        if hla_allele in self.hla_frequencies:
            # More common HLAs tend to be more promiscuous
            hla_factor = 0.8 + 0.4 * (1 - self.hla_frequencies[hla_allele])
        
        binding_factors.append(hla_factor)
        
        # Factor 4: Peptide charge and hydrophobicity
        try:
            analysis = ProteinAnalysis(peptide)
            
            # Hydrophobicity affects binding
            hydrophobicity = analysis.gravy()
            hydro_factor = 1.0 + 0.2 * (hydrophobicity + 2) / 4  # Normalize to ~0.8-1.2
            
            binding_factors.append(hydro_factor)
            
            # Charge can affect binding
            charge_factor = 1.0
            if len(peptide) >= 9:
                middle_peptide = peptide[2:7]  # Middle positions
                if 'R' in middle_peptide or 'K' in middle_peptide:
                    charge_factor = 1.1  # Penalty for positively charged residues
                elif 'D' in middle_peptide or 'E' in middle_peptide:
                    charge_factor = 1.1  # Penalty for negatively charged residues
            
            binding_factors.append(charge_factor)
            
        except Exception:
            # Default values if analysis fails
            binding_factors.extend([1.0, 1.0])
        
        # Factor 5: Random component for realism
        random_factor = 0.7 + 0.6 * np.random.beta(2, 5)  # Beta distribution for more realism
        binding_factors.append(random_factor)
        
        # Calculate final binding affinity
        # Scale factors to realistic nM range (1-5000, lower is better)
        binding_affinity = 50 * np.prod(binding_factors)
        
        # Ensure realistic range
        binding_affinity = max(1, min(5000, binding_affinity))
        
        return binding_affinity
    
    def _predict_binding_mhcflurry(self, peptide, hla_allele):
        """
        Simulate MHCflurry binding prediction.
        
        Args:
            peptide (str): Peptide sequence
            hla_allele (str): HLA allele
            
        Returns:
            float: Predicted binding affinity (nM)
        """
        # Similar to NetMHCpan but with slight variations
        # Base prediction on NetMHCpan with adjustments
        base_prediction = self._predict_binding_netmhcpan(peptide, hla_allele)
        
        # Add MHCflurry-specific adjustments
        adjustment = 0.8 + 0.4 * np.random.random()  # 0.8-1.2 adjustment factor
        
        return base_prediction * adjustment
    
    def _predict_binding_deephlpan(self, peptide, hla_allele):
        """
        Simulate DeepHLApan binding prediction.
        
        Args:
            peptide (str): Peptide sequence
            hla_allele (str): HLA allele
            
        Returns:
            float: Predicted binding affinity (nM)
        """
        # Similar to NetMHCpan but with slight variations
        # Base prediction on NetMHCpan with adjustments
        base_prediction = self._predict_binding_netmhcpan(peptide, hla_allele)
        
        # Add DeepHLApan-specific adjustments
        adjustment = 0.7 + 0.6 * np.random.random()  # 0.7-1.3 adjustment factor
        
        return base_prediction * adjustment
    
    def _predict_binding_generic(self, peptide, hla_allele):
        """
        Generic binding prediction for demonstration.
        
        Args:
            peptide (str): Peptide sequence
            hla_allele (str): HLA allele
            
        Returns:
            float: Predicted binding affinity (nM)
        """
        # Simple binding prediction based on peptide length and composition
        
        # Length factor (9-mers bind best)
        length_factor = abs(len(peptide) - 9) * 100 + 50
        
        # Count hydrophobic residues at positions 2 and C-terminus (common anchors)
        hydrophobic = 'LIVFMYW'
        anchor_factor = 500
        
        if len(peptide) >= 3:
            if peptide[1] in hydrophobic:
                anchor_factor -= 200
            
            if peptide[-1] in hydrophobic:
                anchor_factor -= 200
        
        # Add random component
        random_factor = 50 + 450 * np.random.random()
        
        # Calculate final score
        binding_affinity = length_factor + anchor_factor + random_factor
        
        # Ensure realistic range
        binding_affinity = max(1, min(5000, binding_affinity))
        
        return binding_affinity
    
    def _predict_proteasomal_cleavage(self, peptide):
        """
        Predict proteasomal cleavage likelihood.
        
        Args:
            peptide (str): Peptide sequence
            
        Returns:
            float: Cleavage probability (0-1)
        """
        # Simple proteasomal cleavage prediction
        # Proteasome prefers to cleave after hydrophobic or charged residues
        
        # Calculate C-terminal cleavage score
        c_term = peptide[-1]
        
        if c_term in 'LIVMFYW':  # Hydrophobic C-terminus (good)
            c_score = 0.8 + 0.2 * np.random.random()
        elif c_term in 'RKD':  # Charged C-terminus (moderate)
            c_score = 0.6 + 0.2 * np.random.random()
        elif c_term in 'P':  # Proline (bad - blocks cleavage)
            c_score = 0.2 + 0.3 * np.random.random()
        else:  # Other amino acids (average)
            c_score = 0.4 + 0.3 * np.random.random()
        
        # Calculate N-terminal extension score
        if len(peptide) >= 3:
            n_ext = peptide[:2]
            
            if 'P' in n_ext:  # Proline inhibits cleavage
                n_score = 0.3 + 0.3 * np.random.random()
            elif any(aa in n_ext for aa in 'DE'):  # Acidic residues enhance
                n_score = 0.7 + 0.2 * np.random.random()
            else:
                n_score = 0.5 + 0.3 * np.random.random()
        else:
            n_score = 0.5
        
        # Combine scores (70% C-terminal, 30% N-terminal extension)
        cleavage_score = 0.7 * c_score + 0.3 * n_score
        
        return cleavage_score
    
    def _predict_tap_transport(self, peptide):
        """
        Predict TAP transport efficiency.
        
        Args:
            peptide (str): Peptide sequence
            
        Returns:
            float: Transport efficiency (0-1)
        """
        # TAP transport prediction
        # TAP prefers hydrophobic and basic residues, dislikes acidic ones
        
        # Calculate scores based on first 3 and last residue
        if len(peptide) < 4:
            return 0.5  # Default for very short peptides
        
        # N-terminal score (first 3 residues)
        n_term = peptide[:3]
        
        n_score = 0.5  # Default
        
        # Count favorable residues
        favorable = sum(1 for aa in n_term if aa in 'LIVMFYWRK')
        unfavorable = sum(1 for aa in n_term if aa in 'DE')
        
        n_score = 0.5 + 0.1 * favorable - 0.15 * unfavorable
        
        # C-terminal score (last residue)
        c_term = peptide[-1]
        
        if c_term in 'LIVMFYW':  # Hydrophobic (good)
            c_score = 0.8 + 0.2 * np.random.random()
        elif c_term in 'RK':  # Basic (good)
            c_score = 0.7 + 0.2 * np.random.random()
        elif c_term in 'DE':  # Acidic (bad)
            c_score = 0.2 + 0.3 * np.random.random()
        else:  # Other (moderate)
            c_score = 0.5 + 0.3 * np.random.random()
        
        # Overall peptide charge
        charge = sum(1 for aa in peptide if aa in 'RK') - sum(1 for aa in peptide if aa in 'DE')
        charge_factor = 0.5 + 0.05 * charge  # Positive charge is slightly beneficial
        
        # Combine scores
        transport_score = 0.3 * n_score + 0.5 * c_score + 0.2 * charge_factor
        
        # Ensure valid range
        transport_score = max(0, min(1, transport_score))
        
        return transport_score
    
    def _predict_tcr_probability(self, peptide, mhc_affinity):
        """
        Predict probability of TCR recognition.
        
        Args:
            peptide (str): Peptide sequence
            mhc_affinity (float): MHC binding affinity (nM)
            
        Returns:
            float: TCR recognition probability (0-1)
        """
        # TCR recognition depends on several factors
        
        # Factor 1: MHC binding strength (better binding = better recognition)
        # Transform binding affinity to 0-1 scale (1 = best binding)
        mhc_factor = 1 - (mhc_affinity / 5000)  # 5000 nM as max value
        mhc_factor = max(0, min(1, mhc_factor))
        
        # Factor 2: Peptide properties
        try:
            # Analyze peptide
            analysis = ProteinAnalysis(peptide)
            
            # Hydrophobicity affects recognition (moderate is best)
            hydrophobicity = analysis.gravy()
            
            # Transform to 0-1 scale, with peak around 0
            hydro_optimum = 0  # Optimum hydrophobicity
            hydro_factor = 1 - min(1, abs(hydrophobicity - hydro_optimum) / 2)
            
            # Aromatic residues improve recognition (central positions matter most)
            aromatic_count = 0
            center_start = max(0, len(peptide) // 2 - 2)
            center_end = min(len(peptide), len(peptide) // 2 + 3)
            
            for i, aa in enumerate(peptide):
                if aa in 'FWY':  # Aromatic amino acids
                    # Weight by position (central positions matter more)
                    if center_start <= i < center_end:
                        aromatic_count += 1.5  # Central positions
                    else:
                        aromatic_count += 1.0  # Flanking positions
            
            aromatic_factor = min(1, aromatic_count / (len(peptide) * 0.5))
            
        except Exception:
            # Fallback if analysis fails
            hydro_factor = 0.5
            aromatic_factor = 0.3
        
        # Factor 3: Dissimilarity from self (non-self peptides more likely to be recognized)
        # In a real model, this would compare to human proteome
        # Here we use a random factor as proxy
        nonself_factor = 0.4 + 0.6 * np.random.random()
        
        # Combine factors
        tcr_probability = (
            0.4 * mhc_factor +      # MHC binding
            0.2 * hydro_factor +    # Hydrophobicity
            0.2 * aromatic_factor + # Aromatic content
            0.2 * nonself_factor    # Non-self factor
        )
        
        # Add some randomness for realism
        tcr_probability = tcr_probability * (0.8 + 0.4 * np.random.random())
        
        # Ensure valid range
        tcr_probability = max(0, min(1, tcr_probability))
        
        return tcr_probability
