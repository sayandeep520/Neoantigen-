import pandas as pd
import numpy as np
import streamlit as st
import os
import re
from typing import Dict, List, Union, Tuple, Optional, Any
from Bio.Seq import Seq
from Bio import SeqIO


class NeoantigenPredictor:
    """
    Predict neoantigens from tumor mutations for CRISPR-based cancer immunotherapy.
    Uses MHC binding prediction and TCR affinity estimation.
    """
    
    def __init__(self):
        """Initialize the neoantigen predictor with necessary parameters"""
        # Default parameters
        self.peptide_lengths = [8, 9, 10, 11]  # Common MHC-I peptide lengths
        self.flanking_region = 10  # Amino acids to include on each side
        
        # MHC types to consider
        self.mhc_alleles = [
            'HLA-A*01:01', 'HLA-A*02:01', 'HLA-A*03:01', 'HLA-A*24:02',
            'HLA-B*07:02', 'HLA-B*08:01', 'HLA-B*15:01', 'HLA-B*44:02',
            'HLA-C*07:01', 'HLA-C*07:02'
        ]
        
        # Binding thresholds
        self.strong_binding_threshold = 50  # nM (IC50)
        self.weak_binding_threshold = 500   # nM (IC50)
        
        # Known immunogenic mutations
        self.known_immunogenic_mutations = {
            'KRAS': ['G12D', 'G12V', 'G12C', 'G13D'],
            'TP53': ['R175H', 'R248Q', 'R273H'],
            'BRAF': ['V600E'],
            'EGFR': ['L858R', 'T790M'],
            'PIK3CA': ['E545K', 'H1047R'],
            'IDH1': ['R132H'],
            'IDH2': ['R140Q', 'R172K']
        }
        
        # Amino acid properties (used for immunogenicity prediction)
        self.aa_properties = {
            'A': {'hydrophobicity': 0.31, 'volume': 67, 'charge': 0},
            'R': {'hydrophobicity': -1.01, 'volume': 148, 'charge': 1},
            'N': {'hydrophobicity': -0.6, 'volume': 96, 'charge': 0},
            'D': {'hydrophobicity': -0.77, 'volume': 91, 'charge': -1},
            'C': {'hydrophobicity': 1.54, 'volume': 86, 'charge': 0},
            'Q': {'hydrophobicity': -0.22, 'volume': 114, 'charge': 0},
            'E': {'hydrophobicity': -0.64, 'volume': 109, 'charge': -1},
            'G': {'hydrophobicity': 0.0, 'volume': 48, 'charge': 0},
            'H': {'hydrophobicity': 0.13, 'volume': 118, 'charge': 0.1},
            'I': {'hydrophobicity': 1.8, 'volume': 124, 'charge': 0},
            'L': {'hydrophobicity': 1.7, 'volume': 124, 'charge': 0},
            'K': {'hydrophobicity': -0.99, 'volume': 135, 'charge': 1},
            'M': {'hydrophobicity': 1.23, 'volume': 124, 'charge': 0},
            'F': {'hydrophobicity': 1.79, 'volume': 135, 'charge': 0},
            'P': {'hydrophobicity': 0.72, 'volume': 90, 'charge': 0},
            'S': {'hydrophobicity': -0.04, 'volume': 73, 'charge': 0},
            'T': {'hydrophobicity': 0.26, 'volume': 93, 'charge': 0},
            'W': {'hydrophobicity': 2.25, 'volume': 163, 'charge': 0},
            'Y': {'hydrophobicity': 0.96, 'volume': 141, 'charge': 0},
            'V': {'hydrophobicity': 1.22, 'volume': 105, 'charge': 0}
        }
    
    def predict_neoantigens_from_mutations(self, 
                                          mutations: List[Dict[str, Any]],
                                          mhc_alleles: List[str] = None) -> List[Dict[str, Any]]:
        """
        Predict neoantigens from a list of somatic mutations
        
        Args:
            mutations: List of dictionaries with mutation information
            mhc_alleles: List of MHC alleles to predict binding for
            
        Returns:
            List of dictionaries with neoantigen predictions
        """
        if not mutations:
            st.error("No mutations provided for neoantigen prediction")
            return []
        
        # Use provided MHC alleles or default
        if mhc_alleles is None:
            mhc_alleles = self.mhc_alleles
        
        st.info(f"Predicting neoantigens for {len(mutations)} mutations across {len(mhc_alleles)} MHC alleles")
        
        # Process each mutation to generate candidate peptides
        predicted_neoantigens = []
        
        for mutation in mutations:
            # Check if required fields are present
            required_fields = ['gene', 'mutation_type', 'protein_change']
            if not all(field in mutation for field in required_fields):
                st.warning(f"Missing required fields in mutation: {mutation}")
                continue
            
            # Parse protein change
            protein_change = mutation.get('protein_change', '')
            if not protein_change:
                continue
            
            # Get gene name
            gene = mutation.get('gene', '')
            
            # Different mutation types (missense, frameshift, etc.) need different handling
            mutation_type = mutation.get('mutation_type', '')
            
            # Get wild-type and mutant peptide sequences
            try:
                # For a real system, we would retrieve the actual protein sequence and apply the mutation
                # For demonstration, we'll generate a synthetic peptide sequence
                wild_type_peptide, mutant_peptide = self._generate_peptide_from_mutation(gene, protein_change)
                
                if not wild_type_peptide or not mutant_peptide:
                    continue
                
                # Generate peptides of different lengths containing the mutation
                for peptide_length in self.peptide_lengths:
                    # Generate all possible peptides containing the mutation
                    mutant_peptides = self._generate_peptide_fragments(mutant_peptide, peptide_length)
                    
                    for peptide in mutant_peptides:
                        # For each MHC allele, predict binding
                        for allele in mhc_alleles:
                            # Predict MHC binding (IC50 in nM)
                            mhc_binding = self._predict_mhc_binding(peptide, allele)
                            
                            # If binding is strong or moderate, add to predictions
                            if mhc_binding <= self.weak_binding_threshold:
                                # Additional immunogenicity predictions
                                immunogenicity = self._predict_immunogenicity(peptide, wild_type_peptide)
                                
                                # Estimate TCR recognition probability
                                tcr_probability = self._estimate_tcr_recognition(peptide, allele)
                                
                                # Add neoantigen prediction
                                predicted_neoantigens.append({
                                    'gene': gene,
                                    'mutation': protein_change,
                                    'mutation_type': mutation_type,
                                    'peptide': peptide,
                                    'mhc_allele': allele,
                                    'binding_affinity': mhc_binding,
                                    'binding_level': 'Strong' if mhc_binding <= self.strong_binding_threshold else 'Weak',
                                    'immunogenicity_score': immunogenicity,
                                    'tcr_recognition': tcr_probability,
                                    'priority_score': self._calculate_priority_score(mhc_binding, immunogenicity, tcr_probability)
                                })
                
            except Exception as e:
                st.error(f"Error processing mutation {protein_change} in {gene}: {str(e)}")
                continue
        
        # Sort by priority score (descending)
        predicted_neoantigens.sort(key=lambda x: x['priority_score'], reverse=True)
        
        st.success(f"Predicted {len(predicted_neoantigens)} potential neoantigens")
        return predicted_neoantigens
    
    def _generate_peptide_from_mutation(self, 
                                       gene: str, 
                                       protein_change: str) -> Tuple[str, str]:
        """
        Generate wild-type and mutant peptide sequences from a protein change
        
        Args:
            gene: Gene name
            protein_change: Protein change in standard format (e.g., p.G12D)
            
        Returns:
            Tuple of (wild_type_peptide, mutant_peptide)
        """
        # Parse the protein change
        # Common formats: p.G12D, G12D, p.V600E, etc.
        if protein_change.startswith('p.'):
            protein_change = protein_change[2:]
        
        # Extract wild-type AA, position, and mutant AA
        match = re.match(r'([A-Z])(\d+)([A-Z\*])', protein_change)
        if not match:
            st.warning(f"Could not parse protein change format: {protein_change}")
            return "", ""
        
        wild_aa, position, mutant_aa = match.groups()
        position = int(position)
        
        # In a real implementation, we would fetch the actual protein sequence
        # For demonstration, we'll create a synthetic sequence
        
        # Create a synthetic wild-type sequence with the mutation position
        # Use real flanking sequences when available
        prefix = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRKHKEK"
        
        # Ensure the synthetic sequence has the correct amino acid at the mutation position
        # Extract region around mutation
        start_pos = max(0, position - 1 - self.flanking_region)
        end_pos = min(len(prefix), position - 1 + self.flanking_region + 1)
        
        wild_type_peptide = prefix[start_pos:end_pos]
        
        # Check if the wild-type AA matches what's expected
        mutation_index = min(position - 1 - start_pos, len(wild_type_peptide) - 1)
        if mutation_index >= 0 and mutation_index < len(wild_type_peptide):
            # Create mutant peptide by substituting the amino acid
            mutant_peptide = wild_type_peptide[:mutation_index] + mutant_aa + wild_type_peptide[mutation_index+1:]
        else:
            st.warning(f"Mutation position out of range: {position}")
            return "", ""
        
        return wild_type_peptide, mutant_peptide
    
    def _generate_peptide_fragments(self, peptide: str, length: int) -> List[str]:
        """
        Generate all possible peptide fragments of specified length
        
        Args:
            peptide: Full peptide sequence
            length: Desired fragment length
            
        Returns:
            List of peptide fragments
        """
        fragments = []
        
        if len(peptide) < length:
            return [peptide]
        
        # Generate all possible windows of specified length
        for i in range(len(peptide) - length + 1):
            fragments.append(peptide[i:i+length])
        
        return fragments
    
    def _predict_mhc_binding(self, peptide: str, allele: str) -> float:
        """
        Predict MHC binding affinity (IC50 in nM)
        
        Args:
            peptide: Peptide sequence
            allele: MHC allele
            
        Returns:
            Predicted binding affinity (lower is stronger)
        """
        # In a real implementation, this would call NetMHCpan or similar tool
        # For demonstration, use a simplified model based on peptide properties
        
        # Get peptide length
        peptide_length = len(peptide)
        
        # Check if peptide length is appropriate for MHC class I
        if peptide_length < 8 or peptide_length > 11:
            return 5000.0  # Very weak binding
        
        # Anchor positions for common HLA alleles (simplified)
        anchor_positions = {
            'HLA-A*01:01': [3, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-A*02:01': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-A*03:01': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-A*24:02': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-B*07:02': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-B*08:01': [3, 5, 8] if peptide_length >= 8 else [2, 4, peptide_length],
            'HLA-B*15:01': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-B*44:02': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-C*07:01': [2, 9] if peptide_length >= 9 else [2, peptide_length],
            'HLA-C*07:02': [2, 9] if peptide_length >= 9 else [2, peptide_length]
        }
        
        # Preferred amino acids at anchor positions (simplified)
        preferred_anchors = {
            'HLA-A*01:01': {'3': ['D', 'E'], '9': ['Y']},
            'HLA-A*02:01': {'2': ['L', 'M', 'I', 'V'], '9': ['V', 'L', 'I']},
            'HLA-A*03:01': {'2': ['I', 'L', 'V', 'M'], '9': ['K', 'R']},
            'HLA-A*24:02': {'2': ['Y', 'F', 'W', 'M'], '9': ['F', 'L', 'I', 'W']},
            'HLA-B*07:02': {'2': ['P'], '9': ['L', 'F', 'M']},
            'HLA-B*08:01': {'3': ['K', 'R'], '5': ['K', 'R'], '8': ['L', 'I', 'V']},
            'HLA-B*15:01': {'2': ['Q', 'L', 'I'], '9': ['Y', 'F', 'M']},
            'HLA-B*44:02': {'2': ['E'], '9': ['Y', 'F', 'W']},
            'HLA-C*07:01': {'2': ['Y', 'F'], '9': ['L', 'F', 'Y']},
            'HLA-C*07:02': {'2': ['Y', 'F'], '9': ['L', 'F', 'Y']}
        }
        
        # Start with a base affinity
        base_affinity = 500.0  # Default moderate binding
        
        # Adjust based on peptide length (9-mers typically bind better)
        if peptide_length == 9:
            base_affinity *= 0.7
        elif peptide_length == 10:
            base_affinity *= 1.2
        elif peptide_length == 8:
            base_affinity *= 1.5
        elif peptide_length == 11:
            base_affinity *= 2.0
        
        # Check if allele has defined anchor positions
        if allele in anchor_positions:
            positions = anchor_positions[allele]
            
            # Check if preferred amino acids are at anchor positions
            for pos in positions:
                pos_str = str(pos)
                
                # Adjust for 0-based indexing
                adjusted_pos = pos - 1
                
                if adjusted_pos < len(peptide):
                    aa = peptide[adjusted_pos]
                    
                    # Check if this amino acid is preferred at this position
                    if allele in preferred_anchors and pos_str in preferred_anchors[allele]:
                        if aa in preferred_anchors[allele][pos_str]:
                            # Strong anchor match
                            base_affinity *= 0.3
                        else:
                            # Non-optimal anchor
                            base_affinity *= 1.5
        
        # Add some randomness to simulate natural variation
        variation = np.random.uniform(0.8, 1.2)
        final_affinity = base_affinity * variation
        
        # Ensure the affinity is within a realistic range
        return min(5000.0, max(1.0, final_affinity))
    
    def _predict_immunogenicity(self, peptide: str, wild_type_peptide: str) -> float:
        """
        Predict immunogenicity of a peptide
        
        Args:
            peptide: Neoantigen peptide sequence
            wild_type_peptide: Corresponding wild-type peptide
            
        Returns:
            Immunogenicity score (0-1, higher is more immunogenic)
        """
        # Start with a base score
        base_score = 0.5
        
        # Compare physiochemical properties between mutant and wild-type
        try:
            # Find the position where they differ
            diff_positions = [i for i in range(min(len(peptide), len(wild_type_peptide))) 
                             if i < len(peptide) and i < len(wild_type_peptide) and peptide[i] != wild_type_peptide[i]]
            
            if diff_positions:
                for pos in diff_positions:
                    if pos < len(peptide) and peptide[pos] in self.aa_properties and pos < len(wild_type_peptide) and wild_type_peptide[pos] in self.aa_properties:
                        # Get amino acid properties
                        mutant_aa = peptide[pos]
                        wild_aa = wild_type_peptide[pos]
                        
                        # Calculate property differences
                        hydro_diff = abs(self.aa_properties[mutant_aa]['hydrophobicity'] - 
                                        self.aa_properties[wild_aa]['hydrophobicity'])
                        volume_diff = abs(self.aa_properties[mutant_aa]['volume'] - 
                                        self.aa_properties[wild_aa]['volume'])
                        charge_diff = abs(self.aa_properties[mutant_aa]['charge'] - 
                                        self.aa_properties[wild_aa]['charge'])
                        
                        # Normalize differences
                        hydro_diff = min(1.0, hydro_diff / 3.0)
                        volume_diff = min(1.0, volume_diff / 100.0)
                        
                        # Larger differences often result in higher immunogenicity
                        diff_score = (hydro_diff * 0.4 + volume_diff * 0.3 + charge_diff * 0.3)
                        
                        # Adjust base score
                        base_score += diff_score * 0.3
        except Exception as e:
            st.warning(f"Error comparing peptide properties: {str(e)}")
        
        # Check if the central portion of the peptide (TCR contact residues) has differences
        central_diff = False
        mid_point = len(peptide) // 2
        window = 2
        
        try:
            for i in range(max(0, mid_point - window), min(len(peptide), mid_point + window + 1)):
                if i < len(wild_type_peptide) and peptide[i] != wild_type_peptide[i]:
                    central_diff = True
                    break
            
            if central_diff:
                base_score += 0.1
        except Exception as e:
            st.warning(f"Error checking central differences: {str(e)}")
        
        # Check for presence of aromatic residues (often contribute to immunogenicity)
        aromatic_count = peptide.count('F') + peptide.count('Y') + peptide.count('W')
        base_score += min(0.1, aromatic_count * 0.03)
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, base_score))
    
    def _estimate_tcr_recognition(self, peptide: str, allele: str) -> float:
        """
        Estimate probability of TCR recognition
        
        Args:
            peptide: Peptide sequence
            allele: MHC allele
            
        Returns:
            TCR recognition probability (0-1)
        """
        # In a real implementation, this would use TCR binding prediction algorithms
        # For demonstration, use a simplified model
        
        # Base recognition probability
        base_probability = 0.3  # Moderate chance of recognition
        
        # Length-based adjustment (9-mers often have better TCR recognition)
        if len(peptide) == 9:
            base_probability *= 1.2
        elif len(peptide) == 10:
            base_probability *= 1.1
        elif len(peptide) == 8:
            base_probability *= 0.9
        
        # Adjust based on composition of TCR-facing residues
        # In MHC-I peptides, positions 4-6 often face the TCR
        tcr_facing_positions = range(3, min(6, len(peptide)))
        
        # Count amino acids with distinctive features in TCR-facing positions
        distinctive_aa_count = sum(1 for i in tcr_facing_positions 
                                 if i < len(peptide) and peptide[i] in 'FYWH')  # Aromatic/bulky residues
        
        base_probability += distinctive_aa_count * 0.05
        
        # Add some randomness to simulate natural variation
        variation = np.random.uniform(0.85, 1.15)
        final_probability = base_probability * variation
        
        # Ensure the probability is within a valid range
        return min(0.95, max(0.05, final_probability))
    
    def _calculate_priority_score(self, 
                                 binding_affinity: float, 
                                 immunogenicity: float, 
                                 tcr_recognition: float) -> float:
        """
        Calculate overall priority score for a neoantigen
        
        Args:
            binding_affinity: MHC binding affinity (lower is better)
            immunogenicity: Immunogenicity score (higher is better)
            tcr_recognition: TCR recognition probability (higher is better)
            
        Returns:
            Priority score (0-1, higher is better)
        """
        # Convert binding affinity to a 0-1 scale (lower nM = higher score)
        binding_score = max(0.0, min(1.0, 1.0 - (binding_affinity / self.weak_binding_threshold)))
        
        # Weight the components
        weighted_score = (
            binding_score * 0.4 +
            immunogenicity * 0.3 +
            tcr_recognition * 0.3
        )
        
        return weighted_score
    
    def filter_top_neoantigens(self, neoantigens: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Filter the top neoantigens based on priority score
        
        Args:
            neoantigens: List of neoantigen dictionaries
            top_n: Number of top neoantigens to return
            
        Returns:
            List of top neoantigens
        """
        # Sort by priority score (descending)
        sorted_neoantigens = sorted(neoantigens, key=lambda x: x['priority_score'], reverse=True)
        
        # Return top N
        return sorted_neoantigens[:min(top_n, len(sorted_neoantigens))]
    
    def generate_mhc_allele_coverage(self, neoantigens: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze MHC allele coverage of predicted neoantigens
        
        Args:
            neoantigens: List of neoantigen dictionaries
            
        Returns:
            Dictionary with MHC allele counts
        """
        # Count neoantigens by MHC allele
        allele_counts = {}
        
        for neoantigen in neoantigens:
            allele = neoantigen.get('mhc_allele', '')
            if allele:
                allele_counts[allele] = allele_counts.get(allele, 0) + 1
        
        return allele_counts
    
    def analyze_gene_coverage(self, neoantigens: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze gene coverage of predicted neoantigens
        
        Args:
            neoantigens: List of neoantigen dictionaries
            
        Returns:
            Dictionary with gene counts
        """
        # Count neoantigens by gene
        gene_counts = {}
        
        for neoantigen in neoantigens:
            gene = neoantigen.get('gene', '')
            if gene:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
        
        return gene_counts
    
    def generate_sample_mutations(self, num_mutations: int = 20) -> List[Dict[str, Any]]:
        """
        Generate sample mutations for demonstration
        
        Args:
            num_mutations: Number of mutations to generate
            
        Returns:
            List of mutation dictionaries
        """
        # Common genes and mutations
        common_gene_mutations = []
        
        for gene, mutations in self.known_immunogenic_mutations.items():
            for mutation in mutations:
                common_gene_mutations.append((gene, mutation))
        
        # Generate mutations
        sample_mutations = []
        
        for _ in range(num_mutations):
            # Either use a known mutation or generate a random one
            if np.random.random() < 0.7 and common_gene_mutations:
                gene, protein_change = common_gene_mutations[np.random.randint(0, len(common_gene_mutations))]
            else:
                # Generate a random mutation
                gene = f"GENE_{np.random.randint(1, 100)}"
                
                # Random amino acid positions
                position = np.random.randint(10, 500)
                
                # Random amino acid changes
                amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
                from_aa = amino_acids[np.random.randint(0, len(amino_acids))]
                to_aa = amino_acids[np.random.randint(0, len(amino_acids))]
                
                protein_change = f"{from_aa}{position}{to_aa}"
            
            # Create mutation dictionary
            mutation = {
                'gene': gene,
                'chromosome': f"chr{np.random.randint(1, 23)}",
                'position': np.random.randint(1000000, 100000000),
                'ref': ['A', 'C', 'G', 'T'][np.random.randint(0, 4)],
                'alt': ['A', 'C', 'G', 'T'][np.random.randint(0, 4)],
                'mutation_type': 'missense',
                'protein_change': protein_change
            }
            
            sample_mutations.append(mutation)
        
        return sample_mutations
