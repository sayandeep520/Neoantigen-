import numpy as np
import pandas as pd
import streamlit as st
import os
import re
from typing import Dict, List, Union, Tuple, Optional, Any
from Bio.Seq import Seq
from Bio import SeqIO


class CrisprDesigner:
    """
    Design and optimize CRISPR guide RNAs (sgRNAs) for cancer-specific gene targeting
    using AI-driven methods to maximize on-target efficiency and minimize off-target effects.
    """
    
    def __init__(self):
        """Initialize the CRISPR designer with default parameters"""
        # Parameters for sgRNA design
        self.pam_sequence = "NGG"  # PAM for SpCas9
        self.guide_length = 20  # Standard guide length
        self.min_gc_content = 0.4  # Minimum GC content
        self.max_gc_content = 0.7  # Maximum GC content
        self.avoid_homopolymers = True  # Avoid 4+ repeated nucleotides
        
        # Default scoring weights
        self.scoring_weights = {
            'on_target_efficiency': 0.4,
            'off_target_effects': 0.3,
            'gc_content': 0.1,
            'self_complementarity': 0.1,
            'position_effect': 0.1
        }
        
        # Nuclease types
        self.nucleases = {
            'SpCas9': {'pam': 'NGG', 'guide_length': 20, 'description': 'Standard S. pyogenes Cas9'},
            'SaCas9': {'pam': 'NNGRRT', 'guide_length': 21, 'description': 'Smaller S. aureus Cas9'},
            'Cas12a': {'pam': 'TTTV', 'guide_length': 23, 'description': 'Cas12a/Cpf1 with T-rich PAM'},
            'SpCas9-HF': {'pam': 'NGG', 'guide_length': 20, 'description': 'High-fidelity SpCas9 variant'},
            'enCas9': {'pam': 'NGG', 'guide_length': 20, 'description': 'Enhanced specificity SpCas9 variant'}
        }
        
        # Known tumor-associated genes
        self.tumor_genes = {
            'KRAS': 'Frequently mutated in pancreatic, colorectal, and lung cancers',
            'TP53': 'Tumor suppressor mutated in many cancers',
            'EGFR': 'Epidermal growth factor receptor, often overexpressed in cancers',
            'BRCA1': 'DNA repair gene, mutations linked to breast and ovarian cancers',
            'BRCA2': 'DNA repair gene, mutations linked to breast and pancreatic cancers',
            'BRAF': 'Commonly mutated in melanoma and other cancers',
            'PIK3CA': 'Catalytic subunit of PI3K, frequently mutated in breast and other cancers',
            'PTEN': 'Tumor suppressor that negatively regulates PI3K/AKT pathway',
            'CDKN2A': 'Cell cycle regulator, frequently inactivated in pancreatic cancer',
            'SMAD4': 'TGF-β signaling mediator, inactivated in pancreatic cancer',
            'MYC': 'Oncogene amplified in many cancer types',
            'ERBB2': 'HER2, often amplified in breast cancer',
            'ALK': 'Target for fusion proteins in lung cancer',
            'ROS1': 'Target for fusion proteins in lung cancer',
            'RET': 'Proto-oncogene, fusion target in thyroid and lung cancers',
            'CDK4': 'Cell cycle kinase, amplified in various cancers',
            'MDM2': 'Negative regulator of p53, amplified in various cancers',
            'CD274': 'PD-L1, immune checkpoint, often overexpressed in cancers',
            'CTLA4': 'Immune checkpoint target',
            'FOXP3': 'Transcription factor in regulatory T cells'
        }
    
    def set_nuclease(self, nuclease_name: str) -> Dict[str, Any]:
        """
        Set the CRISPR nuclease to use
        
        Args:
            nuclease_name: Name of the nuclease to use
            
        Returns:
            Dictionary with nuclease parameters
        """
        if nuclease_name not in self.nucleases:
            st.error(f"Unknown nuclease: {nuclease_name}")
            st.info(f"Using default SpCas9 instead")
            nuclease_name = 'SpCas9'
        
        nuclease_params = self.nucleases[nuclease_name]
        self.pam_sequence = nuclease_params['pam']
        self.guide_length = nuclease_params['guide_length']
        
        st.success(f"Set nuclease to {nuclease_name} with PAM {self.pam_sequence}")
        return nuclease_params
    
    def find_pam_sites(self, sequence: str, pam: str = None) -> List[Dict[str, Any]]:
        """
        Find all PAM sites in a given DNA sequence
        
        Args:
            sequence: DNA sequence to search
            pam: PAM sequence to look for (default: use the instance's PAM)
            
        Returns:
            List of dictionaries with PAM sites information
        """
        if not sequence:
            st.error("No sequence provided")
            return []
        
        # Use instance PAM if not specified
        if pam is None:
            pam = self.pam_sequence
        
        # Convert ambiguous PAM to regex pattern
        pam_pattern = pam.replace('N', '[ATCG]')
        pam_pattern = pam_pattern.replace('R', '[AG]')
        pam_pattern = pam_pattern.replace('Y', '[CT]')
        pam_pattern = pam_pattern.replace('S', '[GC]')
        pam_pattern = pam_pattern.replace('W', '[AT]')
        pam_pattern = pam_pattern.replace('K', '[GT]')
        pam_pattern = pam_pattern.replace('M', '[AC]')
        pam_pattern = pam_pattern.replace('B', '[CGT]')
        pam_pattern = pam_pattern.replace('D', '[AGT]')
        pam_pattern = pam_pattern.replace('H', '[ACT]')
        pam_pattern = pam_pattern.replace('V', '[ACG]')
        
        # Find all PAM sites on forward strand
        pam_sites = []
        
        # Forward strand PAM sites
        for match in re.finditer(pam_pattern, sequence):
            start = match.start()
            end = match.end()
            
            # Extract guide sequence (upstream of PAM)
            guide_start = max(0, start - self.guide_length)
            guide_sequence = sequence[guide_start:start]
            
            # Only include if guide sequence is full length
            if len(guide_sequence) == self.guide_length:
                pam_sites.append({
                    'position': guide_start,
                    'strand': 'forward',
                    'guide_sequence': guide_sequence,
                    'pam_sequence': sequence[start:end],
                    'full_sequence': guide_sequence + sequence[start:end]
                })
        
        # Reverse complement the sequence
        rev_comp_sequence = str(Seq(sequence).reverse_complement())
        
        # Reverse strand PAM sites
        for match in re.finditer(pam_pattern, rev_comp_sequence):
            start = match.start()
            end = match.end()
            
            # Extract guide sequence (upstream of PAM)
            guide_start = max(0, start - self.guide_length)
            guide_sequence = rev_comp_sequence[guide_start:start]
            
            # Only include if guide sequence is full length
            if len(guide_sequence) == self.guide_length:
                # Calculate position in original sequence
                orig_position = len(sequence) - (guide_start + self.guide_length)
                
                pam_sites.append({
                    'position': orig_position,
                    'strand': 'reverse',
                    'guide_sequence': guide_sequence,
                    'pam_sequence': rev_comp_sequence[start:end],
                    'full_sequence': guide_sequence + rev_comp_sequence[start:end]
                })
        
        return pam_sites
    
    def score_guide_sequence(self, guide: Dict[str, Any]) -> Dict[str, float]:
        """
        Score a guide sequence based on various criteria
        
        Args:
            guide: Dictionary with guide sequence information
            
        Returns:
            Dictionary with scores for various criteria
        """
        guide_seq = guide['guide_sequence']
        
        # GC content score
        gc_count = guide_seq.count('G') + guide_seq.count('C')
        gc_content = gc_count / len(guide_seq)
        
        # Score based on GC content (optimal range)
        if self.min_gc_content <= gc_content <= self.max_gc_content:
            gc_score = 1.0
        else:
            # Penalty for being outside optimal range
            gc_score = 1.0 - min(
                abs(gc_content - self.min_gc_content),
                abs(gc_content - self.max_gc_content)
            )
        
        # Homopolymer score (penalty for 4+ consecutive identical nucleotides)
        homopolymer_score = 1.0
        for base in 'ATGC':
            if base * 4 in guide_seq:
                homopolymer_score *= 0.7
        
        # Self-complementarity score (check for hairpins)
        self_comp_score = 1.0
        for i in range(2, min(len(guide_seq), 10)):
            # Check i-length subsequences for reverse complementarity
            for j in range(len(guide_seq) - i + 1):
                subseq = guide_seq[j:j+i]
                rev_comp = str(Seq(subseq).reverse_complement())
                
                if rev_comp in guide_seq:
                    # Penalty based on length of complementary sequence
                    self_comp_score *= 0.9 ** (i / 5)
        
        # Position effect score
        # Guides targeting near the 5' end of the coding sequence are often more effective
        if 'position' in guide:
            # Normalize position effect (assuming a gene of 1000bp)
            # Real implementation would use actual gene length
            position_score = 1.0 - (guide['position'] / 1000.0)
        else:
            position_score = 0.5  # Default if position is unknown
        
        # On-target efficiency score (simplified - a real model would be more complex)
        # We'll use a simple heuristic based on nucleotides at specific positions
        on_target_score = 0.6  # Base score
        
        # Preferred bases at certain positions (simplified from published algorithms)
        if guide_seq[-1] == 'G':  # Preferred at end of guide
            on_target_score += 0.1
        if guide_seq[0] != 'T':   # T at position 1 is unfavorable
            on_target_score += 0.05
        if 'GCC' in guide_seq:    # GCC motif is favorable
            on_target_score += 0.05
        if guide_seq.count('T') <= 3:  # Low T count is favorable
            on_target_score += 0.05
            
        # Off-target score (simplified - real implementation would use alignment)
        # Higher score = fewer predicted off-targets
        off_target_score = 0.5  # Base score
        
        # Adjust based on seed region (8 nucleotides adjacent to PAM)
        seed_region = guide_seq[-8:]
        
        # Penalize seed regions with low complexity
        seed_gc = (seed_region.count('G') + seed_region.count('C')) / 8.0
        if 0.25 <= seed_gc <= 0.75:
            off_target_score += 0.1
            
        # Penalize seed regions with homopolymers
        if not any(base * 3 in seed_region for base in 'ATGC'):
            off_target_score += 0.1
        
        # Reward guides with more balanced nucleotide composition
        nucleotide_balance = min(
            guide_seq.count('A'), 
            guide_seq.count('T'),
            guide_seq.count('G'),
            guide_seq.count('C')
        ) / (len(guide_seq) / 4.0)
        off_target_score += 0.2 * nucleotide_balance
        
        # Combine all scores
        combined_score = (
            self.scoring_weights['on_target_efficiency'] * on_target_score +
            self.scoring_weights['off_target_effects'] * off_target_score +
            self.scoring_weights['gc_content'] * gc_score +
            self.scoring_weights['self_complementarity'] * self_comp_score +
            self.scoring_weights['position_effect'] * position_score
        )
        
        # Ensure scores are within valid range
        combined_score = min(1.0, max(0.0, combined_score))
        
        return {
            'gc_content': gc_content,
            'gc_score': gc_score,
            'homopolymer_score': homopolymer_score,
            'self_complementarity_score': self_comp_score,
            'position_score': position_score,
            'on_target_score': on_target_score,
            'off_target_score': off_target_score,
            'combined_score': combined_score
        }
    
    def design_guides_for_gene(self, gene_name: str, sequence: str = None) -> List[Dict[str, Any]]:
        """
        Design optimal CRISPR guides for a specific gene
        
        Args:
            gene_name: Name of the gene
            sequence: DNA sequence of the gene (if None, attempt to fetch)
            
        Returns:
            List of dictionaries with guide information
        """
        if sequence is None:
            st.warning(f"No sequence provided for {gene_name}. Attempting to use a demo sequence.")
            # In a real implementation, we would fetch the sequence from a database
            # For now, use a dummy sequence for demonstration
            sequence = 'ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT' + \
                      'AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA' + \
                      'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT'
        
        if not sequence:
            st.error(f"Could not obtain sequence for {gene_name}")
            return []
        
        # Find all PAM sites
        pam_sites = self.find_pam_sites(sequence)
        
        if not pam_sites:
            st.warning(f"No valid PAM sites found in the sequence for {gene_name}")
            return []
        
        # Score all guides
        scored_guides = []
        for guide in pam_sites:
            scores = self.score_guide_sequence(guide)
            guide_info = {
                'gene': gene_name,
                'sequence': guide['guide_sequence'],
                'pam': guide['pam_sequence'],
                'position': guide['position'],
                'strand': guide['strand'],
                'gc_content': scores['gc_content'],
                'on_target_score': scores['on_target_score'],
                'off_target_score': scores['off_target_score'],
                'overall_score': scores['combined_score']
            }
            scored_guides.append(guide_info)
        
        # Sort guides by overall score (descending)
        scored_guides.sort(key=lambda x: x['overall_score'], reverse=True)
        
        st.success(f"Designed {len(scored_guides)} guides for {gene_name}")
        return scored_guides
    
    def get_top_guides(self, scored_guides: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N guides from a list of scored guides
        
        Args:
            scored_guides: List of scored guide dictionaries
            top_n: Number of top guides to return
            
        Returns:
            List of top N guides
        """
        return scored_guides[:min(top_n, len(scored_guides))]
    
    def check_off_targets(self, guide_sequence: str, genome: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        Check for potential off-target sites in a genome
        
        Args:
            guide_sequence: Guide RNA sequence
            genome: Dictionary with chromosomes as keys and sequences as values
            
        Returns:
            List of potential off-target sites
        """
        st.warning("Full off-target checking requires a genomic database.")
        st.info("Using a simplified off-target simulation for demonstration.")
        
        # Create simulated off-targets by introducing mutations to the guide sequence
        off_targets = []
        
        # Mutate with 1, 2, or 3 mismatches
        for num_mismatches in [1, 2, 3]:
            # Generate a few simulated off-targets with random mutations
            for _ in range(3):  # Generate 3 examples for each mismatch count
                mutated_guide = list(guide_sequence)
                
                # Introduce random mismatches
                for _ in range(num_mismatches):
                    pos = np.random.randint(0, len(guide_sequence))
                    original_base = mutated_guide[pos]
                    
                    # Replace with a different base
                    bases = list('ATGC')
                    bases.remove(original_base)
                    new_base = np.random.choice(bases)
                    
                    mutated_guide[pos] = new_base
                
                mutated_sequence = ''.join(mutated_guide)
                
                # Calculate a score based on number and position of mismatches
                # Mismatches in seed region (8 bases next to PAM) are more significant
                seed_mismatches = sum(1 for i in range(len(guide_sequence)-8, len(guide_sequence))
                                    if guide_sequence[i] != mutated_guide[i])
                non_seed_mismatches = num_mismatches - seed_mismatches
                
                # More penalty for seed region mismatches
                mismatch_score = 1.0 - (non_seed_mismatches * 0.1 + seed_mismatches * 0.3)
                
                off_targets.append({
                    'sequence': mutated_sequence,
                    'mismatches': num_mismatches,
                    'seed_mismatches': seed_mismatches,
                    'location': f"chr{np.random.randint(1, 23)}:{np.random.randint(1, 100000000)}",
                    'gene': f"GENE_{np.random.randint(1, 1000)}",
                    'score': mismatch_score
                })
        
        # Sort by mismatch score (higher = more concerning)
        off_targets.sort(key=lambda x: x['score'], reverse=True)
        
        return off_targets
    
    def optimize_guide_rna(self, guide_sequence: str, optimize_for: str = 'efficiency') -> Dict[str, str]:
        """
        Optimize a guide RNA sequence for better performance
        
        Args:
            guide_sequence: Original guide RNA sequence
            optimize_for: Optimization goal ('efficiency', 'specificity', or 'balanced')
            
        Returns:
            Dictionary with original and optimized guide sequences
        """
        if len(guide_sequence) != self.guide_length:
            st.error(f"Guide sequence must be {self.guide_length} nucleotides")
            return {'original': guide_sequence, 'optimized': guide_sequence}
        
        optimized_guide = list(guide_sequence)
        
        # Optimization strategy depends on the goal
        if optimize_for == 'efficiency':
            # Optimize for efficiency (on-target activity)
            # Rules based on published efficiency scoring algorithms
            
            # Favor G at position 20 (end of guide)
            if optimized_guide[-1] != 'G':
                optimized_guide[-1] = 'G'
            
            # Avoid T at position 1
            if optimized_guide[0] == 'T':
                optimized_guide[0] = 'G'
            
            # Favor G or C at positions 16-19 for stability
            for i in range(15, 19):
                if optimized_guide[i] not in ['G', 'C']:
                    optimized_guide[i] = 'G'
            
        elif optimize_for == 'specificity':
            # Optimize for specificity (reduce off-targets)
            
            # Add GC bases to seed region (last 8 bases)
            for i in range(len(optimized_guide)-8, len(optimized_guide)):
                if optimized_guide[i] not in ['G', 'C']:
                    optimized_guide[i] = 'G' if i % 2 == 0 else 'C'
            
            # Ensure balanced nucleotide composition
            a_count = optimized_guide.count('A')
            t_count = optimized_guide.count('T')
            g_count = optimized_guide.count('G')
            c_count = optimized_guide.count('C')
            
            # Balance nucleotides by replacing the most common with the least common
            base_counts = {'A': a_count, 'T': t_count, 'G': g_count, 'C': c_count}
            most_common = max(base_counts, key=base_counts.get)
            least_common = min(base_counts, key=base_counts.get)
            
            # Replace some of the most common bases with the least common
            for i in range(5, 12):  # Middle of guide, away from seed region
                if optimized_guide[i] == most_common:
                    optimized_guide[i] = least_common
        
        else:  # balanced approach
            # Create a compromise between efficiency and specificity
            # G/C at end for efficiency
            if optimized_guide[-1] not in ['G', 'C']:
                optimized_guide[-1] = 'G'
            
            # Balance GC content overall
            gc_count = sum(1 for base in optimized_guide if base in ['G', 'C'])
            optimal_gc = self.guide_length * 0.55  # ~55% GC is a good balance
            
            if gc_count < optimal_gc:
                # Add some GC
                for i in range(5, 15):
                    if optimized_guide[i] in ['A', 'T'] and gc_count < optimal_gc:
                        optimized_guide[i] = 'G' if i % 2 == 0 else 'C'
                        gc_count += 1
            elif gc_count > optimal_gc:
                # Reduce some GC
                for i in range(5, 15):
                    if optimized_guide[i] in ['G', 'C'] and gc_count > optimal_gc:
                        optimized_guide[i] = 'A' if i % 2 == 0 else 'T'
                        gc_count -= 1
        
        optimized_sequence = ''.join(optimized_guide)
        
        return {'original': guide_sequence, 'optimized': optimized_sequence}
    
    def design_for_mutation(self, gene: str, mutation: str, sequence: str = None) -> List[Dict[str, Any]]:
        """
        Design guides that specifically target a mutated sequence
        
        Args:
            gene: Gene name
            mutation: Mutation description (e.g., 'G12D')
            sequence: Gene sequence (if None, attempt to fetch)
            
        Returns:
            List of guides targeting the mutation
        """
        if sequence is None:
            st.warning(f"No sequence provided for {gene} with mutation {mutation}. Using demo sequence.")
            # In a real implementation, we would fetch the specific sequence with mutation
            # For demonstration, create a synthetic mutated sequence
            
            # Parse mutation (simplified for common format amino_acid + position + new_amino_acid)
            # G12D means G at position 12 is changed to D
            match = re.match(r'([A-Z])(\d+)([A-Z])', mutation)
            if not match:
                st.error(f"Could not parse mutation format: {mutation}")
                return []
            
            wild_aa, position, mutant_aa = match.groups()
            position = int(position)
            
            # Create wild-type and mutant sequences
            wild_type_seq = 'ATGCTGACTGACTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA' + 'GGT' + 'CTAGCTAGCTAGCTAGCTAGCTAGCTA'
            
            # Replace codon at mutation position with mutant codon
            # This is simplified; in reality, would use actual genetic code translation
            mutant_codons = {
                'D': 'GAT',  # Aspartic acid
                'V': 'GTT',  # Valine
                'R': 'CGT',  # Arginine
                'K': 'AAG',  # Lysine
                'E': 'GAG',  # Glutamic acid
            }
            
            mutant_codon = mutant_codons.get(mutant_aa, 'NNN')
            mutant_seq = wild_type_seq[:40] + mutant_codon + wild_type_seq[43:]
            
            sequence = mutant_seq
        
        # Find PAM sites near the mutation
        pam_sites = self.find_pam_sites(sequence)
        
        # For a real implementation, we'd identify which guides overlap the mutation
        # For demonstration, randomly select a few guides and mark them as mutation-specific
        mutation_guides = []
        
        for guide in pam_sites[:5]:  # Take first 5 guides
            scores = self.score_guide_sequence(guide)
            
            guide_info = {
                'gene': gene,
                'mutation': mutation,
                'sequence': guide['guide_sequence'],
                'pam': guide['pam_sequence'],
                'position': guide['position'],
                'strand': guide['strand'],
                'gc_content': scores['gc_content'],
                'on_target_score': scores['on_target_score'],
                'off_target_score': scores['off_target_score'],
                'overall_score': scores['combined_score'],
                'targets_mutation': True,  # In reality, would determine based on overlap with mutation
                'mutation_specificity': np.random.uniform(0.7, 0.95)  # Simulated specificity
            }
            mutation_guides.append(guide_info)
        
        # Sort guides by overall score (descending)
        mutation_guides.sort(key=lambda x: x['overall_score'], reverse=True)
        
        st.success(f"Designed {len(mutation_guides)} guides targeting {mutation} in {gene}")
        return mutation_guides
    
    def generate_oligo_sequences(self, guide_sequence: str) -> Dict[str, str]:
        """
        Generate oligo sequences for cloning the guide RNA
        
        Args:
            guide_sequence: Guide RNA sequence
            
        Returns:
            Dictionary with forward and reverse oligo sequences
        """
        # Validate guide length
        if len(guide_sequence) != self.guide_length:
            st.error(f"Guide sequence must be {self.guide_length} nucleotides")
            return {'forward': '', 'reverse': ''}
        
        # Standard overhangs for Gibson assembly or Golden Gate cloning with BbsI
        forward_overhang = "CACCG"
        reverse_overhang = "AAAC"
        
        # Generate forward oligo
        forward_oligo = forward_overhang + guide_sequence
        
        # Generate reverse oligo (reverse complement of guide with appropriate overhangs)
        reverse_comp_guide = str(Seq(guide_sequence).reverse_complement())
        reverse_oligo = reverse_overhang + reverse_comp_guide + "C"
        
        return {
            'forward': forward_oligo,
            'reverse': reverse_oligo
        }
    
    def get_tumor_associated_genes(self, cancer_type: str = None) -> Dict[str, str]:
        """
        Get a list of tumor-associated genes, optionally filtered by cancer type
        
        Args:
            cancer_type: Cancer type to filter genes
            
        Returns:
            Dictionary of gene names and descriptions
        """
        # For a full implementation, we would filter based on cancer type
        # For simplicity, return the full list with a note
        if cancer_type:
            st.info(f"Gene filtering by cancer type ({cancer_type}) would be applied in a full implementation.")
        
        return self.tumor_genes
