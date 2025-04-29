import numpy as np
# Force fallback mode
TENSORFLOW_AVAILABLE = False
print("Using synthetic CRISPR prediction mode")
import os
import pandas as pd
import random
import re

class CRISPRTargetModel:
    """
    Model for predicting CRISPR-Cas9 target efficiency and off-target effects.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the CRISPR target prediction model.
        
        Args:
            model_path (str): Path to a pre-trained model (optional)
        """
        self.model = None
        self.feature_encoder = None
        self.sequence_encoder = None
        
        # In this version, we're always using synthetic predictions
        # regardless of model path
        print("Using synthetic prediction mode for CRISPR targeting")
    
    def _build_default_model(self):
        """
        This is a stub method since we're using synthetic predictions
        instead of a real model. This avoids TensorFlow dependencies.
        """
        # Intentionally left empty, synthetic predictions are used instead
        pass
    
    def predict_efficiency(self, sequences, features=None):
        """
        Predict on-target efficiency for a list of sgRNA sequences.
        
        Args:
            sequences (list): List of sgRNA sequences
            features (numpy.ndarray): Additional features for each sequence
            
        Returns:
            numpy.ndarray: Predicted efficiency scores
        """
        # Always use synthetic predictions in this version
        efficiency_scores = self._synthetic_efficiency_prediction(sequences)
        return efficiency_scores
    
    def predict_offtargets(self, sequences, features=None):
        """
        Predict off-target effects for a list of sgRNA sequences.
        
        Args:
            sequences (list): List of sgRNA sequences
            features (numpy.ndarray): Additional features for each sequence
            
        Returns:
            numpy.ndarray: Predicted off-target scores
        """
        # Always use synthetic predictions in this version
        offtarget_scores = self._synthetic_offtarget_prediction(sequences)
        return offtarget_scores
    
    def _encode_sequences(self, sequences):
        """
        One-hot encode sgRNA sequences.
        
        Args:
            sequences (list): List of sgRNA sequences
            
        Returns:
            numpy.ndarray: One-hot encoded sequences
        """
        # Define nucleotide mapping
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'N': 4}
        
        # Initialize encoded array
        max_length = max(len(seq) for seq in sequences)
        encoded = np.zeros((len(sequences), max_length, 4))
        
        # Encode each sequence
        for i, seq in enumerate(sequences):
            for j, nucleotide in enumerate(seq[:max_length]):
                if nucleotide in nucleotide_map and nucleotide_map[nucleotide] < 4:
                    encoded[i, j, nucleotide_map[nucleotide]] = 1
        
        return encoded
    
    def _generate_features(self, sequences):
        """
        Generate additional features for sgRNA sequences.
        
        Args:
            sequences (list): List of sgRNA sequences
            
        Returns:
            numpy.ndarray: Feature array
        """
        features = np.zeros((len(sequences), 5))
        
        for i, seq in enumerate(sequences):
            # GC content
            gc_count = seq.count('G') + seq.count('C')
            features[i, 0] = gc_count / len(seq)
            
            # Self-complementarity score (simplified)
            self_comp = 0
            for j in range(len(seq) - 1):
                if seq[j:j+2] in ['GC', 'CG', 'AT', 'TA']:
                    self_comp += 1
            features[i, 1] = self_comp / (len(seq) - 1)
            
            # Homopolymer count
            features[i, 2] = len(re.findall(r'([ACGT])\1{3,}', seq)) / len(seq)
            
            # Position bias (assuming PAM at 3' end)
            features[i, 3] = sum(i * (n in 'GC') for i, n in enumerate(seq)) / (len(seq) * (len(seq) - 1) / 2)
            
            # Secondary structure potential (very simplified)
            features[i, 4] = seq.count('G') * seq.count('C') / (len(seq) ** 2)
        
        return features
    
    def _synthetic_efficiency_prediction(self, sequences):
        """
        Generate synthetic efficiency predictions for demonstration.
        
        Args:
            sequences (list): List of sgRNA sequences
            
        Returns:
            numpy.ndarray: Synthetic efficiency scores
        """
        scores = np.zeros(len(sequences))
        
        for i, seq in enumerate(sequences):
            # GC content factor (optimal around 40-60%)
            gc_content = (seq.count('G') + seq.count('C')) / len(seq)
            gc_factor = 1.0 - 2.0 * abs(gc_content - 0.5)
            
            # Self-complementarity penalty
            self_comp = 0
            for j in range(len(seq) - 1):
                if seq[j:j+2] in ['GC', 'CG', 'AT', 'TA']:
                    self_comp += 1
            self_comp_factor = 1.0 - (self_comp / (len(seq) - 1)) * 0.5
            
            # Position-specific bias (based on literature)
            pos_factors = []
            for j, nt in enumerate(seq):
                if j < len(seq) // 3:
                    # Prefer G at 5' end
                    pos_factors.append(1.2 if nt == 'G' else 1.0)
                elif j > 2 * len(seq) // 3:
                    # Prefer C at 3' end
                    pos_factors.append(1.1 if nt == 'C' else 1.0)
                else:
                    # Balanced middle
                    pos_factors.append(1.0)
            pos_factor = sum(pos_factors) / len(pos_factors)
            
            # Random factor for realism
            random_factor = 0.7 + 0.3 * np.random.random()
            
            # Combine factors
            score = gc_factor * self_comp_factor * pos_factor * random_factor
            
            # Ensure score is in [0, 1] range
            scores[i] = max(0, min(1, score))
        
        return scores
    
    def _synthetic_offtarget_prediction(self, sequences):
        """
        Generate synthetic off-target predictions for demonstration.
        
        Args:
            sequences (list): List of sgRNA sequences
            
        Returns:
            numpy.ndarray: Synthetic off-target scores
        """
        scores = np.zeros(len(sequences))
        
        for i, seq in enumerate(sequences):
            # GC content (higher GC = more off-targets)
            gc_content = (seq.count('G') + seq.count('C')) / len(seq)
            gc_factor = gc_content
            
            # Seed region (last 8bp before PAM) - higher complexity = fewer off-targets
            if len(seq) >= 8:
                seed = seq[-8:]
                seed_complexity = len(set(seed)) / 4.0  # 4 nucleotides max
                seed_factor = 1.0 - seed_complexity
            else:
                seed_factor = 0.5
            
            # Presence of homopolymers (increases off-targets)
            homopolymer_factor = 0
            for nt in 'ACGT':
                homopolymer_factor += 0.2 * len(re.findall(nt + '{3,}', seq))
            homopolymer_factor = min(1, homopolymer_factor)
            
            # Random factor for realism
            random_factor = 0.3 + 0.7 * np.random.beta(2, 5)  # Beta distribution biased toward lower values
            
            # Combine factors
            score = 0.4 * gc_factor + 0.3 * seed_factor + 0.2 * homopolymer_factor + 0.1 * random_factor
            
            # Ensure score is in [0, 1] range
            scores[i] = max(0, min(1, score))
        
        return scores
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        # Stub method since we're not using a real model
        print(f"Model would be saved to {path} (synthetic model only, no actual save performed)")
