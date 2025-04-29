import numpy as np
from typing import Dict, Tuple

class DeepSeekReasoning:
    """
    Advanced reasoning module for CRISPR target analysis using simplified scoring
    """
    def __init__(self):
        self.sequence_length = 23  # Base length for CRISPR sequence

    def analyze_target_compatibility(self, sequence: str, context: Dict) -> Tuple[float, Dict]:
        """
        Analyze target sequence compatibility using advanced reasoning
        """
        # Structural analysis
        structure_score = self._analyze_structural_features(sequence)

        # Target accessibility 
        accessibility = self._evaluate_accessibility(sequence, context)

        # Chromatin state impact
        chromatin_impact = context.get('chromatin_accessibility', 0.5)

        # Combine scores with reasoning
        total_score = 0.4 * structure_score + 0.3 * accessibility + 0.3 * chromatin_impact

        reasoning = {
            'structural_analysis': structure_score,
            'accessibility': accessibility,
            'chromatin_impact': chromatin_impact,
            'explanation': self._generate_reasoning_explanation(
                structure_score, accessibility, chromatin_impact
            )
        }

        return total_score, reasoning

    def _analyze_structural_features(self, sequence: str) -> float:
        """Analyze DNA structural features"""
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        repeats = self._check_sequence_repeats(sequence)
        return 0.7 * gc_content + 0.3 * (1 - repeats)

    def _evaluate_accessibility(self, sequence: str, context: Dict) -> float:
        """Evaluate target site accessibility"""
        local_structure = context.get('local_structure', 0.5)
        melting_temp = self._calculate_melting_temperature(sequence)
        return 0.6 * local_structure + 0.4 * (1 - (melting_temp - 37) / 50)

    def _check_sequence_repeats(self, sequence: str) -> float:
        """Check for sequence repeats that might affect stability"""
        repeat_score = 0
        for i in range(2, 5):
            for j in range(len(sequence) - i):
                if sequence[j:j+i] in sequence[j+i:]:
                    repeat_score += i / len(sequence)
        return min(1.0, repeat_score)

    def _calculate_melting_temperature(self, sequence: str) -> float:
        """Calculate approximate melting temperature"""
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        return 64.9 + 41 * (gc_content - 0.41)

    def _generate_reasoning_explanation(self, 
                                     structure_score: float,
                                     accessibility: float,
                                     chromatin_impact: float) -> str:
        """Generate human-readable explanation of the reasoning"""
        explanations = []

        if structure_score > 0.7:
            explanations.append("Strong structural compatibility")
        elif structure_score > 0.4:
            explanations.append("Moderate structural features")
        else:
            explanations.append("Potential structural concerns")

        if accessibility > 0.7:
            explanations.append("Highly accessible target site")
        elif accessibility > 0.4:
            explanations.append("Moderately accessible region")
        else:
            explanations.append("Limited target accessibility")

        if chromatin_impact > 0.7:
            explanations.append("Favorable chromatin state")
        elif chromatin_impact > 0.4:
            explanations.append("Acceptable chromatin environment")
        else:
            explanations.append("Challenging chromatin context")

        return ". ".join(explanations) + "."