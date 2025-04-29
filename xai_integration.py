import os
import json
import logging
import base64
from typing import Optional, List, Dict, Any, Union
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAIIntegration:
    """Integration with Google's Gemini API for generative AI capabilities"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini integration"""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.vision_model = genai.GenerativeModel('gemini-pro-vision')
                logger.info("Successfully initialized Gemini client")
            except Exception as e:
                logger.error(f"Error initializing Gemini client: {str(e)}")
                self.model = None
                self.vision_model = None

    def is_available(self) -> bool:
        """Check if Gemini integration is available"""
        return self.model is not None and self.api_key is not None

    def analyze_crispr_target(self, sequence: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a CRISPR target sequence using Gemini"""
        if not self.is_available():
            return self._get_fallback_response("crispr_analysis")

        try:
            prompt = f"""
            As a computational biologist, analyze this CRISPR target sequence:

            Sequence: {sequence}

            Provide a detailed analysis including:
            1. Potential on-target efficiency factors
            2. Possible off-target concerns
            3. Sequence composition analysis
            4. Recommendations for optimization

            Format your response as a structured JSON with these keys:
            - efficiency_score: (float between 0-1)
            - off_target_risk: (string, one of "low", "medium", "high")
            - gc_content: (float as percentage)
            - sequence_issues: (array of strings)
            - recommendations: (array of strings)
            """

            if context:
                prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

            response = self.model.generate_content(prompt)
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Error analyzing CRISPR target: {str(e)}")
            return self._get_fallback_response("crispr_analysis")

    def optimize_neoantigen(self, mutation: Dict[str, Any], clinical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize a neoantigen using Gemini"""
        if not self.is_available():
            return self._get_fallback_response("neoantigen_optimization")

        try:
            prompt = f"""
            As an immunology AI specialist, analyze and optimize this candidate neoantigen:

            Mutation: {json.dumps(mutation, indent=2)}

            Provide optimization suggestions including:
            1. MHC binding improvement strategies
            2. Immunogenicity enhancement approaches
            3. Peptide modifications for stability
            4. Specific amino acid substitutions to consider

            Format your response as a structured JSON with these keys:
            - optimized_mutation: (object with modified mutation)
            - mhc_binding_prediction: (float between 0-1)
            - immunogenicity_score: (float between 0-1) 
            - stability_score: (float between 0-1)
            - modifications: (array of objects with type, position, original, substitution, rationale)
            - clinical_recommendations: (array of strings)
            """

            if clinical_context:
                prompt += f"\n\nClinical context:\n{json.dumps(clinical_context, indent=2)}"

            response = self.model.generate_content(prompt)
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Error optimizing neoantigen: {str(e)}")
            return self._get_fallback_response("neoantigen_optimization")

    def analyze_gene_editing_strategy(self, target_genes: List[str], disease_context: str, edit_approach: str) -> Dict[str, Any]:
        """Analyze a gene editing strategy using Gemini"""
        if not self.is_available():
            return self._get_fallback_response("gene_editing_analysis")

        try:
            prompt = f"""
            As a gene therapy expert, analyze this gene editing strategy:

            Target Genes: {', '.join(target_genes)}
            Disease Context: {disease_context}
            Editing Approach: {edit_approach}

            Provide a comprehensive analysis including:
            1. Potential efficacy in the disease context
            2. Safety considerations specific to these genes
            3. Delivery challenges and solutions
            4. Alternative targeting strategies to consider
            5. Potential impacts on related biological pathways

            Format your response as a structured JSON with these keys:
            - efficacy_rating: (float between 0-1)
            - safety_concerns: (array of objects with type, severity, mitigation)
            - delivery_recommendations: (array of strings)
            - alternative_targets: (array of objects with gene, rationale)
            - pathway_impacts: (array of objects with pathway, impact_type, severity)
            - overall_assessment: (string)
            """

            response = self.model.generate_content(prompt)
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Error analyzing gene editing strategy: {str(e)}")
            return self._get_fallback_response("gene_editing_analysis")

    def analyze_image(self, image_path: str, query: str) -> str:
        """Analyze an image using Gemini Vision"""
        if not self.is_available():
            return "Gemini integration is not available. Please check your API key."

        try:
            image_data = None
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            response = self.vision_model.generate_content([query, image_data])
            return response.text

        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Failed to analyze image: {str(e)}"

    def _get_fallback_response(self, request_type: str) -> Dict[str, Any]:
        """Generate a fallback response when the API is not available"""
        return {
            "error": "Gemini API not available",
            "message": f"Please provide a Gemini API key to enable {request_type}.",
            "status": "api_unavailable"
        }

# Example usage
def demonstrate_xai_integration():
    """
    Demonstrate the XAI integration functionality

    This function shows how to use the XAI integration class
    with examples for CRISPR target optimization and neoantigen analysis.
    """
    # Initialize with API key from environment or specify directly
    xai = XAIIntegration()

    if not xai.is_available():
        print("xAI integration not available. Please set GEMINI_API_KEY environment variable.")
        return

    # Example 1: Analyze a CRISPR target
    crispr_sequence = "GTCCCCTCCACCCCACAGTGGGGCCACTAGGGACAGGATTGGTGACAGAAAAGCCCCATCCTTAGGCCTCCCAAGTGCTGGGATTACAGG"
    crispr_analysis = xai.analyze_crispr_target(crispr_sequence)
    print("\nCRISPR Target Analysis:")
    print(json.dumps(crispr_analysis, indent=2))

    # Example 2: Optimize a neoantigen
    mutation = {
        "gene": "KRAS",
        "protein_change": "G12D",
        "mutation_type": "missense"
    }

    clinical_context = {
        "cancer_type": "pancreatic ductal adenocarcinoma",
        "patient_hla": ["HLA-A*02:01", "HLA-B*07:02"],
        "prior_treatments": ["gemcitabine", "FOLFIRINOX"]
    }

    neoantigen_result = xai.optimize_neoantigen(mutation, clinical_context)
    print("\nNeoantigen Optimization:")
    print(json.dumps(neoantigen_result, indent=2))

    # Example 3: Analyze gene editing strategy
    genes = ["TP53", "CDKN2A", "SMAD4"]
    disease = "Pancreatic cancer with metastasis to liver"
    approach = "CRISPR-Cas9 base editing to restore wild-type function"

    editing_analysis = xai.analyze_gene_editing_strategy(genes, disease, approach)
    print("\nGene Editing Strategy Analysis:")
    print(json.dumps(editing_analysis, indent=2))

if __name__ == "__main__":
    demonstrate_xai_integration()