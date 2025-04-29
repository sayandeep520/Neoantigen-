import pandas as pd
import numpy as np
import requests
import os
import time
import json
import streamlit as st
from io import StringIO
from typing import Dict, List, Union, Tuple, Optional


class DataCollector:
    """
    Data collector for multi-omics data from various sources
    including TCGA, ICGC, GTEx, DepMap, etc.
    """
    
    def __init__(self):
        """Initialize the data collector with API endpoints and credentials"""
        self.api_endpoints = {
            "tcga": "https://api.gdc.cancer.gov/",
            "depmap": "https://depmap.org/portal/api/",
            "gtex": "https://gtexportal.org/rest/v1/",
            "cbioportal": "https://www.cbioportal.org/api/",
            "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi",
            "uniprot": "https://www.uniprot.org/uniprot/",
        }
        self.cancer_types = [
            "PAAD",  # Pancreatic cancer
            "LUAD",  # Lung adenocarcinoma
            "BRCA",  # Breast cancer
            "COAD",  # Colorectal cancer
            "PRAD",  # Prostate cancer
        ]
        
    def fetch_tcga_data(self, cancer_type: str = "PAAD", data_type: str = "mutations") -> pd.DataFrame:
        """
        Fetch genomic data from TCGA
        
        Args:
            cancer_type: Cancer type code (e.g., "PAAD" for pancreatic cancer)
            data_type: Type of data to fetch (mutations, cnv, clinical, etc.)
            
        Returns:
            DataFrame with the requested data
        """
        st.info(f"Fetching {data_type} data for {cancer_type} from TCGA...")
        
        if data_type == "mutations":
            endpoint = f"{self.api_endpoints['cbioportal']}molecular-profiles/{cancer_type}_mutations/mutations"
            try:
                response = requests.get(endpoint)
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    st.success(f"Successfully fetched mutation data for {cancer_type}")
                    return df
                else:
                    st.error(f"Failed to fetch data: {response.status_code}")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Error fetching TCGA data: {str(e)}")
                return pd.DataFrame()
        
        elif data_type == "clinical":
            endpoint = f"{self.api_endpoints['cbioportal']}studies/{cancer_type}_tcga/clinical-data"
            try:
                response = requests.get(endpoint)
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    st.success(f"Successfully fetched clinical data for {cancer_type}")
                    return df
                else:
                    st.error(f"Failed to fetch data: {response.status_code}")
                    return pd.DataFrame()
            except Exception as e:
                st.error(f"Error fetching clinical data: {str(e)}")
                return pd.DataFrame()
                
        # Mock data for demonstration
        st.warning("Demo mode: Returning empty dataframe")
        return pd.DataFrame()
    
    def fetch_depmap_data(self, dataset: str = "CRISPR_gene_effect") -> pd.DataFrame:
        """
        Fetch CRISPR screening data from DepMap
        
        Args:
            dataset: Name of the dataset (CRISPR_gene_effect, CRISPR_gene_dependency, etc.)
            
        Returns:
            DataFrame with the requested data
        """
        st.info(f"Fetching {dataset} from DepMap...")
        
        endpoint = f"{self.api_endpoints['depmap']}{dataset}"
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                st.success(f"Successfully fetched {dataset} from DepMap")
                return df
            else:
                st.error(f"Failed to fetch data: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching DepMap data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_gtex_data(self, tissue: str = "pancreas") -> pd.DataFrame:
        """
        Fetch expression data from GTEx
        
        Args:
            tissue: Tissue type (e.g., "pancreas")
            
        Returns:
            DataFrame with the requested data
        """
        st.info(f"Fetching expression data for {tissue} from GTEx...")
        
        endpoint = f"{self.api_endpoints['gtex']}expression/geneExpression?tissueId={tissue}"
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                st.success(f"Successfully fetched expression data for {tissue}")
                return df
            else:
                st.error(f"Failed to fetch data: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching GTEx data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_geo_data(self, gse_id: str) -> pd.DataFrame:
        """
        Fetch gene expression data from GEO
        
        Args:
            gse_id: GEO series ID (e.g., "GSE71729")
            
        Returns:
            DataFrame with the requested data
        """
        st.info(f"Fetching data for {gse_id} from GEO...")
        
        endpoint = f"{self.api_endpoints['geo']}?acc={gse_id}&targ=self&view=brief&form=text"
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                st.success(f"Successfully fetched data for {gse_id}")
                # Parsing GEO data requires specialized libraries like GEOparse
                st.warning("GEO data parsing requires extra processing")
                return pd.DataFrame({'gse_id': [gse_id], 'data': [response.text]})
            else:
                st.error(f"Failed to fetch data: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching GEO data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_uniprot_data(self, protein_id: str) -> pd.DataFrame:
        """
        Fetch protein data from UniProt
        
        Args:
            protein_id: UniProt ID (e.g., "P53_HUMAN")
            
        Returns:
            DataFrame with the requested data
        """
        st.info(f"Fetching data for {protein_id} from UniProt...")
        
        endpoint = f"{self.api_endpoints['uniprot']}{protein_id}.json"
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame([data])
                st.success(f"Successfully fetched data for {protein_id}")
                return df
            else:
                st.error(f"Failed to fetch data: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching UniProt data: {str(e)}")
            return pd.DataFrame()
            
    def get_available_cancer_types(self) -> List[str]:
        """Return the list of available cancer types"""
        return self.cancer_types
    
    def get_sample_data(self, data_type: str) -> pd.DataFrame:
        """
        Get sample data for testing and demonstration
        
        Args:
            data_type: Type of data (mutations, expression, clinical)
            
        Returns:
            DataFrame with sample data
        """
        st.warning("Using demo data for demonstration")
        
        if data_type == "mutations":
            # Sample mutation data structure
            return pd.DataFrame({
                'gene': ['KRAS', 'TP53', 'CDKN2A', 'SMAD4'],
                'mutation': ['G12D', 'R175H', 'R80*', 'D355G'],
                'sample_id': ['TCGA-PAAD-001', 'TCGA-PAAD-002', 'TCGA-PAAD-001', 'TCGA-PAAD-003'],
                'mutation_type': ['Missense', 'Missense', 'Nonsense', 'Missense'],
                'chromosome': ['12', '17', '9', '18'],
                'start_position': [25398284, 7578406, 21968233, 48575673],
                'end_position': [25398284, 7578406, 21968233, 48575673]
            })
        
        elif data_type == "expression":
            # Sample gene expression data structure
            genes = ['KRAS', 'TP53', 'CDKN2A', 'SMAD4', 'BRCA1', 'BRCA2']
            samples = [f'TCGA-PAAD-00{i}' for i in range(1, 5)]
            
            data = {}
            for sample in samples:
                data[sample] = np.random.normal(0, 1, len(genes))
            
            df = pd.DataFrame(data, index=genes)
            return df
        
        elif data_type == "clinical":
            # Sample clinical data structure
            return pd.DataFrame({
                'patient_id': ['TCGA-PAAD-001', 'TCGA-PAAD-002', 'TCGA-PAAD-003', 'TCGA-PAAD-004'],
                'age': [65, 72, 58, 61],
                'gender': ['Male', 'Female', 'Male', 'Female'],
                'stage': ['Stage II', 'Stage III', 'Stage I', 'Stage IV'],
                'survival_months': [12.5, 8.2, 24.3, 6.1],
                'vital_status': ['Dead', 'Dead', 'Alive', 'Dead']
            })
        
        else:
            st.error(f"Unknown data type: {data_type}")
            return pd.DataFrame()
