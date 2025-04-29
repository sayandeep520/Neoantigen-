import os
import requests
import pandas as pd
import streamlit as st
import time
import json
from typing import Dict, List, Union, Tuple, Optional, Any
from io import StringIO
import re


class BioAPIConnector:
    """
    Connector class to handle API interactions with bioinformatics databases.
    Manages authentication, rate limiting, and data retrieval.
    """
    
    def __init__(self):
        """Initialize the API connector with default settings"""
        self.api_keys = {
            'cbioportal': os.getenv('CBIOPORTAL_API_KEY', ''),
            'depmap': os.getenv('DEPMAP_API_KEY', ''),
            'ncbi': os.getenv('NCBI_API_KEY', ''),
            'immport': os.getenv('IMMPORT_API_KEY', ''),
            'ensembl': os.getenv('ENSEMBL_API_KEY', '')
        }
        
        self.api_endpoints = {
            'cbioportal': 'https://www.cbioportal.org/api/',
            'depmap': 'https://depmap.org/portal/api/',
            'gtex': 'https://gtexportal.org/rest/v1/',
            'ncbi': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
            'ensembl': 'https://rest.ensembl.org/',
            'uniprot': 'https://www.uniprot.org/uniprot/',
            'immport': 'https://api.immport.org/data/'
        }
        
        # Rate limiting parameters
        self.rate_limits = {
            'cbioportal': {'requests': 3, 'period': 1},  # 3 requests per second
            'depmap': {'requests': 2, 'period': 1},      # 2 requests per second
            'gtex': {'requests': 2, 'period': 1},        # 2 requests per second
            'ncbi': {'requests': 3, 'period': 1},        # 3 requests per second
            'ensembl': {'requests': 15, 'period': 60},   # 15 requests per minute
            'uniprot': {'requests': 3, 'period': 1},     # 3 requests per second
            'immport': {'requests': 5, 'period': 1}      # 5 requests per second
        }
        
        # Last request timestamps
        self.last_request_times = {api: [] for api in self.api_endpoints}
    
    def _apply_rate_limiting(self, api_name: str):
        """
        Apply rate limiting for the specified API
        
        Args:
            api_name: Name of the API
        """
        if api_name not in self.rate_limits:
            return
        
        # Get rate limit parameters
        limit = self.rate_limits[api_name]
        requests_allowed = limit['requests']
        period = limit['period']
        
        # Get timestamps of previous requests
        timestamps = self.last_request_times[api_name]
        
        # Remove timestamps older than the period
        current_time = time.time()
        timestamps = [t for t in timestamps if current_time - t < period]
        
        # Update timestamps
        self.last_request_times[api_name] = timestamps
        
        # If at the rate limit, wait until the oldest timestamp falls outside the period
        if len(timestamps) >= requests_allowed:
            wait_time = period - (current_time - timestamps[0]) + 0.1  # Add a small buffer
            if wait_time > 0:
                time.sleep(wait_time)
        
        # Add current timestamp
        self.last_request_times[api_name].append(time.time())
    
    def make_request(self, 
                    api_name: str, 
                    endpoint: str, 
                    method: str = 'GET', 
                    params: Dict = None, 
                    data: Dict = None,
                    headers: Dict = None) -> requests.Response:
        """
        Make an API request with rate limiting
        
        Args:
            api_name: Name of the API
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request data
            headers: Request headers
            
        Returns:
            Response object
        """
        # Apply rate limiting
        self._apply_rate_limiting(api_name)
        
        # Build URL
        if api_name not in self.api_endpoints:
            st.error(f"Unknown API: {api_name}")
            return None
        
        base_url = self.api_endpoints[api_name]
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Set up headers
        if headers is None:
            headers = {}
        
        # Add API key if available
        if api_name in self.api_keys and self.api_keys[api_name]:
            headers['Authorization'] = f"Bearer {self.api_keys[api_name]}"
        
        # Make the request
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            )
            
            # Check for errors
            response.raise_for_status()
            
            return response
        
        except requests.exceptions.RequestException as e:
            st.error(f"API request error ({api_name}): {str(e)}")
            return None
    
    def get_cbioportal_studies(self) -> pd.DataFrame:
        """
        Get list of studies from cBioPortal
        
        Returns:
            DataFrame with study information
        """
        response = self.make_request('cbioportal', 'studies')
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved {len(df)} studies from cBioPortal")
            return df
        else:
            st.error("Failed to retrieve studies from cBioPortal")
            return pd.DataFrame()
    
    def get_cbioportal_cancer_types(self) -> pd.DataFrame:
        """
        Get list of cancer types from cBioPortal
        
        Returns:
            DataFrame with cancer type information
        """
        response = self.make_request('cbioportal', 'cancer-types')
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved {len(df)} cancer types from cBioPortal")
            return df
        else:
            st.error("Failed to retrieve cancer types from cBioPortal")
            return pd.DataFrame()
    
    def get_cbioportal_mutations(self, study_id: str, gene_list: List[str] = None) -> pd.DataFrame:
        """
        Get mutation data from cBioPortal
        
        Args:
            study_id: Study identifier
            gene_list: List of genes to filter by
            
        Returns:
            DataFrame with mutation data
        """
        # Build endpoint
        endpoint = f"molecular-profiles/{study_id}_mutations/mutations"
        
        # Build parameters
        params = {}
        if gene_list:
            params['hugoGeneSymbols'] = ','.join(gene_list)
        
        # Make request
        response = self.make_request('cbioportal', endpoint, params=params)
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved {len(df)} mutations from cBioPortal")
            return df
        else:
            st.error(f"Failed to retrieve mutations from cBioPortal for study {study_id}")
            return pd.DataFrame()
    
    def get_cbioportal_clinical_data(self, study_id: str) -> pd.DataFrame:
        """
        Get clinical data from cBioPortal
        
        Args:
            study_id: Study identifier
            
        Returns:
            DataFrame with clinical data
        """
        endpoint = f"studies/{study_id}/clinical-data"
        response = self.make_request('cbioportal', endpoint)
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved clinical data for {len(df)} patients from cBioPortal")
            return df
        else:
            st.error(f"Failed to retrieve clinical data from cBioPortal for study {study_id}")
            return pd.DataFrame()
    
    def get_depmap_datasets(self) -> List[str]:
        """
        Get list of available datasets from DepMap
        
        Returns:
            List of dataset names
        """
        response = self.make_request('depmap', 'datasets')
        
        if response and response.status_code == 200:
            data = response.json()
            datasets = data.get('datasets', [])
            st.success(f"Retrieved {len(datasets)} datasets from DepMap")
            return datasets
        else:
            st.error("Failed to retrieve datasets from DepMap")
            return []
    
    def get_depmap_data(self, dataset: str) -> pd.DataFrame:
        """
        Get dataset from DepMap
        
        Args:
            dataset: Dataset name
            
        Returns:
            DataFrame with dataset
        """
        endpoint = f"{dataset}"
        response = self.make_request('depmap', endpoint)
        
        if response and response.status_code == 200:
            try:
                df = pd.read_csv(StringIO(response.text))
                st.success(f"Retrieved {dataset} from DepMap: {df.shape[0]} rows, {df.shape[1]} columns")
                return df
            except Exception as e:
                st.error(f"Error parsing DepMap data: {str(e)}")
                return pd.DataFrame()
        else:
            st.error(f"Failed to retrieve {dataset} from DepMap")
            return pd.DataFrame()
    
    def get_gtex_tissues(self) -> pd.DataFrame:
        """
        Get list of tissues from GTEx
        
        Returns:
            DataFrame with tissue information
        """
        endpoint = "dataset/tissueSiteDetail"
        response = self.make_request('gtex', endpoint)
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved {len(df)} tissues from GTEx")
            return df
        else:
            st.error("Failed to retrieve tissues from GTEx")
            return pd.DataFrame()
    
    def get_gtex_expression(self, tissue_id: str) -> pd.DataFrame:
        """
        Get expression data from GTEx for a specific tissue
        
        Args:
            tissue_id: Tissue identifier
            
        Returns:
            DataFrame with expression data
        """
        endpoint = f"expression/geneExpression?tissueId={tissue_id}"
        response = self.make_request('gtex', endpoint)
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved expression data for {tissue_id} from GTEx: {df.shape[0]} rows")
            return df
        else:
            st.error(f"Failed to retrieve expression data for {tissue_id} from GTEx")
            return pd.DataFrame()
    
    def search_geo_dataset(self, query: str) -> pd.DataFrame:
        """
        Search for GEO datasets matching a query
        
        Args:
            query: Search query
            
        Returns:
            DataFrame with search results
        """
        # Build esearch query
        params = {
            'db': 'gds',
            'term': query,
            'retmax': 50,
            'retmode': 'json'
        }
        
        response = self.make_request('ncbi', 'esearch.fcgi', params=params)
        
        if response and response.status_code == 200:
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            # If no results, return empty DataFrame
            if not id_list:
                st.warning(f"No GEO datasets found for query: {query}")
                return pd.DataFrame()
            
            # Fetch summary for each dataset
            results = []
            
            for dataset_id in id_list:
                summary = self.get_geo_summary(dataset_id)
                if summary:
                    results.append(summary)
            
            if results:
                df = pd.DataFrame(results)
                st.success(f"Found {len(df)} GEO datasets matching query: {query}")
                return df
            else:
                st.error("Failed to retrieve GEO dataset summaries")
                return pd.DataFrame()
        else:
            st.error(f"Failed to search GEO for query: {query}")
            return pd.DataFrame()
    
    def get_geo_summary(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get summary information for a GEO dataset
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dictionary with dataset summary
        """
        params = {
            'db': 'gds',
            'id': dataset_id,
            'retmode': 'json'
        }
        
        response = self.make_request('ncbi', 'esummary.fcgi', params=params)
        
        if response and response.status_code == 200:
            data = response.json()
            summary = data.get('result', {}).get(dataset_id, {})
            
            if summary:
                # Extract relevant fields
                return {
                    'id': summary.get('accession', ''),
                    'title': summary.get('title', ''),
                    'summary': summary.get('summary', ''),
                    'platform': summary.get('platform', ''),
                    'samples': summary.get('samples', ''),
                    'organism': summary.get('taxon', '')
                }
        
        return {}
    
    def get_uniprot_protein(self, protein_id: str) -> Dict[str, Any]:
        """
        Get protein information from UniProt
        
        Args:
            protein_id: Protein identifier
            
        Returns:
            Dictionary with protein information
        """
        endpoint = f"{protein_id}.json"
        response = self.make_request('uniprot', endpoint)
        
        if response and response.status_code == 200:
            data = response.json()
            st.success(f"Retrieved protein information for {protein_id} from UniProt")
            return data
        else:
            st.error(f"Failed to retrieve protein information for {protein_id} from UniProt")
            return {}
    
    def search_ensembl_gene(self, gene_name: str, species: str = 'human') -> Dict[str, Any]:
        """
        Search for a gene in Ensembl
        
        Args:
            gene_name: Gene name
            species: Species name
            
        Returns:
            Dictionary with gene information
        """
        endpoint = f"lookup/symbol/{species}/{gene_name}"
        headers = {'Content-Type': 'application/json'}
        
        response = self.make_request('ensembl', endpoint, headers=headers)
        
        if response and response.status_code == 200:
            data = response.json()
            st.success(f"Retrieved gene information for {gene_name} from Ensembl")
            return data
        else:
            st.error(f"Failed to retrieve gene information for {gene_name} from Ensembl")
            return {}
    
    def get_ensembl_sequence(self, gene_id: str) -> Dict[str, Any]:
        """
        Get gene sequence from Ensembl
        
        Args:
            gene_id: Ensembl gene identifier
            
        Returns:
            Dictionary with gene sequence
        """
        endpoint = f"sequence/id/{gene_id}"
        headers = {'Content-Type': 'application/json'}
        
        response = self.make_request('ensembl', endpoint, headers=headers)
        
        if response and response.status_code == 200:
            data = response.json()
            st.success(f"Retrieved sequence for {gene_id} from Ensembl")
            return data
        else:
            st.error(f"Failed to retrieve sequence for {gene_id} from Ensembl")
            return {}
    
    def get_immport_data(self, resource_type: str) -> pd.DataFrame:
        """
        Get data from ImmPort
        
        Args:
            resource_type: Type of resource to fetch
            
        Returns:
            DataFrame with data
        """
        endpoint = f"query/{resource_type}"
        headers = {'Content-Type': 'application/json'}
        
        response = self.make_request('immport', endpoint, headers=headers)
        
        if response and response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.success(f"Retrieved {len(df)} {resource_type} entries from ImmPort")
            return df
        else:
            st.error(f"Failed to retrieve {resource_type} from ImmPort")
            return pd.DataFrame()


# Create a singleton instance
api_connector = BioAPIConnector()
