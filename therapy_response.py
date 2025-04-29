import pandas as pd
import numpy as np
import streamlit as st
import os
import time
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class TherapyResponsePredictor:
    """
    Predict patient responses to AI-driven CRISPR therapies using multi-omics data.
    Integrates genomic, transcriptomic, proteomic features with immune profiling.
    """
    
    def __init__(self):
        """Initialize the therapy response predictor"""
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        self.feature_categories = {
            'genomic': [
                'mutation_burden', 'gene_amplifications', 'gene_deletions',
                'signature_1', 'signature_2', 'signature_3'
            ],
            'transcriptomic': [
                'immune_score', 'stromal_score', 'tumor_purity',
                'cell_cycle_score', 'dna_repair_score', 'ifn_gamma_response'
            ],
            'immune': [
                'cd8_infiltration', 'cd4_infiltration', 'treg_infiltration',
                'nk_cell_activity', 'myeloid_score', 'pd1_expression',
                'pdl1_expression', 'ctla4_expression'
            ],
            'clinical': [
                'age', 'gender', 'stage', 'prior_therapy', 'ecog_status'
            ]
        }
        
        self.current_model = None
        self.feature_importances = {}
    
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for therapy response prediction
        
        Args:
            data: DataFrame with patient data
            
        Returns:
            Dictionary with prepared data
        """
        if data.empty:
            st.error("Empty DataFrame provided for therapy response prediction")
            return {}
        
        required_columns = ['patient_id', 'response']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Missing required columns: {required_columns}")
            return {}
        
        # Extract features and target
        st.info("Preparing data for therapy response prediction")
        
        # Identify feature columns (all except patient_id and response)
        feature_cols = [col for col in data.columns if col not in ['patient_id', 'response']]
        
        # Check if there are any feature columns
        if not feature_cols:
            st.error("No feature columns found in data")
            return {}
        
        # Extract features and target
        X = data[feature_cols]
        y = data['response']
        
        # Encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 0:
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        st.success(f"Data prepared with {X_train.shape[1]} features")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_encoded.columns.tolist(),
            'scaler': scaler if len(numerical_cols) > 0 else None,
            'numerical_cols': numerical_cols
        }
    
    def train_model(self, prepared_data: Dict[str, Any], model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Train a model to predict therapy response
        
        Args:
            prepared_data: Dictionary with prepared data
            model_type: Type of model to train
            
        Returns:
            Dictionary with model and performance metrics
        """
        # Check if prepared data contains required keys
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'feature_names']
        if not all(key in prepared_data for key in required_keys):
            st.error("Missing required data for model training")
            return {}
        
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_test = prepared_data['X_test']
        y_test = prepared_data['y_test']
        feature_names = prepared_data['feature_names']
        
        # Check if model type is valid
        if model_type not in self.models:
            st.error(f"Invalid model type: {model_type}")
            model_type = 'xgboost'
            st.info(f"Using default model: {model_type}")
        
        # Create a new instance of the model
        model = self.models[model_type]
        
        st.info(f"Training {model_type} model...")
        
        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate performance metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_[0])
            feature_importance = dict(zip(feature_names, importances))
        else:
            feature_importance = {}
        
        # Sort feature importances
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Save model and feature importances
        self.current_model = model
        self.feature_importances = feature_importance
        
        st.success(f"Model training complete in {training_time:.2f} seconds")
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_time': training_time
        }
    
    def predict_response(self, data: pd.DataFrame, model_result: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Predict therapy response for new patients
        
        Args:
            data: DataFrame with patient data
            model_result: Dictionary with model and preprocessing information
            
        Returns:
            DataFrame with predictions
        """
        if data.empty:
            st.error("Empty DataFrame provided for prediction")
            return pd.DataFrame()
        
        # Use provided model or self.current_model
        model = model_result.get('model') if model_result else self.current_model
        
        if model is None:
            st.error("No trained model available. Please train a model first.")
            return pd.DataFrame()
        
        # Extract patient IDs
        patient_ids = data['patient_id'] if 'patient_id' in data.columns else data.index
        
        # Extract features
        feature_cols = [col for col in data.columns if col != 'patient_id' and col != 'response']
        X = data[feature_cols]
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Scale numerical features if scaler is available
        if model_result and 'scaler' in model_result and model_result['scaler'] is not None:
            scaler = model_result['scaler']
            numerical_cols = model_result.get('numerical_cols', [])
            
            if len(numerical_cols) > 0:
                # Ensure all numerical columns exist in X_encoded
                existing_num_cols = [col for col in numerical_cols if col in X_encoded.columns]
                if existing_num_cols:
                    X_encoded[existing_num_cols] = scaler.transform(X_encoded[existing_num_cols])
        
        # Make predictions
        try:
            y_pred = model.predict(X_encoded)
            y_pred_proba = model.predict_proba(X_encoded)[:, 1]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'patient_id': patient_ids,
                'predicted_response': y_pred,
                'response_probability': y_pred_proba
            })
            
            st.success(f"Generated predictions for {len(results)} patients")
            return results
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return pd.DataFrame()
    
    def explain_predictions(self, data: pd.DataFrame, prediction_result: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain predictions using feature importances
        
        Args:
            data: Original data used for predictions
            prediction_result: DataFrame with predictions
            
        Returns:
            Dictionary with explanation information
        """
        if data.empty or prediction_result.empty:
            st.error("Empty data provided for explanation")
            return {}
        
        if not self.feature_importances:
            st.warning("No feature importances available for explanation")
            return {}
        
        # Get top features
        top_features = list(self.feature_importances.keys())[:10]
        top_importances = list(self.feature_importances.values())[:10]
        
        # Find patients with different predictions
        if 'patient_id' in data.columns and 'patient_id' in prediction_result.columns:
            # Merge original data with predictions
            merged_data = pd.merge(data, prediction_result, on='patient_id')
            
            # Find responders and non-responders
            responders = merged_data[merged_data['predicted_response'] == 1]
            non_responders = merged_data[merged_data['predicted_response'] == 0]
            
            # Calculate feature difference between responders and non-responders
            feature_diffs = {}
            
            for feature in top_features:
                if feature in merged_data.columns:
                    resp_mean = responders[feature].mean() if not responders.empty else 0
                    non_resp_mean = non_responders[feature].mean() if not non_responders.empty else 0
                    
                    feature_diffs[feature] = resp_mean - non_resp_mean
        else:
            feature_diffs = {}
        
        # Group features by category
        categorized_features = {}
        
        for category, features in self.feature_categories.items():
            category_importances = {}
            
            for feature in features:
                # Check for exact match
                if feature in self.feature_importances:
                    category_importances[feature] = self.feature_importances[feature]
                
                # Check for features that contain this feature name (from one-hot encoding)
                for full_feature, importance in self.feature_importances.items():
                    if feature in full_feature and feature != full_feature:
                        category_importances[full_feature] = importance
            
            if category_importances:
                # Calculate average importance for category
                avg_importance = sum(category_importances.values()) / len(category_importances)
                categorized_features[category] = {
                    'average_importance': avg_importance,
                    'features': category_importances
                }
        
        return {
            'top_features': dict(zip(top_features, top_importances)),
            'feature_differences': feature_diffs,
            'feature_categories': categorized_features
        }
    
    def generate_sample_data(self, num_patients: int = 100) -> pd.DataFrame:
        """
        Generate sample patient data for demonstration
        
        Args:
            num_patients: Number of patients to generate
            
        Returns:
            DataFrame with sample patient data
        """
        st.warning("Generating sample data for demonstration purposes")
        
        # Create patient IDs
        patient_ids = [f"PT-{i+1:03d}" for i in range(num_patients)]
        
        # Generate sample data
        data = {
            'patient_id': patient_ids,
            'age': np.random.normal(65, 10, num_patients).astype(int),
            'gender': np.random.choice(['M', 'F'], num_patients),
            'stage': np.random.choice(['I', 'II', 'III', 'IV'], num_patients, p=[0.1, 0.2, 0.4, 0.3]),
            'prior_therapy': np.random.choice(['None', 'Chemotherapy', 'Immunotherapy', 'Combined'], num_patients),
            'ecog_status': np.random.choice([0, 1, 2], num_patients, p=[0.4, 0.4, 0.2]),
            
            # Genomic features
            'mutation_burden': np.random.lognormal(3, 1, num_patients),
            'gene_amplifications': np.random.poisson(5, num_patients),
            'gene_deletions': np.random.poisson(3, num_patients),
            'signature_1': np.random.random(num_patients),
            'signature_2': np.random.random(num_patients),
            'signature_3': np.random.random(num_patients),
            
            # Transcriptomic features
            'immune_score': np.random.normal(0, 1, num_patients),
            'stromal_score': np.random.normal(0, 1, num_patients),
            'tumor_purity': np.random.beta(5, 2, num_patients),
            'cell_cycle_score': np.random.normal(0, 1, num_patients),
            'dna_repair_score': np.random.normal(0, 1, num_patients),
            'ifn_gamma_response': np.random.normal(0, 1, num_patients),
            
            # Immune features
            'cd8_infiltration': np.random.gamma(2, 1, num_patients),
            'cd4_infiltration': np.random.gamma(2, 1, num_patients),
            'treg_infiltration': np.random.gamma(1, 1, num_patients),
            'nk_cell_activity': np.random.gamma(1.5, 1, num_patients),
            'myeloid_score': np.random.normal(0, 1, num_patients),
            'pd1_expression': np.random.gamma(1, 1, num_patients),
            'pdl1_expression': np.random.gamma(1, 1, num_patients),
            'ctla4_expression': np.random.gamma(1, 1, num_patients),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate synthetic response based on features
        # Higher probability of response if:
        # - Higher mutation burden
        # - Higher CD8 infiltration
        # - Higher IFN-gamma response
        # - Lower Treg infiltration
        # - Earlier stage
        
        # Create a probability model
        p_response = 0.3  # Base probability
        
        # Add feature effects
        p_response += 0.3 * (df['mutation_burden'] - df['mutation_burden'].mean()) / df['mutation_burden'].std()
        p_response += 0.2 * (df['cd8_infiltration'] - df['cd8_infiltration'].mean()) / df['cd8_infiltration'].std()
        p_response += 0.15 * (df['ifn_gamma_response'] - df['ifn_gamma_response'].mean()) / df['ifn_gamma_response'].std()
        p_response -= 0.15 * (df['treg_infiltration'] - df['treg_infiltration'].mean()) / df['treg_infiltration'].std()
        
        # Add stage effect
        stage_effect = {'I': 0.2, 'II': 0.1, 'III': -0.05, 'IV': -0.15}
        df['stage_effect'] = df['stage'].map(stage_effect)
        p_response += df['stage_effect']
        df.drop('stage_effect', axis=1, inplace=True)
        
        # Normalize to 0-1 range
        p_response = 1 / (1 + np.exp(-p_response))  # Sigmoid function
        
        # Generate binary response
        df['response'] = np.random.binomial(1, p_response)
        
        return df
    
    def analyze_feature_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlations between features and therapy response
        
        Args:
            data: DataFrame with patient data including response
            
        Returns:
            DataFrame with correlation analysis
        """
        if data.empty or 'response' not in data.columns:
            st.error("Invalid data for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlations with response
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        
        if 'response' not in numeric_data.columns:
            st.error("Response variable is not numeric")
            return pd.DataFrame()
        
        # Calculate correlation with response
        correlations = numeric_data.corr()['response'].sort_values(ascending=False)
        
        # Create correlation DataFrame
        correlation_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        # Filter out response self-correlation
        correlation_df = correlation_df[correlation_df['feature'] != 'response']
        
        return correlation_df
    
    def identify_patient_subgroups(self, data: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Identify patient subgroups based on feature patterns
        
        Args:
            data: DataFrame with patient data
            n_clusters: Number of clusters to identify
            
        Returns:
            Dictionary with clustering results
        """
        if data.empty:
            st.error("Empty data provided for subgroup identification")
            return {}
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Extract features (exclude patient_id and response)
            feature_cols = [col for col in data.columns if col not in ['patient_id', 'response']]
            X = data[feature_cols]
            
            # Handle non-numeric features
            X = pd.get_dummies(X, drop_first=True)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=min(5, X.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Calculate response rates by cluster
            if 'response' in data.columns:
                cluster_response = {}
                
                for cluster in range(n_clusters):
                    cluster_data = data[clusters == cluster]
                    response_rate = cluster_data['response'].mean() if not cluster_data.empty else 0
                    cluster_response[f'Cluster {cluster+1}'] = response_rate
            else:
                cluster_response = {}
            
            # Create a DataFrame with cluster assignments
            cluster_df = pd.DataFrame({
                'patient_id': data['patient_id'] if 'patient_id' in data.columns else data.index,
                'cluster': clusters + 1  # Start clusters at 1 for clarity
            })
            
            # Calculate cluster centers
            cluster_centers = kmeans.cluster_centers_
            
            # Get feature importances for clusters
            feature_importances = {}
            for i in range(n_clusters):
                # Find features that differ most from the overall mean
                center = cluster_centers[i]
                diff = center - X_scaled.mean(axis=0)
                
                # Get top features for this cluster
                top_indices = np.argsort(np.abs(diff))[-5:]  # Top 5 features
                top_features = {X.columns[idx]: diff[idx] for idx in top_indices}
                
                feature_importances[f'Cluster {i+1}'] = top_features
            
            return {
                'clusters': cluster_df,
                'response_rates': cluster_response,
                'pca_components': X_pca,
                'feature_importances': feature_importances
            }
            
        except Exception as e:
            st.error(f"Error in subgroup identification: {str(e)}")
            return {}
