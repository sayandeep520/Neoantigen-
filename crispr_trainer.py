"""
Advanced CRISPR Target Optimization Model Trainer

This module provides functionality for training and optimizing machine learning models
for CRISPR target prediction and optimization. It implements multiple model options
including Random Forest, Gradient Boosting, and Neural Networks.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Default configurations and constants
DATA_DIR = 'data'
MODEL_DIR = 'models'
DATA_PATH = os.path.join(DATA_DIR, 'crispr_data.csv')
OUTPUT_PATH = os.path.join(MODEL_DIR, 'crispr_model.pkl')


class CRISPRModelTrainer:
    """
    Advanced trainer for CRISPR target optimization models
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the trainer with optional data path
        
        Args:
            data_path: Path to CRISPR training data CSV
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
    
    def load_data(self, file_path=None, uploaded_file=None):
        """
        Load data from file path or uploaded file
        
        Args:
            file_path: Path to CSV file (optional)
            uploaded_file: Streamlit uploaded file object (optional)
            
        Returns:
            Loaded DataFrame
        """
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
        elif file_path is not None:
            self.data = pd.read_csv(file_path)
        elif self.data_path is not None:
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("No data source provided")
            
        return self.data
        
    def preprocess_data(self, features=None, target=None, test_size=0.2, val_size=0.25, random_state=42):
        """
        Preprocess data and prepare for model training
        
        Args:
            features: List of feature column names (optional)
            target: Target column name (optional)
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test, X_val, y_val
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Determine features and target if not specified
        if features is None:
            # Assume all columns except the target are features
            if target is None:
                # Try to find a reasonable target column
                target_candidates = ['efficiency', 'activity', 'target_efficiency', 'on_target_efficiency']
                for candidate in target_candidates:
                    if candidate in self.data.columns:
                        target = candidate
                        break
                    
                if target is None:
                    # Default to the last column as target
                    target = self.data.columns[-1]
                    
            features = [col for col in self.data.columns if col != target]
        
        # Extract features and target
        X = self.data[features]
        y = self.data[target]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create validation set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=val_size, random_state=random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_val, self.y_val
    
    def build_feature_preprocessor(self, numeric_features=None, categorical_features=None, 
                                 sequence_feature=None):
        """
        Build a feature preprocessing pipeline
        
        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            sequence_feature: Name of the sequence feature column (DNA/RNA)
            
        Returns:
            Preprocessing pipeline
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call preprocess_data() first.")
        
        # Identify feature types if not specified
        if numeric_features is None:
            numeric_features = self.X_train.select_dtypes(include=['float', 'int']).columns.tolist()
            
        if categorical_features is None:
            categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove sequence feature from categorical if specified
            if sequence_feature is not None and sequence_feature in categorical_features:
                categorical_features.remove(sequence_feature)
        
        # Build transformers list
        transformers = []
        
        # Numeric features
        if numeric_features:
            transformers.append(
                ('num', StandardScaler(), numeric_features)
            )
        
        # Categorical features
        if categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            )
        
        # Sequence feature (if specified)
        if sequence_feature is not None:
            transformers.append(
                ('seq', TfidfVectorizer(analyzer='char', ngram_range=(1, 3)), sequence_feature)
            )
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        return self.preprocessor
    
    def build_models(self, include_rf=True, include_gb=True, include_nn=True):
        """
        Build machine learning models for CRISPR target prediction
        
        Args:
            include_rf: Whether to include Random Forest
            include_gb: Whether to include Gradient Boosting
            include_nn: Whether to include Neural Network
            
        Returns:
            Dictionary of models
        """
        models = {}
        
        if include_rf:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            models['Random Forest'] = rf
            
        if include_gb:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            models['Gradient Boosting'] = gb
            
        if include_nn:
            nn = MLPRegressor(hidden_layer_sizes=(100, 50), solver='adam', max_iter=500, random_state=42)
            models['Neural Network'] = nn
            
        self.models = models
        return models
    
    def train_models(self):
        """
        Train all models on the training data
        
        Returns:
            Dictionary of trained models
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not built. Call build_feature_preprocessor() first.")
            
        if not self.models:
            self.build_models()
            
        trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            pipeline.fit(self.X_train, self.y_train)
            trained_models[name] = pipeline
            
        self.trained_models = trained_models
        return trained_models
    
    def evaluate_models(self):
        """
        Evaluate all trained models on test data
        
        Returns:
            Dictionary of evaluation results
        """
        if not hasattr(self, 'trained_models'):
            raise ValueError("Models not trained. Call train_models() first.")
            
        results = {}
        
        for name, model in self.trained_models.items():
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = mean_squared_error(self.y_test, y_pred, squared=False)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
            }
            
        self.results = results
        return results
    
    def save_best_model(self, metric='r2', output_path=None):
        """
        Save the best model based on a specified metric
        
        Args:
            metric: Metric to use for selecting best model ('r2', 'mse', 'rmse', 'mae')
            output_path: Path to save model
            
        Returns:
            Path to saved model
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate_models() first.")
            
        # Determine best model based on metric
        best_score = float('-inf') if metric == 'r2' else float('inf')
        best_model_name = None
        
        for name, result in self.results.items():
            # For R² higher is better, for error metrics lower is better
            if metric == 'r2':
                score = result[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = name
            else:
                score = result[metric]
                if score < best_score:
                    best_score = score
                    best_model_name = name
                
        if best_model_name is None:
            raise ValueError(f"Could not determine best model using metric: {metric}")
            
        # Save best model
        best_model = self.results[best_model_name]['model']
        
        if output_path is None:
            output_path = OUTPUT_PATH
            
        with open(output_path, 'wb') as f:
            pickle.dump(best_model, f)
            
        print(f"Best model ({best_model_name}, {metric}={best_score:.4f}) saved to {output_path}")
        return output_path
        
    def plot_prediction_vs_actual(self):
        """
        Plot prediction vs actual values for all models
        
        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate_models() first.")
            
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        # Handle case with only one model
        if n_models == 1:
            axes = [axes]
            
        for (name, result), ax in zip(self.results.items(), axes):
            ax.scatter(self.y_test, result['predictions'], alpha=0.5)
            ax.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f"{name}\nR²: {result['r2']:.3f}, RMSE: {result['rmse']:.3f}")
                
        plt.tight_layout()
        return fig
        
    def plot_feature_importance(self, model_name=None):
        """
        Plot feature importance for a specified model
        
        Args:
            model_name: Name of the model to plot importance for
            
        Returns:
            Matplotlib figure or None if not supported
        """
        if not self.results:
            raise ValueError("No evaluation results. Call evaluate_models() first.")
            
        # If model_name not specified, use first model that supports feature importance
        if model_name is None:
            for name, result in self.results.items():
                model = result['model'].named_steps['regressor']
                if hasattr(model, 'feature_importances_'):
                    model_name = name
                    break
                    
        if model_name is None or model_name not in self.results:
            print("No model with feature importance found")
            return None
            
        # Get the model and preprocessor
        pipeline = self.results[model_name]['model']
        model = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Check if model supports feature importance
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not support feature importance")
            return None
            
        # Get feature names
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).head(20)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f"Feature Importance for {model_name}")
        plt.tight_layout()
        
        return plt.gcf()


def run_training_demo(data_path=DATA_PATH):
    """
    Run a complete training demo
    
    Args:
        data_path: Path to training data
        
    Returns:
        Trained model and results
    """
    # Create trainer
    trainer = CRISPRModelTrainer(data_path)
    
    # Try to load data
    try:
        trainer.load_data()
        print(f"Loaded data with {trainer.data.shape[0]} samples and {trainer.data.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    # Preprocess data
    trainer.preprocess_data()
    print(f"Data split into {trainer.X_train.shape[0]} training, {trainer.X_val.shape[0]} validation, and {trainer.X_test.shape[0]} test samples")
    
    # Build feature preprocessor
    trainer.build_feature_preprocessor()
    
    # Build models
    trainer.build_models()
    
    # Train models
    trainer.train_models()
    
    # Evaluate models
    results = trainer.evaluate_models()
    
    # Print results
    for name, result in results.items():
        print(f"\nResults for {name}:")
        print(f"R²: {result['r2']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"MAE: {result['mae']:.4f}")
    
    # Save best model
    trainer.save_best_model(metric='r2')
    
    return trainer


if __name__ == "__main__":
    run_training_demo()