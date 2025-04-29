import numpy as np
import pandas as pd
import streamlit as st
import time
import pickle
import os
from typing import Dict, List, Union, Tuple, Optional, Any

# Machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import tensorflow as tf


class ModelTrainer:
    """
    Train and evaluate AI models for various tasks in cancer immunotherapy:
    - CRISPR target prediction
    - Neoantigen prediction
    - Therapy response prediction
    """
    
    def __init__(self):
        """Initialize the model trainer with common models and evaluation metrics"""
        self.models = {
            'classification': {
                'random_forest': RandomForestClassifier(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'svm': SVC(probability=True, random_state=42),
                'xgboost': xgb.XGBClassifier(random_state=42)
            },
            'regression': {
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': xgb.XGBRegressor(random_state=42),
                'elastic_net': ElasticNet(random_state=42)
            }
        }
        
        self.model_params = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'elastic_net': {
                'alpha': [0.1, 0.5, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        }
        
        # Available transformation models
        self.transformer_available = self._check_tensorflow()
    
    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available for transformer models"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            return True
        except ImportError:
            return False
    
    def preprocess_for_training(self, df: pd.DataFrame, 
                               target_col: str, 
                               feature_cols: List[str] = None,
                               test_size: float = 0.2,
                               encode_target: bool = True) -> Dict[str, Any]:
        """
        Preprocess data for model training
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: Feature column names (if None, use all columns except target)
            test_size: Proportion of data for testing
            encode_target: Whether to encode categorical target
            
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, and feature_names
        """
        if df.empty:
            st.error("Empty DataFrame provided for training")
            return {}
        
        if target_col not in df.columns:
            st.error(f"Target column '{target_col}' not found in DataFrame")
            return {}
        
        # Select features
        if feature_cols is None:
            feature_cols = df.columns.difference([target_col]).tolist()
        
        # Check if all feature columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Feature columns not found: {', '.join(missing_cols)}")
            return {}
        
        # Extract features and target
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Encode categorical target if needed
        if encode_target and y.dtype == 'object':
            st.info("Encoding categorical target variable")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            encoder_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            st.write("Target encoding mapping:", encoder_mapping)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if encode_target else None
        )
        
        st.success(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols
        }
    
    def train_model(self, 
                   train_data: Dict[str, Any],
                   model_type: str = 'classification',
                   model_name: str = 'random_forest',
                   hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            train_data: Dictionary with training data (X_train, X_test, y_train, y_test)
            model_type: Type of model ('classification' or 'regression')
            model_name: Name of the model
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with trained model and evaluation metrics
        """
        # Check if training data is provided
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test']
        if not all(key in train_data for key in required_keys):
            st.error("Missing required training data")
            return {}
        
        X_train = train_data['X_train']
        X_test = train_data['X_test']
        y_train = train_data['y_train']
        y_test = train_data['y_test']
        
        # Check if model type is valid
        if model_type not in self.models:
            st.error(f"Invalid model type: {model_type}")
            return {}
        
        # Check if model name is valid
        if model_name not in self.models[model_type]:
            st.error(f"Invalid model name: {model_name}")
            return {}
        
        # Get the model
        model = self.models[model_type][model_name]
        
        # Train with or without hyperparameter tuning
        if hyperparameter_tuning:
            st.info(f"Performing hyperparameter tuning for {model_name}")
            
            # Get hyperparameters for the model
            params = self.model_params.get(model_name, {})
            
            if not params:
                st.warning(f"No hyperparameters defined for {model_name}, using default settings")
                model.fit(X_train, y_train)
            else:
                # Create grid search
                grid_search = GridSearchCV(
                    model, params, cv=5, scoring='accuracy' if model_type == 'classification' else 'neg_mean_squared_error'
                )
                
                # Train with grid search
                with st.spinner(f"Training {model_name} with hyperparameter tuning..."):
                    start_time = time.time()
                    grid_search.fit(X_train, y_train)
                    training_time = time.time() - start_time
                
                # Get best model and parameters
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                st.success(f"Hyperparameter tuning complete in {training_time:.2f} seconds")
                st.write("Best parameters:", best_params)
        else:
            # Train without hyperparameter tuning
            with st.spinner(f"Training {model_name}..."):
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
            
            st.success(f"Training complete in {training_time:.2f} seconds")
        
        # Evaluate model
        evaluation = self.evaluate_model(model, X_test, y_test, model_type)
        
        # Return results
        return {
            'model': model,
            'evaluation': evaluation,
            'training_time': training_time,
            'feature_importance': self.get_feature_importance(model, train_data.get('feature_names', [])),
        }
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_type: str) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        if model_type == 'classification':
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }
            
            # Add ROC AUC if probability predictions are available
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
        elif model_type == 'regression':
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            }
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from the trained model
        
        Args:
            model: Trained model
            feature_names: Names of the features
            
        Returns:
            Dictionary with feature importance
        """
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importance = model.feature_importances_
            
            # Create dictionary with feature names and importance
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance (descending)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
        elif hasattr(model, 'coef_'):
            # Get coefficients
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            # Create dictionary with feature names and coefficients
            feature_importance = dict(zip(feature_names, np.abs(coef)))
            
            # Sort by importance (descending)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
        else:
            return {}
    
    def save_model(self, model: Any, model_name: str, model_dir: str = 'models') -> str:
        """
        Save trained model to file
        
        Args:
            model: Trained model
            model_name: Name of the model
            model_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create unique filename
        timestamp = int(time.time())
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(model_dir, filename)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        st.success(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """
        Load model from file
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model
        """
        # Check if file exists
        if not os.path.exists(filepath):
            st.error(f"Model file not found: {filepath}")
            return None
        
        # Load model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        st.success(f"Model loaded from {filepath}")
        return model
    
    def create_deep_learning_model(self, input_shape: int, output_shape: int, model_type: str = 'classification') -> tf.keras.Model:
        """
        Create a deep learning model for CRISPR or neoantigen prediction
        
        Args:
            input_shape: Number of input features
            output_shape: Number of output classes or regression targets
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            Keras model
        """
        if not self.transformer_available:
            st.error("TensorFlow is not available. Cannot create deep learning model.")
            return None
        
        # Import TensorFlow modules
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create a simple feedforward neural network
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
        ])
        
        # Add output layer based on model type
        if model_type == 'classification':
            if output_shape == 2:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy', keras.metrics.AUC()]
            else:
                model.add(layers.Dense(output_shape, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(layers.Dense(output_shape, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def train_deep_learning_model(self, 
                                 train_data: Dict[str, Any],
                                 model_type: str = 'classification',
                                 epochs: int = 50,
                                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Train a deep learning model
        
        Args:
            train_data: Dictionary with training data (X_train, X_test, y_train, y_test)
            model_type: Type of model ('classification' or 'regression')
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with trained model and evaluation metrics
        """
        if not self.transformer_available:
            st.error("TensorFlow is not available. Cannot train deep learning model.")
            return {}
        
        # Check if training data is provided
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test']
        if not all(key in train_data for key in required_keys):
            st.error("Missing required training data")
            return {}
        
        X_train = train_data['X_train'].values
        X_test = train_data['X_test'].values
        y_train = train_data['y_train'].values
        y_test = train_data['y_test'].values
        
        # Get input and output shapes
        input_shape = X_train.shape[1]
        output_shape = len(np.unique(y_train)) if model_type == 'classification' else 1
        
        # Create model
        model = self.create_deep_learning_model(input_shape, output_shape, model_type)
        
        if model is None:
            return {}
        
        # Create early stopping callback
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        with st.spinner("Training deep learning model..."):
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            training_time = time.time() - start_time
        
        st.success(f"Deep learning model training complete in {training_time:.2f} seconds")
        
        # Evaluate model
        evaluation = {}
        
        if model_type == 'classification':
            # Get predictions
            y_pred = np.argmax(model.predict(X_test), axis=1) if output_shape > 2 else (model.predict(X_test) > 0.5).astype(int).flatten()
            
            # Calculate metrics
            evaluation = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }
            
            # Add ROC AUC for binary classification
            if output_shape == 2:
                y_pred_proba = model.predict(X_test).flatten()
                evaluation['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
        else:
            # Get predictions
            y_pred = model.predict(X_test).flatten()
            
            # Calculate metrics
            evaluation = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            }
        
        # Return results
        return {
            'model': model,
            'evaluation': evaluation,
            'training_time': training_time,
            'history': history.history,
        }
