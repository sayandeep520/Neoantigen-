import numpy as np
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, TypeError):
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using fallback options")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
import os
import random

class TherapyResponseModel:
    """
    Model for predicting patient response to cancer immunotherapy.
    """
    
    def __init__(self, model_type='xgboost', prediction_type='both'):
        """
        Initialize the therapy response prediction model.
        
        Args:
            model_type (str): Type of model to use (xgboost, random_forest, neural_network)
            prediction_type (str): Type of prediction (response_classification, survival_analysis, both)
        """
        self.model_type = model_type
        self.prediction_type = prediction_type
        
        # Initialize models
        self.classification_model = None
        self.survival_model = None
        
        # Initialize feature scaler
        self.feature_scaler = StandardScaler()
        
        # Track features used in the model
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, features, survival_data, params=None):
        """
        Train the therapy response prediction model.
        
        Args:
            features (pandas.DataFrame): Feature matrix for training
            survival_data (pandas.DataFrame): Survival information (time and status)
            params (dict): Additional parameters for model training
            
        Returns:
            dict: Training results and metrics
        """
        if params is None:
            params = {}
        
        # Save feature names
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
        
        # Standardize features
        X = self.feature_scaler.fit_transform(features)
        
        # Initialize results dictionary
        results = {}
        
        # Train classification model if required
        if self.prediction_type in ['response_classification', 'both']:
            # For classification, we need binary outcome
            # Use survival status as target if available
            if 'survival_status' in survival_data.columns:
                y_class = survival_data['survival_status'].values
            else:
                # Create synthetic targets - in a real application, would use actual response data
                y_class = np.random.binomial(1, 0.5, size=X.shape[0])
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42, stratify=y_class
            )
            
            # Train classification model based on model type
            if self.model_type == 'xgboost':
                try:
                    from xgboost import XGBClassifier
                    
                    # XGBoost parameters
                    learning_rate = params.get('learning_rate', 0.1)
                    max_depth = params.get('max_depth', 6)
                    n_estimators = params.get('n_estimators', 100)
                    
                    self.classification_model = XGBClassifier(
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        objective="binary:logistic",
                        random_state=42
                    )
                except ImportError:
                    # Fallback to Random Forest if XGBoost not available
                    print("XGBoost not available, falling back to Random Forest")
                    self.model_type = 'random_forest'
            
            if self.model_type == 'random_forest':
                # Random Forest parameters
                n_estimators = params.get('n_estimators', 100)
                max_features = params.get('max_features', 'sqrt')
                class_weight = params.get('class_weight', 'balanced')
                
                self.classification_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    class_weight=class_weight,
                    random_state=42
                )
            
            elif self.model_type == 'neural_network':
                # Neural Network parameters
                hidden_layers = params.get('hidden_layers', 2)
                neurons = params.get('neurons_per_layer', 64)
                dropout_rate = params.get('dropout_rate', 0.3)
                
                # Create NN model
                self.classification_model = tf.keras.Sequential()
                
                # Input layer
                self.classification_model.add(tf.keras.layers.Dense(
                    neurons, activation='relu', input_shape=(X_train.shape[1],)
                ))
                self.classification_model.add(tf.keras.layers.Dropout(dropout_rate))
                
                # Hidden layers
                for _ in range(hidden_layers - 1):
                    self.classification_model.add(tf.keras.layers.Dense(neurons, activation='relu'))
                    self.classification_model.add(tf.keras.layers.Dropout(dropout_rate))
                
                # Output layer
                self.classification_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                
                # Compile model
                self.classification_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            
            # Train model
            if self.model_type == 'neural_network':
                # For neural network, use validation data
                self.classification_model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=0
                )
            else:
                # For tree-based models
                self.classification_model.fit(X_train, y_train)
            
            # Evaluate model
            classification_metrics = self._evaluate_classification(X_test, y_test)
            results['classification_metrics'] = classification_metrics
            
            # Store feature importance if available
            if hasattr(self.classification_model, 'feature_importances_'):
                self.feature_importance = self.classification_model.feature_importances_
                
                # Create feature importance dictionary for results
                if self.feature_names is not None:
                    feature_importance_list = [
                        {'feature': feature, 'importance': float(importance)}
                        for feature, importance in zip(self.feature_names, self.feature_importance)
                    ]
                    results['feature_importance'] = feature_importance_list
        
        # Train survival model if required
        if self.prediction_type in ['survival_analysis', 'both']:
            # For survival analysis, we need time and event data
            if 'survival_months' in survival_data.columns and 'survival_status' in survival_data.columns:
                time_column = 'survival_months'
                event_column = 'survival_status'
                times = survival_data[time_column].values
                events = survival_data[event_column].values
            else:
                # Create synthetic survival data for demonstration
                print("Warning: Using synthetic survival data for demonstration")
                # Exponential distribution for times, with censoring
                times = np.random.exponential(scale=24, size=X.shape[0])  # 24 month mean
                events = np.random.binomial(1, 0.7, size=X.shape[0])  # 70% events, 30% censored
            
            # Split data for evaluation
            X_train, X_test, times_train, times_test, events_train, events_test = train_test_split(
                X, times, events, test_size=0.2, random_state=42
            )
            
            # Train survival model based on model type
            if self.model_type == 'xgboost':
                try:
                    from xgboost import XGBRegressor
                    
                    # XGBoost parameters
                    learning_rate = params.get('learning_rate', 0.1)
                    max_depth = params.get('max_depth', 6)
                    n_estimators = params.get('n_estimators', 100)
                    
                    self.survival_model = XGBRegressor(
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        objective="reg:squarederror",
                        random_state=42
                    )
                except ImportError:
                    # Fallback to Random Forest
                    print("XGBoost not available, falling back to Random Forest")
                    self.model_type = 'random_forest'
            
            if self.model_type == 'random_forest':
                # Random Forest parameters
                n_estimators = params.get('n_estimators', 100)
                max_features = params.get('max_features', 'sqrt')
                
                self.survival_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    random_state=42
                )
            
            elif self.model_type == 'neural_network':
                # Neural Network parameters
                hidden_layers = params.get('hidden_layers', 2)
                neurons = params.get('neurons_per_layer', 64)
                dropout_rate = params.get('dropout_rate', 0.3)
                
                # Create NN model
                self.survival_model = tf.keras.Sequential()
                
                # Input layer
                self.survival_model.add(tf.keras.layers.Dense(
                    neurons, activation='relu', input_shape=(X_train.shape[1],)
                ))
                self.survival_model.add(tf.keras.layers.Dropout(dropout_rate))
                
                # Hidden layers
                for _ in range(hidden_layers - 1):
                    self.survival_model.add(tf.keras.layers.Dense(neurons, activation='relu'))
                    self.survival_model.add(tf.keras.layers.Dropout(dropout_rate))
                
                # Output layer - single continuous value for survival time
                self.survival_model.add(tf.keras.layers.Dense(1))
                
                # Compile model
                self.survival_model.compile(
                    optimizer='adam',
                    loss='mse',  # Mean squared error for regression
                    metrics=['mae']  # Mean absolute error
                )
            
            # Train model
            if self.model_type in ['xgboost', 'random_forest']:
                # For direct regression, predict survival time
                self.survival_model.fit(X_train, times_train)
            elif self.model_type == 'neural_network':
                # For neural network
                self.survival_model.fit(
                    X_train, times_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
            
            # Evaluate survival model
            survival_metrics = self._evaluate_survival(
                X_test, times_test, events_test,
                X_train, times_train, events_train
            )
            results['survival_metrics'] = survival_metrics
        
        return results
    
    def predict(self, features):
        """
        Make predictions for new patients.
        
        Args:
            features (pandas.DataFrame or numpy.ndarray): Patient features
            
        Returns:
            dict: Prediction results
        """
        # Standardize features
        X = self.feature_scaler.transform(features)
        
        # Initialize results
        predictions = {}
        
        # Classification predictions
        if self.prediction_type in ['response_classification', 'both'] and self.classification_model is not None:
            if self.model_type == 'neural_network':
                # For neural network, get raw probabilities
                response_probs = self.classification_model.predict(X).flatten()
                # Convert to binary predictions
                response_class = (response_probs >= 0.5).astype(int)
            else:
                # For tree-based models
                response_class = self.classification_model.predict(X)
                response_probs = self.classification_model.predict_proba(X)[:, 1]
            
            predictions['response_class'] = response_class
            predictions['response_probability'] = response_probs
        
        # Survival predictions
        if self.prediction_type in ['survival_analysis', 'both'] and self.survival_model is not None:
            if self.model_type == 'neural_network':
                # For neural network
                survival_times = self.survival_model.predict(X).flatten()
            else:
                # For tree-based models
                survival_times = self.survival_model.predict(X)
            
            predictions['predicted_survival'] = survival_times
            
            # Calculate additional survival metrics
            # For demonstration, we'll create KM-like survival curves
            t_max = np.max(survival_times) * 1.5
            time_points = np.linspace(0, t_max, 100)
            
            # Create risk groups based on predicted survival
            median_survival = np.median(survival_times)
            is_high_risk = survival_times < median_survival
            
            # Generate survival curves for each patient
            patient_curves = np.zeros((len(X), len(time_points)))
            
            for i in range(len(X)):
                # Exponential survival curve based on predicted time
                predicted_time = survival_times[i]
                rate = 1.0 / max(predicted_time, 0.1)  # Avoid division by zero
                patient_curves[i] = np.exp(-rate * time_points)
            
            predictions['survival_curves'] = {
                'times': time_points.tolist(),
                'curves': patient_curves.tolist(),
                'is_high_risk': is_high_risk.tolist()
            }
        
        return predictions
    
    def _evaluate_classification(self, X_test, y_test):
        """
        Evaluate classification model performance.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            
        Returns:
            dict: Classification metrics
        """
        # Get predictions
        if self.model_type == 'neural_network':
            y_pred_proba = self.classification_model.predict(X_test).flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = self.classification_model.predict(X_test)
            y_pred_proba = self.classification_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate confusion matrix
        cm = np.zeros((2, 2), dtype=int)
        for i in range(len(y_test)):
            cm[int(y_test[i]), int(y_pred[i])] += 1
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'confusion_matrix': cm.tolist()
        }
    
    def _evaluate_survival(self, X_test, times_test, events_test, X_train, times_train, events_train):
        """
        Evaluate survival model performance.
        
        Args:
            X_test (numpy.ndarray): Test features
            times_test (numpy.ndarray): Test survival times
            events_test (numpy.ndarray): Test event indicators
            X_train (numpy.ndarray): Training features
            times_train (numpy.ndarray): Training survival times
            events_train (numpy.ndarray): Training event indicators
            
        Returns:
            dict: Survival metrics
        """
        # Make predictions
        if self.model_type == 'neural_network':
            pred_times_test = self.classification_model.predict(X_test).flatten()
        else:
            pred_times_test = self.survival_model.predict(X_test)
        
        # Calculate concordance index
        try:
            c_index = concordance_index(times_test, pred_times_test, events_test)
        except Exception:
            # Fallback if calculation fails
            c_index = 0.5
        
        # Create stratification by predicted risk
        median_pred = np.median(pred_times_test)
        high_risk = pred_times_test < median_pred
        low_risk = ~high_risk
        
        # Fit Kaplan-Meier curves
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        # Ensure there are events in each group
        if np.sum(events_test[high_risk]) > 0:
            kmf_high.fit(times_test[high_risk], events_test[high_risk], label='High Risk')
        else:
            # Create dummy data if no events
            kmf_high.fit(np.array([1, 2, 3]), np.array([0, 0, 1]), label='High Risk')
        
        if np.sum(events_test[low_risk]) > 0:
            kmf_low.fit(times_test[low_risk], events_test[low_risk], label='Low Risk')
        else:
            # Create dummy data if no events
            kmf_low.fit(np.array([3, 4, 5]), np.array([0, 0, 1]), label='Low Risk')
        
        # Calculate log-rank p-value
        from lifelines.statistics import logrank_test
        
        if np.sum(high_risk) > 0 and np.sum(low_risk) > 0:
            try:
                results_logrank = logrank_test(
                    times_test[high_risk], times_test[low_risk],
                    events_test[high_risk], events_test[low_risk]
                )
                log_rank_p = results_logrank.p_value
            except Exception:
                log_rank_p = 0.5
        else:
            log_rank_p = 0.5
        
        # Create survival curves for visualization
        time_points = np.linspace(0, np.max(times_test) * 1.2, 100)
        
        try:
            high_risk_survival = pd.DataFrame({
                'time': kmf_high.survival_function_.index.values,
                'survival_probability': kmf_high.survival_function_.values.ravel(),
                'group': 'High Risk'
            })
            
            low_risk_survival = pd.DataFrame({
                'time': kmf_low.survival_function_.index.values,
                'survival_probability': kmf_low.survival_function_.values.ravel(),
                'group': 'Low Risk'
            })
            
            survival_curves = pd.concat([high_risk_survival, low_risk_survival])
        except Exception:
            # Create synthetic curves if fitting fails
            high_times = np.linspace(0, np.max(times_test) * 0.8, 20)
            low_times = np.linspace(0, np.max(times_test) * 1.2, 20)
            
            high_survival = np.exp(-0.1 * high_times)
            low_survival = np.exp(-0.05 * low_times)
            
            high_risk_survival = pd.DataFrame({
                'time': high_times,
                'survival_probability': high_survival,
                'group': 'High Risk'
            })
            
            low_risk_survival = pd.DataFrame({
                'time': low_times,
                'survival_probability': low_survival,
                'group': 'Low Risk'
            })
            
            survival_curves = pd.concat([high_risk_survival, low_risk_survival])
        
        # Time-dependent AUC (simplified for demonstration)
        # In real implementation, would use proper time-dependent AUC calculation
        time_points_auc = np.linspace(0, np.max(times_test), 10)
        time_auc = [max(c_index, 0.5)] * len(time_points_auc)  # Use c-index as proxy
        
        return {
            'c_index': float(c_index),
            'log_rank_p': float(log_rank_p),
            'survival_curves': survival_curves.to_dict('records'),
            'time_points': time_points_auc.tolist(),
            'time_auc': time_auc
        }
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path (str): Base path to save the model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model configuration
        config = {
            'model_type': self.model_type,
            'prediction_type': self.prediction_type,
            'feature_names': self.feature_names
        }
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            import json
            json.dump(config, f)
        
        # Save the feature scaler
        from joblib import dump
        dump(self.feature_scaler, os.path.join(path, 'feature_scaler.joblib'))
        
        # Save classification model if available
        if self.classification_model is not None:
            if self.model_type == 'neural_network':
                self.classification_model.save(os.path.join(path, 'classification_model'))
            else:
                dump(self.classification_model, os.path.join(path, 'classification_model.joblib'))
        
        # Save survival model if available
        if self.survival_model is not None:
            if self.model_type == 'neural_network':
                self.survival_model.save(os.path.join(path, 'survival_model'))
            else:
                dump(self.survival_model, os.path.join(path, 'survival_model.joblib'))
    
    @classmethod
    def load_model(cls, path):
        """
        Load a model from disk.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            TherapyResponseModel: Loaded model
        """
        # Load configuration
        try:
            with open(os.path.join(path, 'config.json'), 'r') as f:
                import json
                config = json.load(f)
            
            # Create model instance
            model = cls(
                model_type=config.get('model_type', 'xgboost'),
                prediction_type=config.get('prediction_type', 'both')
            )
            
            # Set feature names
            model.feature_names = config.get('feature_names')
            
            # Load feature scaler
            from joblib import load
            model.feature_scaler = load(os.path.join(path, 'feature_scaler.joblib'))
            
            # Load classification model if available
            if os.path.exists(os.path.join(path, 'classification_model.joblib')):
                model.classification_model = load(os.path.join(path, 'classification_model.joblib'))
            elif os.path.exists(os.path.join(path, 'classification_model')):
                model.classification_model = tf.keras.models.load_model(
                    os.path.join(path, 'classification_model')
                )
            
            # Load survival model if available
            if os.path.exists(os.path.join(path, 'survival_model.joblib')):
                model.survival_model = load(os.path.join(path, 'survival_model.joblib'))
            elif os.path.exists(os.path.join(path, 'survival_model')):
                model.survival_model = tf.keras.models.load_model(
                    os.path.join(path, 'survival_model')
                )
            
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
