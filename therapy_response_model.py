import numpy as np
import pandas as pd
import os
import time
import joblib  # For model saving/loading

from typing import Dict, Union, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# TensorFlow support (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Deep learning models will be disabled.")


class TherapyResponsePredictor:
    """
    Predict patient response to AI-driven CRISPR-based cancer immunotherapy.
    Supports ML models and deep learning (if TensorFlow is installed).
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the therapy response model.

        Args:
            model_type (str): Choose from ['random_forest', 'gradient_boosting', 'xgboost', 'deep_learning']
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.scaler = StandardScaler()

    def _initialize_model(self):
        """Initializes the selected model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
        elif self.model_type == "deep_learning" and TENSORFLOW_AVAILABLE:
            return self._build_deep_learning_model()
        else:
            raise ValueError("Invalid model_type. Choose from ['random_forest', 'gradient_boosting', 'xgboost', 'deep_learning'].")

    def _build_deep_learning_model(self):
        """Builds a simple deep learning model using TensorFlow."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(None,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train the selected model."""
        # Handle NaN values before splitting
        X = np.nan_to_num(X, nan=0.0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if self.model_type == "deep_learning" and TENSORFLOW_AVAILABLE:
            self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        else:
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            return self._evaluate_model(y_test, predictions)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def _evaluate_model(self, y_true, y_pred):
        """Evaluate model performance."""
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        
        print(f"Model Performance:\nAccuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nROC-AUC: {roc_auc:.4f}")
        return results

    def save_model(self, path: str):
        """Save the trained model."""
        joblib.dump((self.model, self.scaler), path)
        print(f"Model saved at {path}")

    def load_model(self, path: str):
        """Load a saved model."""
        self.model, self.scaler = joblib.load(path)
        print(f"Model loaded from {path}")