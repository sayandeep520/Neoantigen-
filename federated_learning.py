"""
Federated Learning Utilities for Multi-Institutional Collaboration

This module implements federated learning techniques that allow multiple institutions to
collaboratively train machine learning models without sharing raw patient data, addressing
privacy concerns in medical research.

Key components:
1. Federated averaging for model aggregation
2. Secure model update exchange
3. Differential privacy mechanisms
4. Model evaluation across distributed datasets
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.base import BaseEstimator
import pickle
import copy
from datetime import datetime
import hashlib

class FederatedModelManager:
    """
    Manages the federated learning process for multi-institutional collaboration.
    """
    def __init__(
        self, 
        model_type: str = "random_forest", 
        base_model: Optional[BaseEstimator] = None,
        institution_id: str = "local_institution",
        model_dir: str = "./federated_models",
        add_differential_privacy: bool = True,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        communication_rounds: int = 10
    ):
        """
        Initialize the federated learning manager.
        
        Args:
            model_type: Type of model to use ("random_forest", "linear", etc.)
            base_model: Initial model instance (if None, will be created based on model_type)
            institution_id: Identifier for the local institution
            model_dir: Directory to store federated models
            add_differential_privacy: Whether to add differential privacy to model updates
            privacy_epsilon: Epsilon parameter for differential privacy
            privacy_delta: Delta parameter for differential privacy
            communication_rounds: Number of communication rounds for federated training
        """
        self.model_type = model_type
        self.base_model = base_model
        self.institution_id = institution_id
        self.model_dir = model_dir
        self.add_differential_privacy = add_differential_privacy
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        self.communication_rounds = communication_rounds
        self.current_round = 0
        self.model_history = []
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize local model
        if self.base_model is None:
            self.base_model = self._create_model(model_type)
        
        self.current_model = copy.deepcopy(self.base_model)
    
    def _create_model(self, model_type: str) -> BaseEstimator:
        """
        Create a new model instance based on specified type.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            New model instance
        """
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "linear":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "svm":
            from sklearn.svm import SVC
            return SVC(probability=True, random_state=42)
        elif model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_local_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the local model on local data and prepare model update.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Model update information
        """
        # Train model on local data
        self.current_model.fit(X, y)
        
        # Create model update
        model_update = self._create_model_update()
        
        # Add noise for differential privacy if enabled
        if self.add_differential_privacy:
            model_update = self._add_differential_privacy(model_update)
        
        # Save model update
        update_id = self._save_model_update(model_update)
        
        return {
            "update_id": update_id,
            "institution_id": self.institution_id,
            "round": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "parameters": {k: str(type(v)) for k, v in model_update.items()}
        }
    
    def _create_model_update(self) -> Dict[str, Any]:
        """
        Extract model parameters for sharing.
        
        Returns:
            Dictionary of model parameters
        """
        # For scikit-learn models, we can use direct attribute access
        # This is a simplified approach; in practice, this would be more complex
        if hasattr(self.current_model, "feature_importances_"):
            return {"feature_importances": self.current_model.feature_importances_}
        elif hasattr(self.current_model, "coef_"):
            return {"coefficients": self.current_model.coef_[0]}
        
        # Fallback: serialize the entire model
        return {"model": pickle.dumps(self.current_model)}
    
    def _add_differential_privacy(self, model_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add differential privacy noise to model updates.
        
        Args:
            model_update: Model parameters
            
        Returns:
            Model parameters with added noise
        """
        # Simple implementation: add calibrated Gaussian noise
        # In practice, would use a proper DP library
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                # Scale noise based on sensitivity and privacy parameters
                sensitivity = 1.0  # Assuming bounded parameters
                noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.privacy_delta)) / self.privacy_epsilon
                noise = np.random.normal(0, noise_scale, value.shape)
                model_update[key] = value + noise
        
        return model_update
    
    def _save_model_update(self, model_update: Dict[str, Any]) -> str:
        """
        Save model update to file for sharing.
        
        Args:
            model_update: Model parameters
            
        Returns:
            Update identifier
        """
        # Generate unique ID for this update
        update_id = f"{self.institution_id}_{self.current_round}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_update = {}
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                serializable_update[key] = value.tolist()
            elif key == "model":
                # For binary data, use a separate file
                model_file = os.path.join(self.model_dir, f"{update_id}_model.pkl")
                with open(model_file, 'wb') as f:
                    f.write(value)
                serializable_update[key] = f"{update_id}_model.pkl"
            else:
                serializable_update[key] = value
        
        # Save update metadata
        metadata = {
            "update_id": update_id,
            "institution_id": self.institution_id,
            "round": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "parameters": list(model_update.keys())
        }
        
        metadata_file = os.path.join(self.model_dir, f"{update_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save the update
        update_file = os.path.join(self.model_dir, f"{update_id}_update.json")
        with open(update_file, 'w') as f:
            json.dump(serializable_update, f, indent=2)
        
        return update_id
    
    def load_model_updates(self, update_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load model updates from other institutions.
        
        Args:
            update_ids: List of update identifiers
            
        Returns:
            List of model updates
        """
        updates = []
        
        for update_id in update_ids:
            # Load update metadata
            metadata_file = os.path.join(self.model_dir, f"{update_id}_metadata.json")
            update_file = os.path.join(self.model_dir, f"{update_id}_update.json")
            
            if not os.path.exists(metadata_file) or not os.path.exists(update_file):
                print(f"Warning: Update {update_id} not found")
                continue
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load update
            with open(update_file, 'r') as f:
                update_data = json.load(f)
            
            # Convert lists back to numpy arrays
            for key, value in update_data.items():
                if key != "model" and isinstance(value, list):
                    update_data[key] = np.array(value)
                elif key == "model":
                    # Load binary model data
                    model_file = os.path.join(self.model_dir, value)
                    with open(model_file, 'rb') as f:
                        update_data[key] = pickle.load(f)
            
            updates.append({
                "metadata": metadata,
                "update": update_data
            })
        
        return updates
    
    def aggregate_models(self, updates: List[Dict[str, Any]]) -> None:
        """
        Aggregate model updates using federated averaging.
        
        Args:
            updates: List of model updates from different institutions
        """
        if not updates:
            return
        
        # Check if all updates have the same structure
        update_keys = set()
        for update in updates:
            update_keys.update(update["update"].keys())
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Federated averaging
        for key in update_keys:
            # Collect parameters from all updates that have this key
            params = []
            for update in updates:
                if key in update["update"]:
                    params.append(update["update"][key])
            
            if not params:
                continue
            
            # Check if parameters are numpy arrays
            if isinstance(params[0], np.ndarray):
                # Average the arrays
                aggregated_params[key] = np.mean(params, axis=0)
            elif key == "model":
                # For full model objects, select the most recent one
                # This is a simplified approach; in practice, would do proper model averaging
                most_recent = max(updates, key=lambda u: u["metadata"]["timestamp"])
                aggregated_params[key] = most_recent["update"][key]
        
        # Update the current model with aggregated parameters
        self._update_model_with_params(aggregated_params)
        
        # Increment round counter
        self.current_round += 1
        
        # Save the aggregated model
        self._save_aggregated_model()
    
    def _update_model_with_params(self, params: Dict[str, Any]) -> None:
        """
        Update the current model with new parameters.
        
        Args:
            params: New model parameters
        """
        # Update model parameters
        if "feature_importances" in params and hasattr(self.current_model, "feature_importances_"):
            self.current_model.feature_importances_ = params["feature_importances"]
        elif "coefficients" in params and hasattr(self.current_model, "coef_"):
            self.current_model.coef_[0] = params["coefficients"]
        elif "model" in params:
            # Replace the entire model
            self.current_model = copy.deepcopy(params["model"])
    
    def _save_aggregated_model(self) -> None:
        """
        Save the current aggregated model.
        """
        model_file = os.path.join(
            self.model_dir, 
            f"aggregated_model_round_{self.current_round}.pkl"
        )
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.current_model, f)
        
        # Save metadata
        metadata = {
            "round": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "model_type": self.model_type,
            "institution_id": self.institution_id,
            "participating_institutions": list(set(
                [update["metadata"]["institution_id"] for update in self.model_history]
                + [self.institution_id]
            ))
        }
        
        metadata_file = os.path.join(
            self.model_dir, 
            f"aggregated_model_round_{self.current_round}_metadata.json"
        )
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the current model on local validation data.
        
        Args:
            X: Validation features
            y: Validation labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Make predictions
        y_pred = self.current_model.predict(X)
        
        # For ROC AUC, need probability predictions
        try:
            y_proba = self.current_model.predict_proba(X)[:, 1]
            has_proba = True
        except (AttributeError, IndexError):
            has_proba = False
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        if has_proba:
            metrics["roc_auc"] = roc_auc_score(y, y_proba, multi_class='ovr' if len(np.unique(y)) > 2 else 'raise')
        
        return metrics
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available federated models.
        
        Returns:
            List of model metadata
        """
        models = []
        
        # Find all model metadata files
        for filename in os.listdir(self.model_dir):
            if filename.startswith("aggregated_model_round_") and filename.endswith("_metadata.json"):
                metadata_file = os.path.join(self.model_dir, filename)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                models.append(metadata)
        
        # Sort by round
        models.sort(key=lambda m: m["round"])
        
        return models
    
    def load_model(self, round_number: int) -> bool:
        """
        Load a specific aggregated model.
        
        Args:
            round_number: Training round to load
            
        Returns:
            Whether the model was successfully loaded
        """
        model_file = os.path.join(
            self.model_dir, 
            f"aggregated_model_round_{round_number}.pkl"
        )
        
        if not os.path.exists(model_file):
            return False
        
        try:
            with open(model_file, 'rb') as f:
                self.current_model = pickle.load(f)
            
            self.current_round = round_number
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class FederatedDataSimulator:
    """
    Simulates federated datasets for testing multi-institutional collaboration.
    
    This class creates synthetic datasets representing data from multiple 
    institutions, with varying distributions and features to test federated learning.
    """
    def __init__(
        self, 
        num_institutions: int = 3,
        num_samples_per_institution: int = 500,
        num_features: int = 20,
        num_classes: int = 2,
        feature_overlap: float = 0.7,
        distribution_shift: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize the federated data simulator.
        
        Args:
            num_institutions: Number of institutions to simulate
            num_samples_per_institution: Samples per institution
            num_features: Number of features in the dataset
            num_classes: Number of target classes
            feature_overlap: Fraction of shared features across institutions
            distribution_shift: Amount of distribution shift between institutions
            random_seed: Random seed for reproducibility
        """
        self.num_institutions = num_institutions
        self.num_samples_per_institution = num_samples_per_institution
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_overlap = feature_overlap
        self.distribution_shift = distribution_shift
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Generate institution-specific data
        self.institution_data = {}
        self._generate_data()
    
    def _generate_data(self) -> None:
        """
        Generate synthetic data for each institution.
        """
        # Generate common features shared across institutions
        num_shared_features = int(self.num_features * self.feature_overlap)
        shared_feature_means = np.random.normal(0, 1, num_shared_features)
        shared_feature_stds = np.random.uniform(0.5, 1.5, num_shared_features)
        
        # Define class centers (for classification problems)
        class_centers = np.random.normal(0, 2, (self.num_classes, self.num_features))
        
        for i in range(self.num_institutions):
            # Create institution-specific distribution shift
            institution_shift = np.random.normal(0, self.distribution_shift, self.num_features)
            
            # Generate features
            X = np.zeros((self.num_samples_per_institution, self.num_features))
            
            # Generate shared features
            for j in range(num_shared_features):
                X[:, j] = np.random.normal(
                    shared_feature_means[j] + institution_shift[j],
                    shared_feature_stds[j],
                    self.num_samples_per_institution
                )
            
            # Generate institution-specific features
            for j in range(num_shared_features, self.num_features):
                X[:, j] = np.random.normal(
                    institution_shift[j],
                    np.random.uniform(0.5, 1.5),
                    self.num_samples_per_institution
                )
            
            # Generate labels
            # First, assign samples to classes randomly
            y = np.random.randint(0, self.num_classes, self.num_samples_per_institution)
            
            # Then adjust features based on class labels (make them more separable)
            for sample_idx in range(self.num_samples_per_institution):
                class_idx = y[sample_idx]
                # Move sample closer to its class center
                X[sample_idx, :] = X[sample_idx, :] * 0.3 + class_centers[class_idx, :] * 0.7
            
            # Split into train/test
            train_size = int(0.8 * self.num_samples_per_institution)
            
            indices = np.random.permutation(self.num_samples_per_institution)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            # Generate column names
            feature_names = [f"feature_{j}" for j in range(self.num_features)]
            
            # Create DataFrames
            train_df = pd.DataFrame(X[train_indices, :], columns=feature_names)
            train_df['target'] = y[train_indices]
            
            test_df = pd.DataFrame(X[test_indices, :], columns=feature_names)
            test_df['target'] = y[test_indices]
            
            # Add metadata
            metadata = {
                "institution_id": f"institution_{i}",
                "num_samples": self.num_samples_per_institution,
                "num_features": self.num_features,
                "num_classes": self.num_classes,
                "shared_features": feature_names[:num_shared_features],
                "private_features": feature_names[num_shared_features:],
                "distribution_shift": float(np.mean(np.abs(institution_shift)))
            }
            
            # Store data for this institution
            self.institution_data[f"institution_{i}"] = {
                "train": train_df,
                "test": test_df,
                "metadata": metadata
            }
    
    def get_institution_data(self, institution_id: str) -> Dict[str, Any]:
        """
        Get data for a specific institution.
        
        Args:
            institution_id: Institution identifier
            
        Returns:
            Dictionary with train/test data and metadata
        """
        if institution_id not in self.institution_data:
            raise ValueError(f"Institution {institution_id} not found")
        
        return self.institution_data[institution_id]
    
    def get_all_institutions(self) -> List[str]:
        """
        Get a list of all institution IDs.
        
        Returns:
            List of institution identifiers
        """
        return list(self.institution_data.keys())
    
    def save_data(self, output_dir: str) -> None:
        """
        Save all institution data to disk.
        
        Args:
            output_dir: Directory to save data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for institution_id, data in self.institution_data.items():
            # Create institution directory
            institution_dir = os.path.join(output_dir, institution_id)
            os.makedirs(institution_dir, exist_ok=True)
            
            # Save train/test data
            data["train"].to_csv(os.path.join(institution_dir, "train.csv"), index=False)
            data["test"].to_csv(os.path.join(institution_dir, "test.csv"), index=False)
            
            # Save metadata
            with open(os.path.join(institution_dir, "metadata.json"), 'w') as f:
                json.dump(data["metadata"], f, indent=2)
    
    def load_data(self, input_dir: str) -> None:
        """
        Load institution data from disk.
        
        Args:
            input_dir: Directory to load data from
        """
        self.institution_data = {}
        
        # Find all institution directories
        for item in os.listdir(input_dir):
            institution_dir = os.path.join(input_dir, item)
            
            if not os.path.isdir(institution_dir):
                continue
            
            # Check if this is an institution directory
            if not all(os.path.exists(os.path.join(institution_dir, f)) for f in ["train.csv", "test.csv", "metadata.json"]):
                continue
            
            # Load data
            train_df = pd.read_csv(os.path.join(institution_dir, "train.csv"))
            test_df = pd.read_csv(os.path.join(institution_dir, "test.csv"))
            
            with open(os.path.join(institution_dir, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            self.institution_data[item] = {
                "train": train_df,
                "test": test_df,
                "metadata": metadata
            }


def run_federated_learning_simulation(
    num_institutions: int = 3,
    num_samples: int = 500,
    num_features: int = 20,
    num_classes: int = 2,
    feature_overlap: float = 0.7,
    distribution_shift: float = 0.2,
    communication_rounds: int = 5,
    use_differential_privacy: bool = True,
    model_type: str = "random_forest",
    output_dir: str = "./federated_results"
) -> Dict[str, Any]:
    """
    Run a complete federated learning simulation.
    
    Args:
        num_institutions: Number of institutions to simulate
        num_samples: Samples per institution
        num_features: Number of features in the dataset
        num_classes: Number of target classes
        feature_overlap: Fraction of shared features across institutions
        distribution_shift: Amount of distribution shift between institutions
        communication_rounds: Number of federated training rounds
        use_differential_privacy: Whether to use differential privacy
        model_type: Type of model to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary with simulation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data simulator
    simulator = FederatedDataSimulator(
        num_institutions=num_institutions,
        num_samples_per_institution=num_samples,
        num_features=num_features,
        num_classes=num_classes,
        feature_overlap=feature_overlap,
        distribution_shift=distribution_shift
    )
    
    # Save simulated data
    data_dir = os.path.join(output_dir, "data")
    simulator.save_data(data_dir)
    
    # Initialize federated learning managers for each institution
    federated_managers = {}
    for institution_id in simulator.get_all_institutions():
        model_dir = os.path.join(output_dir, "models", institution_id)
        os.makedirs(model_dir, exist_ok=True)
        
        federated_managers[institution_id] = FederatedModelManager(
            model_type=model_type,
            institution_id=institution_id,
            model_dir=model_dir,
            add_differential_privacy=use_differential_privacy,
            communication_rounds=communication_rounds
        )
    
    # Track metrics for each round
    all_metrics = {institution_id: [] for institution_id in simulator.get_all_institutions()}
    
    # Conduct federated learning
    for round_idx in range(communication_rounds):
        print(f"Starting round {round_idx + 1}/{communication_rounds}")
        
        # Each institution trains its local model and prepares updates
        updates = {}
        for institution_id, manager in federated_managers.items():
            # Get institution data
            institution_data = simulator.get_institution_data(institution_id)
            X_train = institution_data["train"].drop(columns=["target"]).values
            y_train = institution_data["train"]["target"].values
            
            # Train local model
            update_info = manager.train_local_model(X_train, y_train)
            updates[institution_id] = update_info
            
            # Evaluate model on local test data
            X_test = institution_data["test"].drop(columns=["target"]).values
            y_test = institution_data["test"]["target"].values
            metrics = manager.evaluate_model(X_test, y_test)
            
            # Track metrics
            all_metrics[institution_id].append({
                "round": round_idx,
                "local_only": True,
                **metrics
            })
            
            print(f"  {institution_id} local model accuracy: {metrics['accuracy']:.4f}")
        
        # Share updates between institutions
        for institution_id, manager in federated_managers.items():
            # Collect updates from other institutions
            other_updates = [info for other_id, info in updates.items() if other_id != institution_id]
            update_ids = [info["update_id"] for info in other_updates]
            
            # Load and aggregate updates
            loaded_updates = manager.load_model_updates(update_ids)
            manager.aggregate_models(loaded_updates)
            
            # Evaluate federated model
            institution_data = simulator.get_institution_data(institution_id)
            X_test = institution_data["test"].drop(columns=["target"]).values
            y_test = institution_data["test"]["target"].values
            metrics = manager.evaluate_model(X_test, y_test)
            
            # Track metrics
            all_metrics[institution_id].append({
                "round": round_idx,
                "local_only": False,
                **metrics
            })
            
            print(f"  {institution_id} federated model accuracy: {metrics['accuracy']:.4f}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Compute final performance comparison
    final_comparison = {}
    
    # For each institution, compare local-only vs federated performance
    for institution_id, metrics_list in all_metrics.items():
        local_metrics = [m for m in metrics_list if m["local_only"]]
        federated_metrics = [m for m in metrics_list if not m["local_only"]]
        
        if local_metrics and federated_metrics:
            final_local = local_metrics[-1]
            final_federated = federated_metrics[-1]
            
            improvement = {
                metric: final_federated[metric] - final_local[metric]
                for metric in final_local.keys()
                if metric not in ["round", "local_only"]
            }
            
            final_comparison[institution_id] = {
                "local_final": final_local,
                "federated_final": final_federated,
                "improvement": improvement
            }
    
    # Save final comparison
    comparison_file = os.path.join(output_dir, "final_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(final_comparison, f, indent=2)
    
    return {
        "metrics": all_metrics,
        "final_comparison": final_comparison
    }


if __name__ == "__main__":
    # Example usage
    results = run_federated_learning_simulation(
        num_institutions=3,
        num_samples=500,
        num_features=20,
        num_classes=2,
        communication_rounds=3,
        output_dir="./federated_results_example"
    )
    
    # Print summary
    for institution_id, comparison in results["final_comparison"].items():
        print(f"{institution_id}:")
        print(f"  Local accuracy: {comparison['local_final']['accuracy']:.4f}")
        print(f"  Federated accuracy: {comparison['federated_final']['accuracy']:.4f}")
        print(f"  Improvement: {comparison['improvement']['accuracy']:.4f}")
        print()