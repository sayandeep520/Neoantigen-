"""
Test script for federated learning module

This script runs a simple test of the federated learning module to verify functionality.
It creates simulated data, trains models, and evaluates performance.
"""

import os
import numpy as np
import pandas as pd
from utils.federated_learning import (
    FederatedModelManager, 
    FederatedDataSimulator,
    run_federated_learning_simulation
)

def test_federated_data_simulator():
    """Test the federated data simulator"""
    print("\n=== Testing FederatedDataSimulator ===")
    
    # Create simulator
    simulator = FederatedDataSimulator(
        num_institutions=3,
        num_samples_per_institution=100,
        num_features=10,
        num_classes=2,
        feature_overlap=0.7,
        distribution_shift=0.2,
        random_seed=42
    )
    
    # Check institutions
    institutions = simulator.get_all_institutions()
    print(f"Generated {len(institutions)} institutions: {institutions}")
    
    # Check data for an institution
    institution_data = simulator.get_institution_data(institutions[0])
    print(f"Institution {institutions[0]} train data shape: {institution_data['train'].shape}")
    print(f"Institution {institutions[0]} test data shape: {institution_data['test'].shape}")
    
    # Check metadata
    print(f"Institution {institutions[0]} metadata: {institution_data['metadata']}")
    
    # Test save/load
    output_dir = "./test_federated_data"
    os.makedirs(output_dir, exist_ok=True)
    
    simulator.save_data(output_dir)
    print(f"Saved data to {output_dir}")
    
    # Create a new simulator and load data
    new_simulator = FederatedDataSimulator()
    new_simulator.load_data(output_dir)
    
    new_institutions = new_simulator.get_all_institutions()
    print(f"Loaded {len(new_institutions)} institutions: {new_institutions}")
    
    return simulator

def test_federated_model_manager(simulator):
    """Test the federated model manager"""
    print("\n=== Testing FederatedModelManager ===")
    
    institutions = simulator.get_all_institutions()
    
    # Create managers for each institution
    managers = {}
    for institution_id in institutions:
        managers[institution_id] = FederatedModelManager(
            model_type="random_forest",
            institution_id=institution_id,
            model_dir=f"./test_federated_models/{institution_id}",
            add_differential_privacy=True,
            privacy_epsilon=1.0,
            communication_rounds=3
        )
    
    # Train local models
    updates = {}
    for institution_id, manager in managers.items():
        institution_data = simulator.get_institution_data(institution_id)
        X_train = institution_data["train"].drop(columns=["target"]).values
        y_train = institution_data["train"]["target"].values
        
        update_info = manager.train_local_model(X_train, y_train)
        updates[institution_id] = update_info
        
        print(f"Trained model for {institution_id}, update ID: {update_info['update_id']}")
    
    # Evaluate local models
    for institution_id, manager in managers.items():
        institution_data = simulator.get_institution_data(institution_id)
        X_test = institution_data["test"].drop(columns=["target"]).values
        y_test = institution_data["test"]["target"].values
        
        metrics = manager.evaluate_model(X_test, y_test)
        print(f"{institution_id} local model accuracy: {metrics['accuracy']:.4f}")
    
    # Federated learning
    for institution_id, manager in managers.items():
        # Get updates from other institutions
        other_updates = [info for other_id, info in updates.items() if other_id != institution_id]
        update_ids = [info["update_id"] for info in other_updates]
        
        loaded_updates = manager.load_model_updates(update_ids)
        manager.aggregate_models(loaded_updates)
        
        # Evaluate federated model
        institution_data = simulator.get_institution_data(institution_id)
        X_test = institution_data["test"].drop(columns=["target"]).values
        y_test = institution_data["test"]["target"].values
        
        metrics = manager.evaluate_model(X_test, y_test)
        print(f"{institution_id} federated model accuracy: {metrics['accuracy']:.4f}")
    
    return managers

def test_full_simulation():
    """Test the full simulation function"""
    print("\n=== Testing Full Simulation ===")
    
    results = run_federated_learning_simulation(
        num_institutions=3,
        num_samples=100,
        num_features=10,
        num_classes=2,
        feature_overlap=0.7,
        distribution_shift=0.2,
        communication_rounds=2,
        use_differential_privacy=True,
        model_type="random_forest",
        output_dir="./test_federated_results"
    )
    
    # Print summary
    print("\nFinal comparison:")
    for institution_id, comparison in results["final_comparison"].items():
        print(f"{institution_id}:")
        print(f"  Local accuracy: {comparison['local_final']['accuracy']:.4f}")
        print(f"  Federated accuracy: {comparison['federated_final']['accuracy']:.4f}")
        print(f"  Improvement: {comparison['improvement']['accuracy']:.4f}")

if __name__ == "__main__":
    # Run tests
    simulator = test_federated_data_simulator()
    managers = test_federated_model_manager(simulator)
    test_full_simulation()
    
    print("\n=== All tests completed ===")