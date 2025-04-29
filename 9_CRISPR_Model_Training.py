"""
CRISPR Model Training

This module provides a Streamlit interface for training and optimizing 
machine learning models for CRISPR target prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import matplotlib.pyplot as plt
from io import BytesIO

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from models.crispr_trainer import CRISPRModelTrainer, DATA_PATH, MODEL_DIR, OUTPUT_PATH

# Page configuration
st.set_page_config(
    page_title="CRISPR Model Training",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 CRISPR Target Optimization Model Training")
st.markdown("""
This page allows you to train and optimize machine learning models for CRISPR guide RNA efficiency prediction. 
Upload your own CRISPR activity data or use the provided example dataset to train models.

The trained models can then be used to predict the efficiency of new guide RNAs for your target genes.
""")

# Create sidebar for settings
st.sidebar.header("Training Settings")

# Main content area
st.markdown("## Data Selection")

# Option to upload own data or use example data
data_option = st.radio(
    "Select data source:",
    ["Use example data", "Upload custom data"]
)

df = None
trainer = CRISPRModelTrainer()

if data_option == "Upload custom data":
    uploaded_file = st.file_uploader("Upload CRISPR training data (CSV format)", type="csv")
    if uploaded_file is not None:
        try:
            df = trainer.load_data(uploaded_file=uploaded_file)
            st.success(f"Successfully loaded data with {df.shape[0]} samples and {df.shape[1]} features.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
else:
    # Use example data
    try:
        df = trainer.load_data(file_path=DATA_PATH)
        st.success(f"Using example data with {df.shape[0]} samples and {df.shape[1]} features.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading example data: {str(e)}")
        st.info("Please make sure example data exists at data/crispr_data.csv or upload your own data.")

# Data preprocessing settings (if data loaded successfully)
if df is not None:
    st.markdown("## Feature & Target Selection")
    
    # Select target column
    target_col = st.selectbox(
        "Select target variable (efficiency/activity):",
        options=[col for col in df.columns if col.lower() in ['activity', 'efficiency', 'on_target_efficiency', 'target_efficiency', 'score']],
        index=0
    )
    
    # Select features (multi-select)
    st.markdown("### Select features for training")
    all_features = list(df.columns)
    default_features = [col for col in all_features if col != target_col]
    
    features = st.multiselect(
        "Select features to include in the model:",
        options=all_features,
        default=default_features
    )
    
    # Train-test split settings
    st.markdown("### Data Split Settings")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    with col2:
        val_size = st.slider("Validation set size (% of training)", 10, 40, 25, 5) / 100
        
    random_state = st.sidebar.number_input("Random seed", 0, 1000, 42)
    
    # Advanced settings in sidebar
    st.sidebar.markdown("### Model Selection")
    include_rf = st.sidebar.checkbox("Include Random Forest", True)
    include_gb = st.sidebar.checkbox("Include Gradient Boosting", True)
    include_nn = st.sidebar.checkbox("Include Neural Network", True)
    
    # Feature preprocessing
    st.sidebar.markdown("### Feature Preprocessing")
    
    # Determine feature types
    numeric_features = df[features].select_dtypes(include=['float', 'int']).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Find potential sequence features
    sequence_candidates = [col for col in features if 'sequence' in col.lower() or 'seq' in col.lower()]
    
    # Let user select sequence feature
    sequence_feature = None
    if sequence_candidates:
        sequence_feature = st.sidebar.selectbox(
            "Select DNA/RNA sequence feature (optional):",
            options=[None] + sequence_candidates,
            index=0
        )
    
    # Train models button
    if st.button("Train Models"):
        with st.spinner("Preprocessing data..."):
            # Preprocess data
            X_train, X_test, y_train, y_test, X_val, y_val = trainer.preprocess_data(
                features=features,
                target=target_col,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state
            )
            
            st.info(f"Data split into {X_train.shape[0]} training, {X_val.shape[0]} validation, and {X_test.shape[0]} test samples.")
            
            # Build feature preprocessor
            trainer.build_feature_preprocessor(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                sequence_feature=sequence_feature
            )
            
            # Build models
            trainer.build_models(
                include_rf=include_rf,
                include_gb=include_gb,
                include_nn=include_nn
            )
        
        with st.spinner("Training models... This may take a moment."):
            # Train models
            trainer.train_models()
            
            # Evaluate models
            results = trainer.evaluate_models()
            
            # Show results
            st.success("Training complete! Model evaluation results:")
            
            # Create tabs for results
            model_tabs = st.tabs([name for name in results.keys()])
            
            for i, (name, result) in enumerate(results.items()):
                with model_tabs[i]:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"### {name} Performance Metrics")
                        report = result['report']
                        
                        # Extract and format class metrics
                        class_metrics = {k: v for k, v in report.items() 
                                        if k not in ['accuracy', 'macro avg', 'weighted avg']}
                        
                        # Format overall metrics
                        st.metric("Accuracy", f"{report['accuracy']:.4f}")
                        
                        if result['roc_auc'] is not None:
                            st.metric("ROC AUC", f"{result['roc_auc']:.4f}")
                        
                        # Show class metrics (for classification)
                        st.markdown("#### Class-wise Performance")
                        class_df = pd.DataFrame(class_metrics).T
                        st.dataframe(class_df.style.format("{:.4f}"))
                        
                    with col2:
                        st.markdown("### Visualizations")
                        
                        # For ROC curve (if applicable)
                        if result['roc_auc'] is not None:
                            fig, ax = plt.subplots()
                            from sklearn.metrics import roc_curve
                            fpr, tpr, _ = roc_curve(trainer.y_test, result['probabilities'])
                            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {result['roc_auc']:.3f})")
                            ax.plot([0, 1], [0, 1], 'k--')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve')
                            ax.legend(loc='best')
                            st.pyplot(fig)
                        
                        # Feature importance (if supported)
                        model = result['model'].named_steps['classifier']
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("#### Feature Importance")
                            
                            feature_importance = trainer.plot_feature_importance(name)
                            if feature_importance is not None:
                                st.pyplot(feature_importance)
            
            # Final metrics comparison
            st.markdown("## Model Comparison")
            metrics_data = {}
            
            for name, result in results.items():
                metrics_data[name] = {
                    'Accuracy': result['report']['accuracy'],
                    'F1-Score (macro)': result['report']['macro avg']['f1-score'],
                    'Precision (macro)': result['report']['macro avg']['precision'],
                    'Recall (macro)': result['report']['macro avg']['recall']
                }
                
                if result['roc_auc'] is not None:
                    metrics_data[name]['ROC AUC'] = result['roc_auc']
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(metrics_data).T
            
            # Highlight the best model for each metric
            st.dataframe(metrics_df.style.highlight_max(axis=0).format("{:.4f}"))
            
            # Save best model
            st.markdown("## Save Best Model")
            
            metric_options = ['f1-score', 'precision', 'recall', 'accuracy']
            if any([results[name]['roc_auc'] is not None for name in results.keys()]):
                metric_options.append('roc_auc')
                
            save_metric = st.selectbox(
                "Select metric to determine best model:",
                options=metric_options,
                index=0
            )
            
            if st.button("Save Best Model"):
                with st.spinner("Saving best model..."):
                    try:
                        saved_path = trainer.save_best_model(metric=save_metric)
                        st.success(f"Best model saved to {saved_path}")
                        
                        # Add download button
                        with open(saved_path, 'rb') as f:
                            model_bytes = f.read()
                            
                        st.download_button(
                            label="Download Model",
                            data=model_bytes,
                            file_name="crispr_model.pkl",
                            mime="application/octet-stream"
                        )
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
else:
    st.info("Please load data to continue.")

# Add information about how to use the trained model
st.markdown("---")
st.markdown("## How to Use Trained Models")
st.markdown("""
Once you've trained and saved your model, you can use it for predicting CRISPR guide RNA efficiency in two ways:

1. **Direct Integration**: The trained model is automatically saved and can be used directly from the CRISPR Target Optimization page.

2. **Custom Usage**: Download the model file and load it in your own code:

```python
import pickle

# Load the model
with open('crispr_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Prepare your data (must match training format)
# ...

# Make predictions
predictions = model.predict(your_data)
```

For best results, ensure your new data has the same features and format as the training data.
""")