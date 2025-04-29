import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from models.therapy_response_model import TherapyResponsePredictor
from utils.model_trainer import train_response_model, evaluate_model
from utils.visualization import plot_feature_importance, plot_survival_curves, plot_confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Therapy Response Prediction | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Session state check for required data
if 'preprocessed_data' not in st.session_state or not st.session_state['preprocessed_data']:
    st.warning("No preprocessed data available. Please complete data preprocessing first.")
    preprocess_button = st.button("Go to Data Preprocessing")
    if preprocess_button:
        st.switch_page("pages/2_Data_Preprocessing.py")
    st.stop()

# Initialize session states for this page
if 'response_model' not in st.session_state:
    st.session_state['response_model'] = None
if 'response_predictions' not in st.session_state:
    st.session_state['response_predictions'] = None

# Main header
st.title("🔮 Therapy Response Prediction")
st.markdown("""
This module uses multi-omics data to predict patient response to the engineered CRISPR-based cancer immunotherapy.
The AI model integrates genomic, transcriptomic, and clinical features to forecast treatment outcomes.
""")

# Check for available data types
available_data_types = list(st.session_state['preprocessed_data'].keys())
required_data = ["tcga"]
missing_data = [data for data in required_data if data not in available_data_types]

if missing_data:
    st.warning(f"Missing required data: {', '.join(missing_data)}. Some features may not work properly.")

# Model configuration
st.header("Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Select Prediction Model",
        options=["XGBoost", "Random Forest", "Neural Network", "Ensemble"],
        index=0,
        help="Machine learning algorithm for response prediction"
    )
    
    prediction_type = st.radio(
        "Prediction Type",
        options=["Response Classification", "Survival Analysis", "Both"],
        index=2,
        help="Type of prediction to perform"
    )

with col2:
    st.subheader("Feature Selection")
    
    include_genomic = st.checkbox("Include Genomic Features", value=True, 
                                 help="Mutation and copy number data")
    include_transcriptomic = st.checkbox("Include Transcriptomic Features", value=True,
                                        help="Gene expression data")
    include_immune = st.checkbox("Include Immune Features", value=True,
                                help="Immune cell infiltration and cytokine levels")
    include_clinical = st.checkbox("Include Clinical Features", value=True,
                                  help="Patient age, sex, stage, etc.")

# Advanced model settings
with st.expander("Advanced Model Settings"):
    col1, col2 = st.columns(2)
    
    with col1:
        if model_type == "XGBoost":
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                help="Step size shrinkage to prevent overfitting"
            )
            
            max_depth = st.slider(
                "Maximum Tree Depth",
                min_value=3,
                max_value=15,
                value=6,
                step=1,
                help="Maximum depth of decision trees"
            )
        
        elif model_type == "Random Forest":
            n_estimators = st.slider(
                "Number of Trees",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of decision trees in the forest"
            )
            
            max_features = st.selectbox(
                "Feature Selection Strategy",
                options=["sqrt", "log2", "None"],
                index=0,
                help="Method to select features for each tree"
            )
        
        elif model_type == "Neural Network":
            layers = st.slider(
                "Hidden Layers",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                help="Number of hidden layers in the neural network"
            )
            
            neurons = st.slider(
                "Neurons per Layer",
                min_value=16,
                max_value=256,
                value=64,
                step=16,
                help="Number of neurons in each hidden layer"
            )
    
    with col2:
        validation_strategy = st.selectbox(
            "Validation Strategy",
            options=["5-Fold Cross-Validation", "10-Fold Cross-Validation", "Hold-out Validation"],
            index=0,
            help="Method to validate model performance"
        )
        
        class_weight = st.selectbox(
            "Class Weighting",
            options=["Balanced", "None"],
            index=0,
            help="Adjust weights inversely proportional to class frequencies"
        )
        
        feature_selection = st.checkbox(
            "Perform Automated Feature Selection",
            value=True,
            help="Automatically select most informative features"
        )
        
        if feature_selection:
            n_features = st.slider(
                "Number of Features to Select",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Number of top features to include in the model"
            )

# Model training section
st.header("Model Training")

# Prepare feature data
if st.button("Train Response Prediction Model"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Prepare training data
    status_text.text("Preparing multi-omics data for model training...")
    
    # Combine available data sources
    training_features = []
    target_variable = None
    
    # TCGA genomic and clinical data
    if 'tcga' in st.session_state['preprocessed_data']:
        tcga_data = st.session_state['preprocessed_data']['tcga']
        
        if include_genomic:
            status_text.text("Extracting genomic features...")
            # Extract genomic features (assuming they exist in the data)
            if 'TMB' in tcga_data.columns:
                training_features.append(tcga_data[['TMB']].copy())
            
            # Extract mutation features (simplifying for demonstration)
            mutation_cols = [col for col in tcga_data.columns if 'mutation' in col.lower()]
            if mutation_cols:
                training_features.append(tcga_data[mutation_cols].copy())
        
        if include_clinical:
            status_text.text("Extracting clinical features...")
            # Extract clinical features if available
            clinical_cols = [col for col in tcga_data.columns if col.lower() in ['age', 'gender', 'stage', 'grade']]
            if clinical_cols:
                training_features.append(tcga_data[clinical_cols].copy())
        
        # Extract target variable (response or survival)
        if 'survival_months' in tcga_data.columns:
            survival_data = tcga_data[['survival_months', 'survival_status']].copy()
        else:
            # Create synthetic survival data for demonstration
            st.warning("Survival data not found in TCGA data. Using synthetic data for demonstration.")
            survival_data = pd.DataFrame({
                'survival_months': np.random.exponential(24, size=len(tcga_data)),
                'survival_status': np.random.binomial(1, 0.5, size=len(tcga_data))
            })
    else:
        st.error("TCGA data not available. Cannot train response prediction model.")
        st.stop()
    
    # Transcriptomic data
    if include_transcriptomic and 'gtex' in st.session_state['preprocessed_data']:
        status_text.text("Extracting transcriptomic features...")
        gtex_data = st.session_state['preprocessed_data']['gtex']
        
        # Select gene expression features (simplifying for demonstration)
        expression_cols = [col for col in gtex_data.columns if col.startswith('ENSG') or col.lower().startswith('gene_')]
        if expression_cols:
            # Take a subset of expression features to avoid high dimensionality
            selected_expression_cols = expression_cols[:min(50, len(expression_cols))]
            training_features.append(gtex_data[selected_expression_cols].copy())
    
    # Immune features
    if include_immune:
        status_text.text("Extracting immune features...")
        # Check if immune features exist in any dataset
        immune_features = None
        
        for data_type, data in st.session_state['preprocessed_data'].items():
            immune_cols = [col for col in data.columns if any(term in col.lower() for term in ['immune', 'infiltration', 'cytokine', 'tcell', 'bcell'])]
            if immune_cols:
                immune_features = data[immune_cols].copy()
                break
        
        if immune_features is not None:
            training_features.append(immune_features)
        else:
            st.warning("No immune features found in available data.")
    
    # Progress update
    progress_bar.progress(0.3)
    
    # Step 2: Combine all features
    status_text.text("Combining all features for model training...")
    
    if not training_features:
        st.error("No features available for model training.")
        st.stop()
    
    # Combine all feature dataframes
    # Note: In a real implementation, this would require careful alignment of samples
    combined_features = pd.concat(training_features, axis=1)
    
    # Progress update
    progress_bar.progress(0.4)
    
    # Step 3: Initialize and train model
    status_text.text(f"Training {model_type} model for therapy response prediction...")
    
    # Set up model parameters based on user selections
    model_params = {}
    
    if model_type == "XGBoost":
        model_params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": 100,
            "objective": "binary:logistic" if "Classification" in prediction_type else "survival:cox"
        }
    elif model_type == "Random Forest":
        model_params = {
            "n_estimators": n_estimators,
            "max_features": None if max_features == "None" else max_features,
            "class_weight": "balanced" if class_weight == "Balanced" else None
        }
    elif model_type == "Neural Network":
        model_params = {
            "hidden_layers": layers,
            "neurons_per_layer": neurons,
            "activation": "relu",
            "dropout_rate": 0.3
        }
    elif model_type == "Ensemble":
        model_params = {
            "base_models": ["xgboost", "random_forest", "neural_network"],
            "meta_learner": "logistic"
        }
    
    # Add validation strategy
    if "5-Fold" in validation_strategy:
        model_params["cv_folds"] = 5
    elif "10-Fold" in validation_strategy:
        model_params["cv_folds"] = 10
    else:
        model_params["test_size"] = 0.2
    
    # Add feature selection if enabled
    if feature_selection:
        model_params["feature_selection"] = True
        model_params["n_features"] = n_features
    
    # Train the model
    model, results = train_response_model(
        combined_features,
        survival_data,
        model_type=model_type.lower().replace(" ", "_"),
        prediction_type=prediction_type.lower().replace(" ", "_"),
        params=model_params
    )
    
    # Store model and results
    st.session_state['response_model'] = model
    st.session_state['response_predictions'] = results
    
    # Progress update
    progress_bar.progress(0.8)
    
    # Step 4: Evaluate model performance
    status_text.text("Evaluating model performance...")
    
    # Evaluate model (this would be done automatically in the train_response_model function)
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Response prediction model trained successfully!")

# Display model results if available
if st.session_state['response_predictions'] is not None:
    st.header("Model Performance")
    
    # Extract results based on prediction type
    results = st.session_state['response_predictions']
    
    # Create tabs for different result types
    if "Both" in prediction_type:
        result_tabs = st.tabs(["Response Classification", "Survival Analysis"])
        
        with result_tabs[0]:
            st.subheader("Response Classification Results")
            
            # Display classification metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{results['classification_metrics']['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{results['classification_metrics']['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{results['classification_metrics']['recall']:.3f}")
            with col4:
                st.metric("F1 Score", f"{results['classification_metrics']['f1']:.3f}")
            
            # ROC Curve
            st.write("ROC Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['classification_metrics']['fpr'],
                y=results['classification_metrics']['tpr'],
                mode='lines',
                name=f'ROC (AUC = {results["classification_metrics"]["auc"]:.3f})'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random'
            ))
            fig.update_layout(
                title='Receiver Operating Characteristic',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700,
                height=500
            )
            st.plotly_chart(fig)
            
            # Confusion Matrix
            st.write("Confusion Matrix")
            plot_confusion_matrix(
                results['classification_metrics']['confusion_matrix'],
                ["Non-responder", "Responder"]
            )
        
        with result_tabs[1]:
            st.subheader("Survival Analysis Results")
            
            # Display survival metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Concordance Index", f"{results['survival_metrics']['c_index']:.3f}")
            with col2:
                st.metric("Log-rank p-value", f"{results['survival_metrics']['log_rank_p']:.4f}")
            
            # Survival Curves
            st.write("Kaplan-Meier Survival Curves")
            plot_survival_curves(results['survival_metrics']['survival_curves'])
            
            # Time-dependent AUC
            st.write("Time-dependent AUC")
            fig = px.line(
                x=results['survival_metrics']['time_points'],
                y=results['survival_metrics']['time_auc'],
                labels={'x': 'Time (months)', 'y': 'AUC'},
                title='Time-dependent AUC'
            )
            st.plotly_chart(fig)
    
    elif "Classification" in prediction_type:
        st.subheader("Response Classification Results")
        
        # Display classification metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['classification_metrics']['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{results['classification_metrics']['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{results['classification_metrics']['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{results['classification_metrics']['f1']:.3f}")
        
        # ROC Curve
        st.write("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['classification_metrics']['fpr'],
            y=results['classification_metrics']['tpr'],
            mode='lines',
            name=f'ROC (AUC = {results["classification_metrics"]["auc"]:.3f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random'
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700,
            height=500
        )
        st.plotly_chart(fig)
        
        # Confusion Matrix
        st.write("Confusion Matrix")
        plot_confusion_matrix(
            results['classification_metrics']['confusion_matrix'],
            ["Non-responder", "Responder"]
        )
    
    else:  # Survival Analysis
        st.subheader("Survival Analysis Results")
        
        # Display survival metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Concordance Index", f"{results['survival_metrics']['c_index']:.3f}")
        with col2:
            st.metric("Log-rank p-value", f"{results['survival_metrics']['log_rank_p']:.4f}")
        
        # Survival Curves
        st.write("Kaplan-Meier Survival Curves")
        plot_survival_curves(results['survival_metrics']['survival_curves'])
        
        # Time-dependent AUC
        st.write("Time-dependent AUC")
        fig = px.line(
            x=results['survival_metrics']['time_points'],
            y=results['survival_metrics']['time_auc'],
            labels={'x': 'Time (months)', 'y': 'AUC'},
            title='Time-dependent AUC'
        )
        st.plotly_chart(fig)
    
    # Feature importance
    st.header("Feature Importance")
    
    if 'feature_importance' in results:
        plot_feature_importance(results['feature_importance'])
    
    # Patient-specific predictions
    st.header("Patient-Specific Predictions")
    st.markdown("""
    Upload patient multi-omics data or use the form below to enter key biomarkers
    to predict therapy response for a specific patient.
    """)
    
    prediction_method = st.radio(
        "Prediction Method",
        options=["Upload Patient Data", "Enter Key Biomarkers"],
        index=1
    )
    
    if prediction_method == "Upload Patient Data":
        uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            # Load and process uploaded data
            patient_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(patient_data.head())
            
            predict_button = st.button("Predict Response")
            
            if predict_button:
                # This would process the data and make a prediction
                st.info("Patient data prediction functionality would be implemented here.")
                
                # Placeholder for patient prediction results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Response Prediction**")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=0.73,
                        title={'text': "Response Probability"},
                        gauge={'axis': {'range': [0, 1]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 0.4], 'color': "red"},
                                   {'range': [0.4, 0.7], 'color': "yellow"},
                                   {'range': [0.7, 1], 'color': "green"}
                               ]}
                    ))
                    st.plotly_chart(fig)
                
                with col2:
                    st.write("**Survival Prediction**")
                    # Placeholder survival curve
                    time_points = np.linspace(0, 60, 100)
                    survival_prob = np.exp(-0.02 * time_points)
                    
                    fig = px.line(
                        x=time_points,
                        y=survival_prob,
                        labels={'x': 'Time (months)', 'y': 'Survival Probability'},
                        title='Predicted Survival Curve'
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig)
                
                # Prediction explanation
                st.write("**Key Factors Influencing Prediction**")
                explanation_data = pd.DataFrame({
                    'Feature': ['TMB', 'CD8+ T-cell Infiltration', 'PD-L1 Expression', 'KRAS Mutation'],
                    'Importance': [0.35, 0.25, 0.20, 0.10],
                    'Direction': ['Positive', 'Positive', 'Negative', 'Negative']
                })
                
                fig = px.bar(
                    explanation_data,
                    x='Importance',
                    y='Feature',
                    color='Direction',
                    orientation='h',
                    title='Feature Contribution to Prediction',
                    color_discrete_map={'Positive': 'green', 'Negative': 'red'}
                )
                st.plotly_chart(fig)
    
    else:  # Enter Key Biomarkers
        st.subheader("Enter Patient Biomarker Values")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tmb = st.number_input("Tumor Mutation Burden (mutations/Mb)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            cd8_infiltration = st.number_input("CD8+ T-cell Infiltration Score", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
            pdl1_expression = st.number_input("PD-L1 Expression (% positive cells)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        
        with col2:
            age = st.number_input("Patient Age", min_value=18, max_value=100, value=65, step=1)
            has_kras_mutation = st.checkbox("KRAS Mutation", value=False)
            has_tp53_mutation = st.checkbox("TP53 Mutation", value=True)
        
        predict_button = st.button("Predict Response")
        
        if predict_button:
            # This would process the entered values and make a prediction
            st.info("The model would make a prediction based on the entered biomarker values.")
            
            # Placeholder for patient prediction results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Response Prediction**")
                
                # Response prediction value - this would come from the model
                response_prob = 0.73
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=response_prob,
                    title={'text': "Response Probability"},
                    gauge={'axis': {'range': [0, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 0.4], 'color': "red"},
                               {'range': [0.4, 0.7], 'color': "yellow"},
                               {'range': [0.7, 1], 'color': "green"}
                           ]}
                ))
                st.plotly_chart(fig)
                
                response_category = "Likely Responder" if response_prob >= 0.5 else "Likely Non-responder"
                st.markdown(f"**Prediction: {response_category}**")
            
            with col2:
                st.write("**Survival Prediction**")
                
                # Placeholder survival curve - this would come from the model
                time_points = np.linspace(0, 60, 100)
                survival_prob = np.exp(-0.02 * time_points)
                
                fig = px.line(
                    x=time_points,
                    y=survival_prob,
                    labels={'x': 'Time (months)', 'y': 'Survival Probability'},
                    title='Predicted Survival Curve'
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=0.5, x1=60, y1=0.5,
                    line=dict(color="red", dash="dash")
                )
                
                # Add median survival
                median_survival = -np.log(0.5) / 0.02
                fig.add_shape(
                    type="line",
                    x0=median_survival, y0=0, x1=median_survival, y1=0.5,
                    line=dict(color="red", dash="dash")
                )
                fig.add_annotation(
                    x=median_survival,
                    y=0.25,
                    text=f"Median: {median_survival:.1f} months",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
                
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)
            
            # Prediction explanation
            st.write("**Key Factors Influencing Prediction**")
            explanation_data = pd.DataFrame({
                'Feature': ['TMB', 'CD8+ T-cell Infiltration', 'PD-L1 Expression', 'KRAS Mutation', 'TP53 Mutation', 'Age'],
                'Importance': [0.35, 0.25, 0.20, 0.10, 0.05, 0.05],
                'Direction': ['Positive', 'Positive', 'Positive', 'Negative', 'Positive', 'Negative'],
                'Value': [tmb, cd8_infiltration, pdl1_expression, has_kras_mutation, has_tp53_mutation, age]
            })
            
            fig = px.bar(
                explanation_data,
                x='Importance',
                y='Feature',
                color='Direction',
                orientation='h',
                title='Feature Contribution to Prediction',
                color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                text='Value'
            )
            st.plotly_chart(fig)
            
            # Recommendations
            st.subheader("Clinical Recommendations")
            
            if response_prob >= 0.7:
                st.success("""
                **Recommended for CRISPR-based Immunotherapy**
                
                The patient is likely to respond well to the therapy based on:
                - High TMB indicates more neoantigens
                - Good CD8+ T-cell infiltration suggests active immune response
                - PD-L1 expression indicates potential for checkpoint inhibition
                """)
            elif response_prob >= 0.4:
                st.warning("""
                **May Benefit from CRISPR-based Immunotherapy with Close Monitoring**
                
                The patient has moderate response potential:
                - Consider combination with other therapies
                - Monitor frequently for response
                - Watch for potential resistance mechanisms
                """)
            else:
                st.error("""
                **Alternative Therapy Recommended**
                
                The patient is unlikely to respond well to this therapy:
                - Consider traditional approaches first
                - May need genomic profiling for targeted therapies
                - Consider clinical trial options
                """)
    
    # Export options
    st.header("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_json = {
            "model_type": model_type,
            "prediction_type": prediction_type,
            "performance_metrics": results
        }
        
        st.download_button(
            "Download Model Results (JSON)",
            data=pd.io.json.dumps(model_json),
            file_name="therapy_response_model_results.json",
            mime="application/json"
        )
    
    with col2:
        # If there are predictions for a cohort, allow downloading
        if 'patient_predictions' in results:
            st.download_button(
                "Download Patient Predictions (CSV)",
                data=results['patient_predictions'].to_csv(index=False),
                file_name="patient_predictions.csv",
                mime="text/csv"
            )
