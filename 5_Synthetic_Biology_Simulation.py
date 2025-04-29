import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.synthetic_biology_utils import (
    simulate_bacterial_circuit,
    optimize_circuit_parameters,
    generate_circuit_diagram,
    simulate_drug_release
)

# Page configuration
st.set_page_config(
    page_title="Synthetic Biology Simulation | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Initialize session states for this page
if 'circuit_config' not in st.session_state:
    st.session_state['circuit_config'] = None
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None

# Main header
st.title("🔄 Synthetic Biology Circuit Simulation")
st.markdown("""
This module simulates and optimizes synthetic gene circuits in bacteria for targeted drug delivery in cancer therapy.
Design circuits with quorum sensing, pH sensitivity, and controlled drug release mechanisms.
""")

# Circuit design section
st.header("Bacterial Circuit Design")

circuit_type = st.radio(
    "Select Circuit Type",
    options=["pH-Sensitive Drug Release", "Quorum Sensing Drug Release", "Combined pH & Quorum Sensing", "Custom Circuit"],
    index=2
)

# Circuit parameters
st.subheader("Circuit Parameters")

col1, col2 = st.columns(2)

with col1:
    if "pH" in circuit_type:
        st.write("**pH Sensitivity Parameters**")
        ph_threshold = st.slider(
            "pH Threshold", 
            min_value=4.0, 
            max_value=8.0, 
            value=6.5, 
            step=0.1,
            help="pH level at which the circuit activates"
        )
        
        ph_sensitivity = st.slider(
            "pH Sensitivity", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.5, 
            step=0.1,
            help="Sensitivity of the pH response (lower = more sensitive)"
        )
    
    if circuit_type != "pH-Sensitive Drug Release":
        st.write("**Quorum Sensing Parameters**")
        qs_threshold = st.slider(
            "Quorum Sensing Threshold", 
            min_value=0.1, 
            max_value=5.0, 
            value=1.0, 
            step=0.1,
            help="Population density threshold for activation"
        )
        
        qs_cooperativity = st.slider(
            "Quorum Sensing Cooperativity", 
            min_value=1.0, 
            max_value=5.0, 
            value=2.0, 
            step=0.1,
            help="Hill coefficient for quorum sensing response"
        )

with col2:
    st.write("**Drug Release Parameters**")
    max_drug_production = st.slider(
        "Maximum Drug Production Rate", 
        min_value=0.1, 
        max_value=10.0, 
        value=2.0, 
        step=0.1,
        help="Maximum rate of drug production when fully activated"
    )
    
    drug_degradation = st.slider(
        "Drug Degradation Rate", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.1, 
        step=0.01,
        help="Rate of drug degradation/clearance"
    )
    
    growth_rate = st.slider(
        "Bacterial Growth Rate", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.5, 
        step=0.1,
        help="Growth rate of the bacterial population"
    )

# Advanced options
with st.expander("Advanced Circuit Configuration"):
    col1, col2 = st.columns(2)
    
    with col1:
        leakiness = st.slider(
            "Circuit Leakiness", 
            min_value=0.0, 
            max_value=0.5, 
            value=0.05, 
            step=0.01,
            help="Baseline activity in the absence of input signals"
        )
        
        payload_type = st.selectbox(
            "Therapeutic Payload",
            options=["Cytokine (IL-12)", "Antibody Fragments", "Chemotherapeutic", "CRISPR-Cas9", "Custom"],
            index=0,
            help="Type of therapeutic payload delivered by the bacteria"
        )
    
    with col2:
        promoter_strength = st.slider(
            "Promoter Strength", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Strength of the promoter controlling drug expression"
        )
        
        ribosome_binding_strength = st.slider(
            "Ribosome Binding Strength", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Efficiency of translation initiation"
        )

# Simulation parameters
st.header("Simulation Parameters")

col1, col2 = st.columns(2)

with col1:
    simulation_time = st.slider(
        "Simulation Time (hours)", 
        min_value=1, 
        max_value=96, 
        value=48, 
        step=1
    )
    
    time_step = st.slider(
        "Time Step (minutes)", 
        min_value=1, 
        max_value=60, 
        value=10, 
        step=1
    )

with col2:
    population_initial = st.slider(
        "Initial Bacterial Population", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.1, 
        step=0.01,
        help="Initial population density (relative to carrying capacity)"
    )
    
    stochastic_simulation = st.checkbox(
        "Stochastic Simulation", 
        value=False,
        help="Include random noise in the simulation"
    )

# Optimization options
st.header("AI-Driven Circuit Optimization")

perform_optimization = st.checkbox(
    "Optimize Circuit Parameters", 
    value=True,
    help="Use AI to optimize circuit parameters for better performance"
)

if perform_optimization:
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_objective = st.selectbox(
            "Optimization Objective",
            options=["Maximize Tumor Specificity", "Maximize Drug Release", "Minimize Off-Target Release", "Balance Specificity & Efficacy"],
            index=3,
            help="Primary objective for optimization"
        )
        
        optimization_iterations = st.slider(
            "Optimization Iterations", 
            min_value=10, 
            max_value=500, 
            value=100, 
            step=10,
            help="Number of iterations for the optimization algorithm"
        )
    
    with col2:
        environment_conditions = st.multiselect(
            "Simulate Environmental Conditions",
            options=["Tumor Environment (Low pH, High Density)", "Normal Tissue (Neutral pH, Low Density)", "Boundary Zone (Gradient)", "Blood Circulation"],
            default=["Tumor Environment (Low pH, High Density)", "Normal Tissue (Neutral pH, Low Density)"],
            help="Environments to test during optimization"
        )

# Run simulation button
if st.button("Run Circuit Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare circuit configuration
    status_text.text("Configuring bacterial circuit...")
    
    # Create circuit configuration based on selected parameters
    circuit_config = {
        "circuit_type": circuit_type,
        "parameters": {
            "growth_rate": growth_rate,
            "max_drug_production": max_drug_production,
            "drug_degradation": drug_degradation,
            "leakiness": leakiness,
            "promoter_strength": promoter_strength,
            "ribosome_binding_strength": ribosome_binding_strength,
            "payload_type": payload_type
        }
    }
    
    # Add pH parameters if applicable
    if "pH" in circuit_type:
        circuit_config["parameters"]["ph_threshold"] = ph_threshold
        circuit_config["parameters"]["ph_sensitivity"] = ph_sensitivity
    
    # Add quorum sensing parameters if applicable
    if circuit_type != "pH-Sensitive Drug Release":
        circuit_config["parameters"]["qs_threshold"] = qs_threshold
        circuit_config["parameters"]["qs_cooperativity"] = qs_cooperativity
    
    # Add simulation parameters
    simulation_params = {
        "simulation_time": simulation_time,  # hours
        "time_step": time_step / 60,  # convert minutes to hours
        "population_initial": population_initial,
        "stochastic": stochastic_simulation
    }
    
    progress_bar.progress(0.1)
    
    # Optimize circuit parameters if requested
    if perform_optimization:
        status_text.text("Optimizing circuit parameters with AI...")
        
        # Define environments based on selection
        environments = []
        if "Tumor Environment (Low pH, High Density)" in environment_conditions:
            environments.append({"name": "Tumor", "ph": 6.0, "population_density": 0.8})
        if "Normal Tissue (Neutral pH, Low Density)" in environment_conditions:
            environments.append({"name": "Normal", "ph": 7.4, "population_density": 0.2})
        if "Boundary Zone (Gradient)" in environment_conditions:
            environments.append({"name": "Boundary", "ph": 6.7, "population_density": 0.5})
        if "Blood Circulation" in environment_conditions:
            environments.append({"name": "Blood", "ph": 7.4, "population_density": 0.1})
        
        # Default environment if none selected
        if not environments:
            environments = [
                {"name": "Tumor", "ph": 6.0, "population_density": 0.8},
                {"name": "Normal", "ph": 7.4, "population_density": 0.2}
            ]
        
        # Perform optimization
        optimized_config = optimize_circuit_parameters(
            circuit_config,
            optimization_objective,
            environments,
            iterations=optimization_iterations
        )
        
        circuit_config = optimized_config
        
        progress_bar.progress(0.5)
        status_text.text("Circuit optimization completed. Running simulation...")
    else:
        progress_bar.progress(0.5)
        status_text.text("Running bacterial circuit simulation...")
    
    # Store circuit configuration
    st.session_state['circuit_config'] = circuit_config
    
    # Run simulation
    environments = [
        {"name": "Tumor", "ph": 6.0, "population_density": 0.8},
        {"name": "Normal", "ph": 7.4, "population_density": 0.2}
    ]
    
    simulation_results = {}
    
    for env in environments:
        env_name = env["name"]
        status_text.text(f"Simulating circuit in {env_name} environment...")
        
        # Run simulation for this environment
        simulation_results[env_name] = simulate_bacterial_circuit(
            circuit_config,
            env,
            simulation_params
        )
    
    # Store simulation results
    st.session_state['simulation_results'] = simulation_results
    
    progress_bar.progress(1.0)
    status_text.text("✅ Bacterial circuit simulation completed successfully!")

# Display simulation results
if st.session_state['simulation_results'] is not None:
    st.header("Simulation Results")
    
    # Display circuit diagram
    st.subheader("Synthetic Circuit Diagram")
    circuit_diagram = generate_circuit_diagram(st.session_state['circuit_config'])
    st.image(circuit_diagram, use_column_width=True)
    
    # Show simulation results for different environments
    st.subheader("Dynamic Simulation")
    
    # Create tabs for different environments
    env_tabs = st.tabs([f"{env_name} Environment" for env_name in st.session_state['simulation_results'].keys()])
    
    for i, (env_name, sim_data) in enumerate(st.session_state['simulation_results'].items()):
        with env_tabs[i]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot bacterial population dynamics
                fig = px.line(
                    sim_data, 
                    x='time', 
                    y='population', 
                    title=f"Bacterial Population Dynamics in {env_name} Environment",
                    labels={'time': 'Time (hours)', 'population': 'Relative Population Density'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot drug concentration
                fig = px.line(
                    sim_data, 
                    x='time', 
                    y='drug_concentration', 
                    title=f"Drug Concentration in {env_name} Environment",
                    labels={'time': 'Time (hours)', 'drug_concentration': 'Relative Drug Concentration'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display additional simulation metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                peak_drug = sim_data['drug_concentration'].max()
                peak_drug = float(peak_drug)  # Ensure it's a float, not a Series
                time_to_peak = sim_data.loc[sim_data['drug_concentration'].idxmax(), 'time']
                time_to_peak = float(time_to_peak)  # Ensure it's a float, not a Series
                
                st.metric(
                    "Peak Drug Concentration",
                    f"{peak_drug:.2f}",
                    help="Maximum drug concentration reached during simulation"
                )
                st.metric(
                    "Time to Peak",
                    f"{time_to_peak:.1f} hours",
                    help="Time required to reach peak drug concentration"
                )
            
            with col2:
                final_population = float(sim_data['population'].iloc[-1])
                population_change = final_population - float(sim_data['population'].iloc[0])
                st.metric(
                    "Final Population Density",
                    f"{final_population:.2f}",
                    f"{population_change:+.2f}",
                    help="Bacterial population density at the end of simulation"
                )
                auc = float(np.trapz(sim_data['drug_concentration'], sim_data['time']))
                st.metric(
                    "Drug Exposure (AUC)",
                    f"{auc:.2f}",
                    help="Area under the curve of drug concentration over time"
                )
            
            with col3:
                if "pH" in st.session_state['circuit_config']['circuit_type']:
                    ph_value = sim_data.get('ph', sim_data.get('pH', [env_name == "Tumor" and 6.0 or 7.4] * len(sim_data)))
                    # Handle different types of ph_value
                    if isinstance(ph_value, pd.Series):
                        avg_ph = ph_value.mean()
                    elif isinstance(ph_value, list):
                        avg_ph = np.mean(ph_value)
                    else:
                        avg_ph = ph_value
                    
                    st.metric(
                        "Environmental pH",
                        f"{float(avg_ph):.1f}",
                        help="pH level in the simulated environment"
                    )
                
                if st.session_state['circuit_config']['circuit_type'] != "pH-Sensitive Drug Release":
                    # Safe calculation of circuit efficiency
                    population_max = float(sim_data['population'].max())
                    max_drug_production = st.session_state['circuit_config']['parameters']['max_drug_production']
                    circuit_efficiency = peak_drug / (population_max * max_drug_production)
                    
                    st.metric(
                        "Circuit Efficiency",
                        f"{float(circuit_efficiency):.2%}",
                        help="Efficiency of drug production relative to theoretical maximum"
                    )
    
    # Compare environments
    st.subheader("Environment Comparison")
    
    # Prepare data for comparison
    env_comparison = pd.DataFrame({
        'Environment': [],
        'Peak Drug Concentration': [],
        'Time to Peak (hours)': [],
        'Drug Exposure (AUC)': [],
        'Final Population': []
    })
    
    for env_name, sim_data in st.session_state['simulation_results'].items():
        peak_drug = sim_data['drug_concentration'].max()
        time_to_peak = sim_data.loc[sim_data['drug_concentration'].idxmax(), 'time']
        auc = np.trapz(sim_data['drug_concentration'], sim_data['time'])
        final_population = sim_data['population'].iloc[-1]
        
        env_comparison = pd.concat([
            env_comparison,
            pd.DataFrame({
                'Environment': [env_name],
                'Peak Drug Concentration': [peak_drug],
                'Time to Peak (hours)': [time_to_peak],
                'Drug Exposure (AUC)': [auc],
                'Final Population': [final_population]
            })
        ], ignore_index=True)
    
    # Display comparison table
    st.dataframe(env_comparison)
    
    # Plot comparison chart
    fig = px.bar(
        env_comparison,
        x='Environment',
        y=['Peak Drug Concentration', 'Drug Exposure (AUC)'],
        barmode='group',
        title="Drug Delivery Comparison Across Environments",
        labels={'value': 'Value', 'variable': 'Metric'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Drug release simulation
    st.subheader("Tumor-Specific Drug Release Simulation")
    
    # Simulate drug release over time in tumor vs normal tissue
    drug_release_data = simulate_drug_release(
        st.session_state['circuit_config'],
        st.session_state['simulation_results']
    )
    
    # Plot drug release over time
    fig = px.line(
        drug_release_data,
        x='time',
        y=['tumor_drug_level', 'normal_drug_level'],
        title="Tumor-Specific Drug Release Over Time",
        labels={
            'time': 'Time (hours)',
            'value': 'Drug Level',
            'variable': 'Tissue Type'
        }
    )
    fig.update_layout(legend_title_text='Tissue Type')
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("_drug_level", "").title()))
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate tumor specificity
    tumor_sum = float(drug_release_data['tumor_drug_level'].sum())
    normal_sum = float(drug_release_data['normal_drug_level'].sum())
    tumor_normal_ratio = tumor_sum / max(normal_sum, 0.001)
    
    st.metric(
        "Tumor Specificity Ratio",
        f"{float(tumor_normal_ratio):.2f}",
        help="Ratio of drug exposure in tumor vs normal tissue (higher is better)"
    )
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare data for export
        export_data = {
            "circuit_config": st.session_state['circuit_config'],
            "simulation_results": {
                env: data.to_dict('records') for env, data in st.session_state['simulation_results'].items()
            }
        }
        
        st.download_button(
            "Download Circuit Design & Simulation (JSON)",
            data=pd.io.json.dumps(export_data),
            file_name="bacterial_circuit_simulation.json",
            mime="application/json"
        )
    
    with col2:
        # Export drug release data
        st.download_button(
            "Download Drug Release Data (CSV)",
            data=drug_release_data.to_csv(index=False),
            file_name="drug_release_data.csv",
            mime="text/csv"
        )
    
    # Next steps
    st.markdown("---")
    st.header("Next Steps")
    st.markdown("""
    With the bacterial circuit designed and simulated, you can now:
    1. **Therapy Response Prediction** - Predict patient responses to the designed therapy
    """)
    
    next_button = st.button("Proceed to Therapy Response Prediction")
    if next_button:
        st.switch_page("pages/9_Therapy_Response_Prediction.py")
