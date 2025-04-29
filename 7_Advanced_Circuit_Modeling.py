import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.advanced_circuit_modeling import (
    simulate_advanced_circuit,
    evolutionary_circuit_optimization,
    simulate_multicellular_dynamics,
    generate_advanced_circuit_diagram
)

# Page configuration
st.set_page_config(
    page_title="Advanced Circuit Modeling | AI-Driven CRISPR Cancer Immunotherapy Platform",
    page_icon="🧬",
    layout="wide"
)

# Initialize session states for this page
if 'advanced_circuit_config' not in st.session_state:
    st.session_state['advanced_circuit_config'] = None
if 'advanced_simulation_results' not in st.session_state:
    st.session_state['advanced_simulation_results'] = None
if 'multicellular_results' not in st.session_state:
    st.session_state['multicellular_results'] = None

# Main header
st.title("🔄 Advanced Synthetic Biology Circuit Modeling")
st.markdown("""
This advanced module provides sophisticated modeling of synthetic genetic circuits in bacteria 
for targeted drug delivery in cancer therapy. Design complex circuits with multiple regulatory 
layers, cell-cell communication, and evolutionary optimization.
""")

# Circuit design section
st.header("Advanced Bacterial Circuit Design")

tabs = st.tabs(["Basic Circuit Configuration", "Advanced Settings", "Evolutionary Optimization", "Multicellular Dynamics"])

with tabs[0]:
    # Basic circuit configuration
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

with tabs[1]:
    # Advanced circuit settings
    st.subheader("Genetic Circuit Components")
    
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
        
        protein_degradation_rate = st.slider(
            "Protein Degradation Rate", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.1, 
            step=0.01,
            help="Rate of therapeutic protein degradation"
        )
        
        protein_maturation_time = st.slider(
            "Protein Maturation Time (hours)", 
            min_value=0.0, 
            max_value=5.0, 
            value=0.5, 
            step=0.1,
            help="Time delay for protein folding and maturation"
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
        
        # Additional sensor options
        use_oxygen_sensing = st.checkbox(
            "Add Oxygen Sensing",
            value=False,
            help="Add oxygen-responsive elements to the circuit"
        )
        
        use_glucose_sensing = st.checkbox(
            "Add Glucose Sensing", 
            value=False,
            help="Add glucose-responsive elements to the circuit"
        )
    
    if use_oxygen_sensing:
        st.subheader("Oxygen Sensing Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            oxygen_threshold = st.slider(
                "Oxygen Threshold (%)", 
                min_value=0.1, 
                max_value=15.0, 
                value=5.0, 
                step=0.1,
                help="Oxygen concentration threshold for activation"
            )
            
            oxygen_sensitivity = st.slider(
                "Oxygen Sensitivity", 
                min_value=0.1, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Sensitivity of the oxygen response"
            )
        
        with col2:
            oxygen_sensor_type = st.radio(
                "Oxygen Sensor Type",
                options=["aerobic", "anaerobic"],
                index=1,
                help="Aerobic sensors activate in high oxygen, anaerobic in low oxygen"
            )
    
    if use_glucose_sensing:
        st.subheader("Glucose Sensing Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            glucose_threshold = st.slider(
                "Glucose Threshold", 
                min_value=0.1, 
                max_value=5.0, 
                value=1.0, 
                step=0.1,
                help="Glucose concentration threshold for activation"
            )
        
        with col2:
            glucose_sensitivity = st.slider(
                "Glucose Sensitivity", 
                min_value=0.1, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Sensitivity of the glucose response"
            )

with tabs[2]:
    # Evolutionary optimization settings
    st.subheader("Evolutionary Circuit Optimization")
    
    perform_evolution = st.checkbox(
        "Use Evolutionary Algorithm",
        value=True,
        help="Use a differential evolution algorithm to optimize circuit parameters"
    )
    
    if perform_evolution:
        col1, col2 = st.columns(2)
        
        with col1:
            evolution_objective = st.selectbox(
                "Optimization Objective",
                options=["Maximize Tumor Specificity", "Maximize Drug Release", "Minimize Off-Target Release", "Balance Specificity & Efficacy"],
                index=3,
                help="Primary objective for optimization"
            )
            
            evolution_iterations = st.slider(
                "Optimization Iterations", 
                min_value=10, 
                max_value=1000, 
                value=300, 
                step=10,
                help="Number of iterations for the evolutionary algorithm"
            )
        
        with col2:
            evolution_environments = st.multiselect(
                "Simulate Environmental Conditions",
                options=["Tumor Environment (Low pH, High Density)", "Normal Tissue (Neutral pH, Low Density)", "Boundary Zone (Gradient)", "Blood Circulation"],
                default=["Tumor Environment (Low pH, High Density)", "Normal Tissue (Neutral pH, Low Density)"],
                help="Environments to test during optimization"
            )
            
            st.info("The evolutionary algorithm uses differential evolution to find optimal circuit parameters that maximize the selected objective across all selected environments.")

with tabs[3]:
    # Multicellular dynamics settings
    st.subheader("Multicellular Population Dynamics")
    
    simulate_multicellular = st.checkbox(
        "Simulate Multicellular Dynamics",
        value=False,
        help="Model cell-cell communication and population heterogeneity"
    )
    
    if simulate_multicellular:
        col1, col2 = st.columns(2)
        
        with col1:
            population_count = st.slider(
                "Number of Cells", 
                min_value=3, 
                max_value=20, 
                value=10, 
                step=1,
                help="Number of cells in the simulation"
            )
            
            diffusion_rate = st.slider(
                "Diffusion Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3, 
                step=0.05,
                help="Rate of signal molecule diffusion between cells"
            )
        
        with col2:
            st.info("Multicellular simulation models a population of bacteria with cell-cell communication via diffusible signal molecules. Each cell has slightly different properties.")

# Simulation parameters
st.header("Simulation Settings")

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

# Run simulation button
if st.button("Run Advanced Circuit Simulation"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare circuit configuration
    status_text.text("Configuring advanced bacterial circuit...")
    
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
            "payload_type": payload_type,
            "protein_degradation_rate": protein_degradation_rate,
            "protein_maturation_time": protein_maturation_time,
            "use_oxygen_sensing": use_oxygen_sensing,
            "use_glucose_sensing": use_glucose_sensing
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
    
    # Add additional sensing parameters if enabled
    if use_oxygen_sensing:
        circuit_config["parameters"]["oxygen_threshold"] = oxygen_threshold
        circuit_config["parameters"]["oxygen_sensitivity"] = oxygen_sensitivity
        circuit_config["parameters"]["oxygen_sensor_type"] = oxygen_sensor_type
    
    if use_glucose_sensing:
        circuit_config["parameters"]["glucose_threshold"] = glucose_threshold
        circuit_config["parameters"]["glucose_sensitivity"] = glucose_sensitivity
    
    # Add multicellular parameters if enabled
    if simulate_multicellular:
        circuit_config["parameters"]["diffusion_rate"] = diffusion_rate
    
    # Add simulation parameters
    simulation_params = {
        "simulation_time": simulation_time,  # hours
        "time_step": time_step / 60,  # convert minutes to hours
        "population_initial": population_initial,
        "stochastic": stochastic_simulation
    }
    
    # Add population count for multicellular simulation
    if simulate_multicellular:
        simulation_params["population_count"] = population_count
    
    progress_bar.progress(0.1)
    
    # Evolutionary optimization if requested
    if perform_evolution:
        status_text.text("Performing evolutionary circuit optimization...")
        
        # Define environments based on selection
        environments = []
        if "Tumor Environment (Low pH, High Density)" in evolution_environments:
            environments.append({"name": "Tumor", "ph": 6.0, "population_density": 0.8, "oxygen": 2.0, "glucose": 0.5})
        if "Normal Tissue (Neutral pH, Low Density)" in evolution_environments:
            environments.append({"name": "Normal", "ph": 7.4, "population_density": 0.2, "oxygen": 8.0, "glucose": 1.0})
        if "Boundary Zone (Gradient)" in evolution_environments:
            environments.append({"name": "Boundary", "ph": 6.7, "population_density": 0.5, "oxygen": 5.0, "glucose": 0.8})
        if "Blood Circulation" in evolution_environments:
            environments.append({"name": "Blood", "ph": 7.4, "population_density": 0.1, "oxygen": 10.0, "glucose": 1.2})
        
        # Default environment if none selected
        if not environments:
            environments = [
                {"name": "Tumor", "ph": 6.0, "population_density": 0.8, "oxygen": 2.0, "glucose": 0.5},
                {"name": "Normal", "ph": 7.4, "population_density": 0.2, "oxygen": 8.0, "glucose": 1.0}
            ]
        
        # Perform evolutionary optimization
        optimized_config = evolutionary_circuit_optimization(
            circuit_config,
            evolution_objective,
            environments,
            iterations=evolution_iterations
        )
        
        circuit_config = optimized_config
        
        progress_bar.progress(0.5)
        status_text.text("Evolutionary optimization completed. Running advanced simulation...")
    else:
        progress_bar.progress(0.5)
        status_text.text("Running advanced bacterial circuit simulation...")
    
    # Store circuit configuration
    st.session_state['advanced_circuit_config'] = circuit_config
    
    # Run simulation
    environments = [
        {"name": "Tumor", "ph": 6.0, "population_density": 0.8, "oxygen": 2.0, "glucose": 0.5},
        {"name": "Normal", "ph": 7.4, "population_density": 0.2, "oxygen": 8.0, "glucose": 1.0}
    ]
    
    advanced_simulation_results = {}
    
    for env in environments:
        env_name = env["name"]
        status_text.text(f"Simulating advanced circuit in {env_name} environment...")
        
        # Run simulation for this environment
        advanced_simulation_results[env_name] = simulate_advanced_circuit(
            circuit_config,
            env,
            simulation_params
        )
    
    # Store simulation results
    st.session_state['advanced_simulation_results'] = advanced_simulation_results
    
    # Run multicellular simulation if requested
    if simulate_multicellular:
        status_text.text("Simulating multicellular population dynamics...")
        
        # Run multicellular simulation for tumor environment
        tumor_env = environments[0]
        multicellular_results = simulate_multicellular_dynamics(
            circuit_config,
            tumor_env,
            simulation_params
        )
        
        # Store multicellular results
        st.session_state['multicellular_results'] = multicellular_results
    
    progress_bar.progress(1.0)
    status_text.text("✅ Advanced circuit simulation completed successfully!")

# Display simulation results
if st.session_state['advanced_simulation_results'] is not None:
    st.header("Simulation Results")
    
    # Display circuit diagram
    st.subheader("Advanced Synthetic Circuit Diagram")
    circuit_diagram = generate_advanced_circuit_diagram(st.session_state['advanced_circuit_config'])
    st.image(circuit_diagram, use_column_width=True)
    
    # Show simulation results for different environments
    st.subheader("Advanced Circuit Dynamics")
    
    # Create tabs for different environments
    env_tabs = st.tabs([f"{env_name} Environment" for env_name in st.session_state['advanced_simulation_results'].keys()])
    
    for i, (env_name, sim_data) in enumerate(st.session_state['advanced_simulation_results'].items()):
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
            
            # Display sensor responses if available
            sensor_cols = [col for col in sim_data.columns if '_response' in col]
            if sensor_cols:
                st.subheader("Sensor Responses")
                fig = px.line(
                    sim_data,
                    x='time',
                    y=sensor_cols,
                    title=f"Sensor Responses in {env_name} Environment",
                    labels={'time': 'Time (hours)', 'value': 'Response (0-1)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display additional simulation metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                peak_drug = float(sim_data['drug_concentration'].max())
                time_to_peak = float(sim_data.loc[sim_data['drug_concentration'].idxmax(), 'time'])
                
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
                if "pH" in st.session_state['advanced_circuit_config']['circuit_type']:
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
                
                if st.session_state['advanced_circuit_config']['circuit_type'] != "pH-Sensitive Drug Release":
                    # Safe calculation of circuit efficiency
                    population_max = float(sim_data['population'].max())
                    max_drug_production = st.session_state['advanced_circuit_config']['parameters']['max_drug_production']
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
    
    for env_name, sim_data in st.session_state['advanced_simulation_results'].items():
        peak_drug = float(sim_data['drug_concentration'].max())
        time_to_peak = float(sim_data.loc[sim_data['drug_concentration'].idxmax(), 'time'])
        auc = float(np.trapz(sim_data['drug_concentration'], sim_data['time']))
        final_population = float(sim_data['population'].iloc[-1])
        
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
    
    # Display comparison
    st.dataframe(env_comparison)
    
    # Calculate tumor specificity if both tumor and normal results exist
    if 'Tumor' in st.session_state['advanced_simulation_results'] and 'Normal' in st.session_state['advanced_simulation_results']:
        tumor_data = st.session_state['advanced_simulation_results']['Tumor']
        normal_data = st.session_state['advanced_simulation_results']['Normal']
        
        # Calculate tumor specificity ratio
        tumor_auc = float(np.trapz(tumor_data['drug_concentration'], tumor_data['time']))
        normal_auc = float(np.trapz(normal_data['drug_concentration'], normal_data['time']))
        
        # Avoid division by zero
        if normal_auc > 0:
            tumor_specificity = tumor_auc / normal_auc
            st.metric(
                "Tumor Specificity Ratio",
                f"{tumor_specificity:.2f}x",
                help="Ratio of drug exposure in tumor vs. normal tissue. Higher is better for targeted therapy."
            )
    
    # Display multicellular dynamics results if available
    if st.session_state['multicellular_results'] is not None:
        st.header("Multicellular Population Dynamics")
        
        multi_data = st.session_state['multicellular_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot population dynamics with error bars
            fig = go.Figure()
            
            # Mean population
            fig.add_trace(go.Scatter(
                x=multi_data['time'],
                y=multi_data['population_mean'],
                mode='lines',
                name='Mean Population',
                line=dict(color='blue')
            ))
            
            # Upper and lower bounds
            fig.add_trace(go.Scatter(
                x=multi_data['time'],
                y=multi_data['population_mean'] + multi_data['population_std'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=multi_data['time'],
                y=multi_data['population_mean'] - multi_data['population_std'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                title='Population Dynamics with Variability',
                xaxis_title='Time (hours)',
                yaxis_title='Population Density',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Plot drug concentration dynamics with error bars
            fig = go.Figure()
            
            # Mean drug concentration
            fig.add_trace(go.Scatter(
                x=multi_data['time'],
                y=multi_data['drug_concentration_mean'],
                mode='lines',
                name='Mean Drug Conc.',
                line=dict(color='red')
            ))
            
            # Upper and lower bounds
            fig.add_trace(go.Scatter(
                x=multi_data['time'],
                y=multi_data['drug_concentration_mean'] + multi_data['drug_concentration_std'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=multi_data['time'],
                y=multi_data['drug_concentration_mean'] - multi_data['drug_concentration_std'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                title='Drug Concentration with Variability',
                xaxis_title='Time (hours)',
                yaxis_title='Drug Concentration',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show cell heterogeneity
        st.subheader("Cell Population Heterogeneity")
        
        # Get individual cell columns
        cell_pop_cols = [col for col in multi_data.columns if col.startswith('cell_') and col.endswith('_population')]
        cell_drug_cols = [col for col in multi_data.columns if col.startswith('cell_') and col.endswith('_drug')]
        
        if cell_pop_cols and cell_drug_cols:
            # Only show a subset of cells if there are many
            max_cells_to_show = 5
            if len(cell_pop_cols) > max_cells_to_show:
                cell_pop_cols = cell_pop_cols[:max_cells_to_show]
                cell_drug_cols = cell_drug_cols[:max_cells_to_show]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot individual cell populations
                fig = px.line(
                    multi_data,
                    x='time',
                    y=cell_pop_cols,
                    title='Individual Cell Population Dynamics',
                    labels={'time': 'Time (hours)', 'value': 'Population Density'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot individual cell drug concentrations
                fig = px.line(
                    multi_data,
                    x='time',
                    y=cell_drug_cols,
                    title='Individual Cell Drug Production',
                    labels={'time': 'Time (hours)', 'value': 'Drug Concentration'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate cell-to-cell variability metrics
            final_time_point = multi_data['time'].iloc[-1]
            end_pop_values = [float(multi_data[col].iloc[-1]) for col in cell_pop_cols]
            end_drug_values = [float(multi_data[col].iloc[-1]) for col in cell_drug_cols]
            
            # Calculate coefficient of variation (CV = std/mean)
            pop_cv = np.std(end_pop_values) / np.mean(end_pop_values) if np.mean(end_pop_values) > 0 else 0
            drug_cv = np.std(end_drug_values) / np.mean(end_drug_values) if np.mean(end_drug_values) > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Population Heterogeneity (CV)",
                    f"{pop_cv:.2f}",
                    help="Coefficient of variation in population density across cells. Higher values indicate more heterogeneity."
                )
            
            with col2:
                st.metric(
                    "Drug Production Heterogeneity (CV)",
                    f"{drug_cv:.2f}",
                    help="Coefficient of variation in drug production across cells. Higher values indicate more heterogeneity."
                )
            
            # Display correlation between population and drug production
            corr = np.corrcoef(end_pop_values, end_drug_values)[0, 1]
            st.metric(
                "Population-Drug Correlation",
                f"{corr:.2f}",
                help="Correlation between population density and drug production across cells. Values near 1 indicate strong positive correlation."
            )
    
    # Add download button for simulation results
    if st.button("Download Simulation Results"):
        # Prepare data for download
        results_dict = {}
        for env_name, sim_data in st.session_state['advanced_simulation_results'].items():
            results_dict[f"{env_name}_results"] = sim_data.to_dict()
        
        if st.session_state['multicellular_results'] is not None:
            results_dict["multicellular_results"] = st.session_state['multicellular_results'].to_dict()
        
        # Create JSON string
        import json
        results_json = json.dumps(results_dict)
        
        # Create download link
        import base64
        b64 = base64.b64encode(results_json.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="advanced_circuit_simulation_results.json">Download JSON Results</a>'
        st.markdown(href, unsafe_allow_html=True)