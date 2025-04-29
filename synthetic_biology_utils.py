import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow, PathPatch
from matplotlib.path import Path
import io
from PIL import Image

def simulate_bacterial_circuit(circuit_config, environment, simulation_params):
    """
    Simulate a synthetic bacterial circuit under specific environmental conditions.
    
    Args:
        circuit_config (dict): Configuration of the bacterial circuit
        environment (dict): Environmental conditions (pH, population density)
        simulation_params (dict): Simulation parameters (time, step size, etc.)
        
    Returns:
        pandas.DataFrame: Simulation results over time
    """
    # Extract circuit parameters
    circuit_type = circuit_config['circuit_type']
    params = circuit_config['parameters']
    
    # Extract simulation parameters
    simulation_time = simulation_params['simulation_time']  # hours
    time_step = simulation_params['time_step']  # hours
    population_initial = simulation_params['population_initial']
    stochastic = simulation_params.get('stochastic', False)
    
    # Extract environment parameters
    env_name = environment['name']
    env_ph = environment.get('ph', 7.4 if env_name == "Normal" else 6.0)
    env_population_density = environment.get('population_density', 0.2 if env_name == "Normal" else 0.8)
    
    # Create time points
    time_points = np.arange(0, simulation_time + time_step, time_step)
    
    # Initialize arrays for results
    population = np.zeros(len(time_points))
    drug_concentration = np.zeros(len(time_points))
    
    # Set initial conditions
    population[0] = population_initial
    drug_concentration[0] = 0
    
    # Extract circuit parameters
    growth_rate = params['growth_rate']
    max_drug_production = params['max_drug_production']
    drug_degradation = params['drug_degradation']
    leakiness = params['leakiness']
    
    # Circuit-specific parameters
    ph_sensitive = "pH" in circuit_type
    quorum_sensing = circuit_type != "pH-Sensitive Drug Release"
    
    if ph_sensitive:
        ph_threshold = params['ph_threshold']
        ph_sensitivity = params['ph_sensitivity']
    
    if quorum_sensing:
        qs_threshold = params['qs_threshold']
        qs_cooperativity = params['qs_cooperativity']
    
    # Create pH array (could vary over time in more complex simulations)
    ph = np.ones(len(time_points)) * env_ph
    
    # Simulate circuit dynamics
    for i in range(1, len(time_points)):
        # Population growth (logistic model)
        # dP/dt = r*P*(1-P/K) where K is carrying capacity (set to 1.0)
        population_growth = growth_rate * population[i-1] * (1 - population[i-1])
        
        # Add randomness if stochastic
        if stochastic:
            population_growth += np.random.normal(0, 0.02)
        
        population[i] = population[i-1] + population_growth * time_step
        
        # Ensure population stays in valid range
        population[i] = np.clip(population[i], 0, 1)
        
        # Calculate circuit activation based on type
        circuit_activation = leakiness  # Baseline activity
        
        if ph_sensitive and quorum_sensing:
            # pH response (sigmoidal)
            ph_response = 1 / (1 + np.exp((ph[i-1] - ph_threshold) / ph_sensitivity))
            
            # Quorum sensing response (Hill function)
            qs_response = population[i-1]**qs_cooperativity / (qs_threshold**qs_cooperativity + population[i-1]**qs_cooperativity)
            
            # Combine responses (can be adjusted based on circuit design)
            circuit_activation += (1 - leakiness) * ph_response * qs_response
            
        elif ph_sensitive:
            # pH response only
            ph_response = 1 / (1 + np.exp((ph[i-1] - ph_threshold) / ph_sensitivity))
            circuit_activation += (1 - leakiness) * ph_response
            
        elif quorum_sensing:
            # Quorum sensing only
            qs_response = population[i-1]**qs_cooperativity / (qs_threshold**qs_cooperativity + population[i-1]**qs_cooperativity)
            circuit_activation += (1 - leakiness) * qs_response
        
        # Drug production and degradation
        # dD/dt = production - degradation
        drug_production = max_drug_production * circuit_activation * population[i-1]
        drug_degradation_rate = drug_degradation * drug_concentration[i-1]
        
        drug_concentration[i] = drug_concentration[i-1] + (drug_production - drug_degradation_rate) * time_step
        
        # Add randomness if stochastic
        if stochastic:
            drug_concentration[i] += np.random.normal(0, 0.05 * drug_concentration[i-1])
            drug_concentration[i] = max(0, drug_concentration[i])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'time': time_points,
        'population': population,
        'drug_concentration': drug_concentration,
        'pH': ph
    })
    
    return results

def optimize_circuit_parameters(circuit_config, objective, environments, iterations=100):
    """
    Optimize bacterial circuit parameters using a simple algorithm.
    
    Args:
        circuit_config (dict): Initial circuit configuration
        objective (str): Optimization objective
        environments (list): List of environments to test
        iterations (int): Number of optimization iterations
        
    Returns:
        dict: Optimized circuit configuration
    """
    # Make a copy of the initial configuration
    best_config = circuit_config.copy()
    best_config['parameters'] = circuit_config['parameters'].copy()
    
    # Set up simulation parameters
    simulation_params = {
        'simulation_time': 48,  # hours
        'time_step': 0.5,  # hours
        'population_initial': 0.1,
        'stochastic': False
    }
    
    # Define parameter ranges for optimization
    param_ranges = {
        'growth_rate': (0.1, 2.0),
        'max_drug_production': (0.1, 10.0),
        'drug_degradation': (0.01, 1.0),
        'leakiness': (0.0, 0.3),
        'promoter_strength': (0.1, 10.0),
        'ribosome_binding_strength': (0.1, 10.0)
    }
    
    # Add circuit-specific parameters
    if "pH" in circuit_config['circuit_type']:
        param_ranges['ph_threshold'] = (4.0, 8.0)
        param_ranges['ph_sensitivity'] = (0.1, 2.0)
    
    if circuit_config['circuit_type'] != "pH-Sensitive Drug Release":
        param_ranges['qs_threshold'] = (0.1, 5.0)
        param_ranges['qs_cooperativity'] = (1.0, 5.0)
    
    # Run simulations with the initial configuration
    best_score = evaluate_circuit(best_config, environments, simulation_params, objective)
    
    # Simple optimization loop
    for i in range(iterations):
        # Create a new configuration with random perturbations
        new_config = best_config.copy()
        new_config['parameters'] = best_config['parameters'].copy()
        
        # Randomly select a parameter to modify
        param_to_modify = np.random.choice(list(new_config['parameters'].keys()))
        
        # Skip non-numeric parameters
        if param_to_modify not in param_ranges:
            continue
        
        # Get current value and range
        current_value = new_config['parameters'][param_to_modify]
        param_min, param_max = param_ranges.get(param_to_modify, (0, 1))
        
        # Apply random perturbation
        perturbation = np.random.normal(0, (param_max - param_min) * 0.1)
        new_value = current_value + perturbation
        
        # Ensure value stays in valid range
        new_value = max(param_min, min(param_max, new_value))
        
        # Update parameter
        new_config['parameters'][param_to_modify] = new_value
        
        # Evaluate new configuration
        new_score = evaluate_circuit(new_config, environments, simulation_params, objective)
        
        # Update best configuration if better
        if new_score > best_score:
            best_config = new_config
            best_score = new_score
    
    return best_config

def evaluate_circuit(circuit_config, environments, simulation_params, objective):
    """
    Evaluate a circuit configuration against the optimization objective.
    
    Args:
        circuit_config (dict): Circuit configuration
        environments (list): List of environments to test
        simulation_params (dict): Simulation parameters
        objective (str): Optimization objective
        
    Returns:
        float: Score representing how well the circuit meets the objective
    """
    # Run simulations in each environment
    results = {}
    for env in environments:
        results[env['name']] = simulate_bacterial_circuit(
            circuit_config,
            env,
            simulation_params
        )
    
    # Calculate scores based on objective
    if objective == "Maximize Tumor Specificity":
        # Compare drug concentration in tumor vs. normal tissue
        tumor_env = next((env for env in environments if env['name'] == "Tumor"), environments[0])
        normal_env = next((env for env in environments if env['name'] == "Normal"), environments[-1])
        
        tumor_auc = np.trapz(results[tumor_env['name']]['drug_concentration'], results[tumor_env['name']]['time'])
        normal_auc = np.trapz(results[normal_env['name']]['drug_concentration'], results[normal_env['name']]['time'])
        
        # Avoid division by zero
        normal_auc = max(normal_auc, 0.001)
        
        # Calculate tumor specificity ratio
        specificity_ratio = tumor_auc / normal_auc
        
        return specificity_ratio
    
    elif objective == "Maximize Drug Release":
        # Calculate total drug release across all environments
        total_auc = 0
        for env_name, result in results.items():
            auc = np.trapz(result['drug_concentration'], result['time'])
            
            # Weight tumor environment more heavily
            if "Tumor" in env_name:
                auc *= 2
                
            total_auc += auc
            
        return total_auc
    
    elif objective == "Minimize Off-Target Release":
        # Calculate drug release in non-tumor environments
        off_target_auc = 0
        for env_name, result in results.items():
            if "Normal" in env_name or "Blood" in env_name:
                auc = np.trapz(result['drug_concentration'], result['time'])
                off_target_auc += auc
        
        # Invert so lower off-target release gives higher score
        return 1 / (off_target_auc + 0.1)  # Add small constant to avoid division by zero
    
    else:  # "Balance Specificity & Efficacy"
        # Calculate both specificity and efficacy
        tumor_env = next((env for env in environments if env['name'] == "Tumor"), environments[0])
        normal_env = next((env for env in environments if env['name'] == "Normal"), environments[-1])
        
        tumor_auc = np.trapz(results[tumor_env['name']]['drug_concentration'], results[tumor_env['name']]['time'])
        normal_auc = np.trapz(results[normal_env['name']]['drug_concentration'], results[normal_env['name']]['time'])
        
        # Avoid division by zero
        normal_auc = max(normal_auc, 0.001)
        
        # Calculate tumor specificity ratio
        specificity_ratio = tumor_auc / normal_auc
        
        # Calculate efficacy (absolute drug level in tumor)
        efficacy = tumor_auc
        
        # Combine scores (can adjust weights)
        combined_score = 0.7 * min(specificity_ratio, 10) / 10 + 0.3 * min(efficacy, 50) / 50
        
        return combined_score

def generate_circuit_diagram(circuit_config):
    """
    Generate a visual diagram of the synthetic biology circuit.
    
    Args:
        circuit_config (dict): Circuit configuration
        
    Returns:
        PIL.Image: Circuit diagram image
    """
    circuit_type = circuit_config['circuit_type']
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Circuit elements
    promoters = []
    rbs = []
    coding_sequences = []
    terminators = []
    arrows = []
    text_elements = []
    
    # Set up basic circuit components based on type
    if "pH" in circuit_type and "Quorum" in circuit_type:
        # Combined pH and QS circuit
        
        # pH sensing module
        promoters.append({"x": 1, "y": 2, "color": "green", "label": "pH-sensitive\nPromoter"})
        rbs.append({"x": 2, "y": 2, "label": "RBS"})
        coding_sequences.append({"x": 3, "y": 2, "width": 1.5, "label": "Sensor\nProtein", "color": "lightgreen"})
        terminators.append({"x": 5, "y": 2})
        
        # Quorum sensing module
        promoters.append({"x": 1, "y": 1, "color": "blue", "label": "QS\nPromoter"})
        rbs.append({"x": 2, "y": 1, "label": "RBS"})
        coding_sequences.append({"x": 3, "y": 1, "width": 1.5, "label": "QS\nSensor", "color": "lightblue"})
        terminators.append({"x": 5, "y": 1})
        
        # Drug release module (AND gate)
        promoters.append({"x": 6, "y": 1.5, "color": "red", "label": "AND Gate\nPromoter"})
        rbs.append({"x": 7, "y": 1.5, "label": "RBS"})
        coding_sequences.append({"x": 8, "y": 1.5, "width": 2, "label": "Drug\nEffector", "color": "pink"})
        terminators.append({"x": 10.5, "y": 1.5})
        
        # Arrows showing regulation
        arrows.append({"start": (5, 2), "end": (6, 1.7), "color": "green"})
        arrows.append({"start": (5, 1), "end": (6, 1.3), "color": "blue"})
        
        # Environmental inputs
        text_elements.append({"x": 0.5, "y": 2.3, "text": "Low pH\nInput", "color": "green"})
        text_elements.append({"x": 0.5, "y": 0.7, "text": "High Cell\nDensity", "color": "blue"})
        
        # Output
        text_elements.append({"x": 11, "y": 1.5, "text": "Drug\nRelease", "color": "red"})
        
    elif "pH" in circuit_type:
        # pH sensitive circuit
        
        # pH sensing module
        promoters.append({"x": 1, "y": 1.5, "color": "green", "label": "pH-sensitive\nPromoter"})
        rbs.append({"x": 2, "y": 1.5, "label": "RBS"})
        coding_sequences.append({"x": 3, "y": 1.5, "width": 2, "label": "pH Sensor &\nDrug Effector", "color": "lightgreen"})
        terminators.append({"x": 5.5, "y": 1.5})
        
        # Environmental input
        text_elements.append({"x": 0.5, "y": 1.8, "text": "Low pH\nInput", "color": "green"})
        
        # Output
        text_elements.append({"x": 6, "y": 1.5, "text": "Drug\nRelease", "color": "red"})
        
    elif "Quorum" in circuit_type:
        # Quorum sensing circuit
        
        # Quorum sensing module
        promoters.append({"x": 1, "y": 1.5, "color": "blue", "label": "QS\nPromoter"})
        rbs.append({"x": 2, "y": 1.5, "label": "RBS"})
        coding_sequences.append({"x": 3, "y": 1.5, "width": 1.5, "label": "QS\nSensor", "color": "lightblue"})
        terminators.append({"x": 5, "y": 1.5})
        
        # Drug release module
        promoters.append({"x": 6, "y": 1.5, "color": "purple", "label": "Response\nPromoter"})
        rbs.append({"x": 7, "y": 1.5, "label": "RBS"})
        coding_sequences.append({"x": 8, "y": 1.5, "width": 2, "label": "Drug\nEffector", "color": "pink"})
        terminators.append({"x": 10.5, "y": 1.5})
        
        # Arrow showing regulation
        arrows.append({"start": (5, 1.5), "end": (6, 1.5), "color": "blue"})
        
        # Environmental input
        text_elements.append({"x": 0.5, "y": 1.8, "text": "High Cell\nDensity", "color": "blue"})
        
        # Output
        text_elements.append({"x": 11, "y": 1.5, "text": "Drug\nRelease", "color": "red"})
        
    else:  # Custom circuit
        # Generic custom circuit
        
        # Sensor module
        promoters.append({"x": 1, "y": 1.5, "color": "gray", "label": "Sensor\nPromoter"})
        rbs.append({"x": 2, "y": 1.5, "label": "RBS"})
        coding_sequences.append({"x": 3, "y": 1.5, "width": 1.5, "label": "Custom\nSensor", "color": "lightgray"})
        terminators.append({"x": 5, "y": 1.5})
        
        # Response module
        promoters.append({"x": 6, "y": 1.5, "color": "purple", "label": "Response\nPromoter"})
        rbs.append({"x": 7, "y": 1.5, "label": "RBS"})
        coding_sequences.append({"x": 8, "y": 1.5, "width": 2, "label": "Drug\nEffector", "color": "pink"})
        terminators.append({"x": 10.5, "y": 1.5})
        
        # Arrow showing regulation
        arrows.append({"start": (5, 1.5), "end": (6, 1.5), "color": "gray"})
        
        # Custom input
        text_elements.append({"x": 0.5, "y": 1.8, "text": "Custom\nInput", "color": "gray"})
        
        # Output
        text_elements.append({"x": 11, "y": 1.5, "text": "Drug\nRelease", "color": "red"})
    
    # Draw backbone
    # Simplified for diagram - just draw lines for each module
    for idx, promoter in enumerate(promoters):
        y = promoter["y"]
        
        # Find the last element in this row
        if idx < len(terminators):
            end_x = terminators[idx]["x"]
        else:
            end_x = promoter["x"] + 4
        
        # Draw backbone line
        ax.plot([promoter["x"] - 0.5, end_x + 0.5], [y, y], color="black", linewidth=2, zorder=1)
    
    # Draw circuit elements
    # Promoters (arrow-like triangles)
    for p in promoters:
        x, y = p["x"], p["y"]
        color = p.get("color", "black")
        
        # Draw promoter arrow
        vertices = [
            (x - 0.2, y - 0.3),  # bottom left
            (x + 0.2, y),        # point
            (x - 0.2, y + 0.3)   # top left
        ]
        
        promoter_polygon = Polygon(vertices, facecolor=color, edgecolor="black", zorder=2)
        ax.add_patch(promoter_polygon)
        
        # Add label
        if "label" in p:
            ax.text(x, y + 0.4, p["label"], ha="center", va="bottom", fontsize=8)
    
    # RBS (half circles)
    for r in rbs:
        x, y = r["x"], r["y"]
        
        # Draw RBS semicircle
        rbs_arc = plt.Circle((x, y), 0.15, facecolor="yellow", edgecolor="black", zorder=2)
        ax.add_patch(rbs_arc)
        
        # Add label
        if "label" in r:
            ax.text(x, y - 0.3, r["label"], ha="center", va="top", fontsize=8)
    
    # Coding sequences (rectangles)
    for c in coding_sequences:
        x, y = c["x"], c["y"]
        width = c.get("width", 1.0)
        color = c.get("color", "skyblue")
        
        # Draw CDS rectangle
        cds_rect = Rectangle((x, y - 0.25), width, 0.5, facecolor=color, edgecolor="black", zorder=2)
        ax.add_patch(cds_rect)
        
        # Add label
        if "label" in c:
            ax.text(x + width/2, y, c["label"], ha="center", va="center", fontsize=8)
    
    # Terminators (T shapes)
    for t in terminators:
        x, y = t["x"], t["y"]
        
        # Draw terminator
        ax.plot([x, x], [y - 0.3, y + 0.3], color="black", linewidth=2, zorder=2)
        ax.plot([x - 0.15, x + 0.15], [y + 0.3, y + 0.3], color="black", linewidth=2, zorder=2)
    
    # Arrows for regulation
    for a in arrows:
        start, end = a["start"], a["end"]
        color = a.get("color", "black")
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        ax.arrow(start[0], start[1], dx, dy, head_width=0.1, head_length=0.1, 
                 fc=color, ec=color, zorder=3, length_includes_head=True)
    
    # Text elements
    for t in text_elements:
        x, y = t["x"], t["y"]
        text = t["text"]
        color = t.get("color", "black")
        
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")
    
    # Add circuit parameters as text
    params = circuit_config['parameters']
    param_text = "\n".join([
        f"{k.replace('_', ' ').title()}: {v:.2f}" for k, v in params.items()
        if k not in ['payload_type']
    ])
    
    # Add payload type
    if 'payload_type' in params:
        param_text += f"\nPayload: {params['payload_type']}"
    
    # Add parameters box
    ax.text(11, 2.5, "Circuit Parameters:\n" + param_text, ha="left", va="top", fontsize=8,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Set plot properties
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    ax.set_title(f"{circuit_type} Bacterial Circuit", fontsize=14, fontweight="bold")
    
    # Add bacterial cell outline
    cell_outline = plt.Rectangle((0.2, 0.2), 11.6, 2.6, fill=False, linestyle='--', 
                                ec='gray', linewidth=2, zorder=0)
    ax.add_patch(cell_outline)
    ax.text(0.5, 0.4, "E. coli Cell", fontsize=10, color='gray')
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    
    # Convert to image
    buf.seek(0)
    img = Image.open(buf)
    
    return img

def simulate_drug_release(circuit_config, simulation_results):
    """
    Simulate drug release profiles in tumor vs. normal tissue.
    
    Args:
        circuit_config (dict): Circuit configuration
        simulation_results (dict): Results from bacterial circuit simulations
        
    Returns:
        pandas.DataFrame: Drug release data over time in different tissues
    """
    # Extract results
    tumor_results = simulation_results.get("Tumor", None)
    normal_results = simulation_results.get("Normal", None)
    
    # If we don't have both tumor and normal results, use the first two results
    if tumor_results is None or normal_results is None:
        results = list(simulation_results.values())
        if len(results) >= 2:
            tumor_results = results[0]
            normal_results = results[1]
        else:
            # Only one environment, duplicate with lower values for demonstration
            tumor_results = results[0]
            normal_results = results[0].copy()
            normal_results['drug_concentration'] = normal_results['drug_concentration'] * 0.2
    
    # Create drug release DataFrame
    drug_release_data = pd.DataFrame({
        'time': tumor_results['time'],
        'tumor_drug_level': tumor_results['drug_concentration'],
        'normal_drug_level': normal_results['drug_concentration']
    })
    
    # Calculate cumulative drug release
    drug_release_data['tumor_cumulative'] = np.cumsum(drug_release_data['tumor_drug_level']) * (
        drug_release_data['time'].iloc[1] - drug_release_data['time'].iloc[0])
    
    drug_release_data['normal_cumulative'] = np.cumsum(drug_release_data['normal_drug_level']) * (
        drug_release_data['time'].iloc[1] - drug_release_data['time'].iloc[0])
    
    return drug_release_data
