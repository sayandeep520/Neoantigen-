"""
Advanced Synthetic Biology Circuit Modeling Module

This module extends the basic synthetic biology simulation capabilities with:
1. Modular genetic circuit design with interchangeable parts
2. Multi-layer regulatory networks
3. Advanced dynamics including time delays and stochastic effects
4. Evolutionary circuit optimization
5. Cell-cell communication models
6. Integration with CRISPR targeting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow, PathPatch
from matplotlib.path import Path
import io
from PIL import Image
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# Genetic part definitions for modular circuit design
class GeneticPart:
    """Base class for genetic parts used in synthetic circuits"""
    def __init__(self, name, efficiency=1.0, leakiness=0.0):
        self.name = name
        self.efficiency = efficiency
        self.leakiness = leakiness
        
    def get_activity(self, inputs=None, env=None):
        """Get the activity level of this genetic part (0.0-1.0)"""
        return self.leakiness

class Promoter(GeneticPart):
    """Promoter element that initiates transcription"""
    def __init__(self, name, efficiency=1.0, leakiness=0.0, regulators=None):
        super().__init__(name, efficiency, leakiness)
        self.regulators = regulators or []
        
    def get_activity(self, inputs=None, env=None):
        if not inputs or not self.regulators:
            return self.leakiness
        
        # Calculate regulation effect
        regulation = 0
        for regulator in self.regulators:
            if regulator['type'] == 'activator':
                if regulator['name'] in inputs:
                    # Hill function for activation
                    conc = inputs[regulator['name']]
                    Kd = regulator.get('Kd', 0.5)
                    n = regulator.get('n', 2)
                    regulation += conc**n / (Kd**n + conc**n)
            elif regulator['type'] == 'repressor':
                if regulator['name'] in inputs:
                    # Hill function for repression
                    conc = inputs[regulator['name']]
                    Kd = regulator.get('Kd', 0.5)
                    n = regulator.get('n', 2)
                    regulation += Kd**n / (Kd**n + conc**n)
        
        # Normalize regulation effect
        if len(self.regulators) > 0:
            regulation /= len(self.regulators)
        
        # Apply efficiency
        activity = self.leakiness + (1 - self.leakiness) * regulation * self.efficiency
        return min(1.0, max(0.0, activity))

class RBS(GeneticPart):
    """Ribosome Binding Site controlling translation initiation"""
    def __init__(self, name, efficiency=1.0, leakiness=0.0):
        super().__init__(name, efficiency, leakiness)

class CodingSequence(GeneticPart):
    """Coding sequence for a protein or RNA product"""
    def __init__(self, name, efficiency=1.0, degradation_rate=0.1, mature_time=0.0):
        super().__init__(name, efficiency)
        self.degradation_rate = degradation_rate
        self.mature_time = mature_time  # Time delay for protein maturation

class Terminator(GeneticPart):
    """Terminator sequence that stops transcription"""
    def __init__(self, name, efficiency=0.95):
        super().__init__(name, efficiency)

# Environmental sensors
class PhSensor:
    """pH-responsive genetic element"""
    def __init__(self, threshold=6.5, sensitivity=0.5):
        self.threshold = threshold
        self.sensitivity = sensitivity
        
    def get_response(self, ph):
        """Calculate response to pH (0.0-1.0)"""
        return 1.0 / (1.0 + np.exp((ph - self.threshold) / self.sensitivity))

class QuorumSensor:
    """Quorum sensing genetic element"""
    def __init__(self, threshold=0.5, cooperativity=2.0):
        self.threshold = threshold
        self.cooperativity = cooperativity
        
    def get_response(self, density):
        """Calculate response to population density (0.0-1.0)"""
        return density**self.cooperativity / (self.threshold**self.cooperativity + density**self.cooperativity)

class OxygenSensor:
    """Oxygen-responsive genetic element"""
    def __init__(self, threshold=5.0, sensitivity=1.0, is_aerobic=True):
        self.threshold = threshold  # Oxygen concentration threshold in %
        self.sensitivity = sensitivity
        self.is_aerobic = is_aerobic  # True if active in high oxygen, False if active in low oxygen
        
    def get_response(self, oxygen_conc):
        """Calculate response to oxygen concentration (0.0-1.0)"""
        if self.is_aerobic:
            return oxygen_conc**2 / (self.threshold**2 + oxygen_conc**2)
        else:
            return self.threshold**2 / (self.threshold**2 + oxygen_conc**2)

class GlucoseSensor:
    """Glucose-responsive genetic element"""
    def __init__(self, threshold=1.0, sensitivity=1.0):
        self.threshold = threshold  # Glucose concentration threshold
        self.sensitivity = sensitivity
        
    def get_response(self, glucose_conc):
        """Calculate response to glucose concentration (0.0-1.0)"""
        return glucose_conc**2 / (self.threshold**2 + glucose_conc**2)

# Advanced genetic circuit
class GeneticCircuit:
    """A genetic circuit composed of multiple genetic parts"""
    def __init__(self, name, parts=None):
        self.name = name
        self.parts = parts or []
        
    def add_part(self, part):
        """Add a genetic part to the circuit"""
        self.parts.append(part)
        
    def simulate(self, environment, simulation_params):
        """Simulate the genetic circuit behavior"""
        # Implementation depends on circuit type and complexity
        pass

# Multi-layer regulatory circuit with time delays and stochastic effects
class AdvancedCircuitSimulator:
    """Simulator for advanced genetic circuits with multi-layer regulation"""
    
    def __init__(self, circuit_config=None):
        self.circuit_config = circuit_config or {}
        self.initialize_circuit()
        
    def initialize_circuit(self):
        """Initialize circuit components based on configuration"""
        circuit_type = self.circuit_config.get('circuit_type', 'Custom')
        params = self.circuit_config.get('parameters', {})
        
        # Create sensors
        self.sensors = {}
        
        if "pH" in circuit_type:
            self.sensors['ph'] = PhSensor(
                threshold=params.get('ph_threshold', 6.5),
                sensitivity=params.get('ph_sensitivity', 0.5)
            )
            
        if circuit_type != "pH-Sensitive Drug Release":
            self.sensors['quorum'] = QuorumSensor(
                threshold=params.get('qs_threshold', 0.5),
                cooperativity=params.get('qs_cooperativity', 2.0)
            )
            
        if params.get('use_oxygen_sensing', False):
            self.sensors['oxygen'] = OxygenSensor(
                threshold=params.get('oxygen_threshold', 5.0),
                sensitivity=params.get('oxygen_sensitivity', 1.0),
                is_aerobic=params.get('oxygen_sensor_type', 'anaerobic') == 'aerobic'
            )
            
        if params.get('use_glucose_sensing', False):
            self.sensors['glucose'] = GlucoseSensor(
                threshold=params.get('glucose_threshold', 1.0),
                sensitivity=params.get('glucose_sensitivity', 1.0)
            )
            
        # Create genetic parts and circuits
        self.create_circuit_from_config()
        
    def create_circuit_from_config(self):
        """Create the genetic circuit based on configuration"""
        circuit_type = self.circuit_config.get('circuit_type', 'Custom')
        params = self.circuit_config.get('parameters', {})
        
        # Initialize circuit
        self.circuit = GeneticCircuit(name=circuit_type)
        
        # Architecture depends on circuit type
        if "pH" in circuit_type and "Quorum" in circuit_type:
            # Combined pH and QS circuit (AND gate)
            self._create_combined_circuit(params)
        elif "pH" in circuit_type:
            # pH-responsive circuit
            self._create_ph_circuit(params)
        elif "Quorum" in circuit_type:
            # Quorum sensing circuit
            self._create_quorum_circuit(params)
        else:
            # Custom circuit
            self._create_custom_circuit(params)
            
    def _create_combined_circuit(self, params):
        """Create a combined pH and quorum sensing circuit"""
        # pH-responsive promoter
        ph_promoter = Promoter(
            name="pH_responsive_promoter",
            efficiency=params.get('promoter_strength', 1.0),
            leakiness=params.get('leakiness', 0.0)
        )
        
        # Quorum sensing promoter
        qs_promoter = Promoter(
            name="quorum_sensing_promoter",
            efficiency=params.get('promoter_strength', 1.0),
            leakiness=params.get('leakiness', 0.0)
        )
        
        # AND gate promoter
        and_promoter = Promoter(
            name="AND_gate_promoter",
            efficiency=params.get('promoter_strength', 1.0),
            leakiness=params.get('leakiness', 0.0),
            regulators=[
                {'name': 'pH_sensor', 'type': 'activator', 'Kd': 0.3, 'n': 2},
                {'name': 'quorum_sensor', 'type': 'activator', 'Kd': 0.3, 'n': 2}
            ]
        )
        
        # RBS elements
        ph_rbs = RBS(
            name="pH_sensor_RBS",
            efficiency=params.get('ribosome_binding_strength', 1.0)
        )
        
        qs_rbs = RBS(
            name="quorum_sensor_RBS",
            efficiency=params.get('ribosome_binding_strength', 1.0)
        )
        
        drug_rbs = RBS(
            name="drug_effector_RBS",
            efficiency=params.get('ribosome_binding_strength', 1.0)
        )
        
        # Coding sequences
        ph_sensor_cds = CodingSequence(
            name="pH_sensor",
            efficiency=1.0,
            degradation_rate=params.get('protein_degradation_rate', 0.1)
        )
        
        qs_sensor_cds = CodingSequence(
            name="quorum_sensor",
            efficiency=1.0,
            degradation_rate=params.get('protein_degradation_rate', 0.1)
        )
        
        drug_effector_cds = CodingSequence(
            name="drug_effector",
            efficiency=1.0,
            degradation_rate=params.get('protein_degradation_rate', 0.1),
            mature_time=params.get('protein_maturation_time', 0.5)
        )
        
        # Terminators
        ph_terminator = Terminator(name="pH_module_terminator")
        qs_terminator = Terminator(name="QS_module_terminator")
        drug_terminator = Terminator(name="drug_module_terminator")
        
        # Add parts to circuit
        for part in [
            ph_promoter, ph_rbs, ph_sensor_cds, ph_terminator,
            qs_promoter, qs_rbs, qs_sensor_cds, qs_terminator,
            and_promoter, drug_rbs, drug_effector_cds, drug_terminator
        ]:
            self.circuit.add_part(part)
    
    def _create_ph_circuit(self, params):
        """Create a pH-responsive circuit"""
        # Add implementation here
        pass
        
    def _create_quorum_circuit(self, params):
        """Create a quorum sensing circuit"""
        # Add implementation here
        pass
        
    def _create_custom_circuit(self, params):
        """Create a custom circuit"""
        # Add implementation here
        pass
        
    def simulate_circuit(self, environment, simulation_params):
        """
        Simulate the genetic circuit under specific environmental conditions
        
        Args:
            environment (dict): Environmental conditions (pH, population density, etc.)
            simulation_params (dict): Simulation parameters (time, step size, etc.)
            
        Returns:
            pandas.DataFrame: Simulation results over time
        """
        # Extract simulation parameters
        simulation_time = simulation_params.get('simulation_time', 48)  # hours
        time_step = simulation_params.get('time_step', 0.5)  # hours
        stochastic = simulation_params.get('stochastic', False)
        population_initial = simulation_params.get('population_initial', 0.1)
        
        # Create time points
        time_points = np.arange(0, simulation_time + time_step, time_step)
        
        # Extract environment parameters
        env_name = environment.get('name', 'Generic')
        env_ph = environment.get('ph', 7.4 if env_name == "Normal" else 6.0)
        env_population_density = environment.get('population_density', 0.2 if env_name == "Normal" else 0.8)
        env_oxygen = environment.get('oxygen', 8.0 if env_name == "Normal" else 2.0)  # oxygen in %
        env_glucose = environment.get('glucose', 1.0 if env_name == "Normal" else 0.5)  # relative glucose level
        
        # Initialize arrays for results
        population = np.zeros(len(time_points))
        drug_concentration = np.zeros(len(time_points))
        sensor_proteins = {}
        
        # Set initial conditions
        population[0] = population_initial
        drug_concentration[0] = 0
        
        # Extract parameters
        params = self.circuit_config.get('parameters', {})
        growth_rate = params.get('growth_rate', 0.5)
        max_drug_production = params.get('max_drug_production', 2.0)
        drug_degradation = params.get('drug_degradation', 0.1)
        
        # Simulate circuit dynamics
        for i in range(1, len(time_points)):
            # Calculate time delta
            dt = time_step
            
            # Population growth (logistic model)
            population_growth = growth_rate * population[i-1] * (1 - population[i-1])
            
            # Add randomness if stochastic
            if stochastic:
                population_growth += np.random.normal(0, 0.02)
            
            population[i] = population[i-1] + population_growth * dt
            
            # Ensure population stays in valid range
            population[i] = np.clip(population[i], 0, 1)
            
            # Calculate sensor responses
            sensor_responses = {}
            
            if 'ph' in self.sensors:
                sensor_responses['ph'] = self.sensors['ph'].get_response(env_ph)
                
            if 'quorum' in self.sensors:
                sensor_responses['quorum'] = self.sensors['quorum'].get_response(population[i])
                
            if 'oxygen' in self.sensors:
                sensor_responses['oxygen'] = self.sensors['oxygen'].get_response(env_oxygen)
                
            if 'glucose' in self.sensors:
                sensor_responses['glucose'] = self.sensors['glucose'].get_response(env_glucose)
            
            # Calculate circuit activation based on type
            circuit_type = self.circuit_config.get('circuit_type', 'Custom')
            leakiness = params.get('leakiness', 0.0)
            
            circuit_activation = leakiness  # Baseline activity
            
            if "pH" in circuit_type and "Quorum" in circuit_type:
                # Combined AND-gate circuit
                if 'ph' in sensor_responses and 'quorum' in sensor_responses:
                    ph_response = sensor_responses['ph']
                    qs_response = sensor_responses['quorum']
                    
                    # AND gate logic (both inputs must be high)
                    circuit_activation += (1 - leakiness) * ph_response * qs_response
            elif "pH" in circuit_type:
                # pH-only circuit
                if 'ph' in sensor_responses:
                    ph_response = sensor_responses['ph']
                    circuit_activation += (1 - leakiness) * ph_response
            elif "Quorum" in circuit_type:
                # Quorum sensing-only circuit
                if 'quorum' in sensor_responses:
                    qs_response = sensor_responses['quorum']
                    circuit_activation += (1 - leakiness) * qs_response
            
            # Drug production and degradation
            drug_production = max_drug_production * circuit_activation * population[i]
            drug_degradation_rate = drug_degradation * drug_concentration[i-1]
            
            drug_concentration[i] = drug_concentration[i-1] + (drug_production - drug_degradation_rate) * dt
            
            # Add randomness if stochastic
            if stochastic:
                drug_concentration[i] += np.random.normal(0, 0.05 * drug_concentration[i-1])
                drug_concentration[i] = max(0, drug_concentration[i])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'time': time_points,
            'population': population,
            'drug_concentration': drug_concentration,
            'pH': np.ones(len(time_points)) * env_ph
        })
        
        # Add sensor response data if available
        for sensor_name, response in sensor_responses.items():
            results[f'{sensor_name}_response'] = response
        
        return results

def simulate_advanced_circuit(circuit_config, environment, simulation_params):
    """
    Wrapper function to simulate an advanced genetic circuit
    
    Args:
        circuit_config (dict): Configuration of the genetic circuit
        environment (dict): Environmental conditions
        simulation_params (dict): Simulation parameters
        
    Returns:
        pandas.DataFrame: Simulation results over time
    """
    simulator = AdvancedCircuitSimulator(circuit_config)
    return simulator.simulate_circuit(environment, simulation_params)

def generate_advanced_circuit_diagram(circuit_config):
    """
    Generate a more detailed visual diagram of the synthetic biology circuit
    
    Args:
        circuit_config (dict): Circuit configuration
        
    Returns:
        PIL.Image: Circuit diagram image
    """
    # This is a placeholder for a more advanced implementation
    # that would generate a more detailed circuit diagram
    # For now, we'll use the existing function
    from utils.synthetic_biology_utils import generate_circuit_diagram
    return generate_circuit_diagram(circuit_config)

def evolutionary_circuit_optimization(circuit_config, objective, environments, iterations=100):
    """
    Optimize bacterial circuit parameters using differential evolution
    
    Args:
        circuit_config (dict): Initial circuit configuration
        objective (str): Optimization objective
        environments (list): List of environments to test
        iterations (int): Number of optimization iterations
        
    Returns:
        dict: Optimized circuit configuration
    """
    # Make a copy of the initial configuration
    initial_config = circuit_config.copy()
    initial_params = circuit_config.get('parameters', {}).copy()
    
    # Define parameter ranges for optimization
    param_ranges = {}
    
    # Add common parameters
    param_ranges.update({
        'growth_rate': (0.1, 2.0),
        'max_drug_production': (0.1, 10.0),
        'drug_degradation': (0.01, 1.0),
        'leakiness': (0.0, 0.3),
        'promoter_strength': (0.1, 10.0),
        'ribosome_binding_strength': (0.1, 10.0),
        'protein_degradation_rate': (0.01, 1.0),
        'protein_maturation_time': (0.0, 2.0)
    })
    
    # Add circuit-specific parameters
    if "pH" in circuit_config.get('circuit_type', ''):
        param_ranges.update({
            'ph_threshold': (4.0, 8.0),
            'ph_sensitivity': (0.1, 2.0)
        })
    
    if circuit_config.get('circuit_type', '') != "pH-Sensitive Drug Release":
        param_ranges.update({
            'qs_threshold': (0.1, 5.0),
            'qs_cooperativity': (1.0, 5.0)
        })
    
    # Set up simulation parameters
    simulation_params = {
        'simulation_time': 48,  # hours
        'time_step': 0.5,  # hours
        'population_initial': 0.1,
        'stochastic': False
    }
    
    # Get keys and bounds for parameters actually present in the config
    keys = []
    bounds = []
    for key, (min_val, max_val) in param_ranges.items():
        if key in initial_params:
            keys.append(key)
            bounds.append((min_val, max_val))
    
    # Define the objective function for optimization
    def objective_function(x):
        # Create a new configuration with the proposed parameters
        new_config = initial_config.copy()
        new_params = initial_params.copy()
        
        # Update parameters
        for i, key in enumerate(keys):
            new_params[key] = x[i]
        
        new_config['parameters'] = new_params
        
        # Evaluate the new configuration
        simulator = AdvancedCircuitSimulator(new_config)
        
        # Run simulations in each environment
        results = {}
        for env in environments:
            results[env['name']] = simulator.simulate_circuit(env, simulation_params)
        
        # Calculate score based on objective
        if objective == "Maximize Tumor Specificity":
            # Compare drug concentration in tumor vs. normal tissue
            tumor_env = next((env for env in environments if env['name'] == "Tumor"), environments[0])
            normal_env = next((env for env in environments if env['name'] == "Normal"), environments[-1])
            
            tumor_auc = np.trapz(results[tumor_env['name']]['drug_concentration'], results[tumor_env['name']]['time'])
            normal_auc = np.trapz(results[normal_env['name']]['drug_concentration'], results[normal_env['name']]['time'])
            
            # Avoid division by zero
            normal_auc = max(normal_auc, 0.001)
            
            # Calculate tumor specificity ratio (maximize)
            specificity_ratio = tumor_auc / normal_auc
            
            # We want to maximize, but differential_evolution minimizes
            return -specificity_ratio
        
        elif objective == "Maximize Drug Release":
            # Calculate total drug release across all environments
            total_auc = 0
            for env_name, result in results.items():
                auc = np.trapz(result['drug_concentration'], result['time'])
                
                # Weight tumor environment more heavily
                if "Tumor" in env_name:
                    auc *= 2
                    
                total_auc += auc
                
            # We want to maximize, but differential_evolution minimizes
            return -total_auc
        
        elif objective == "Minimize Off-Target Release":
            # Calculate drug release in non-tumor environments
            off_target_auc = 0
            for env_name, result in results.items():
                if "Normal" in env_name or "Blood" in env_name:
                    auc = np.trapz(result['drug_concentration'], result['time'])
                    off_target_auc += auc
            
            # We want to minimize off-target release
            return off_target_auc
        
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
            
            # We want to maximize, but differential_evolution minimizes
            return -combined_score
    
    # Run differential evolution
    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=iterations // 10,  # Fewer iterations needed for differential evolution
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42
    )
    
    # Create optimized configuration
    optimized_config = initial_config.copy()
    optimized_params = initial_params.copy()
    
    # Update parameters with optimized values
    for i, key in enumerate(keys):
        optimized_params[key] = result.x[i]
    
    optimized_config['parameters'] = optimized_params
    
    return optimized_config

def simulate_multicellular_dynamics(circuit_config, environment, simulation_params):
    """
    Simulate bacterial circuit with cell-cell communication
    
    Args:
        circuit_config (dict): Circuit configuration
        environment (dict): Environmental conditions
        simulation_params (dict): Simulation parameters
        
    Returns:
        pandas.DataFrame: Simulation results over time
    """
    # Extract parameters
    params = circuit_config.get('parameters', {})
    population_count = simulation_params.get('population_count', 5)
    simulation_time = simulation_params.get('simulation_time', 48)
    time_step = simulation_params.get('time_step', 0.5)
    
    # Create time points
    time_points = np.arange(0, simulation_time + time_step, time_step)
    
    # Initialize population array [time, cell]
    populations = np.zeros((len(time_points), population_count))
    drug_concentrations = np.zeros((len(time_points), population_count))
    
    # Set initial populations (slightly different for each cell)
    initial_population = simulation_params.get('population_initial', 0.1)
    for i in range(population_count):
        populations[0, i] = initial_population * (0.8 + 0.4 * np.random.random())
    
    # Diffusion rate for communication molecules
    diffusion_rate = params.get('diffusion_rate', 0.1)
    
    # Simulate dynamics
    for t in range(1, len(time_points)):
        # Calculate global quorum sensing molecule concentration
        total_population = np.sum(populations[t-1, :])
        global_qs_conc = total_population / population_count
        
        for cell in range(population_count):
            # Local population density effects
            local_density = populations[t-1, cell]
            
            # Combined density (weighted average of local and global)
            effective_density = (1 - diffusion_rate) * local_density + diffusion_rate * global_qs_conc
            
            # Create local environment with cell-specific values
            local_env = environment.copy()
            local_env['population_density'] = effective_density
            
            # Get single-time-step simulation for this cell
            simulator = AdvancedCircuitSimulator(circuit_config)
            cell_results = simulator.simulate_circuit(
                local_env,
                {'simulation_time': time_step, 'time_step': time_step, 'population_initial': local_density}
            )
            
            # Update population and drug concentration
            populations[t, cell] = cell_results['population'].iloc[-1]
            drug_concentrations[t, cell] = cell_results['drug_concentration'].iloc[-1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'time': time_points,
        'population_mean': np.mean(populations, axis=1),
        'population_std': np.std(populations, axis=1),
        'drug_concentration_mean': np.mean(drug_concentrations, axis=1),
        'drug_concentration_std': np.std(drug_concentrations, axis=1)
    })
    
    # Store all cell data
    for i in range(population_count):
        results[f'cell_{i}_population'] = populations[:, i]
        results[f'cell_{i}_drug'] = drug_concentrations[:, i]
    
    return results