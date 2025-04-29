import numpy as np
import pandas as pd
import streamlit as st
import time
from typing import Dict, List, Union, Tuple, Optional, Any

# Import simulation libraries if available
try:
    import simplesbml
except ImportError:
    pass

from utils.molecule_generation import MoleculeGenerator, setup_cuda_environment

class SyntheticBiologySimulator:
    """
    Simulator for synthetic biology circuits in E. coli for cancer immunotherapy.
    Focuses on pH-sensitive promoters, quorum sensing, and controlled drug release.
    """

    def __init__(self):
        """Initialize the simulator with default parameters"""
        self.molecule_gen = MoleculeGenerator()
        self.cuda_available = setup_cuda_environment()
        self.ph_response_promoters = {
            'asr': {'min_ph': 3.5, 'max_ph': 5.5, 'leakiness': 0.05},
            'cadBA': {'min_ph': 5.0, 'max_ph': 6.5, 'leakiness': 0.02},
            'hdeAB': {'min_ph': 3.0, 'max_ph': 6.0, 'leakiness': 0.01},
            'gadAB': {'min_ph': 3.8, 'max_ph': 5.8, 'leakiness': 0.03},
        }

        self.quorum_sensing_systems = {
            'las': {'threshold': 20, 'max_activation': 0.95, 'cooperativity': 2.5},
            'rhl': {'threshold': 15, 'max_activation': 0.85, 'cooperativity': 2.0},
            'lux': {'threshold': 10, 'max_activation': 0.90, 'cooperativity': 1.8},
            'tra': {'threshold': 25, 'max_activation': 0.92, 'cooperativity': 3.0},
        }

        self.drug_release_modules = {
            'direct_lysis': {'efficiency': 0.80, 'timing': 'immediate', 'controllability': 'low'},
            'secretion': {'efficiency': 0.65, 'timing': 'sustained', 'controllability': 'medium'},
            'outer_membrane_vesicles': {'efficiency': 0.75, 'timing': 'delayed', 'controllability': 'high'},
            'nanobody_display': {'efficiency': 0.70, 'timing': 'continuous', 'controllability': 'high'},
        }

        self.sbml_model = None
        self.model_parameters = {}

    def create_circuit_model(self, 
                             ph_promoter: str, 
                             quorum_module: str,
                             drug_release: str,
                             tumor_targeting: bool = True) -> Dict[str, Any]:
        """
        Create a synthetic biology circuit model for cancer therapy

        Args:
            ph_promoter: pH-sensitive promoter (e.g., 'asr', 'cadBA')
            quorum_module: Quorum sensing system (e.g., 'las', 'rhl')
            drug_release: Drug release method (e.g., 'direct_lysis', 'secretion')
            tumor_targeting: Whether to include tumor-targeting mechanisms

        Returns:
            Dictionary with circuit design parameters
        """
        # Validate inputs
        if ph_promoter not in self.ph_response_promoters:
            st.error(f"Unknown pH-sensitive promoter: {ph_promoter}")
            ph_promoter = list(self.ph_response_promoters.keys())[0]
            st.info(f"Using {ph_promoter} as default")

        if quorum_module not in self.quorum_sensing_systems:
            st.error(f"Unknown quorum sensing system: {quorum_module}")
            quorum_module = list(self.quorum_sensing_systems.keys())[0]
            st.info(f"Using {quorum_module} as default")

        if drug_release not in self.drug_release_modules:
            st.error(f"Unknown drug release module: {drug_release}")
            drug_release = list(self.drug_release_modules.keys())[0]
            st.info(f"Using {drug_release} as default")

        # Create the circuit model
        circuit = {
            'ph_promoter': {
                'name': ph_promoter,
                'parameters': self.ph_response_promoters[ph_promoter]
            },
            'quorum_module': {
                'name': quorum_module,
                'parameters': self.quorum_sensing_systems[quorum_module]
            },
            'drug_release': {
                'name': drug_release,
                'parameters': self.drug_release_modules[drug_release]
            },
            'tumor_targeting': tumor_targeting,
            'efficiency': self._calculate_circuit_efficiency(
                ph_promoter, quorum_module, drug_release, tumor_targeting
            )
        }

        # Store model parameters
        self.model_parameters = circuit

        # Attempt to create SBML model if simplesbml is available
        try:
            self.sbml_model = self._create_sbml_model(circuit)
        except:
            self.sbml_model = None
            st.warning("SBML model creation failed. simplesbml library not available.")

        return circuit

    def _calculate_circuit_efficiency(self, 
                                     ph_promoter: str, 
                                     quorum_module: str,
                                     drug_release: str,
                                     tumor_targeting: bool) -> Dict[str, float]:
        """
        Calculate the efficiency of the synthetic circuit

        Args:
            ph_promoter: pH-sensitive promoter
            quorum_module: Quorum sensing system
            drug_release: Drug release method
            tumor_targeting: Whether tumor targeting is included

        Returns:
            Dictionary with efficiency metrics
        """
        # Get parameters for each component
        ph_params = self.ph_response_promoters[ph_promoter]
        qs_params = self.quorum_sensing_systems[quorum_module]
        dr_params = self.drug_release_modules[drug_release]

        # Base efficiencies
        ph_efficiency = 1.0 - (ph_params['leakiness'] * 2)  # Higher leakiness reduces specificity
        qs_efficiency = qs_params['max_activation'] * (qs_params['cooperativity'] / 3.0)  # Normalized to [0, 1]
        dr_efficiency = dr_params['efficiency']

        # Tumor targeting bonus (if included)
        targeting_factor = 1.2 if tumor_targeting else 1.0

        # Final efficiency metrics
        specificity = ph_efficiency * targeting_factor
        potency = dr_efficiency * 0.9 + qs_efficiency * 0.1  # Drug release dominates potency
        controllability = qs_efficiency * 0.7 + 0.3  # Quorum sensing gives control

        # Safety is inverse of leakiness
        safety = 1.0 - (ph_params['leakiness'] * 3)  # More sensitive to leakiness

        # Overall score (weighted average)
        overall = 0.3 * specificity + 0.3 * potency + 0.2 * controllability + 0.2 * safety

        return {
            'specificity': min(1.0, max(0.0, specificity)),
            'potency': min(1.0, max(0.0, potency)),
            'controllability': min(1.0, max(0.0, controllability)),
            'safety': min(1.0, max(0.0, safety)),
            'overall': min(1.0, max(0.0, overall))
        }

    def _create_sbml_model(self, circuit: Dict[str, Any]) -> Any:
        """
        Create an SBML model of the synthetic circuit

        Args:
            circuit: Circuit design parameters

        Returns:
            SimpleSBML model object
        """
        try:
            # Check if simplesbml is available
            import simplesbml

            # Create a new SBML model
            model = simplesbml.SbmlModel()

            # Add compartments
            model.addCompartment(volume=1, comp_id='cell')
            model.addCompartment(volume=10, comp_id='environment')

            # Add species for key components
            # Promoter components
            model.addSpecies('pHsensor', 0.0, comp='cell')
            model.addSpecies('ActivePromoter', 0.0, comp='cell')

            # Quorum sensing components
            model.addSpecies('SignalMolecule', 0.0, comp='cell')
            model.addSpecies('ExternalSignal', 0.0, comp='environment')
            model.addSpecies('BoundReceptor', 0.0, comp='cell')

            # Drug release components
            model.addSpecies('TherapeuticAgent', 0.0, comp='cell')
            model.addSpecies('ReleasedDrug', 0.0, comp='environment')

            # Add parameters for circuit
            ph_params = circuit['ph_promoter']['parameters']
            qs_params = circuit['quorum_module']['parameters']
            dr_params = circuit['drug_release']['parameters']

            model.addParameter('min_ph', ph_params['min_ph'])
            model.addParameter('max_ph', ph_params['max_ph'])
            model.addParameter('leakiness', ph_params['leakiness'])

            model.addParameter('qs_threshold', qs_params['threshold'])
            model.addParameter('qs_cooperativity', qs_params['cooperativity'])
            model.addParameter('qs_max_activation', qs_params['max_activation'])

            model.addParameter('dr_efficiency', dr_params['efficiency'])

            # Add parameters for environmental conditions
            model.addParameter('environment_ph', 6.8)  # Normal pH
            model.addParameter('tumor_ph', 6.0)  # Tumor pH

            model.addParameter('cell_density', 0.1)  # Initial cell density
            model.addParameter('growth_rate', 0.05)  # Cell growth rate

            # Add parameter for tumor targeting
            model.addParameter('tumor_targeting', 1.0 if circuit['tumor_targeting'] else 0.0)

            # Add reactions for pH sensing
            model.addReaction(
                ['environment_ph'], ['pHsensor'],
                'pHsensor_activation', 
                '1/(1 + exp((environment_ph - (min_ph + max_ph)/2)/(max_ph - min_ph)))'
            )

            # Add reactions for promoter activation
            model.addReaction(
                ['pHsensor'], ['ActivePromoter'],
                'promoter_activation',
                'pHsensor * (1 - leakiness) + leakiness'
            )

            # Add reactions for signal molecule production
            model.addReaction(
                ['ActivePromoter'], ['SignalMolecule'],
                'signal_production',
                'ActivePromoter * 2.0 * (1 + tumor_targeting * 0.5)'
            )

            # Add reactions for signal export
            model.addReaction(
                ['SignalMolecule'], ['ExternalSignal'],
                'signal_export',
                'SignalMolecule * 0.5'
            )

            # Add reactions for quorum sensing
            model.addReaction(
                ['ExternalSignal'], ['BoundReceptor'],
                'quorum_sensing',
                '(ExternalSignal^qs_cooperativity)/((qs_threshold^qs_cooperativity) + (ExternalSignal^qs_cooperativity)) * qs_max_activation'
            )

            # Add reactions for therapeutic agent production
            model.addReaction(
                ['BoundReceptor', 'ActivePromoter'], ['TherapeuticAgent'],
                'therapeutic_production',
                'BoundReceptor * ActivePromoter * 3.0'
            )

            # Add reactions for drug release
            model.addReaction(
                ['TherapeuticAgent'], ['ReleasedDrug'],
                'drug_release',
                'TherapeuticAgent * dr_efficiency * (1 + tumor_targeting * 0.2)'
            )

            return model

        except (ImportError, Exception) as e:
            st.error(f"Failed to create SBML model: {str(e)}")
            return None

    def simulate_circuit(self, 
                        ph_values: List[float] = None,
                        population_density: List[float] = None,
                        time_points: int = 100,
                        simulation_time: float = 24.0) -> pd.DataFrame:
        """
        Simulate the behavior of the synthetic circuit

        Args:
            ph_values: List of pH values to simulate
            population_density: List of bacterial population densities
            time_points: Number of time points to simulate
            simulation_time: Total simulation time (in hours)

        Returns:
            DataFrame with simulation results
        """
        if not self.model_parameters:
            st.error("No circuit model created. Please create a circuit first.")
            return pd.DataFrame()

        # Default pH values if not provided
        if ph_values is None:
            ph_values = [7.4, 7.0, 6.5, 6.0, 5.5, 5.0]

        # Default population densities if not provided
        if population_density is None:
            population_density = [0.01, 0.1, 0.5, 1.0]

        # Create time points
        times = np.linspace(0, simulation_time, time_points)

        # Create result storage
        results = []

        # Simulate for each combination of pH and population density
        for ph in ph_values:
            for density in population_density:
                # Get circuit parameters
                ph_params = self.model_parameters['ph_promoter']['parameters']
                qs_params = self.model_parameters['quorum_module']['parameters']
                dr_params = self.model_parameters['drug_release']['parameters']

                # Calculate pH response
                ph_min = ph_params['min_ph']
                ph_max = ph_params['max_ph']
                leakiness = ph_params['leakiness']

                # Sigmoid pH response function
                ph_response = 1.0 / (1.0 + np.exp((ph - (ph_min + ph_max)/2) / ((ph_max - ph_min)/4)))
                ph_response = ph_response * (1.0 - leakiness) + leakiness

                # Calculate growth dynamics
                growth_rate = 0.05 * (1.0 - abs(ph - 7.0) / 2.0)  # Growth reduced at extreme pH
                population = density * np.exp(growth_rate * times)

                # Calculate quorum sensing activation
                qs_threshold = qs_params['threshold']
                qs_cooperativity = qs_params['cooperativity']
                qs_max = qs_params['max_activation']

                # Hill function for quorum sensing
                qs_activation = qs_max * (population**qs_cooperativity) / (qs_threshold**qs_cooperativity + population**qs_cooperativity)

                # Calculate drug production and release
                # Initial delay in drug production
                delay_mask = times < 1.0
                drug_production = np.zeros_like(times)
                drug_production[~delay_mask] = ph_response * qs_activation[~delay_mask] * (1.0 - np.exp(-(times[~delay_mask] - 1.0)))

                # Drug release efficiency
                drug_efficiency = dr_params['efficiency']

                # Calculate drug release based on timing characteristic
                if dr_params['timing'] == 'immediate':
                    drug_release = drug_production * drug_efficiency
                elif dr_params['timing'] == 'sustained':
                    # Apply a moving average for sustained release
                    window_size = max(1, int(time_points / 10))
                    drug_release = np.convolve(drug_production, np.ones(window_size)/window_size, mode='same') * drug_efficiency
                elif dr_params['timing'] == 'delayed':
                    # Apply a delay to release
                    delay = int(time_points / 5)
                    drug_release = np.zeros_like(drug_production)
                    drug_release[delay:] = drug_production[:-delay] * drug_efficiency if delay < len(drug_production) else drug_production * 0
                else:
                    # Default to continuous release
                    drug_release = drug_production * drug_efficiency * (0.8 + 0.2 * np.sin(times / 2.0))

                # Store results
                for i, t in enumerate(times):
                    results.append({
                        'time': t,
                        'pH': ph,
                        'population_density': population[i],
                        'pH_response': ph_response,
                        'quorum_activation': qs_activation[i],
                        'drug_production': drug_production[i],
                        'drug_release': drug_release[i]
                    })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def optimize_circuit(self, 
                        ph_target: float, 
                        population_target: float,
                        iterations: int = 10) -> Dict[str, Any]:
        """
        Optimize the synthetic circuit for specific target conditions

        Args:
            ph_target: Target pH value (e.g., tumor microenvironment pH)
            population_target: Target bacterial population density
            iterations: Number of optimization iterations

        Returns:
            Dictionary with optimized circuit parameters
        """
        st.info(f"Optimizing circuit for pH {ph_target} and population density {population_target}")

        # Initialize tracking of best parameters and performance
        best_score = 0.0
        best_circuit = None
        best_params = {}

        # All available options
        ph_promoters = list(self.ph_response_promoters.keys())
        quorum_modules = list(self.quorum_sensing_systems.keys())
        drug_releases = list(self.drug_release_modules.keys())

        # Track optimization progress
        progress_bar = st.progress(0)

        # Perform optimization iterations
        for i in range(iterations):
            # Update progress
            progress_bar.progress((i + 1) / iterations)

            # Try different combinations of parameters
            for ph_promoter in ph_promoters:
                for quorum_module in quorum_modules:
                    for drug_release in drug_releases:
                        for tumor_targeting in [True, False]:
                            # Create circuit with these parameters
                            circuit = self.create_circuit_model(
                                ph_promoter=ph_promoter,
                                quorum_module=quorum_module,
                                drug_release=drug_release,
                                tumor_targeting=tumor_targeting
                            )

                            # Simulate the circuit
                            sim_results = self.simulate_circuit(
                                ph_values=[ph_target],
                                population_density=[population_target],
                                time_points=50,
                                simulation_time=12.0
                            )

                            # Calculate performance score
                            if not sim_results.empty:
                                # Get final drug release value
                                final_drug_release = sim_results['drug_release'].iloc[-1]

                                # Get circuit efficiency metrics
                                specificity = circuit['efficiency']['specificity']
                                potency = circuit['efficiency']['potency']
                                controllability = circuit['efficiency']['controllability']
                                safety = circuit['efficiency']['safety']

                                # Calculate score based on drug release and efficiency metrics
                                score = (final_drug_release * 0.5 + 
                                        specificity * 0.2 + 
                                        potency * 0.1 + 
                                        controllability * 0.1 + 
                                        safety * 0.1)

                                # Update best parameters if score is better
                                if score > best_score:
                                    best_score = score
                                    best_circuit = circuit
                                    best_params = {
                                        'ph_promoter': ph_promoter,
                                        'quorum_module': quorum_module,
                                        'drug_release': drug_release,
                                        'tumor_targeting': tumor_targeting,
                                        'final_drug_release': final_drug_release,
                                        'score': score
                                    }

        # Clear progress bar
        progress_bar.empty()

        if best_circuit:
            st.success(f"Optimization complete! Best score: {best_score:.4f}")

            # Update model with best parameters
            self.create_circuit_model(
                ph_promoter=best_params['ph_promoter'],
                quorum_module=best_params['quorum_module'],
                drug_release=best_params['drug_release'],
                tumor_targeting=best_params['tumor_targeting']
            )

            return best_params
        else:
            st.error("Optimization failed. No valid circuit found.")
            return {}

    def export_circuit_design(self) -> Dict[str, Any]:
        """
        Export the synthetic biology circuit design

        Returns:
            Dictionary with full circuit specification
        """
        if not self.model_parameters:
            st.error("No circuit model created. Please create a circuit first.")
            return {}

        # Get model parameters
        circuit = self.model_parameters

        # Create detailed design specifications
        design = {
            'circuit_components': {
                'ph_sensor': {
                    'name': circuit['ph_promoter']['name'],
                    'min_ph': circuit['ph_promoter']['parameters']['min_ph'],
                    'max_ph': circuit['ph_promoter']['parameters']['max_ph'],
                    'leakiness': circuit['ph_promoter']['parameters']['leakiness'],
                    'description': f"pH-sensitive promoter active between pH {circuit['ph_promoter']['parameters']['min_ph']} and {circuit['ph_promoter']['parameters']['max_ph']}"
                },
                'quorum_sensing': {
                    'name': circuit['quorum_module']['name'],
                    'threshold': circuit['quorum_module']['parameters']['threshold'],
                    'cooperativity': circuit['quorum_module']['parameters']['cooperativity'],
                    'max_activation': circuit['quorum_module']['parameters']['max_activation'],
                    'description': f"Quorum sensing system with threshold at {circuit['quorum_module']['parameters']['threshold']} cell density"
                },
                'drug_delivery': {
                    'name': circuit['drug_release']['name'],
                    'efficiency': circuit['drug_release']['parameters']['efficiency'],
                    'timing': circuit['drug_release']['parameters']['timing'],
                    'controllability': circuit['drug_release']['parameters']['controllability'],
                    'description': f"{circuit['drug_release']['parameters']['timing'].capitalize()} drug release with {circuit['drug_release']['parameters']['efficiency']*100:.0f}% efficiency"
                }
            },
            'targeting': {
                'tumor_targeting': circuit['tumor_targeting'],
                'mechanism': "Surface display of tumor-targeting peptides" if circuit['tumor_targeting'] else "None"
            },
            'performance': {
                'specificity': circuit['efficiency']['specificity'],
                'potency': circuit['efficiency']['potency'],
                'controllability': circuit['efficiency']['controllability'],
                'safety': circuit['efficiency']['safety'],
                'overall': circuit['efficiency']['overall']
            },
            'genetic_components': {
                'promoters': [
                    f"{circuit['ph_promoter']['name']} (pH-sensitive)",
                    "pLac (Inducible)",
                    "pBAD (Arabinose-inducible)"
                ],
                'coding_sequences': [
                    f"{circuit['quorum_module']['name']}I (Signal synthase)",
                    f"{circuit['quorum_module']['name']}R (Receptor)",
                    "Therapeutic protein",
                    "Lysis protein" if circuit['drug_release']['name'] == 'direct_lysis' else "Secretion signal"
                ],
                'terminators': [
                    "rrnB T1", "T7 terminator", "synthetic strong terminator"
                ],
                'regulatory_elements': [
                    "RBS.E (Strong)", "RBS.J (Medium)", "RBS.H (Weak)"
                ]
            },
            'assembly_strategy': {
                'method': "Golden Gate Assembly",
                'plasmid_backbone': "pSB1C3 (High copy)",
                'selection_marker': "Chloramphenicol resistance"
            }
        }

        return design