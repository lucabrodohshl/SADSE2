"""
MILP solver integration for drone mission planning - V2 with Sim-to-Real Perturbation.

This module extends the original milp_solver.py with robustness analysis capabilities
for the docs/experiments.md guidelines. It adds:
- Perturbed energy models for sim-to-real mismatch testing
- Support for degradation factors and noise

Original functions remain unchanged in milp_solver.py.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .domain import Configuration


def drone_energy_model_v2_perturbed(
    config: Configuration, 
    task, 
    degradation_factor: float = 0.0,
    noise_std: float = 0.02,
    apply_perturbation: bool = True,
    random_state: Optional[np.random.RandomState] = None
) -> float:
    """
    Perturbed drone energy model for robustness testing (Experiment A from docs/experiments.md).
    
    This implements the sim-to-real mismatch where the Physical Twin's energy consumption
    deviates from the Digital Twin's model:
    
    E_real(x) = E_model(x) * (1 + δ) * (1 + ε)
    
    where:
    - δ (degradation_factor): Systematic error (e.g., battery wear, rotor degradation)
    - ε (noise): Random fluctuation ~ N(0, noise_std)
    
    Args:
        config: Drone configuration
        task: Task specification (dict with 'length' or numeric value)
        degradation_factor: Systematic bias (0.0 = no bias, 0.10 = 10% more energy)
        noise_std: Standard deviation of Gaussian noise (default 0.02 = 2%)
        apply_perturbation: If False, returns ideal model (for DT); if True, applies perturbation (for PT)
        random_state: Optional random state for reproducibility
        
    Returns:
        Energy consumption in Wh
    """
    from .milp_solver import drone_energy_model
    
    # Get the ideal/modeled energy consumption
    energy_model = drone_energy_model(config, task)
    
    # If no perturbation requested, return the ideal model (Digital Twin view)
    if not apply_perturbation:
        return energy_model
    
    # Apply sim-to-real mismatch (Physical Twin view)
    
    # 1. Systematic bias (degradation)
    bias_multiplier = 1.0 + float(degradation_factor)
    
    # 2. Random noise
    if random_state is None:
        random_state = np.random.RandomState()
    
    noise_multiplier = random_state.normal(0.0, float(noise_std))
    
    # Combine both perturbations
    energy_real = energy_model * bias_multiplier + noise_multiplier
    
    return max(0.0, energy_real)  # Energy cannot be negative




def drone_energy_model_v3_perturbed(
    config: Configuration, 
    task, 
    degradation_factor: float = 0.0,
    noise_multiplier: float = 0.02,
    apply_perturbation: bool = True
) -> float:
    """
    Perturbed drone energy model for robustness testing (Experiment A from docs/experiments.md).
    
    This implements the sim-to-real mismatch where the Physical Twin's energy consumption
    deviates from the Digital Twin's model:
    
    E_real(x) = E_model(x) * (1 + δ) * (1 + ε)
    
    where:
    - δ (degradation_factor): Systematic error (e.g., battery wear, rotor degradation)
    - ε (noise): Random fluctuation ~ N(0, noise_std)
    
    Args:
        config: Drone configuration
        task: Task specification (dict with 'length' or numeric value)
        degradation_factor: Systematic bias (0.0 = no bias, 0.10 = 10% more energy)
        noise_std: Standard deviation of Gaussian noise (default 0.02 = 2%)
        apply_perturbation: If False, returns ideal model (for DT); if True, applies perturbation (for PT)
        random_state: Optional random state for reproducibility
        
    Returns:
        Energy consumption in Wh
    """
    from .milp_solver import drone_energy_model
    
    # Get the ideal/modeled energy consumption
    energy_model = drone_energy_model(config, task)
    
    # If no perturbation requested, return the ideal model (Digital Twin view)
    if not apply_perturbation:
        return energy_model
    
    # Apply sim-to-real mismatch (Physical Twin view)
    
    # 1. Systematic bias (degradation)
    bias_multiplier = 1.0 + float(degradation_factor)
    
    # Combine both perturbations
    energy_real = energy_model * bias_multiplier + noise_multiplier
    
    return max(0.0, energy_real)  # Energy cannot be negative




def calculate_real_cost(
    predicted_cost: float, 
    degradation_factor: float = 0.0,
    noise_std: float = 0.02,
    random_state: Optional[np.random.RandomState] = None
) -> float:
    """
    Calculate real cost from predicted cost with perturbation.
    
    This is a simplified version for when you already have the predicted cost
    and just want to apply the sim-to-real transformation.
    
    Args:
        predicted_cost: The cost predicted by the Digital Twin model
        degradation_factor: Systematic bias (0.0 to 0.20 typical range)
        noise_std: Standard deviation of random noise (default 0.02 = ±2%)
        random_state: Optional random state for reproducibility
        
    Returns:
        Real cost with perturbation applied
    """
    # Bias: Systematic error
    bias = 1.0 + float(degradation_factor)
    
    # Noise: Random fluctuation
    if random_state is None:
        random_state = np.random.RandomState()
    
    noise = random_state.normal(1.0, float(noise_std))
    
    real_cost = predicted_cost * bias * noise
    
    return max(0.0, real_cost)


def check_feasibility_violation(
    real_cost: float,
    battery_capacity: float,
    energy_reserve_ratio: float = 0.20,
    energy_budget_safety: float = 1.05
) -> Tuple[bool, float]:
    """
    Check if a real energy consumption violates feasibility constraints.
    
    Args:
        real_cost: Actual energy consumption measured in Physical Twin
        battery_capacity: Battery capacity in Wh
        energy_reserve_ratio: Reserve ratio (default 20%)
        energy_budget_safety: Safety factor (default 1.05)
        
    Returns:
        (is_violation, margin) where:
        - is_violation: True if constraint is violated
        - margin: How much over/under the limit (positive = violation)
    """
    usable_budget = (1.0 - energy_reserve_ratio) * battery_capacity / energy_budget_safety
    
    margin = real_cost - usable_budget
    is_violation = margin > 0
    
    return is_violation, margin


class RobustnessMetrics:
    """
    Container for robustness analysis metrics.
    """
    
    def __init__(self):
        self.degradation_factors = []
        self.predicted_energies = []
        self.real_energies = []
        self.violations = []
        self.margins = []
        self.timestamps = []
        
    def add_measurement(self, 
                       degradation_factor: float,
                       predicted_energy: float,
                       real_energy: float,
                       is_violation: bool,
                       margin: float,
                       timestamp: float):
        """Add a single measurement."""
        self.degradation_factors.append(degradation_factor)
        self.predicted_energies.append(predicted_energy)
        self.real_energies.append(real_energy)
        self.violations.append(is_violation)
        self.margins.append(margin)
        self.timestamps.append(timestamp)
        
    def get_summary(self, degradation_factor: Optional[float] = None) -> Dict:
        """
        Get summary statistics.
        
        Args:
            degradation_factor: If provided, filter to this degradation level
            
        Returns:
            Dictionary with summary metrics
        """
        # Filter data if degradation_factor specified
        if degradation_factor is not None:
            indices = [i for i, df in enumerate(self.degradation_factors) 
                      if abs(df - degradation_factor) < 1e-6]
        else:
            indices = list(range(len(self.degradation_factors)))
        
        if not indices:
            return {
                'count': 0,
                'avg_predicted': 0.0,
                'avg_real': 0.0,
                'avg_error_pct': 0.0,
                'violation_rate': 0.0,
                'avg_margin': 0.0
            }
        
        predicted = [self.predicted_energies[i] for i in indices]
        real = [self.real_energies[i] for i in indices]
        violations = [self.violations[i] for i in indices]
        margins = [self.margins[i] for i in indices]
        
        avg_predicted = np.mean(predicted)
        avg_real = np.mean(real)
        avg_error_pct = ((avg_real - avg_predicted) / avg_predicted * 100) if avg_predicted > 0 else 0.0
        violation_rate = np.mean(violations) * 100  # As percentage
        avg_margin = np.mean(margins)
        
        return {
            'count': len(indices),
            'avg_predicted': avg_predicted,
            'avg_real': avg_real,
            'avg_error_pct': avg_error_pct,
            'violation_rate': violation_rate,
            'avg_margin': avg_margin
        }
    
    def get_all_degradation_levels(self) -> List[float]:
        """Get unique degradation factor levels used."""
        return sorted(list(set(self.degradation_factors)))
    
    def to_dict(self) -> Dict:
        """Export all data as dictionary."""
        return {
            'degradation_factors': self.degradation_factors,
            'predicted_energies': self.predicted_energies,
            'real_energies': self.real_energies,
            'violations': self.violations,
            'margins': self.margins,
            'timestamps': self.timestamps
        }
