"""
MILP solver integration for drone mission planning.

This module provides MILP-based optimization for multi-drone task assignment
with general n-dimensional configuration spaces.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pulp import *
import warnings

from .domain import DomainSpec, ConstrainedDomain, Configuration


def solve_task_assignment_milp(
    configurations: List[Configuration],
    tasks: List[Dict[str, float]],
    num_agents: int,
    objective_function: Optional[callable] = None,
    constraints: Optional[List[callable]] = None,
    solver: Optional[LpSolver] = None,
    # NEW: optional per-agent battery constraint (Wh)
    battery_capacity_wh: Optional[float] = 300.0,
    energy_reserve_ratio: float = 0.20,
    energy_budget_safety: float = 1.05,
) -> Tuple[Optional[Configuration], Optional[Dict[int, List[int]]], float]:
    """
    Multi-agent task assignment MILP.

    If battery_capacity_wh is provided, enforce per-agent energy budget:
        sum_j sum_k x[i,j,k] * E[k,j] <= usable_budget,
    where usable_budget = (1 - reserve) * capacity / safety.
    """
    if not configurations:
        warnings.warn("No configurations provided to MILP solver")
        return None, None, float('inf')
    if not tasks:
        warnings.warn("No tasks provided to MILP solver")
        return None, None, float('inf')

    # Prefer the real drone model if none provided
    if objective_function is None:
        objective_function = _default_energy_objective

    # Choose a solver if none provided
    if solver is None:
        solver = None
        for solver_class in [GUROBI_CMD, GLPK_CMD, COIN_CMD]:
            try:
                test_solver = solver_class(msg=0)
                if test_solver.available():
                    solver = test_solver
                    break
            except:
                continue
        if solver is None:
            warnings.warn("No local MILP solver found. Using PuLP default solver (may be slow).")
            solver = None  # let PuLP choose

    num_configs = len(configurations)
    num_tasks = len(tasks)

    # --- Precompute constant energies E[k, j] for objective + constraints ---
    E = np.zeros((num_configs, num_tasks), dtype=float)
    BIG_M = 1e9  # penalty if an evaluation fails
    for k in range(num_configs):
        for j in range(num_tasks):
            try:
                e = float(objective_function(configurations[k], tasks[j]))
                if not np.isfinite(e) or e < 0:
                    e = BIG_M
            except Exception as ex:
                warnings.warn(f"Objective failed for config {k}, task {j}: {ex}")
                e = BIG_M
            E[k, j] = e

    # Build MILP
    prob = LpProblem("TaskAssignment", LpMinimize)

    # Decision variables
    x = {}
    for i in range(num_agents):
        for j in range(num_tasks):
            for k in range(num_configs):
                x[i, j, k] = LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    y = {}
    for k in range(num_configs):
        y[k] = LpVariable(f"y_{k}", cat='Binary')

    # Objective: minimize total energy (use precomputed constants)
    prob += lpSum(x[i, j, k] * E[k, j]
                  for i in range(num_agents)
                  for j in range(num_tasks)
                  for k in range(num_configs))

    # Each task assigned exactly once
    for j in range(num_tasks):
        prob += lpSum(x[i, j, k]
                      for i in range(num_agents)
                      for k in range(num_configs)) == 1, f"Task_{j}_assigned"

    # Only one configuration can be selected
    prob += lpSum(y[k] for k in range(num_configs)) == 1, "One_config"

    # Can only use selected configuration
    for i in range(num_agents):
        for j in range(num_tasks):
            for k in range(num_configs):
                prob += x[i, j, k] <= y[k], f"Use_selected_config_{i}_{j}_{k}"

    # Per-agent battery capacity (binds because E is now filled)
    if battery_capacity_wh is not None and np.isfinite(battery_capacity_wh):
        usable = (1.0 - float(energy_reserve_ratio)) * float(battery_capacity_wh) / float(energy_budget_safety)
        usable = max(0.0, usable)
        for i in range(num_agents):
            prob += lpSum(x[i, j, k] * E[k, j]
                          for j in range(num_tasks)
                          for k in range(num_configs)) <= usable, f"BatteryBudget_agent_{i}"

    # Extra user constraints (optional)
    if constraints:
      for idx, constraint_func in enumerate(constraints):
          constraint_func(prob, x, y, configurations, tasks, num_agents)

    # Solve
    try:
        status = prob.solve(solver)
        if status != LpStatusOptimal:
            warnings.warn(f"MILP solver status: {LpStatus[status]}")
            return None, None, float('inf')

        # Selected configuration
        selected_config_idx = None
        for k in range(num_configs):
            if value(y[k]) > 0.5:
                selected_config_idx = k
                break
        if selected_config_idx is None:
            warnings.warn("No configuration selected in MILP solution")
            return None, None, float('inf')

        optimal_config = configurations[selected_config_idx]

        # Assignment
        assignment = {i: [] for i in range(num_agents)}
        for i in range(num_agents):
            for j in range(num_tasks):
                for k in range(num_configs):
                    if value(x[i, j, k]) > 0.5:
                        assignment[i].append(j)
                        break

        objective_value = value(prob.objective)
        return optimal_config, assignment, objective_value

    except Exception as e:
        warnings.warn(f"MILP solver failed: {e}")
        return None, None, float('inf')


def _default_energy_objective(config: Configuration, task: Dict[str, float]) -> float:
    """
    Default energy-based objective function.
    
    This is a simple model that can be replaced with domain-specific functions.
    
    Args:
        config: Configuration (contains all parameter values)
        task: Task properties (e.g., length, complexity)
        
    Returns:
        Energy cost for this config/task combination
    """
    # Base energy
    base_energy = 10.0
    
    # Get task length (default to 100 if not specified)
    task_length = task.get('length', 100.0)
    
    # Get configuration values
    config_dict = config.as_dict()
    
    # Speed affects time (inverse relationship)
    speed = config_dict.get('speed', 10.0)
    time_required = task_length / max(speed, 0.1)
    
    # Power consumption model (simplified)
    power = base_energy
    
    # Speed affects power (quadratic due to drag)
    if 'speed' in config_dict:
        power += 0.1 * (config_dict['speed'] ** 2)
    
    # Altitude affects power (linear)
    if 'altitude' in config_dict:
        power += 0.01 * config_dict['altitude']
    
    # Other parameters add to power
    for param_name, param_value in config_dict.items():
        if param_name not in ['speed', 'altitude']:
            power += 0.05 * param_value
    
    # Energy = Power × Time
    energy = power * time_required
    
    return energy


def solve_simple_optimization(
    domain_spec: DomainSpec,
    ds: ConstrainedDomain,
    objective_function: callable,
    num_levels: int = 3
) -> Tuple[Optional[Configuration], float]:
    """
    Simple optimization over discretized configuration space.
    
    This is a simplified version without task assignment, useful for
    single-agent problems or testing.
    
    Args:
        domain_spec: Domain specification
        ds: Constrained domain (design space)
        objective_function: Function mapping Configuration -> float
        num_levels: Discretization levels per dimension
        
    Returns:
        (optimal_config, objective_value)
    """
    if ds.is_empty():
        return None, float('inf')
    
    # Generate configurations
    configs = ds.discretize(num_levels=num_levels)
    
    if not configs:
        return None, float('inf')
    
    # Evaluate each configuration
    best_config = None
    best_objective = float('inf')
    
    for config in configs:
        try:
            obj_value = objective_function(config)
            if obj_value < best_objective:
                best_objective = obj_value
                best_config = config
        except Exception as e:
            warnings.warn(f"Objective evaluation failed for {config}: {e}")
            continue
    
    return best_config, best_objective


# Example: Drone-specific energy model
def drone_energy_model(config: Configuration, task) -> float:
    """
    Realistic drone energy consumption model (monitoring/scouting focused).

    Based on physical principles:
    - Propulsion power ~ v^2 (scaled by a throttle/power cap)
    - Altitude slightly increases power (air density effects)
    - Camera/sensor sampling add compute/IO load
    - Spray pump adds mechanical load (if used)

    Args:
        config: Drone configuration
        task: dict or numeric; if dict, expects task['length'] in meters

    Returns:
        Energy consumption in Wh
    """
    config_dict = config.as_dict()

    # ---- Extract parameters (with defaults) ----
    speed        = float(config_dict.get('speed', 10.0))                 # m/s
    altitude     = float(config_dict.get('altitude', 50.0))              # m
    camera_res   = float(config_dict.get('camera_res', 8.0))             # MP
    spray_rate   = float(config_dict.get('spray_rate', 0.0))             # L/min  (0 for monitoring)
    power_limit  = float(config_dict.get('power_limit_factor', 1.0))     # 0.4..1.0 throttle cap
    sensor_hz    = float(config_dict.get('sensor_sampling_hz',
                          config_dict.get('sensor_sampling', 10.0)))      # Hz (alias supported)

    # ---- Task length (meters) ----
    if isinstance(task, dict):
        task_length = float(task.get('length', 100.0))
    else:
        task_length = float(task)

    # =====================================================================
    # Platform profile (keep "old structure" variable names, but realistic)
    # =====================================================================
    # You can switch these few constants for different classes:
    # Small multirotor (Mavic/Phantom-class) realistic defaults:
    P0 = 120.0                # W, baseline hover/cruise + avionics housekeeping
    v_ref = 10.0              # m/s, reference speed for drag scaling
    k_drag_frac = 0.6         # fraction of P0 at v_ref due to parasitic/induced drag
    k_alt_frac_per_m = 0.001  # +0.1% of P0 per meter (light density effect)
    P_cam_per_mp = 0.25       # W / MP (image pipeline load)
    P_sensor_per_hz = 0.10    # W / Hz (sampling/IO/compute)
    P_pump_per_lpm = 20.0     # W / (L/min), if spraying (0 for monitoring)

    # If you fly a heavier quad: set P0≈300, k_drag_frac≈0.5, P_sensor_per_hz≈0.15.
    # For a fixed-wing: P0≈80, k_drag_frac≈0.3, k_alt_frac_per_m≈0.0005

    # Optional per-config overrides (keep structure simple)
    P0 = float(config_dict.get('P0_W', P0))

    base_power = P0  # W  ← previously was scaling with distance; now a constant platform power

    # Propulsion power (quadratic in speed) scaled by the power limit factor
    # (only the *increment* is capped; P0 remains baseline systems draw)
    propulsion_power = P0 * k_drag_frac * (max(0.0, speed) / v_ref) ** 2 * max(0.4, min(1.0, power_limit))

    # Altitude power (fractional increase of P0)
    altitude_power = P0 * k_alt_frac_per_m * max(0.0, altitude)

    # Camera & sensor processing power
    camera_power   = P_cam_per_mp * max(0.0, camera_res)
    sensor_power   = P_sensor_per_hz * max(0.0, sensor_hz)

    # Spray pump power (0 unless you’re simulating spraying passes)
    spray_power    = P_pump_per_lpm * max(0.0, spray_rate)

    # ======================
    # Total power & energy
    # ======================
    # Keep your original names/flow: total_power then time then energy
    total_power = (
        base_power
        + propulsion_power
        + altitude_power
        + camera_power
        + sensor_power
        + spray_power
    )  # [W]

    # Time to complete task
    speed_eff = max(0.1, speed)
    task_time_hours = task_length / (speed_eff * 3600.0)

    # Energy consumption
    energy_wh = total_power * task_time_hours
    return energy_wh
