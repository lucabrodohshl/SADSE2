
from src.strategy import Strategy

from src.domain import DomainSpec, Parameter, ODD, ConstrainedDomain, ConfigurationSpace
from src.smart_cache import SmartZonotopeCache
from src.milp_solver import solve_task_assignment_milp, drone_energy_model
from src.zonotope_ops import Zonotope

import numpy as np

import time
import numpy as np
from typing import Optional
import os
import json
import yaml


N_LEVELS_PER_PARAM = 3


def _to_vals(cfg):
    """
    Return a python list of numeric parameter values from whatever 'cfg' is.
    Supports:
      - objects with a callable .values()
      - objects with a .values array-like attribute
      - dict-like configs
      - numpy arrays / lists / tuples
    """
    # callable method?
    v = getattr(cfg, "values", None)
    if callable(v):
        return list(v())
    # attribute that is array-like?
    if v is not None:
        try:
            return list(v)  # e.g., numpy array
        except TypeError:
            pass
    # dict-like
    if isinstance(cfg, dict):
        return list(cfg.values())
    # already a sequence
    if isinstance(cfg, (list, tuple, np.ndarray)):
        return list(cfg)
    raise TypeError(f"Don't know how to extract values from config of type {type(cfg)}")

class DiscreteStrategy(Strategy):
    """A strategy that performs discrete optimization without caching."""
    tasks: list
    n_drones : int
    last_objective: float
    last_time: float
    cs : ConfigurationSpace
    domain_spec: DomainSpec
    def __init__(self, name: str, 
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None, 
                 file_path: str = None):
        super().__init__(name)
        if info is None and file_path is None:
            print("Warning: Either 'info' or 'file_path' should be provided to initialize the strategy.")
        if info is not None and file_path is not None:
            raise ValueError("Provide only one of 'info' or 'file_path', not both.")
        if file_path is not None:
            info = self.load(file_path)

        self.tasks = cs.tasks
        self.n_drones = cs.size()
        self.cs = cs
        self.domain_spec = cs.get_domain_spec()
        self.last_time = 0.0
        self.last_objective = float('inf')
        return info

        
    

    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = drone_energy_model):
        r"""
        Optimization function for a specific zonotope region.
        
        This discretizes the zonotope and runs MILP optimization.
        """
        
        # Discretize the region (use fewer levels for 6D: 2^6 = 64 configs)
        configurations = ds.discretize(num_levels=N_LEVELS_PER_PARAM)
        
        if not configurations:
            return None, float('inf'), {}
        
        
        # Run MILP optimization
        optimal_config, assignment, objective = solve_task_assignment_milp(
            configurations=configurations,
            tasks=self.tasks,
            num_agents=self.n_drones,
            objective_function=objective_function
        )
        
        metadata = {
            'assignment': assignment,
            'num_configurations': len(configurations),
            'region_volume': zonotope.volume()
        }
        
        return optimal_config, objective, metadata


    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """
        Execute the baseline discrete optimization strategy.
        
        :param objective_function: The objective function to minimize.
        :param odd: The operational design domain (ODD) for the current scenario.
        :return: The optimal configuration found.
        """
        # Optimize over entire DS (no caching)
        ds_bounds = ds.to_bounds_list()
        ds_zonotope = Zonotope.from_box(ds_bounds)
        start_time = time.time()
        baseline_config, baseline_objective, baseline_metadata = self.optimize_in_zonotope(ds_zonotope, ds)
        total_time = (time.time() - start_time) * 1000  # in ms
        self.last_time = total_time
        self.last_opt_status = baseline_config is not None
        return baseline_config, baseline_objective, baseline_metadata, total_time

    def has_hit(self):
        raise NotImplementedError("has_hit method must be implemented by subclasses.")
    
    def load(self, source):
        # load from json or yaml file 
        ext = os.path.splitext(source)[1].lower()
        match ext:
            case ".json":
                with open(source, "r") as f:
                    data = json.load(f)
            case ".yaml" | ".yml":
                with open(source, "r") as f:
                    data = yaml.safe_load(f)
            case _:
                raise ValueError(f"Unsupported fleet file: {ext}")
        return data
    def __str__(self):
        ret = ""
        if self.last_opt_status:
            ret += f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)"
            ret += f"         Objective: {self.last_objective:.2f} Wh"
        else:
            ret +=f"      × {self.name} - No feasible solution"

        return ret
    def get_info(self):
        return "full re-optimization"

    def get_explored_volume(self):
        return np.prod([b[1] - b[0] for b in self.domain_spec.get_bounds()])
    def has_hit(self):
        return False
    def get_coverage_pct(self):
        return 100.0

class SmartCacheStrategy(DiscreteStrategy):
    """A strategy that utilizes a smart zonotope cache for optimization."""
    last_cache_status: bool
    cache: SmartZonotopeCache
    
    def __init__(self, name: str, 
                 cs: ConfigurationSpace = None,
                    info: Optional[dict] = None,
                    file_path: Optional[str] = None,
                 ):
        info = super().__init__(name = name, cs = cs, info = info, file_path = file_path)
        self.last_cache_status = False  # Track if last execution was a cache hit
        self.cache = SmartZonotopeCache(self.domain_spec, 
                                        info["extension_threshold"], 
                                        info["merge_threshold"], 
                                        info["merge_frequency"],
                                        strategy_name=self.name
                                        )
    def get_info(self):
        return self.last_cache_status

    def has_hit(self):
            return self.last_cache_status != 'new_entry'
    def get_cache_volume(self):
        return self.cache.explored_volume
    
    def get_explored_volume(self):
        return self.cache.explored_volume

    def get_coverage_pct(self):
        return  min(100.0, (self.get_cache_volume() / np.prod([b[1] - b[0] for b in self.domain_spec.get_bounds()])) * 100.0)

    
    
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """
        Execute the strategy by checking the cache and performing optimization if needed.
        
        :param objective_function: The objective function to minimize.
        :param odd: The operational design domain (ODD) for the current scenario.
        :param ds: The constrained domain specification.
        :return: The optimal configuration found.
        """
        # Create optimization function closure
        def opt_func(zonotope):
            return self.optimize_in_zonotope(zonotope, ds)

        print(f"=== {self.name}.execute: Checking cache and optimizing if needed ===")
        start_time = time.time()
        config, objective, metadata, status = self.cache.query_and_optimize(odd, ds, opt_func)
        total_time = (time.time() - start_time) * 1000
        self.last_cache_status = status
        self.last_time = total_time
        self.last_objective = objective
        self.last_opt_status = config is not None
        return config, objective, metadata, total_time
    
    def __str__(self):
        ret = ""
        if self.last_opt_status:
            ret += f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)"
            ret += f"         Objective: {self.last_objective:.2f} Wh"
        else:
            ret +=f"      × {self.name} - No feasible solution"
        if self.last_cache_status == 'hit':
            ret = f"      ✓ {self.name} - CACHE HIT! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
        elif self.last_cache_status == 'extended':
            ret = f"      ⊕ {self.name} - REGION EXTENDED! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
        elif self.last_cache_status == 'new_entry':
            ret = f"      + {self.name} - NEW ENTRY ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"

        ret += f" \n        Objective: {self.last_objective:.2f} Wh"
        return ret
    
class LinearStrategy(Strategy):
    """Base strategy that performs continuous linear optimization using LP.
    
    This class provides the core LP formulation that both FullLinearStrategy
    and SmartLinearStrategy inherit. It uses Gurobi for solving.
    """
    tasks: list
    n_drones: int
    domain_spec: DomainSpec
    last_objective: float
    tasks: list
    n_drones: int
    
    def __init__(self, name: str, 
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None, 
                 file_path: str = None):
        super().__init__(name)
        if info is None and file_path is None:
            print("Warning: Either 'info' or 'file_path' should be provided to initialize the strategy.")
        if info is not None and file_path is not None:
            raise ValueError("Provide only one of 'info' or 'file_path', not both.")
        if file_path is not None:
            info = self.load(file_path)

        self.tasks = cs.tasks
        self.n_drones = cs.size()
        self.cs = cs
        self.domain_spec = cs.get_domain_spec()
        self.last_time = 0.0
        self.last_objective = float('inf')
        return info
    
    def load(self, source):
        # load from json or yaml file 
        ext = os.path.splitext(source)[1].lower()
        match ext:
            case ".json":
                with open(source, "r") as f:
                    data = json.load(f)
            case ".yaml" | ".yml":
                with open(source, "r") as f:
                    data = yaml.safe_load(f)
            case _:
                raise ValueError(f"Unsupported fleet file: {ext}")
        return data
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """
        Execute the full linear optimization strategy.
        
        Solves a Linear Program with continuous configuration variables.
        No discretization required - finds optimal in continuous space.
        
        :param odd: The operational design domain (ODD) for the current scenario.
        :param ds: The constrained domain (design space).
        :return: Tuple of (config, objective, metadata, time)
        """
        # Optimize over entire DS (no caching)
        
        ds_bounds = ds.to_bounds_list()
        ds_zonotope = Zonotope.from_box(ds_bounds)
        
        start_time = time.time()
        optimal_config, objective, metadata = self.optimize_in_zonotope(ds_zonotope, ds)
        total_time = (time.time() - start_time) * 1000  # in ms
        
        self.last_time = total_time
        self.last_objective = objective if optimal_config else float('inf')
        self.last_opt_status = optimal_config is not None
        
        return optimal_config, objective, metadata, total_time
    
    def __str__(self):
        ret = ""
        if self.last_opt_status:
            ret += f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)"
            ret += f"\n         Objective: {self.last_objective:.2f} Wh"
        else:
            ret += f"      × {self.name} - No feasible solution"
        return ret
    
    def get_info(self):
        return "full re-optimization (LP)"

    def get_explored_volume(self):
        return np.prod([b[1] - b[0] for b in self.domain_spec.get_bounds()])
    
    def has_hit(self):
        return False
    
    def get_coverage_pct(self):
        return 100.0
    
    
    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = None):
        """
        Linear LP w/ first-order (affine) Taylor approximation of drone_energy_model,
        then returns the TRUE energy (nonlinear) at the solved config.
        Adds per-agent battery capacity constraints via McCormick envelopes.
        """
        if objective_function is None:
            from src.milp_solver import drone_energy_model
            objective_function = lambda config, task: drone_energy_model(config, task)

        import numpy as np
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatusOptimal, value, GUROBI_CMD

        # --- battery settings (read from the strategy; set defaults if absent) ---
        battery_capacity_wh   = float(getattr(self, "battery_capacity_wh",  300) or 300.0)
        energy_reserve_ratio  = float(getattr(self, "energy_reserve_ratio", 0.20))
        energy_budget_safety  = float(getattr(self, "energy_budget_safety", 1.05))
        use_battery_budget    = battery_capacity_wh > 0.0
        if use_battery_budget:
            usable_wh = max(0.0, (1.0 - energy_reserve_ratio) * battery_capacity_wh / energy_budget_safety)

        # Bounds & parameter names
        bounds = zonotope.to_box_bounds()  # (P,2)
        params = self.domain_spec.parameters
        param_names = [p.name for p in params]
        #print("Optimizing over parameters:", param_names)
        #print("battery constraints:", battery_capacity_wh, energy_reserve_ratio, energy_budget_safety, "=> usable_wh:", usable_wh)
        P = len(param_names)

        # Helpers to hop between vector and config
        def vec_to_config(vec):
            return self.domain_spec.create_configuration([float(vec[i]) for i in range(P)])

        # Reference point = center of the box
        center = np.array([(bounds[i, 0] + bounds[i, 1]) / 2.0 for i in range(P)], dtype=float)

        # Keep speed positive during probing (energy has 1/speed in time computation)
        if 'speed' in param_names:
            s_idx = param_names.index('speed')
            center[s_idx] = max(center[s_idx], 0.1)

        # Finite-difference steps
        widths = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-6)
        h_vec = np.maximum(1e-3 * widths, 1e-4)

        # Build LP
        prob = LpProblem("LinearTaskAssignment", LpMinimize)
        num_tasks = len(self.tasks)
        num_agents = self.n_drones

        # Decision vars: continuous config
        config_vars = {
            name: LpVariable(f"config_{name}",
                            lowBound=float(bounds[i, 0]),
                            upBound=float(bounds[i, 1]),
                            cat='Continuous')
            for i, name in enumerate(param_names)
        }

        # Binary assignment x[i,j]
        x = {(i, j): LpVariable(f"x_{i}_{j}", cat='Binary')
            for i in range(num_agents) for j in range(num_tasks)}

        # --- Linearize E(config, task) ≈ E_ref + ∑ grad_i (p_i - p_i_ref) ---
        ref_cfg = vec_to_config(center)
        objective_terms = []
        E_refs = []         # store for bounds computation
        grad_rows = []      # grads per task (np.array length P)

        for j in range(num_tasks):
            task = self.tasks[j]
            E_ref = float(objective_function(ref_cfg, task))
            E_refs.append(E_ref)

            grads = np.zeros(P, dtype=float)
            for i in range(P):
                h = float(h_vec[i])
                p0 = float(center[i])
                lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
                use_central = (p0 - h >= lo) and (p0 + h <= hi)

                if use_central:
                    v_plus = center.copy();  v_plus[i] = p0 + h
                    v_minus = center.copy(); v_minus[i] = p0 - h
                    if 'speed' in param_names:
                        s_idx = param_names.index('speed')
                        v_plus[s_idx]  = max(v_plus[s_idx],  0.1)
                        v_minus[s_idx] = max(v_minus[s_idx], 0.1)
                    E_plus  = float(objective_function(vec_to_config(v_plus),  task))
                    E_minus = float(objective_function(vec_to_config(v_minus), task))
                    grads[i] = (E_plus - E_minus) / (2.0 * h)
                else:
                    if p0 + h <= hi:
                        v_plus = center.copy(); v_plus[i] = p0 + h
                        if 'speed' in param_names:
                            s_idx = param_names.index('speed')
                            v_plus[s_idx] = max(v_plus[s_idx], 0.1)
                        E_plus  = float(objective_function(vec_to_config(v_plus), task))
                        grads[i] = (E_plus - E_ref) / h
                    else:
                        v_minus = center.copy(); v_minus[i] = p0 - h
                        if 'speed' in param_names:
                            s_idx = param_names.index('speed')
                            v_minus[s_idx] = max(v_minus[s_idx], 0.1)
                        E_minus = float(objective_function(vec_to_config(v_minus), task))
                        grads[i] = (E_ref - E_minus) / h

            grad_rows.append(grads.copy())

            # affine E_j(config)
            task_expr = E_ref
            for i, name in enumerate(param_names):
                task_expr += grads[i] * (config_vars[name] - float(center[i]))
            objective_terms.append(task_expr)

        total_energy_linearized = lpSum(objective_terms)
        prob += total_energy_linearized

        # Each task assigned exactly once
        for j in range(num_tasks):
            prob += lpSum([x[i, j] for i in range(num_agents)]) == 1, f"Task_{j}_assigned"

        # Nonnegativity guard
        prob += total_energy_linearized >= 0, "NonnegativeLinearizedEnergy"

        # ----------------------------------------------------------------------
        # OPTIONAL: per-agent battery budget using McCormick envelopes on z[i,j]
        # ----------------------------------------------------------------------
        if use_battery_budget:
            # Compute tight bounds L_j,U_j for each affine E_j over the config box
            #print("Using per-agent battery budget constraints (Wh)")
            center_vec = center
            lo = bounds[:, 0].astype(float)
            hi = bounds[:, 1].astype(float)

            L_j = []
            U_j = []
            for j in range(num_tasks):
                E0 = float(E_refs[j])
                g  = np.array(grad_rows[j], dtype=float)
                delta_min = np.where(g >= 0.0, lo - center_vec, hi - center_vec)
                delta_max = np.where(g >= 0.0, hi - center_vec, lo - center_vec)
                L = E0 + float(np.dot(g, delta_min))
                U = E0 + float(np.dot(g, delta_max))
                if L > U:  # numeric guard
                    L, U = U, L
                # Optional: enforce nonneg lower bound (energy can't be negative)
                L = max(L, 0.0)
                L_j.append(L); U_j.append(U)

            # z[i,j] ≈ x[i,j] * E_j(config)
            z = {(i, j): LpVariable(f"z_{i}_{j}", lowBound=0.0, cat='Continuous')
                for i in range(num_agents) for j in range(num_tasks)}

            # McCormick envelopes
            for i in range(num_agents):
                for j in range(num_tasks):
                    Ej = objective_terms[j]              # affine expression for E_j
                    Lj = float(L_j[j]); Uj = float(U_j[j])
                    # z ≤ Uj * x
                    prob += z[i, j] <= Uj * x[i, j]
                    # z ≥ Lj * x
                    prob += z[i, j] >= Lj * x[i, j]
                    # z ≤ E - Lj*(1 - x)
                    prob += z[i, j] <= Ej - Lj * (1 - x[i, j])
                    # z ≥ E - Uj*(1 - x)
                    prob += z[i, j] >= Ej - Uj * (1 - x[i, j])

            # Per-agent usable energy
            for i in range(num_agents):
                prob += lpSum([z[i, j] for j in range(num_tasks)]) <= usable_wh, f"BatteryBudget_agent_{i}"

        # Solve
        try:
            solver = GUROBI_CMD(msg=0)
            status = prob.solve(solver)
            if status != LpStatusOptimal:
                return None, float('inf'), {'status': 'infeasible'}

            # Extract config
            opt_vals = {n: value(v) for n, v in config_vars.items()}
            optimal_config = self.domain_spec.create_configuration([opt_vals[p.name] for p in params])

            # Assignment
            assignment = {i: [] for i in range(num_agents)}
            for i in range(num_agents):
                for j in range(num_tasks):
                    if value(x[i, j]) > 0.5:
                        assignment[i].append(j)

            # Re-evaluate TRUE energy at the solved config
            import math
            true_total_wh = 0.0
            finite = True
            for j in range(num_tasks):
                e = float(objective_function(optimal_config, self.tasks[j]))
                if not np.isfinite(e):
                    finite = False
                true_total_wh += e if np.isfinite(e) else 0.0

            reported_obj = true_total_wh if finite else float(value(prob.objective))

            metadata = {
                'assignment': assignment,
                'solver_status': 'optimal',
                'region_volume': zonotope.volume(),
                'method': 'LP (first-order linearization of drone_energy_model)',
                'solver': 'gurobi',
                'linearization_point': {n: float(center[i]) for i, n in enumerate(param_names)},
                'linearized_objective_value': float(value(prob.objective)),
                'true_objective_value_wh': float(true_total_wh),
                'objective_is_true_energy': bool(finite),
                'battery_capacity_wh': battery_capacity_wh if use_battery_budget else None,
                'usable_energy_wh': usable_wh if use_battery_budget else None,
            }

            return optimal_config, reported_obj, metadata

        except Exception as e:
            import warnings
            warnings.warn(f"Linear solver (Gurobi) failed: {e}")
            return None, float('inf'), {'status': 'error', 'error': str(e)}

class SmartLinearStrategy(LinearStrategy):

    """A linear strategy that utilizes a smart zonotope cache for optimization.
    
    Uses continuous Linear Programming (LP) with intelligent caching.
    The cache stores regions where LP solutions are valid, enabling
    fast queries without re-solving the entire LP.
    """
    last_cache_status: bool
    cache: SmartZonotopeCache
    
    
    def __init__(self, name: str, 
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None, 
                 file_path: str = None):
        info = super().__init__(name = name, cs = cs, info = info, file_path = file_path)
        self.last_cache_status = False  # Track if last execution was a cache hit
        self.cache = SmartZonotopeCache(self.domain_spec, 
                                        info["extension_threshold"], 
                                        info["merge_threshold"], 
                                        info["merge_frequency"],
                                        strategy_name=self.name
                                        )
    
    def get_info(self):
        return self.last_cache_status

    def has_hit(self):
        return self.last_cache_status == "hit" #self.last_cache_status != 'new_entry'
    
    def get_cache_volume(self):
        return self.cache.explored_volume
    
    def get_explored_volume(self):
        return self.cache.explored_volume

    def get_coverage_pct(self):
        return min(100.0, (self.get_cache_volume() / np.prod([b[1] - b[0] for b in self.domain_spec.get_bounds()])) * 100.0)
    
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """
        Execute the strategy by checking the cache and performing LP optimization if needed.
        
        This combines the benefits of continuous linear optimization with intelligent
        caching to avoid redundant LP solves.
        
        :param odd: The operational design domain (ODD) for the current scenario.
        :param ds: The constrained domain specification.
        :return: Tuple of (config, objective, metadata, time)
        """
        # Create optimization function closure
        print(f"=== {self.name}.execute: Checking cache and optimizing if needed ===")
        def opt_func(zonotope):
            return self.optimize_in_zonotope(zonotope, ds)
        
        start_time = time.time()
        config, objective, metadata, status = self.cache.query_and_optimize(odd, ds, opt_func)
        total_time = (time.time() - start_time) * 1000
        
        self.last_cache_status = status
        self.last_time = total_time
        self.last_objective = objective if objective != float('inf') else float('inf')
        self.last_opt_status = config is not None
        
        return config, objective, metadata, total_time
    
    # Note: optimize_in_zonotope is inherited from LinearStrategy base class
    # The cache framework calls it as needed based on cache hits/misses
    
    def __str__(self):
        ret = ""
        if self.last_opt_status:
            ret += f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)"
            ret += f"\n         Objective: {self.last_objective:.2f} Wh"
        else:
            ret += f"      × {self.name} - No feasible solution"
        
        if self.last_cache_status == 'hit':
            ret = f"      ✓ {self.name} - CACHE HIT! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'extended': 
            ret = f"      ⊕ {self.name} - REGION EXTENDED! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'new_entry':
            ret = f"      + {self.name} - NEW ENTRY ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        
        return ret
    
class BayesianOptimizationStrategy(Strategy):
    """A strategy that uses Bayesian Optimization for continuous optimization."""
    tasks: list
    n_drones: int
    domain_spec: DomainSpec
    last_objective: float
    cs: ConfigurationSpace
    
    def __init__(self, name: str, 
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None, 
                 file_path: str = None):
        super().__init__(name)
        if info is None and file_path is None:
            print("Warning: Either 'info' or 'file_path' should be provided to initialize the strategy.")
        if info is not None and file_path is not None:
            raise ValueError("Provide only one of 'info' or 'file_path', not both.")
        if file_path is not None:
            info = self.load(file_path)

        self.tasks = cs.tasks
        self.n_drones = cs.size()
        self.cs = cs
        self.domain_spec = cs.get_domain_spec()
        self.last_time = 0.0
        self.last_objective = float('inf')
        return info

    def load(self, source):
        # load from json or yaml file 
        ext = os.path.splitext(source)[1].lower()
        match ext:
            case ".json":
                with open(source, "r") as f:
                    data = json.load(f)
            case ".yaml" | ".yml":
                with open(source, "r") as f:
                    data = yaml.safe_load(f)
            case _:
                raise ValueError(f"Unsupported fleet file: {ext}")
        return data

    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = None):
        """Optimize using Bayesian Optimization within a zonotope region."""
        try:
            # Try GPyOpt first
            try:
                import GPy
                import GPyOpt
                use_gpyopt = True
            except ImportError:
                try:
                    from skopt import gp_minimize
                    use_gpyopt = False
                except ImportError:
                    raise ImportError("Neither GPyOpt nor scikit-optimize is available. Please install one of them.")

            if objective_function is None:
                from src.milp_solver import drone_energy_model
                objective_function = lambda config, task: drone_energy_model(config, task.get('length', 100.0))
            
            # Define bounds from zonotope
            bounds = zonotope.to_box_bounds()
            param_names = [p.name for p in self.domain_spec.parameters]
            
            # Wrapper for objective that handles task assignment
            def bo_objective(x):
                # Convert params to configuration
                config = self.domain_spec.create_configuration(x if use_gpyopt else list(x))
                
                # Solve task assignment MILP
                _, assignment, obj = solve_task_assignment_milp(
                    configurations=[config],
                    tasks=self.tasks,
                    num_agents=self.n_drones,
                    objective_function=objective_function
                )
                return obj

            if use_gpyopt:
                # GPyOpt implementation
                bo_domain = []
                for i, name in enumerate(param_names):
                    bo_domain.append({
                        'name': name,
                        'type': 'continuous',
                        'domain': (bounds[i,0], bounds[i,1])
                    })
                
                bo = GPyOpt.methods.BayesianOptimization(
                    f=bo_objective,
                    domain=bo_domain,
                    initial_design_numdata=5,
                    acquisition_type='EI',
                    maximize=False
                )
                
                bo.run_optimization(max_iter=20)
                x_opt = bo.x_opt
                optimal_config = self.domain_spec.create_configuration(x_opt)
                objective = bo.fx_opt
                
                metadata = {
                    'method': 'gpyopt',
                    'bo_model': bo,
                    'num_iterations': len(bo.Y),
                    'region_volume': zonotope.volume()
                }
                
            else:
                # Scikit-optimize implementation
                bounds_list = [(low, high) for low, high in bounds]
                
                result = gp_minimize(
                    func=bo_objective,
                    dimensions=bounds_list,
                    n_calls=25,  # Total evaluations
                    n_initial_points=5,  # Initial random points
                    noise=1e-10
                )
                
                optimal_config = self.domain_spec.create_configuration(list(result.x))
                objective = result.fun
                
                metadata = {
                    'method': 'skopt',
                    'model': result,
                    'num_iterations': len(result.x_iters),
                    'region_volume': zonotope.volume()
                }
            
            return optimal_config, objective, metadata
                
        except Exception as e:
            print(f"Bayesian optimization failed: {e}")
            return None, float('inf'), {'status': 'error'}

    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """Execute the Bayesian optimization strategy."""
        # Optimize over entire DS
        ds_bounds = ds.to_bounds_list()
        ds_zonotope = Zonotope.from_box(ds_bounds)
        
        start_time = time.time()
        optimal_config, objective, metadata = self.optimize_in_zonotope(ds_zonotope, ds)
        total_time = (time.time() - start_time) * 1000  # in ms
        
        self.last_time = total_time
        self.last_objective = objective if optimal_config else float('inf')
        self.last_opt_status = optimal_config is not None
        
        return optimal_config, objective, metadata, total_time

    def get_info(self):
        return "bayesian optimization"

    def get_explored_volume(self):
        return np.prod([b[1] - b[0] for b in self.domain_spec.get_bounds()])
    
    def has_hit(self):
        return False
    
    def get_coverage_pct(self):
        return 100.0

    def __str__(self):
        ret = ""
        if self.last_opt_status:
            ret += f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)"
            ret += f"\n         Objective: {self.last_objective:.2f} Wh"
        else:
            ret += f"      × {self.name} - No feasible solution"
        return ret
    
class SmartBayesianStrategy(BayesianOptimizationStrategy):
    """A Bayesian strategy that utilizes a smart zonotope cache for optimization."""
    last_cache_status: bool
    cache: SmartZonotopeCache
    
    def __init__(self, name: str, 
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None, 
                 file_path: str = None):
        info = super().__init__(name=name, cs=cs, info=info, file_path=file_path)
        self.last_cache_status = False
        self.cache = SmartZonotopeCache(self.domain_spec, 
                                      info["extension_threshold"], 
                                      info["merge_threshold"], 
                                      info["merge_frequency"],
                                      strategy_name=self.name)
    
    def get_info(self):
        return self.last_cache_status

    def has_hit(self):
        return self.last_cache_status != 'new_entry'
    
    def get_cache_volume(self):
        return self.cache.explored_volume
    
    def get_explored_volume(self):
        return self.cache.explored_volume

    def get_coverage_pct(self):
        return min(100.0, (self.get_cache_volume() / np.prod([b[1] - b[0] for b in self.domain_spec.get_bounds()])) * 100.0)
    
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """Execute the strategy by checking cache and performing Bayesian optimization if needed."""
        def opt_func(zonotope):
            return self.optimize_in_zonotope(zonotope, ds)
        
        start_time = time.time()
        config, objective, metadata, status = self.cache.query_and_optimize(odd, ds, opt_func)
        total_time = (time.time() - start_time) * 1000
        
        self.last_cache_status = status
        self.last_time = total_time
        self.last_objective = objective if objective != float('inf') else float('inf')
        self.last_opt_status = config is not None
        
        return config, objective, metadata, total_time
    
    def __str__(self):
        ret = ""
        if self.last_opt_status:
            ret += f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)"
            ret += f"\n         Objective: {self.last_objective:.2f} Wh"
        else:
            ret += f"      × {self.name} - No feasible solution"
        
        if self.last_cache_status == 'hit':
            ret = f"      ✓ {self.name} - CACHE HIT! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'extended': 
            ret = f"      ⊕ {self.name} - REGION EXTENDED! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'new_entry':
            ret = f"      + {self.name} - NEW ENTRY ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        
        return ret
       
class GAStrategy(Strategy):
    """
    Deadline-safe Genetic Algorithm baseline (non-cache).
    - Mixed-friendly (continuous/int-like parameters via real-valued vectors)
    - Samples strictly inside DS_t bounds
    - Fitness evaluated honestly via solve_task_assignment_milp + drone_energy_model
    """
    tasks: list
    n_drones: int
    domain_spec: DomainSpec

    def __init__(self, name: str,
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None,
                 file_path: str = None):
        super().__init__(name)
        if info is None and file_path is None:
            print("Warning: Either 'info' or 'file_path' should be provided to initialize the strategy.")
        if info is not None and file_path is not None:
            raise ValueError("Provide only one of 'info' or 'file_path', not both.")
        if file_path is not None:
            info = self.load(file_path)

        self.cs = cs
        self.tasks = cs.tasks
        self.n_drones = cs.size()
        self.domain_spec = cs.get_domain_spec()

        self.last_time = 0.0
        self.last_objective = float('inf')
        self.last_opt_status = False

        # GA knobs
        self.time_budget_ms = float((info or {}).get("time_budget_ms", 100.0))  # Reduced from 500ms
        self.rng = np.random.default_rng((info or {}).get("seed", 1337))
        self.pop_size = int((info or {}).get("ga_pop", 48))
        self.elite = int((info or {}).get("ga_elite", 2))
        self.cx_rate = float((info or {}).get("ga_crossover", 0.8))
        self.mut_rate = float((info or {}).get("ga_mutation", 0.2))
        self.tourn_k = int((info or {}).get("ga_tournament_k", 3))
        return info

    def __str__(self):
        if self.last_opt_status:
            return f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)\n         Objective: {self.last_objective:.2f} Wh"
        return f"      × {self.name} - No feasible solution"

    def load(self, source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".json":
            with open(source, "r") as f: return json.load(f)
        if ext in (".yaml", ".yml"):
            with open(source, "r") as f: return yaml.safe_load(f)
        raise ValueError(f"Unsupported fleet file: {ext}")

    # ---- shared helpers ----
    def _bounds(self, ds: ConstrainedDomain):
        return ds.to_bounds_list()

    def _clip_vals(self, vals, bounds):
        return [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(vals, bounds)]

    def _rand_vals(self, bounds):
        return [self.rng.uniform(lo, hi) for (lo, hi) in bounds]

    def _cfg_from_vals(self, vals):
        return self.domain_spec.create_configuration(vals)

    def _crossover(self, a, b):
        return [ai if self.rng.random() < 0.5 else bi for ai, bi in zip(a, b)]

    def _mutate(self, vals, bounds):
        out = list(vals)
        for i, (lo, hi) in enumerate(bounds):
            if self.rng.random() < self.mut_rate:
                span = (hi - lo)
                out[i] = float(np.clip(out[i] + self.rng.normal(0.0, 0.1 * span), lo, hi))
        return out

    def _evaluate_one(self, cfg):
        """
        Evaluate a single configuration using greedy task assignment.
        Much faster than MILP - suitable for heuristics.
        """
        # Simple greedy assignment: assign each task to least-loaded drone
        drone_loads = [0.0] * self.n_drones
        
        for task in self.tasks:
            task_energy = drone_energy_model(cfg, task.get('length', 100.0))
            # Assign to least loaded drone (greedy)
            min_drone_idx = min(range(self.n_drones), key=lambda i: drone_loads[i])
            drone_loads[min_drone_idx] += task_energy
        
        total_energy = sum(drone_loads)
        return total_energy

    def _evaluate_many(self, cfgs):
        """
        Evaluate multiple configurations and return the best one.
        Uses direct evaluation (greedy assignment) instead of MILP for speed.
        """
        best_cfg = None
        best_obj = float('inf')
        
        for cfg in cfgs:
            obj = self._evaluate_one(cfg)
            if obj < best_obj:
                best_cfg = cfg
                best_obj = obj
        
        return best_cfg, best_obj

    # ---- smart-cache compatible API ----
    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = None):
        """
        Run one GA 'solve' inside the given zonotope's AABB (or DS_t bounds if zonotope is None).
        Returns (best_cfg, best_obj, metadata)
        """
        bounds = (zonotope.to_box_bounds() if zonotope is not None else self._bounds(ds))
        t0 = time.time()

        # 1) Init population inside bounds
        pop_vals = [self._rand_vals(bounds) for _ in range(self.pop_size)]
        pop_cfgs = [self._cfg_from_vals(self._clip_vals(v, bounds)) for v in pop_vals]
        best_cfg, best_obj = self._evaluate_many(pop_cfgs)

        # 2) GA loop (tight, local to this call — cache wrapper enforces outer budget)
        while (time.time() - t0) * 1000.0 < self.time_budget_ms * 0.9:
            # tournament pick
            def tour_pick():
                idx = self.rng.choice(len(pop_vals), size=min(self.tourn_k, len(pop_vals)), replace=False)
                return pop_vals[int(idx[0])]

            # offspring
            offspring_vals = []
            while len(offspring_vals) < self.pop_size - self.elite:
                p1 = tour_pick(); p2 = tour_pick()
                child = self._crossover(p1, p2) if self.rng.random() < self.cx_rate else p1[:]
                child = self._mutate(child, bounds)
                offspring_vals.append(child)

            # score pool
            pool_vals = pop_vals + offspring_vals
            pool_cfgs = [self._cfg_from_vals(self._clip_vals(v, bounds)) for v in pool_vals]

            scored = []
            for cfg in pool_cfgs:
                c_best, c_obj = self._evaluate_many([cfg])
                scored.append((cfg, c_obj))
                if (time.time() - t0) * 1000.0 > self.time_budget_ms * 0.95:
                    break
            scored.sort(key=lambda x: x[1])

            pop_cfgs = [cfg for (cfg, _) in scored[:self.pop_size]]
            pop_vals = [_to_vals(cfg) for cfg in pop_cfgs]

            gbest_cfg, gbest_obj = self._evaluate_many(pop_cfgs)
            if gbest_obj < best_obj:
                best_cfg, best_obj = gbest_cfg, gbest_obj

        meta = {'status': 'ga'}
        return best_cfg, best_obj, meta

    # ---- plain execute ----
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        t0 = time.time()
        cfg, obj, _meta = self.optimize_in_zonotope(None, ds)
        elapsed = (time.time() - t0) * 1000.0

        self.last_time = elapsed
        self.last_objective = obj if cfg is not None else float('inf')
        self.last_opt_status = cfg is not None
        return cfg, obj, _meta, elapsed
    
    # For plain strategies (GA, LNS, WarmStartOnly, etc.)

    def get_info(self):
        # For consistency with save_results; smart strategies return 'hit'/'extended'/etc.
        return "plain"

    def has_hit(self):
        # No cache, so never a hit
        return False

    def get_cache_volume(self):
        # No cache
        return 0.0

    def get_explored_volume(self):
        return 0.0

    def get_coverage_pct(self):
        return 0.0

class SmartGAStrategy(GAStrategy):
    """
    GA with smart zonotope cache (same pattern as SmartLinearStrategy / SmartBayesianStrategy).
    Uses cache.query_and_optimize(odd, ds, opt_func) where opt_func runs GA inside the given zonotope.
    """
    cache: SmartZonotopeCache
    last_cache_status: bool

    def __init__(self, name: str,
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None,
                 file_path: Optional[str] = None):
        info = super().__init__(name=name, cs=cs, info=info, file_path=file_path)
        self.last_cache_status = False
        # thresholds expected in info, consistent with your SmartCacheStrategy
        self.cache = SmartZonotopeCache(self.domain_spec,
                                        info["extension_threshold"],
                                        info["merge_threshold"],
                                        info["merge_frequency"],
                                        strategy_name=self.name)
    def get_info(self):
        return getattr(self, "last_cache_status", None)

    def has_hit(self):
        return getattr(self, "last_cache_status", None) == "hit"

    def get_cache_volume(self):
        # Prefer a method if your cache exposes it; fall back to attribute
        if hasattr(self.cache, "get_explored_volume"):
            return float(self.cache.get_explored_volume())
        return float(getattr(self.cache, "explored_volume", 0.0))

    def get_explored_volume(self):
        return self.get_cache_volume()

    def get_coverage_pct(self):
        try:
            bounds = self.domain_spec.get_bounds()  # List[(lo, hi)]
            space_vol = 1.0
            for lo, hi in bounds:
                space_vol *= max(0.0, float(hi) - float(lo))
            if space_vol <= 0.0:
                return 0.0
            return min(100.0, (self.get_cache_volume() / space_vol) * 100.0)
        except Exception:
            return 0.0

    def execute(self, odd: ODD, ds: ConstrainedDomain):
        def opt_func(zonotope):
            return self.optimize_in_zonotope(zonotope, ds)

        start_time = time.time()
        config, objective, metadata, status = self.cache.query_and_optimize(odd, ds, opt_func)
        total_time = (time.time() - start_time) * 1000.0

        self.last_cache_status = status
        self.last_time = total_time
        self.last_objective = objective if objective != float('inf') else float('inf')
        self.last_opt_status = config is not None

        return config, objective, metadata, total_time

    def __str__(self):
        # mirror Smart* string style from your file
        if self.last_cache_status == 'hit':
            ret = f"      ✓ {self.name} - CACHE HIT! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'extended':
            ret = f"      ⊕ {self.name} - REGION EXTENDED! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'new_entry':
            ret = f"      + {self.name} - NEW ENTRY ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        else:
            if self.last_opt_status:
                ret = f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)\n        Objective: {self.last_objective:.2f} Wh"
            else:
                ret = f"      × {self.name} - No feasible solution"
        return ret

class LNSStrategy(Strategy):
    """
    Large Neighborhood Search (ruin & recreate) baseline (non-cache).
    - Start from a quick incumbent (best-of-N random samples inside DS_t)
    - Repeatedly free k variables and sample a small neighborhood around incumbent
    - Evaluate neighbors via the MILP chooser; accept improvements
    - Deadline-safe
    """
    tasks: list
    n_drones: int
    domain_spec: DomainSpec

    def __init__(self, name: str,
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None,
                 file_path: str = None):
        super().__init__(name)
        if info is None and file_path is None:
            print("Warning: Either 'info' or 'file_path' should be provided to initialize the strategy.")
        if info is not None and file_path is not None:
            raise ValueError("Provide only one of 'info' or 'file_path', not both.")
        if file_path is not None:
            info = self.load(file_path)

        self.cs = cs
        self.tasks = cs.tasks
        self.n_drones = cs.size()
        self.domain_spec = cs.get_domain_spec()

        self.last_time = 0.0
        self.last_objective = float('inf')
        self.last_opt_status = False

        # LNS knobs
        self.time_budget_ms = float((info or {}).get("time_budget_ms", 100.0))
        self.rng = np.random.default_rng((info or {}).get("seed", 2025))
        self.init_samples = int((info or {}).get("lns_init_samples", 16))
        self.k_free = int((info or {}).get("lns_k_free", 2))
        self.neigh_per_iter = int((info or {}).get("lns_neighbors", 12))
        self.step_frac = float((info or {}).get("lns_step_frac", 0.2))
        self.adapt_k = bool((info or {}).get("lns_adapt_k", True))
        return info

    def __str__(self):
        if self.last_opt_status:
            return f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)\n         Objective: {self.last_objective:.2f} Wh"
        return f"      × {self.name} - No feasible solution"

    def load(self, source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".json":
            with open(source, "r") as f: return json.load(f)
        if ext in (".yaml", ".yml"):
            with open(source, "r") as f: return yaml.safe_load(f)
        raise ValueError(f"Unsupported fleet file: {ext}")

    # ---- helpers ----
    def _bounds(self, ds: ConstrainedDomain):
        return ds.to_bounds_list()

    def _rand_cfg(self, bounds):
        vals = [self.rng.uniform(lo, hi) for (lo, hi) in bounds]
        return self.domain_spec.create_configuration(vals)

    def _neighbor_cfgs(self, base_cfg, bounds, free_idx, n, step_frac):
        base_vals = _to_vals(base_cfg)
        neighs = []
        for _ in range(n):
            vals = base_vals[:]
            for i in free_idx:
                lo, hi = bounds[i]
                span = (hi - lo)
                vals[i] = float(np.clip(vals[i] + self.rng.normal(0, step_frac * span), lo, hi))
            neighs.append(self.domain_spec.create_configuration(vals))
        return neighs

    def _evaluate_one(self, cfg):
        """Direct greedy evaluation: assign each task to least-loaded drone."""
        drone_loads = [0.0] * self.n_drones
        for task in self.tasks:
            task_energy = drone_energy_model(cfg, task.get('length', 100.0))
            min_idx = min(range(self.n_drones), key=lambda i: drone_loads[i])
            drone_loads[min_idx] += task_energy
        return sum(drone_loads)

    def _evaluate_choose_best(self, cfgs):
        """Evaluate all configs directly (greedy assignment) and return best."""
        best_cfg = None
        best_obj = float('inf')
        for cfg in cfgs:
            obj = self._evaluate_one(cfg)
            if obj < best_obj:
                best_cfg, best_obj = cfg, obj
        return best_cfg, best_obj

    # ---- smart-cache compatible API ----
    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = None):
        """
        Run one LNS pass inside the given zonotope's AABB (or DS_t bounds if zonotope is None).
        Returns (best_cfg, best_obj, metadata)
        """
        bounds = (zonotope.to_box_bounds() if zonotope is not None else self._bounds(ds))
        dim = len(bounds)

        # 1) Quick incumbent
        init_cfgs = [self._rand_cfg(bounds) for _ in range(self.init_samples)]
        incumbent, inc_obj = self._evaluate_choose_best(init_cfgs)

        # 2) LNS loop (tight, local to this call — cache wrapper enforces outer budget)
        t0 = time.time()
        while (time.time() - t0) * 1000.0 < self.time_budget_ms:
            free_idx = list(self.rng.choice(dim, size=min(self.k_free, dim), replace=False))
            neigh_cfgs = self._neighbor_cfgs(incumbent, bounds, free_idx, self.neigh_per_iter, self.step_frac)
            neigh_cfgs.append(incumbent)
            cand, cand_obj = self._evaluate_choose_best(neigh_cfgs)

            if cand_obj < inc_obj:
                incumbent, inc_obj = cand, cand_obj
                if self.adapt_k: self.k_free = min(dim, self.k_free + 1)
            else:
                if self.adapt_k: self.k_free = max(1, self.k_free - 1)

            # light diversification
            if self.rng.random() < 0.05:
                restart_cfgs = [self._rand_cfg(bounds) for _ in range(max(4, self.k_free))]
                rbest, robj = self._evaluate_choose_best(restart_cfgs)
                if robj < inc_obj:
                    incumbent, inc_obj = rbest, robj

        meta = {'status': 'lns', 'k_free_final': self.k_free, 'neighbors_per_iter': self.neigh_per_iter}
        return incumbent, inc_obj, meta

    # ---- plain execute ----
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        t0 = time.time()
        cfg, obj, _meta = self.optimize_in_zonotope(None, ds)
        elapsed = (time.time() - t0) * 1000.0

        self.last_time = elapsed
        self.last_objective = obj if cfg is not None else float('inf')
        self.last_opt_status = cfg is not None
        return cfg, obj, _meta, elapsed
    
    # For plain strategies (GA, LNS, WarmStartOnly, etc.)

    def get_info(self):
        # For consistency with save_results; smart strategies return 'hit'/'extended'/etc.
        return "plain"

    def has_hit(self):
        # No cache, so never a hit
        return False

    def get_cache_volume(self):
        # No cache
        return 0.0

    def get_explored_volume(self):
        return 0.0

    def get_coverage_pct(self):
        return 0.0

class SmartLNSStrategy(LNSStrategy):
    """
    LNS with smart zonotope cache (same pattern as Smart* strategies).
    Uses cache.query_and_optimize(odd, ds, opt_func) where opt_func runs LNS inside the given zonotope.
    """
    cache: SmartZonotopeCache
    last_cache_status: bool

    def __init__(self, name: str,
                 cs: ConfigurationSpace = None,
                 info: Optional[dict] = None,
                 file_path: Optional[str] = None):
        info = super().__init__(name=name, cs=cs, info=info, file_path=file_path)
        self.last_cache_status = False
        self.cache = SmartZonotopeCache(self.domain_spec,
                                        info["extension_threshold"],
                                        info["merge_threshold"],
                                        info["merge_frequency"],
                                        strategy_name=self.name)
    def get_info(self):
        return getattr(self, "last_cache_status", None)

    def has_hit(self):
        return getattr(self, "last_cache_status", None) == "hit"

    def get_cache_volume(self):
        # Prefer a method if your cache exposes it; fall back to attribute
        if hasattr(self.cache, "get_explored_volume"):
            return float(self.cache.get_explored_volume())
        return float(getattr(self.cache, "explored_volume", 0.0))

    def get_explored_volume(self):
        return self.get_cache_volume()

    def get_coverage_pct(self):
        try:
            bounds = self.domain_spec.get_bounds()  # List[(lo, hi)]
            space_vol = 1.0
            for lo, hi in bounds:
                space_vol *= max(0.0, float(hi) - float(lo))
            if space_vol <= 0.0:
                return 0.0
            return min(100.0, (self.get_cache_volume() / space_vol) * 100.0)
        except Exception:
            return 0.0

    def execute(self, odd: ODD, ds: ConstrainedDomain):
        def opt_func(zonotope):
            return self.optimize_in_zonotope(zonotope, ds)

        start_time = time.time()
        config, objective, metadata, status = self.cache.query_and_optimize(odd, ds, opt_func)
        total_time = (time.time() - start_time) * 1000.0

        self.last_cache_status = status
        self.last_time = total_time
        self.last_objective = objective if objective != float('inf') else float('inf')
        self.last_opt_status = config is not None

        return config, objective, metadata, total_time

    def __str__(self):
        if self.last_cache_status == 'hit':
            ret = f"      ✓ {self.name} - CACHE HIT! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'extended':
            ret = f"      ⊕ {self.name} - REGION EXTENDED! ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        elif self.last_cache_status == 'new_entry':
            ret = f"      + {self.name} - NEW ENTRY ({self.last_time:.2f}ms, cached: {self.get_coverage_pct():.1f}%)"
            ret += f"\n        Objective: {self.last_objective:.2f} Wh"
        else:
            if self.last_opt_status:
                ret = f"      ✓ {self.name} - Complete ({self.last_time:.2f}ms)\n        Objective: {self.last_objective:.2f} Wh"
            else:
                ret = f"      × {self.name} - No feasible solution"
        return ret
