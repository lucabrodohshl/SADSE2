


from src.strategy import Strategy

from src.domain import DomainSpec, Parameter, ODD, ConstrainedDomain
from src.smart_cache import SmartZonotopeCache
from src.milp_solver import solve_task_assignment_milp, drone_energy_model
from src.zonotope_ops import Zonotope

import time
import numpy as np
class DiscreteStrategy(Strategy):
    """A strategy that performs discrete optimization without caching."""
    tasks: list
    n_drones : int
    
    last_time: float
    def __init__(self, name: str):
        super().__init__(name)
    # Simple energy objective: minimize based on configuration parameters
    def simple_energy_objective(config, task):
            """Simplified energy model for 6D optimization."""
            try:
                speed = config.get('speed')
                altitude = config.get('altitude')
                camera_res = config.get('camera_res')
                spray_rate = config.get('spray_rate')
                battery = config.get('battery_capacity')
                payload = config.get('payload_weight')
                
                task_length = task.get('length', 100.0)
                
                # Simple energy model: higher speed/altitude/payload → more energy
                # Normalized to 0-100 Wh range
                base_energy = task_length * 0.1  # 10 Wh per 100m
                speed_factor = (speed / 40.0) ** 2
                altitude_factor = altitude / 100.0
                payload_factor = payload / 8.0
                
                # Camera and spray also consume energy
                camera_factor = camera_res / 12.0
                spray_factor = spray_rate / 10.0
                
                # Battery constraint (lower battery = penalty)
                battery_factor = 1.0 if battery >= 8000 else 1.5
                
                energy = base_energy * (1.0 + speed_factor + altitude_factor * 0.3 + 
                                    payload_factor * 0.5 + camera_factor * 0.2 + 
                                    spray_factor * 0.3) * battery_factor
                
                return energy
            except Exception as e:
                return 1000.0  # High penalty for errors
        
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """
        Execute the discrete optimization strategy.
        
        :param objective_function: The objective function to minimize.
        :param odd: The operational design domain (ODD) for the current scenario.
        :return: The optimal configuration found.
        """
        #print(f"Executing discrete optimization for ODD: {odd}")
        # Placeholder for actual discrete optimization logic
        # In a real implementation, this would involve setting up and solving an MILP
        optimal_config = None  # Replace with actual optimal configuration
        return optimal_config

    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = simple_energy_objective):
        r"""
        Optimization function for a specific zonotope region.
        
        This discretizes the zonotope and runs MILP optimization.
        """
        
        # Discretize the region (use fewer levels for 6D: 2^6 = 64 configs)
        configurations = ds.discretize(num_levels=2)
        
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


    def has_hit(self):
        raise NotImplementedError("has_hit method must be implemented by subclasses.")
class BaseLineStrategy(DiscreteStrategy):
    """A baseline strategy that performs discrete optimization without caching."""
    last_time: float
    last_objective: float
    def __init__(self, name: str, 
                 tasks: list,
                 n_drones : int,
                domain_spec: DomainSpec):
        super().__init__(name)
        self.tasks = tasks
        self.n_drones = n_drones
        self.domain_spec = domain_spec
        self.last_time = 0.0
        self.last_objective = float('inf')
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
    last_objective: float
    def __init__(self, name: str, 
                 tasks: list,
                 n_drones : int,
                domain_spec: DomainSpec,
                extension_threshold: float = 0.05,  # 5% worse is acceptable for extension
                merge_threshold: float = 0.10,       # 10% difference for merging
                merge_frequency: int = 5):
        super().__init__(name)
        self.tasks = tasks
        self.n_drones = n_drones
        self.domain_spec = domain_spec
        self.last_cache_status = False  # Track if last execution was a cache hit
        self.cache = SmartZonotopeCache(domain_spec, extension_threshold, merge_threshold, merge_frequency)
    def get_info(self):
        return self.last_cache_status

    def has_hit(self):
            return self.last_cache_status == 'hit'
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
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def execute(self, odd: ODD, ds: ConstrainedDomain):
        """
        Execute the linear optimization strategy.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("execute method must be implemented by subclasses.")
    
    def __str__(self):
        return f"LinearStrategy: {self.name}"
    
    def get_explored_volume(self):
        raise NotImplementedError("get_explored_volume method must be implemented by subclasses.")
    
    def has_hit(self):
        raise NotImplementedError("has_hit method must be implemented by subclasses.")
    
    def optimize_in_zonotope(self, zonotope, ds, objective_function: callable = None):
        """
        Solve Linear Program for task assignment with continuous configuration.
        
        This is the core LP formulation shared by all linear strategies.
        Unlike discrete strategies that discretize the configuration space,
        this formulates a continuous LP where configuration parameters are
        continuous decision variables.
        
        :param zonotope: The zonotope region to optimize within
        :param ds: The constrained domain
        :param objective_function: Optional custom objective (uses drone_energy_model by default)
        :return: Tuple of (optimal_config, objective_value, metadata)
        """
        if objective_function is None:
            from src.milp_solver import drone_energy_model
            objective_function = lambda config, task: drone_energy_model(config, task.get('length', 100.0))
        
        # Get bounds from zonotope (use AABB for simplicity)
        bounds = zonotope.to_box_bounds()
        param_names = [p.name for p in self.domain_spec.parameters]
        
        # Create Linear Program
        from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatusOptimal, value, GUROBI_CMD
        
        prob = LpProblem("LinearTaskAssignment", LpMinimize)
        
        num_tasks = len(self.tasks)
        num_agents = self.n_drones
        num_params = len(param_names)
        
        # Decision variables:
        # 1. Configuration parameters (CONTINUOUS)
        config_vars = {}
        for i, param_name in enumerate(param_names):
            config_vars[param_name] = LpVariable(
                f"config_{param_name}",
                lowBound=bounds[i, 0],
                upBound=bounds[i, 1],
                cat='Continuous'
            )
        
        # 2. Task assignments (BINARY): x[i,j] = 1 if agent i does task j
        x = {}
        for i in range(num_agents):
            for j in range(num_tasks):
                x[i, j] = LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Objective: Minimize total energy
        # Since we can't multiply x[i,j] * energy[j] (both variables), we reformulate:
        # Instead of auxiliary energy variables, we directly express the objective
        # in terms of the task assignments and configuration parameters.
        
        # Get reference values (middle of bounds) for linearization
        ref_speed = (bounds[0, 0] + bounds[0, 1]) / 2.0 if 'speed' in param_names else 10.0
        
        # Build objective as sum over all tasks
        # For each task j, compute its energy contribution based on configuration
        objective_terms = []
        
        for j in range(num_tasks):
            task = self.tasks[j]
            task_length = task.get('length', 100.0)
            
            # Linear approximation of drone energy model
            base_energy = 10.0 * task_length / 100.0
            
            # Build energy expression for this task
            # E_j = base + c1*(speed-ref) + c2*altitude + ...
            task_energy_expr = base_energy
            
            if 'speed' in param_names:
                # Speed contribution (linearized around reference)
                speed_coeff = 0.2 * ref_speed * task_length / 100.0
                task_energy_expr += speed_coeff * (config_vars['speed'] - ref_speed)
            
            if 'altitude' in param_names:
                # Altitude contribution (linear)
                altitude_coeff = 0.01 * task_length / 100.0
                task_energy_expr += altitude_coeff * config_vars['altitude']
            
            if 'camera_res' in param_names:
                camera_coeff = 0.4 * task_length / 100.0
                task_energy_expr += camera_coeff * config_vars['camera_res']
            
            if 'spray_rate' in param_names:
                spray_coeff = 1.8 * task_length / 100.0
                task_energy_expr += spray_coeff * config_vars['spray_rate']
            
            if 'battery_capacity' in param_names:
                battery_coeff = 0.005 * task_length / 100.0
                task_energy_expr += battery_coeff * config_vars['battery_capacity']
            
            if 'sensor_sensitivity' in param_names:
                # Inverse relationship - higher sensitivity = less power
                sensor_coeff = -0.001 * task_length / 100.0
                task_energy_expr += sensor_coeff * config_vars['sensor_sensitivity']
            
            # For each agent, add the energy cost if that agent performs this task
            # Since energy depends only on config (shared by all agents), we can write:
            # Σ_i (x[i,j] * E_j) = E_j * Σ_i x[i,j] = E_j * 1 = E_j (due to constraint Σ_i x[i,j] = 1)
            # So the total energy is just Σ_j E_j
            objective_terms.append(task_energy_expr)
        
        # Set objective: minimize total energy across all tasks
        total_energy = lpSum(objective_terms)
        prob += total_energy
        
        # Constraints:
        # 1. Each task assigned to exactly one agent
        for j in range(num_tasks):
            prob += lpSum([x[i, j] for i in range(num_agents)]) == 1, f"Task_{j}_assigned"
        
        # 2. Domain-specific constraints (if any)
        # Add any additional linear constraints from domain spec
        
        # Solve the Linear Program using Gurobi
        try:
            # Use Gurobi solver
            solver = GUROBI_CMD(msg=0)
            status = prob.solve(solver)
            
            if status != LpStatusOptimal:
                return None, float('inf'), {'status': 'infeasible'}
            
            # Extract solution
            optimal_values = {name: value(var) for name, var in config_vars.items()}
            optimal_config = self.domain_spec.create_configuration(
                [optimal_values[p.name] for p in self.domain_spec.parameters]
            )
            
            # Extract assignment
            assignment = {i: [] for i in range(num_agents)}
            for i in range(num_agents):
                for j in range(num_tasks):
                    if value(x[i, j]) > 0.5:
                        assignment[i].append(j)
            
            objective_value = value(prob.objective)
            
            metadata = {
                'assignment': assignment,
                'solver_status': 'optimal',
                'region_volume': zonotope.volume(),
                'method': 'linear_program',
                'solver': 'gurobi'
            }
            
            return optimal_config, objective_value, metadata
        
        except Exception as e:
            import warnings
            warnings.warn(f"Linear solver (Gurobi) failed: {e}")
            return None, float('inf'), {'status': 'error', 'error': str(e)}


class FullLinearStrategy(LinearStrategy):
    """A linear strategy that always re-optimizes without caching.
    
    Uses continuous Linear Programming (LP) instead of discretization.
    Solves a continuous optimization problem over the entire design space.
    """
    last_time: float
    last_objective: float
    tasks: list
    n_drones: int
    domain_spec: DomainSpec
    
    def __init__(self, name: str, 
                 tasks: list,
                 n_drones: int,
                 domain_spec: DomainSpec):
        super().__init__(name)
        self.tasks = tasks
        self.n_drones = n_drones
        self.domain_spec = domain_spec
        self.last_time = 0.0
        self.last_objective = float('inf')
        self.last_opt_status = False
    
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
    
    # Note: optimize_in_zonotope is inherited from LinearStrategy base class
    
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
    



class SmartLinearStrategy(LinearStrategy):
    """A linear strategy that utilizes a smart zonotope cache for optimization.
    
    Uses continuous Linear Programming (LP) with intelligent caching.
    The cache stores regions where LP solutions are valid, enabling
    fast queries without re-solving the entire LP.
    """
    last_cache_status: bool
    cache: SmartZonotopeCache
    last_objective: float
    tasks: list
    n_drones: int
    
    def __init__(self, name: str,
                 tasks: list,
                 n_drones: int,
                 domain_spec: DomainSpec,
                 extension_threshold: float = 0.05,  # 5% worse is acceptable for extension
                 merge_threshold: float = 0.10,       # 10% difference for merging
                 merge_frequency: int = 5):
        super().__init__(name)
        self.tasks = tasks
        self.n_drones = n_drones
        self.domain_spec = domain_spec
        self.last_cache_status = False  # Track if last execution was a cache hit
        self.last_time = 0.0
        self.last_objective = float('inf')
        self.last_opt_status = False
        self.cache = SmartZonotopeCache(domain_spec, extension_threshold, merge_threshold, merge_frequency)
    
    def get_info(self):
        return self.last_cache_status

    def has_hit(self):
        return self.last_cache_status == 'hit'
    
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