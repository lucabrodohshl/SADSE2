"""
Zonotope-based caching system for MILP optimization.

A professional, general n-dimensional caching implementation using zonotopes
for efficient storage and retrieval of optimization results.
"""

__version__ = "1.0.0"

# Core imports
from .zonotope_ops import Zonotope, zonotope_intersection, zonotope_union, zonotope_subtract
from .domain import (
    DomainSpec, Parameter, ParameterType,
    ConstrainedDomain, Configuration, ODD, ConfigurationSpace
)
from .zonotope_cache import ZonotopeCache, CacheEntry
from .milp_solver import (
    solve_task_assignment_milp,
    solve_simple_optimization,
    drone_energy_model
)
from .utils import (
    visualize_2d_cache,
    plot_cache_statistics,
    compare_configurations,
    export_cache_to_json,
    print_cache_summary
)


from .environment import Environment
from .strategy import Strategy

__all__ = [
    # Zonotope operations
    'Zonotope',
    'zonotope_intersection',
    'zonotope_union',
    'zonotope_subtract',
    
    # Domain specification
    'DomainSpec',
    'Parameter',
    'ParameterType',
    'ConstrainedDomain',
    'Configuration',
    'ODD',
    'DroneODD',
    'ConfigurationSpace',
    
    # Cache
    'ZonotopeCache',
    'CacheEntry',
    
    # MILP solving
    'solve_task_assignment_milp',
    'solve_simple_optimization',
    'drone_energy_model',
    
    # Utilities
    'visualize_2d_cache',
    'plot_cache_statistics',
    'compare_configurations',
    'export_cache_to_json',
    'print_cache_summary',

    # Strategy pattern
    'Strategy',

    # Environment
    'Environment',
]
