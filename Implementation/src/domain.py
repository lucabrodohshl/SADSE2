"""
Domain specification for general n-dimensional design spaces.

Defines:
- DomainSpec: General n-dimensional parameter space
- ODD: Operational Design Domain (environmental constraints)
- Configuration: Specific parameter values
- Methods for computing CS, DS, and applying ODD constraints

This module is completely general and works with any number of dimensions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

class ParameterType(Enum):
    """Type of design parameter."""
    CONTINUOUS = "continuous"    # e.g., speed, altitude
    DISCRETE = "discrete"        # e.g., camera resolution levels
    INTEGER = "integer"          # e.g., number of sensors


@dataclass
class Parameter:
    """
    Definition of a single design parameter.
    
    Attributes:
        name: Parameter name (e.g., 'speed', 'altitude')
        bounds: (min, max) values for this parameter
        unit: Physical unit (e.g., 'm/s', 'm', 'MP')
        param_type: Type of parameter (continuous, discrete, integer)
        description: Human-readable description
    """
    name: str
    bounds: Tuple[float, float]
    unit: str = ""
    param_type: ParameterType = ParameterType.CONTINUOUS
    description: str = ""
    
    def __post_init__(self):
        """Validate parameter definition."""
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"Invalid bounds for {self.name}: "
                f"min ({self.bounds[0]}) >= max ({self.bounds[1]})"
            )
    
    @property
    def range_size(self) -> float:
        """Return the size of the parameter range."""
        return self.bounds[1] - self.bounds[0]
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1]."""
        return (value - self.bounds[0]) / self.range_size
    
    def denormalize(self, normalized: float) -> float:
        """Convert from [0, 1] back to parameter range."""
        return self.bounds[0] + normalized * self.range_size


@dataclass
class DomainSpec:
    """
    General n-dimensional domain specification.
    
    This class defines the parameter space for optimization problems.
    It's completely general and works with any number of dimensions.
    
    Attributes:
        parameters: List of Parameter definitions
        dimension: Number of dimensions (automatically inferred)
        constraint_functions: Optional list of functions that constrain the domain
        
    Example:
        >>> spec = DomainSpec(parameters=[
        ...     Parameter('speed', (0, 50), 'm/s'),
        ...     Parameter('altitude', (10, 100), 'm'),
        ...     Parameter('camera_res', (4, 12), 'MP'),
        ... ])
        >>> spec.dimension
        3
    """
    parameters: List[Parameter]
    constraint_functions: List[Callable] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate domain specification."""
        if not self.parameters:
            raise ValueError("Domain must have at least one parameter")
        
        # Check for duplicate names
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names found: {names}")
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the domain."""
        return len(self.parameters)
    
    @property
    def parameter_names(self) -> List[str]:
        """Return list of parameter names."""
        return [p.name for p in self.parameters]
    
    def get_parameter(self, name: str) -> Parameter:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        raise KeyError(f"Parameter '{name}' not found")
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for all parameters.
        
        Returns:
            List of (min, max) tuples, one per dimension
        """
        return [p.bounds for p in self.parameters]
    
    def get_bounds_array(self) -> np.ndarray:
        """
        Get bounds as numpy array.
        
        Returns:
            Array of shape (n, 2) with (min, max) for each dimension
        """
        return np.array(self.get_bounds(), dtype=float)
    
    def apply_odd_constraints(self, odd: 'ODD') -> 'ConstrainedDomain':
        """
        Apply ODD constraints to create design space (DS).
        
        This method applies environmental constraints from the ODD to
        narrow down the feasible parameter ranges.
        
        Args:
            odd: Operational Design Domain with environmental conditions
            
        Returns:
            ConstrainedDomain representing DS = CS ∩ ODD
        """
        # Start with base bounds
        constrained_bounds = self.get_bounds_array().copy()
        
        # Apply each constraint function
        for constraint_func in self.constraint_functions:
            constrained_bounds = constraint_func(constrained_bounds, odd)
        
        # Apply ODD-specific constraints if they exist
        if hasattr(odd, 'apply_constraints'):
            constrained_bounds = odd.apply_constraints(constrained_bounds, self)
        
        return ConstrainedDomain(
            domain_spec=self,
            bounds=constrained_bounds,
            odd=odd
        )
    
    def create_configuration(self, values: Union[np.ndarray, Dict[str, float]]) -> 'Configuration':
        """
        Create a configuration from values.
        
        Args:
            values: Either array of values or dict mapping parameter names to values
            
        Returns:
            Configuration object
        """
        if isinstance(values, dict):
            # Convert dict to array in correct order
            values_array = np.array([values[name] for name in self.parameter_names])
        else:
            values_array = np.asarray(values, dtype=float)
        
        if len(values_array) != self.dimension:
            raise ValueError(
                f"Wrong number of values: expected {self.dimension}, got {len(values_array)}"
            )
        
        return Configuration(domain_spec=self, values=values_array)
    
    def discretize(self, num_levels: int = 3) -> List['Configuration']:
        """
        Generate discrete set of configurations by sampling parameter space.
        
        Args:
            num_levels: Number of levels per dimension (total configs = num_levels^dimension)
            
        Returns:
            List of Configuration objects
            
        Warning:
            For high dimensions, this can generate many configurations!
            num_configs = num_levels^dimension
        """
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2")
        
        # Generate linearly spaced values for each dimension
        param_values = []
        for param in self.parameters:
            if param.param_type == ParameterType.DISCRETE:
                # For discrete parameters, use only min and max
                values = np.array([param.bounds[0], param.bounds[1]])
            else:
                values = np.linspace(param.bounds[0], param.bounds[1], num_levels)
            param_values.append(values)
        
        # Generate all combinations
        configs = []
        
        def generate_recursive(dim: int, current_values: List[float]):
            """Recursively generate all combinations."""
            if dim == self.dimension:
                configs.append(self.create_configuration(np.array(current_values)))
                return
            
            for value in param_values[dim]:
                generate_recursive(dim + 1, current_values + [value])
        
        generate_recursive(0, [])
        
        return configs
    
    def volume(self) -> float:
        """Compute total volume of the parameter space."""
        volume = 1.0
        for param in self.parameters:
            volume *= param.range_size
        return volume
    
    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join([f"{p.name}: [{p.bounds[0]}, {p.bounds[1]}] {p.unit}" 
                                for p in self.parameters])
        return f"DomainSpec({self.dimension}D: {params_str})"


@dataclass
class ConstrainedDomain:
    """
    Domain with applied constraints (represents DS = CS ∩ ODD).
    
    This is the result of applying ODD constraints to a DomainSpec.
    It maintains a reference to the original spec and the constrained bounds.
    
    Attributes:
        domain_spec: Original domain specification
        bounds: Constrained bounds (n × 2 array)
        odd: ODD that was applied
    """
    domain_spec: DomainSpec
    bounds: np.ndarray
    odd: 'ODD'
    
    def __post_init__(self):
        """Validate constrained domain."""
        if self.bounds.shape != (self.domain_spec.dimension, 2):
            raise ValueError(
                f"Bounds shape {self.bounds.shape} doesn't match dimension {self.domain_spec.dimension}"
            )
        
        # Check that bounds are valid
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            # Some dimension has min >= max (empty domain)
            pass  # This is allowed - represents infeasible DS
    
    @property
    def dimension(self) -> int:
        """Return dimension."""
        return self.domain_spec.dimension
    
    def is_empty(self, tol: float = 1e-12) -> bool:
        """Check if domain is empty (infeasible)."""
        return np.any(self.bounds[:, 1] - self.bounds[:, 0] < tol)
    
    def volume(self) -> float:
        """Compute volume of constrained domain."""
        if self.is_empty():
            return 0.0
        
        sizes = self.bounds[:, 1] - self.bounds[:, 0]
        return np.prod(np.maximum(sizes, 0))
    
    def contains(self, config: 'Configuration') -> bool:
        """Check if configuration is within this constrained domain."""
        values = config.as_array()
        return np.all(values >= self.bounds[:, 0]) and np.all(values <= self.bounds[:, 1])
    
    def discretize(self, num_levels: int = 3) -> List['Configuration']:
        """
        Generate discrete configurations within constrained bounds.
        
        Args:
            num_levels: Number of levels per dimension
            
        Returns:
            List of Configuration objects within the constrained bounds
        """
        if self.is_empty():
            return []
        
        # Generate linearly spaced values for each dimension using constrained bounds
        param_values = []
        for i in range(self.dimension):
            values = np.linspace(self.bounds[i, 0], self.bounds[i, 1], num_levels)
            param_values.append(values)
        
        # Generate all combinations
        configs = []
        
        def generate_recursive(dim: int, current_values: List[float]):
            """Recursively generate all combinations."""
            if dim == self.dimension:
                configs.append(self.domain_spec.create_configuration(np.array(current_values)))
                return
            
            for value in param_values[dim]:
                generate_recursive(dim + 1, current_values + [value])
        
        generate_recursive(0, [])
        
        return configs
    
    def to_bounds_list(self) -> List[Tuple[float, float]]:
        """Convert bounds array to list of tuples."""
        return [(self.bounds[i, 0], self.bounds[i, 1]) for i in range(self.dimension)]
    
    def __repr__(self) -> str:
        """String representation."""
        bounds_str = ", ".join([f"{self.domain_spec.parameters[i].name}: "
                                f"[{self.bounds[i, 0]:.2f}, {self.bounds[i, 1]:.2f}]"
                                for i in range(self.dimension)])
        return f"ConstrainedDomain({bounds_str})"


@dataclass
class Configuration:
    """
    Specific parameter configuration.
    
    Represents a point in the n-dimensional parameter space.
    
    Attributes:
        domain_spec: Domain specification this configuration belongs to
        values: Parameter values (n-dimensional array)
    """
    domain_spec: DomainSpec
    values: np.ndarray
    
    def __post_init__(self):
        """Validate configuration."""
        self.values = np.asarray(self.values, dtype=float)
        
        if len(self.values) != self.domain_spec.dimension:
            raise ValueError(
                f"Wrong number of values: expected {self.domain_spec.dimension}, "
                f"got {len(self.values)}"
            )
    
    def as_array(self) -> np.ndarray:
        """Return values as numpy array."""
        return self.values
    
    def as_dict(self) -> Dict[str, float]:
        """Return as dictionary mapping parameter names to values."""
        return {name: val for name, val in zip(self.domain_spec.parameter_names, self.values)}
    
    def get(self, param_name: str) -> float:
        """Get value of specific parameter."""
        try:
            idx = self.domain_spec.parameter_names.index(param_name)
            return self.values[idx]
        except ValueError:
            raise KeyError(f"Parameter '{param_name}' not found")
    
    def is_feasible(self) -> bool:
        """Check if configuration is within domain bounds."""
        bounds = self.domain_spec.get_bounds_array()
        return np.all(self.values >= bounds[:, 0]) and np.all(self.values <= bounds[:, 1])
    
    def __repr__(self) -> str:
        """String representation."""
        config_str = ", ".join([f"{name}={val:.2f}"
                                for name, val in zip(self.domain_spec.parameter_names, self.values)])
        return f"Configuration({config_str})"
    def __eq__(self, other) -> bool:
        """Equality check based on values and domain spec."""
        if not isinstance(other, Configuration):
            return False
        return (self.domain_spec == other.domain_spec and
                np.allclose(self.values, other.values))

@dataclass(eq=False)
class ODD:
    """
    Operational Design Domain - environmental constraints.
    
    This class is flexible and can represent various environmental conditions.
    Users can subclass this to add domain-specific constraints.
    
    Note: We use eq=False to allow custom __hash__ implementation.
    
    Attributes:
        timestamp: Time of this ODD
        conditions: Dictionary of environmental conditions
    """
    timestamp: float
    conditions: Dict[str, float] = field(default_factory=dict)
    
    def get(self, condition_name: str, default: float = 0.0) -> float:
        """Get value of environmental condition."""
        return self.conditions.get(condition_name, default)
    
    def apply_constraints(self, bounds: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        """
        Apply ODD-specific constraints to bounds.
        
        Override this method in subclasses to implement domain-specific logic.
        
        Args:
            bounds: Current bounds (n × 2 array)
            domain_spec: Domain specification
            
        Returns:
            Modified bounds (n × 2 array)
        """
        # Default: no modification
        return bounds
    
    def __hash__(self) -> int:
        """
        Hash for caching based on conditions only (not timestamp).
        
        Timestamp is excluded from hash because we want to merge entries
        with the same environmental conditions but different timestamps.
        
        Converts numpy types to Python types for hashability.
        """
        # Round conditions to 2 decimal places for hashing
        # Convert numpy types to Python types for hashability
        rounded = tuple(sorted(
            (k, round(float(v), 2)) for k, v in self.conditions.items()
        ))
        return hash(rounded)
    
    def __repr__(self) -> str:
        """String representation."""
        cond_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(self.conditions.items())])
        return f"ODD(t={self.timestamp:.1f}s, {cond_str})"
    def contains(self, other: 'ODD', threshold: float = 0.1) -> bool:
        """
        Check if this ODD contains another ODD.
        
        This means ALL conditions in 'other' must be within the threshold
        ranges of this ODD's corresponding conditions.
        
        Args:
            other: Another ODD to compare
            threshold: Tolerance for condition matching

        Returns:
            True if this ODD contains the other ODD, False otherwise.
        """
        # Check that all conditions in 'other' are contained in this ODD
        for key, value in other.conditions.items():
            if key not in self.conditions:
                return False  # Missing condition means no containment
            if not (self.conditions[key] - threshold <= value <= self.conditions[key] + threshold):
                return False  # Condition outside threshold range
        return True
    
    def intersects(self, other: 'ODD', threshold: float = 0.1) -> bool:
        """
        Check if this ODD intersects with another ODD in the set-theoretic sense.
        
        Two ODDs intersect if ALL their common conditions are compatible
        (within threshold). If any common condition conflicts, there's no intersection.
        
        Args:
            other: Another ODD to compare
            threshold: Tolerance for condition matching

        Returns:
            True if ODDs have a non-empty intersection, False otherwise.
        """
        # Find common condition keys
        common_keys = set(self.conditions.keys()) & set(other.conditions.keys())
        #print("DEBUG: ODDs intersect based on common conditions.")
        #print(f"DEBUG: Self conditions: {self.conditions}")
        #print(f"DEBUG: Other conditions: {other.conditions}")
        #input("Press Enter to continue...")
        if not common_keys:
            return False  # No common conditions 
        
        # Check that ALL common conditions are compatible
        for key in common_keys:
            self_value = self.conditions[key]
            other_value = other.conditions[key]
            
            # If any condition is incompatible, intersection is empty
            if abs(self_value - other_value) > threshold:
                return False  # Conflicting condition -> empty intersection
        #print("DEBUG: ODDs have a non-empty intersection.")
        return True  # All common conditions are compatible -> non-empty intersection


from typing import List, Optional
import json
import yaml  # make sure PyYAML is installed

class ConfigurationSpace:
    """
    Configuration Space (CS) for a given DomainSpec.
    
    Can be initialized either directly with a DomainSpec or by loading from a JSON/YAML file.
    """

    def __init__(self, domain_spec: Optional['DomainSpec'] = None, file_path: Optional[str] = None):
        if domain_spec is None and file_path is None:
            raise ValueError("Either 'domain_spec' or 'file_path' must be provided.")
        if domain_spec is not None and file_path is not None:
            raise ValueError("Provide only one of 'domain_spec' or 'file_path', not both.")

        if file_path is not None:
            self.domain_spec = self._load_from_file(file_path)
        else:
            self.domain_spec = domain_spec

    def get_domain_spec(self) -> 'DomainSpec':
        """Return the domain specification."""
        return self.domain_spec

    def volume(self) -> float:
        """Compute the volume of the configuration space."""
        return self.domain_spec.volume()

    def __repr__(self) -> str:
        return f"ConfigurationSpace({self.domain_spec})"

    def get_parameters(self) -> List['Parameter']:
        """Return the list of parameters in the configuration space."""
        return self.domain_spec.parameters

    # ---------------------
    # File loading logic
    # ---------------------
    def _load_from_file(self, file_path: str) -> 'DomainSpec':
        """Load domain specification from a JSON or YAML file based on extension."""
        import os
        ext = os.path.splitext(file_path)[1].lower()

        match ext:
            case ".json":
                data = self._load_json(file_path)
            case ".yaml" | ".yml":
                data = self._load_yaml(file_path)
            case _:
                raise ValueError(f"Unsupported file extension: {ext}. Use .json or .yaml/.yml")

        return self._create_domain_spec(data)

    def _load_json(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            return json.load(f)

    def _load_yaml(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_domain_spec(self, data: dict) -> 'DomainSpec':
        """Convert raw data dict into a DomainSpec object."""
        from .domain import DomainSpec, Parameter, ParameterType  # adjust import path as needed

        parameters = []
        for p in data.get("parameters", []):
            param = Parameter(
                name=p["name"],
                bounds=tuple(p["bounds"]),
                unit=p.get("unit", ""),
                param_type=ParameterType(p.get("param_type", "continuous")),
                description=p.get("description", "")
            )
            parameters.append(param)

        return DomainSpec(parameters=parameters)
