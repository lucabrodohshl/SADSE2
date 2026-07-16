"""Test SmartZonotopeCache with TightDroneODD (recreated)

This script creates a small DomainSpec matching the TightDroneODD
parameters, builds a SmartZonotopeCache, and performs a sequence of
queries to exercise new entry creation, extension, and cache hits.
"""

from src.domain import DomainSpec, Parameter, ParameterType, ConstrainedDomain
from src.smart_cache import SmartZonotopeCache
from evaluate.implementation_specifics.domain_specific import TightDroneODD
import numpy as np


def make_drone_domain_spec() -> DomainSpec:
    params = [
        Parameter('speed', (0.0, 50.0), unit='m/s', param_type=ParameterType.CONTINUOUS),
        Parameter('altitude', (0.0, 150.0), unit='m', param_type=ParameterType.CONTINUOUS),
        Parameter('camera_res', (2.0, 24.0), unit='MP', param_type=ParameterType.CONTINUOUS),
        Parameter('spray_rate', (1.0, 15.0), unit='l/s', param_type=ParameterType.CONTINUOUS),
        Parameter('power_limit_factor', (0.4, 1.0), unit='fraction', param_type=ParameterType.CONTINUOUS),
        Parameter('sensor_sampling', (1.0, 12.0), unit='Hz', param_type=ParameterType.CONTINUOUS),
    ]
    return DomainSpec(parameters=params)


def optimization_function_factory(domain_spec: DomainSpec):
    def optimize(zonotope):
        bounds = zonotope.to_box_bounds()
        mid = np.mean(bounds, axis=1)
        config = domain_spec.create_configuration(mid)
        objective = float(np.sum(mid))
        metadata = {'midpoint': mid.tolist()}
        return config, objective, metadata

    return optimize


def main():
    domain_spec = make_drone_domain_spec()

    cache = SmartZonotopeCache(domain_spec=domain_spec, extension_threshold=1.0, merge_frequency=10)

    odd = TightDroneODD(timestamp=0.0, conditions={
        'wind': 3.0,
        'temperature': 20.0,
        'visibility': 10.0,
        'humidity': 50.0,
    })

    optimizer = optimization_function_factory(domain_spec)

    ds1 = domain_spec.apply_odd_constraints(odd)
    print("=== Query 1: initial constrained DS (expect new_entry) ===")
    result1 = cache.query_and_optimize(odd, ds1, optimizer)
    print("Result 1:")
    print(result1)
    print()

    full_bounds = domain_spec.get_bounds_array()
    ds2 = ConstrainedDomain(domain_spec=domain_spec, bounds=full_bounds.copy(), odd=odd)
    print("=== Query 2: larger DS (expect extension if objective similar) ===")
    result2 = cache.query_and_optimize(odd, ds2, optimizer)
    print("Result 2:")
    print(result2)
    print()

    print("=== Query 3: original DS again (expect hit) ===")
    result3 = cache.query_and_optimize(odd, ds1, optimizer)
    print("Result 3:")
    print(result3)
    print()

    print("\nCache summary:")
    print(cache)


if __name__ == '__main__':
    main()
