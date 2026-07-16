"""
Basic 2D Example: Introduction to Zonotope Caching

This example demonstrates the fundamental concepts using a simple 2D domain.
Perfect for understanding how the cache works before moving to higher dimensions.
"""

import numpy as np
from src.domain import DomainSpec, Parameter, ODD
from src.zonotope_cache import ZonotopeCache
from src.milp_solver import solve_simple_optimization
from src.utils import visualize_2d_cache, print_cache_summary
import matplotlib.pyplot as plt


def simple_objective(config):
    """
    Simple objective function: minimize x² + y².
    
    This has a clear optimal point at the origin.
    """
    values = config.as_array()
    return np.sum(values ** 2)


def main():
    print("=" * 70)
    print("BASIC 2D ZONOTOPE CACHE EXAMPLE")
    print("=" * 70)
    print()
    
    # 1. Define 2D domain
    print("1. Creating 2D domain: speed × altitude")
    domain_spec = DomainSpec(parameters=[
        Parameter('speed', (0, 50), 'm/s'),
        Parameter('altitude', (10, 100), 'm')
    ])
    print(f"   {domain_spec}")
    print()
    
    # 2. Create cache
    print("2. Initializing zonotope cache")
    cache = ZonotopeCache(
        domain_spec=domain_spec,
        enable_merging=True,
        merge_config_threshold=0.2,
        merge_objective_threshold=0.15
    )
    print(f"   {cache}")
    print()
    
    # 3. Run optimization for different ODDs
    print("3. Running optimizations for different environmental conditions")
    print()
    
    num_scenarios = 5
    for i in range(num_scenarios):
        # Create ODD with varying conditions
        odd = ODD(
            timestamp=i * 60.0,
            conditions={
                'wind': 2.0 + i * 1.0,
                'temperature': 20.0 + i * 2.0
            }
        )
        
        # Apply constraints
        ds = domain_spec.apply_odd_constraints(odd)
        
        print(f"   Scenario {i+1}: {odd}")
        print(f"      DS bounds: speed=[{ds.bounds[0,0]:.1f}, {ds.bounds[0,1]:.1f}], "
              f"altitude=[{ds.bounds[1,0]:.1f}, {ds.bounds[1,1]:.1f}]")
        
        # Check cache first
        result = cache.query(odd, ds, method='exact_odd')
        
        if result:
            config, objective, metadata = result
            print(f"      ✓ CACHE HIT! Objective: {objective:.2f}")
        else:
            print(f"      × Cache miss - optimizing...")
            
            # Check for unexplored regions
            unexplored = cache.compute_unexplored(ds)
            print(f"      Unexplored regions: {len(unexplored)}")
            
            # Optimize
            config, objective = solve_simple_optimization(
                domain_spec, ds, simple_objective, num_levels=5
            )
            
            if config:
                print(f"      Found optimal: {config}")
                print(f"      Objective: {objective:.2f}")
                
                # Add to cache
                cache.add(odd, ds, config, objective)
                print(f"      Added to cache (now {len(cache)} entries)")
            else:
                print(f"      No feasible solution found")
        
        print()
    
    # 4. Show cache statistics
    print("4. Cache Performance Summary")
    print()
    print_cache_summary(cache)
    print()
    
    # 5. Visualize cache
    print("5. Generating visualizations...")
    
    fig1 = visualize_2d_cache(cache, show_explored=True, show_entries=True)
    plt.savefig('basic_2d_cache_vis.png', dpi=150, bbox_inches='tight')
    print("   Saved: basic_2d_cache_vis.png")
    
    # Show plots
    plt.show()
    
    print()
    print("=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    print()
    print("Key observations:")
    print("- Zonotope cache dramatically reduces redundant optimizations")
    print("- Intelligent merging keeps cache compact")
    print("- Fast queries enable real-time adaptive planning")
    print()


if __name__ == "__main__":
    main()
