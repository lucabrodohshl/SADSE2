"""
Drone 4D Example: Realistic Agricultural Drone Mission Planning

This example demonstrates a realistic 4D drone optimization problem with:
- 4 design parameters: speed, altitude, camera_res, spray_rate
- Environmental constraints (wind, temperature, visibility)
- Multi-drone task assignment using MILP
- Zonotope-based caching for fast replanning

KEY FEATURE: Weather conditions repeat in cycles to demonstrate cache hits!
This simulates a realistic scenario where similar conditions recur throughout
the day, allowing the cache to dramatically reduce computation time.
"""

import os

import numpy as np
from src.domain import DomainSpec, Parameter
from src.zonotope_cache import ZonotopeCache
from src.milp_solver import solve_task_assignment_milp, drone_energy_model
from src.utils import visualize_2d_cache, plot_cache_statistics, print_cache_summary
import matplotlib.pyplot as plt
import time


from examples.implementation_specifics.vis_utils import print_cache_performance
from examples.implementation_specifics.domain_specific import DroneODD

def create_field_tasks(num_rows: int = 20, row_length: float = 100.0):
    """Create list of field row tasks."""
    tasks = []
    for i in range(num_rows):
        tasks.append({
            'id': i,
            'length': row_length,
            'priority': 1.0
        })
    return tasks


def main():
    print("=" * 70)
    print("4D DRONE MISSION PLANNING WITH ZONOTOPE CACHING")
    print("=" * 70)
    print()
    
    output_dir = "results_drone_4d"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Define 4D drone domain
    print("1. Creating 4D drone domain")
    domain_spec = DomainSpec(parameters=[
        Parameter('speed', (5, 40), 'm/s', description='Flight speed'),
        Parameter('altitude', (10, 100), 'm', description='Flight altitude'),
        Parameter('camera_res', (4, 12), 'MP', description='Camera resolution'),
        Parameter('spray_rate', (2, 10), 'L/min', description='Spray application rate')
    ])
    print(f"   {domain_spec}")
    print()
    
    # 2. Create tasks
    print("2. Creating field tasks")
    tasks = create_field_tasks(num_rows=25, row_length=150.0)
    num_drones = 3
    print(f"   Tasks: {len(tasks)} field rows (150m each)")
    print(f"   Drones: {num_drones}")
    print()
    
    # 3. Create cache
    print("3. Initializing zonotope cache")
    cache = ZonotopeCache(
        domain_spec=domain_spec,
        enable_merging=True,
        merge_config_threshold=0.15,
        merge_objective_threshold=0.10,
        merge_frequency=3
    )
    print(f"   {cache}")
    print()
    
    # 4. Simulate time-varying conditions
    print("4. Simulating adaptive mission planning over time")
    print("   (Weather cycles through 4 scenarios to demonstrate cache hits)")
    print()
    
    duration_minutes = 60
    adaptation_interval = 5  # minutes
    num_adaptations = duration_minutes // adaptation_interval
    
    results = []
    
    # Define a few weather scenarios that will repeat
    # This demonstrates how caching helps when conditions recur
    weather_scenarios = [
        {'wind': 3.0, 'temperature': 22.0, 'visibility': 10.0, 'humidity': 55.0},  # Good conditions
        {'wind': 5.0, 'temperature': 18.0, 'visibility': 8.0, 'humidity': 65.0},   # Windy
        {'wind': 2.0, 'temperature': 25.0, 'visibility': 9.0, 'humidity': 60.0},   # Warm
        {'wind': 4.0, 'temperature': 20.0, 'visibility': 8.5, 'humidity': 70.0},   # Moderate
    ]
    
    for t in range(num_adaptations):
        timestamp = t * adaptation_interval * 60.0  # Convert to seconds
        
        # Cycle through weather scenarios - EXACT repeats for perfect cache demo!
        base_scenario = weather_scenarios[t % len(weather_scenarios)]
        
        odd = DroneODD(
            timestamp=timestamp,
            conditions=base_scenario.copy()  # Exact copy, no variation
        )
        
        print(f"   t={timestamp/60:.0f}min: {odd}")
        
        # Apply constraints to get DS
        ds = domain_spec.apply_odd_constraints(odd)
        
        if ds.is_empty():
            print(f"      WARNING: DS is empty (infeasible conditions)")
            continue
        
        print(f"      DS volume: {ds.volume():.2e}")
        
        # Query cache
        start_time = time.time()
        result = cache.query(odd, ds, method='exact_odd')
        query_time = (time.time() - start_time) * 1000
        
        if result:
            config, objective, metadata = result
            print(f"      ✓ CACHE HIT! ({query_time:.2f}ms)")
            print(f"         Config: speed={config.get('speed'):.1f} m/s, "
                  f"alt={config.get('altitude'):.1f} m, "
                  f"cam={config.get('camera_res'):.1f} MP, "
                  f"spray={config.get('spray_rate'):.1f} L/min")
            print(f"         Objective: {objective:.2f} Wh")
            
            results.append({
                'timestamp': timestamp,
                'cache_hit': True,
                'objective': objective,
                'query_time_ms': query_time
            })
        
        else:
            print(f"      × Cache miss ({query_time:.2f}ms) - optimizing...")
            
            # Check unexplored regions
            unexplored = cache.compute_unexplored(ds)
            print(f"         Unexplored regions: {len(unexplored)}")
            
            # Discretize and optimize
            configs = ds.discretize(num_levels=3)  # 3^4 = 81 configs
            print(f"         Evaluating {len(configs)} configurations...")
            
            # Define objective function
            def task_energy(config, task):
                return drone_energy_model(config, task['length'])
            
            start_opt = time.time()
            config, assignment, objective = solve_task_assignment_milp(
                configurations=configs,
                tasks=tasks,
                num_agents=num_drones,
                objective_function=task_energy
            )
            opt_time = (time.time() - start_opt) * 1000
            
            if config:
                print(f"         ✓ Optimization complete ({opt_time:.0f}ms)")
                print(f"         Config: speed={config.get('speed'):.1f} m/s, "
                      f"alt={config.get('altitude'):.1f} m, "
                      f"cam={config.get('camera_res'):.1f} MP, "
                      f"spray={config.get('spray_rate'):.1f} L/min")
                print(f"         Objective: {objective:.2f} Wh")
                print(f"         Assignment: drone0={len(assignment[0])} tasks, "
                      f"drone1={len(assignment[1])} tasks, "
                      f"drone2={len(assignment[2])} tasks")
                
                # Add to cache
                cache.add(odd, ds, config, objective, metadata={'assignment': assignment})
                print(f"         Added to cache (now {len(cache)} entries)")
                
                results.append({
                    'timestamp': timestamp,
                    'cache_hit': False,
                    'objective': objective,
                    'query_time_ms': query_time,
                    'opt_time_ms': opt_time
                })
            else:
                print(f"         × No feasible solution found")
        
        print()
    
    # 5. Analyze results
    print("5. Mission Planning Results")
    print()
    
    if not results:
        print("   ⚠️  No successful optimizations (all scenarios were infeasible)")
        print("   This may be due to:")
        print("   - CBC solver not available (install: brew install coin-or-tools/coinor/cbc)")
        print("   - Constraints too restrictive")
        print("   - Configuration discretization too coarse")
        print()
    else:
        cache_hits = sum(1 for r in results if r['cache_hit'])
        cache_misses = len(results) - cache_hits
        avg_objective = np.mean([r['objective'] for r in results])
        
        print(f"   Total adaptations: {len(results)}")
        print(f"   Cache hits: {cache_hits} ({100*cache_hits/len(results):.1f}%)")
        print(f"   Cache misses: {cache_misses}")
        print(f"   Average energy: {avg_objective:.2f} Wh")
        print()
    
    # 6. Cache statistics
    print("6. Final Cache Statistics")
    print()
    print_cache_summary(cache)
    print()
    
    # 7. Visualizations
    if results:
        print("7. Generating visualizations...")
        
        # Cache visualization (2D projection: speed × altitude)
        fig1 = visualize_2d_cache(cache, param1_idx=0, param2_idx=1,
                                 show_explored=True, show_entries=True)
        plt.savefig(os.path.join(output_dir, 'drone_4d_cache_speed_altitude.png'), dpi=150, bbox_inches='tight')
        print("   Saved: drone_4d_cache_speed_altitude.png")
        
        # Cache visualization (2D projection: camera × spray)
        fig2 = visualize_2d_cache(cache, param1_idx=2, param2_idx=3,
                                 show_explored=True, show_entries=True)
        plt.savefig(os.path.join(output_dir, 'drone_4d_cache_camera_spray.png'), dpi=150, bbox_inches='tight')
        print("   Saved: drone_4d_cache_camera_spray.png")
        
        # Performance statistics
        fig3 = plot_cache_statistics(cache)
        plt.savefig(os.path.join(output_dir, 'drone_4d_statistics.png'), dpi=150, bbox_inches='tight')
        print("   Saved: drone_4d_statistics.png")
        
        # Show plots
        plt.show()
        print()
    
    print("=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    
    if results:
        print()
        print("Key Results:")
        print(f"- {len(cache)} cache entries cover the explored design space")
        print(f"- {cache.cache_hit_rate():.1f}% cache hit rate reduces computation")
        print(f"- Average energy consumption: {avg_objective:.2f} Wh per mission")
        print(f"- Zonotope merging keeps cache compact (vs 100+ with hyperrectangles)")
    else:
        print()
        print("⚠️  Example did not produce results due to solver issues.")
        print("To fix: Install CBC solver with: brew install coin-or-tools/coinor/cbc")
    print()


if __name__ == "__main__":
    main()
