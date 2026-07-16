"""
Advanced 6D Drone Mission Planning with LARGE Configuration Space

This evaluation uses a much larger CS with tighter ODD constraints to demonstrate:
- Progressive cache coverage growth (10% → 80%+)
- More realistic cache hit progression
- Clear learning behavior over time

Key differences from drone_6d_advanced.py:
- WIDER parameter ranges (larger CS)
- TIGHTER ODD constraints (smaller DS relative to CS)
- More weather scenarios to demonstrate gradual exploration
"""

from paths import ASSETS_DIR, CACHE_RESULTS_DIR

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple
import argparse
import random

from src.domain import DomainSpec, Parameter, ConfigurationSpace
from src.zonotope_ops import Zonotope
from src.milp_solver import solve_task_assignment_milp, drone_energy_model
from src.utils import print_cache_summary
from src.smart_cache import SmartZonotopeCache, SmartCacheEntry
from src.visualization import generate_visualizations, visualize_optimal_configs, plot_energy_comparison
from src.zonotope_cache import ZonotopeCache
from src.memory_utils import MemoryComparator, StrategyMemoryTracker

from evaluate.implementation_specifics.domain_specific import DroneODD, TightDroneODD, Drone, Fleet
from evaluate.implementation_specifics.strategies import  SmartBayesianStrategy, SmartCacheStrategy,DiscreteStrategy, SmartLinearStrategy, LinearStrategy, BayesianOptimizationStrategy, GAStrategy, SmartGAStrategy, LNSStrategy, SmartLNSStrategy
from evaluate.implementation_specifics.vis_utils import print_cache_performance, convert_numpy_types

from evaluate.implementation_specifics.weather import WeatherEnvironmentFixed






def save_results(results, events, optimal_configs,timestamp, s, obj, t_time, config,):
    results.append({
                'timestamp': float(timestamp),
                'status': str(s.get_info()),
                'objective': float(obj),
                'time_ms': float(t_time),
                'approach': str(s.name)
            })

    events.append({
                'timestamp': float(timestamp),
                't': float(timestamp),
                'strategy': str(s.name),
                'latency': float(t_time),
                'energy': float(obj),
                'cache_hit': s.has_hit(),
                'explored_volume': float(s.get_explored_volume()),
                'coverage_pct': s.get_coverage_pct()
            })
    if config is not None:
        optimal_configs[s.name].append({
                    't': float(timestamp),
                    'config': {
                        'speed': float(config.get('speed')),
                        'altitude': float(config.get('altitude')),
                        'camera_res': float(config.get('camera_res')),
                        'spray_rate': float(config.get('spray_rate')),
                        'power_limit_factor': float(config.get('power_limit_factor')),
                        'sensor_sampling': float(config.get('sensor_sampling'))
                    },
                    'energy': float(obj)
                })
    return results, events, optimal_configs




def create_visualizations(strategies, events, optimal_configs, drone_domain, results, output_dir, memory_comparator=None):
    if events and optimal_configs[strategies[0].name]:
        
        # Prepare results dictionary for visualization
        viz_results = {
            'events': events,
            'optimal_configs': optimal_configs,
            'cs_volume': drone_domain.volume(),
            'num_evaluations': len(results)/len(strategies),
            'total_time': sum(r['time_ms'] for r in results) / 1000.0
        }
        
        # Add memory data if available
        if memory_comparator:
            viz_results['memory_analysis'] = memory_comparator.get_all_data()
        
        # Convert numpy types
        viz_results = convert_numpy_types(viz_results)
        
        # Save results to JSON
        json_path = output_dir / 'drone_6d_large_cs_results.json'
        with open(json_path, 'w') as f:
            json.dump(viz_results, f, indent=2)
        print(f"   ✓ Saved results to: {json_path}")
        
        # Generate all visualizations
        generate_visualizations(viz_results, strategies,output_dir)
        
        
        
        print()
        print(f"    All visualizations saved to: {output_dir}")
        print()



def run_evaluation(strategies: List, 
                   drone_domain: Fleet, 
                   env: WeatherEnvironmentFixed, 
                   duration_minutes: int, 
                   adaptation_interval: int, 
                   output_dir: Path,
                   track_memory: bool = False):
    
    print("=" * 70)
    print("6D DRONE EVALUATION: LARGE CONFIGURATION SPACE")
    print("=" * 70)
    print()

    results = []
    events = []  # Track events for visualization
    optimal_configs = {}  #tracks optimal configs according to strategy names
    for s in strategies:
        optimal_configs[s.name] = []
    
    # Initialize memory tracking
    # Initialize memory tracking components only if requested
    memory_comparator = None
    strategy_memory_trackers = {}
    
    if track_memory:
        memory_comparator = MemoryComparator()
        
        # Initialize memory trackers for each strategy
        for s in strategies:
            # Create individual memory tracker for baseline strategies
            if not hasattr(s, 'cache') or s.cache is None:
                strategy_memory_trackers[s.name] = StrategyMemoryTracker(s.name)
                strategy_memory_trackers[s.name].snapshot(f"init_{s.name}", 0)
        
        print("Memory tracking initialized for all strategies")
    else:
        print("Memory tracking disabled")

    num_adaptations = duration_minutes // adaptation_interval
    for t in range(num_adaptations):
        timestamp = t * adaptation_interval * 60.0  # Convert to seconds
        
        odd = env.step(timestamp)
        
        print(f"   t={timestamp/60:.0f}min: ODD(wind={odd.wind_ms:.1f}, temp={odd.temperature_c:.1f}, vis={odd.visibility_km:.1f}, hum={odd.humidity_pct:.1f})")
        
        
        ds = drone_domain.get_domain_spec().apply_odd_constraints(odd)
        if ds.is_empty():
            print(f"      WARNING: DS is empty (infeasible conditions)")
            continue
        
        ds_coverage_pct = (ds.volume() / drone_domain.volume()) * 100
        #print(f"      DS volume: {ds.volume():.2e} ({ds_coverage_pct:.1f}% of CS)")


        previous_result = {}
        for  s in strategies:
            if s.name in previous_result.keys():
                config, obj, t_time = previous_result[s.name]
                print(f"      {s.name}: Reusing previous result - Energy={obj:.2f}J, Time={t_time:.1f}ms")
                results, events, optimal_configs = save_results(results, events, optimal_configs, timestamp, s, obj, t_time, config)
                continue
            
            # Take memory snapshot before execution
            if track_memory and s.name in strategy_memory_trackers:
                strategy_memory_trackers[s.name].snapshot(f"before_exec_{t}", 0)
            elif hasattr(s, 'cache') and s.cache is not None:
                s.cache.sample_memory(f"before_exec_{t}")
            
            config, obj, _, t_time = s.execute(odd, ds)
            previous_result[s.name] = (config, obj, t_time)
            
            # Take memory snapshot after execution
            if track_memory and s.name in strategy_memory_trackers:
                strategy_memory_trackers[s.name].snapshot(f"after_exec_{t}", 0)
            elif hasattr(s, 'cache') and s.cache is not None:
                s.cache.sample_memory(f"after_exec_{t}")

            results, events, optimal_configs,  = save_results(results, events, optimal_configs, timestamp, s, obj, t_time, config)



    print_cache_performance(results, strategies, strategies[0], drone_domain.volume(), output_dir)
    create_visualizations(strategies, events, optimal_configs, drone_domain, results, output_dir, memory_comparator)
    
    # Memory Analysis and Reporting (only if tracking enabled)
    if track_memory:
        print("=" * 70)
        print("MEMORY USAGE ANALYSIS")
        print("=" * 70)
        
        # Collect memory data from all strategies
        for s in strategies:
            if hasattr(s, 'cache') and s.cache is not None:
                # Smart cache strategy - get memory data from cache
                memory_data = s.cache.get_memory_usage()
                memory_comparator.add_strategy(s.name, memory_data)
                print(f"\n{s.name} Memory Summary:")
                print(s.cache.get_memory_summary())
            elif s.name in strategy_memory_trackers:
                # Baseline strategy - get memory data from dedicated tracker
                tracker = strategy_memory_trackers[s.name]
                final_snapshot = tracker.snapshot(f"final_{s.name}", 0)
                
                memory_data = {
                    'strategy': s.name,
                    'current_rss_kb': final_snapshot.rss_kb if tracker.has_psutil else 0,
                    'current_vms_kb': final_snapshot.vms_kb if tracker.has_psutil else 0,
                    'current_python_kb': final_snapshot.python_kb,
                    'cache_data_kb': 0.0,  # No cache for baseline
                    'entries_count': 0,
                    'avg_entry_size_kb': 0.0,
                    'peak_memory': tracker.get_peak_memory(),
                    'memory_growth': tracker.get_memory_growth()
                }
                memory_comparator.add_strategy(s.name, memory_data)
                
                print(f"\n{s.name} Memory Summary:")
                print(f"Current Python: {final_snapshot.python_kb:.1f} KB")
                if tracker.has_psutil:
                    print(f"Current RSS: {final_snapshot.rss_kb:.1f} KB")
                    print(f"Current VMS: {final_snapshot.vms_kb:.1f} KB")
                peak = tracker.get_peak_memory()
                if peak:
                    print(f"Peak Python: {peak.python_kb:.1f} KB")
                    if tracker.has_psutil:
                        print(f"Peak RSS: {peak.rss_kb:.1f} KB")
                print(f"Memory growth: {tracker.get_memory_growth():+.1f} KB")
                print(f"Snapshots taken: {len(tracker.snapshots)}")
        
        # Generate memory comparison report
        print("\n" + "=" * 70)
        print("MEMORY COMPARISON BETWEEN STRATEGIES")
        print("=" * 70)
        print(memory_comparator.get_comparison_summary())
        
        # Export memory data
        memory_json_path = output_dir / 'memory_analysis.json'
        memory_comparator.export_to_json(str(memory_json_path))
        print(f"\n✓ Memory analysis data exported to: {memory_json_path}")
        
        # Generate memory efficiency report
        memory_report_path = output_dir / 'memory_efficiency_report.txt'
        with open(memory_report_path, 'w') as f:
            f.write("MEMORY EFFICIENCY ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(memory_comparator.get_comparison_summary())
            f.write("\n\nDETAILED STRATEGY MEMORY USAGE:\n")
            f.write("-" * 40 + "\n")
            
            for s in strategies:
                f.write(f"\n{s.name}:\n")
                if hasattr(s, 'cache') and s.cache is not None:
                    f.write(s.cache.get_memory_summary())
                elif s.name in strategy_memory_trackers:
                    tracker = strategy_memory_trackers[s.name]
                    final_snapshot = tracker.snapshots[-1] if tracker.snapshots else None
                    if final_snapshot:
                        f.write(f"Final Python Memory: {final_snapshot.python_kb:.1f} KB\n")
                        if tracker.has_psutil:
                            f.write(f"Final RSS: {final_snapshot.rss_kb:.1f} KB\n")
                        f.write(f"Memory Growth: {tracker.get_memory_growth():+.1f} KB\n")
                        f.write(f"Snapshots: {len(tracker.snapshots)}\n")
        
        print(f"✓ Detailed memory report saved to: {memory_report_path}")
        
    print("="*70)
    print("LARGE CS EVALUATION COMPLETE")
    print("="*70)



def main():

    #get arguments from command line
    parser = argparse.ArgumentParser(description="6D Drone Evaluation")
    parser.add_argument("--duration", type=int, default=600, help="Duration of the evaluation in minutes")
    parser.add_argument("--adaptation_interval", type=int, default=5, help="The adaptation happens every N minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--configuration_file", type=str, default=str(ASSETS_DIR / "fleet.json"), help="Path to the cs configuration file")
    parser.add_argument("--is_drone", action='store_true', help="Indicates if the configuration file is for a drone")
    parser.add_argument("--output_dir", type=str, default="latest_run", help="Sub-directory under results/cache/ to save results and visualizations")
    parser.add_argument("--weather_scenarios", type=list, default=[330, 180, 90], help="Number of weather scenarios to generate")
    parser.add_argument("--temperatures", type=list, default=[5, 25, 15], help="Number of weather scenarios to generate")
    parser.add_argument("--track_memory", action='store_true', help="Enable memory usage tracking and analysis")
    args = parser.parse_args()


    random.seed(args.seed)

    if not args.is_drone:
        drone_domain = Fleet(file_path=args.configuration_file)
    else:
        drone_domain = Drone(file_path=args.configuration_file)
    weather_scenarios = args.weather_scenarios  # Different seeds for weather scenarios


    """  
        
        #ATTENTION
        # For a cache hit vs baseline, we assume the first (odd) strategies is the cache one, while even is WITHOUT baseline
        #Therefore, to create the comparisons, we need to have an even number of strategies. 
        # If you want to avoid rerunning the same strategy twice, just name them with same names, for example two strategies "SMART CACHE (DISCRETE)" and "SMART CACHE (DISCRETE)" will not be run twice, but will be compared with the next one in the list. 

        Example, a list of strategies to run, represented with names A, B, C, D, E, F
        List = [A, B, C, D, A, F]
        The pairs (A, B), (C, D), (A, F) will be compared. However, A will NOT be run twice, nor its results will be included twice in the total graphs, like cumulative latency, energy consumption, etc.

        
        """
    strategies = [
                    SmartCacheStrategy(name = "SMART CACHE (DISCRETE)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")
                                ),
                    DiscreteStrategy("BASELINE DISCRETE",cs=drone_domain),
                    SmartLinearStrategy("SMART CACHE (LINEAR)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")
                                ),
                    LinearStrategy("BASELINE LINEAR",cs=drone_domain),
                    #SmartBayesianStrategy(name = "SMART CACHE (BAYESIAN)",
                    #                 cs=drone_domain,
                    #                 file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")),
                    #BayesianOptimizationStrategy("BAYESIAN OPTIMIZATION",cs=drone_domain),
                    SmartGAStrategy(name = "SMART CACHE (GA)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")),
                    GAStrategy("GENETIC ALGORITHM",cs=drone_domain),
                    SmartLNSStrategy(name = "SMART CACHE (LNS)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")),
                    LNSStrategy("LARGE NEIGHBORHOOD SEARCH",cs=drone_domain)
                ]

    output_dir = CACHE_RESULTS_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for w, t in zip(weather_scenarios, args.temperatures):
        env = WeatherEnvironmentFixed("Field Weather", data_function='realistic')
        env.set_seed(args.seed + t)
        env.duration_hours = args.duration * 12
        env.dt_minutes = 60
        env.lat_deg = 52.5        # Berlin-ish; affects diurnal amplitude a
        env.day_of_year = w  # midsummer default
        env.base_temp_c = t  # seasonal mean near-surface temperature
        env.data = env.generate_realistic_data()
       
        
        output_dir = CACHE_RESULTS_DIR / args.output_dir / f"weather_scenario_{w}_temp_{t}_realistic"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_evaluation(strategies, drone_domain, env,  args.duration, args.adaptation_interval, output_dir, args.track_memory)
        


if __name__ == "__main__":
    main()
