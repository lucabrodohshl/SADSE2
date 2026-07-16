

import numpy as np
from typing import List, Dict, Any

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def print_cache_performance(results: List[Dict[str, Any]], strategies: List, smart_cache_strategy, cs_volume: float, output_dir):
    """Print and save performance comparison between strategies.
    
    Pairs smart cache strategies with their corresponding baseline strategies based on 
    optimization method (DISCRETE, LINEAR, GA, LNS, etc.).
    
    Args:
        results: List of results from strategy executions
        strategies: List of strategy objects
        smart_cache_strategy: The smart cache strategy to use as reference (unused in new logic)
        cs_volume: Configuration space volume
        output_dir: Directory to save output files
    """
    if not results:
        print("   [ERROR] No successful optimizations (all scenarios were infeasible)")
        print()
        return

    # Group results by strategy
    strategy_results = {s.name: [r for r in results if r.get('approach') == s.name] for s in strategies}
    
    # Group strategies by their optimization method (DISCRETE, LINEAR, GA, etc.)
    smart_cache_strategies = {}
    baseline_strategies = {}
    
    for strategy in strategies:
        name = strategy.name
        if "SMART CACHE" in name:
            # Extract method from "SMART CACHE (METHOD)"
            method = name.replace("SMART CACHE (", "").replace(")", "")
            smart_cache_strategies[method] = strategy
        elif "BASELINE" in name:
            # Extract method from "BASELINE METHOD"
            method = name.replace("BASELINE ", "")
            baseline_strategies[method] = strategy
        else:
            # For other naming patterns, use the full name as method
            method = name
            if name not in smart_cache_strategies:
                baseline_strategies[method] = strategy
    
    # Pair smart cache strategies with their corresponding baselines
    for method in smart_cache_strategies.keys():
        if method not in baseline_strategies:
            print(f"   [WARNING] No baseline strategy found for method '{method}'. Skipping.")
            continue
            
        smart_strategy = smart_cache_strategies[method]
        baseline_strategy = baseline_strategies[method]
        
        if smart_strategy.name not in strategy_results:
            print(f"   [WARNING] Smart cache strategy '{smart_strategy.name}' not found in results. Skipping.")
            continue
            
        if baseline_strategy.name not in strategy_results:
            print(f"   [WARNING] Baseline strategy '{baseline_strategy.name}' not found in results. Skipping.")
            continue
        
        smart_results = strategy_results[smart_strategy.name]
        baseline_results = strategy_results[baseline_strategy.name]
        
        num_adaptations = len(smart_results)
        
        # Smart cache statistics
        cache_hits = sum(1 for r in smart_results if r['status'] == 'hit')
        extensions = sum(1 for r in smart_results if r['status'] == 'extended')
        new_entries = sum(1 for r in smart_results if r['status'] == 'new_entry')
        smart_avg_time = np.mean([r['time_ms'] for r in smart_results])
        smart_total_time = sum(r['time_ms'] for r in smart_results)
        
        # Baseline statistics
        baseline_avg_time = np.mean([r['time_ms'] for r in baseline_results])
        baseline_total_time = sum(r['time_ms'] for r in baseline_results)
        
        # Speedup calculation
        speedup = baseline_total_time / smart_total_time if smart_total_time > 0 else 1.0
        time_saved = baseline_total_time - smart_total_time
        time_saved_pct = (time_saved / baseline_total_time * 100) if baseline_total_time > 0 else 0
        
        # Coverage progression
        final_coverage = min(100.0, (smart_strategy.get_cache_volume() / cs_volume) * 100.0)
        
        # Save comparison to file
        with open(output_dir / f"{smart_strategy.name}_vs_{baseline_strategy.name}.txt", "w") as f:
            f.write("="*70 + "\n")
            f.write(f"COMPARISON: {smart_strategy.name} vs {baseline_strategy.name}\n")
            f.write("="*70 + "\n")
            f.write(f"Configuration Space Volume: {cs_volume:.2e}\n")
            f.write(f"Final Cached Coverage: {final_coverage:.1f}%\n")
            f.write(f"Total adaptations: {num_adaptations}\n")
            f.write("\n")

            f.write(f"SMART CACHE APPROACH:\n")
            f.write(f"  • Cache hits: {cache_hits} ({100*cache_hits/num_adaptations:.1f}%)\n")
            f.write(f"  • Region extensions: {extensions} ({100*extensions/num_adaptations:.1f}%)\n")
            f.write(f"  • New entries: {new_entries}\n")
            f.write(f"  • Cache entries: {len(smart_strategy.cache)}\n")
            f.write(f"  • Average latency: {smart_avg_time:.2f} ms\n")
            f.write(f"  • Total time: {smart_total_time:.2f} ms\n")
            f.write("\n")

            f.write(f"COMPARISON ({baseline_strategy.name}):\n")
            f.write(f"  • Total optimizations: {len(baseline_results)}\n")
            f.write(f"  • Average latency: {baseline_avg_time:.2f} ms\n")
            f.write(f"  • Total time: {baseline_total_time:.2f} ms\n")
            f.write("\n")

            f.write("PERFORMANCE COMPARISON:\n")
            f.write(f"   Speedup: {speedup:.2f}x faster\n")
            f.write(f"   Time saved: {time_saved:.2f} ms ({time_saved_pct:.1f}%)\n")
            f.write(f"   Cache efficiency: {100*(cache_hits + extensions)/num_adaptations:.1f}% queries optimized\n")
            f.write(f"   Coverage progression: 0% -> {final_coverage:.1f}%\n")
            f.write("="*70 + "\n")

            # Cache statistics
            f.write("Final Cache Statistics\n")
            f.write("\n")
            f.write(str(smart_strategy.cache) + "\n")
            f.write("\n")