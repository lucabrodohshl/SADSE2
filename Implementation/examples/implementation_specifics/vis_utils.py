


import numpy as np

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


def print_cache_performance(results, strategies, smart_cache_strategy, cs_volume, output_dir):
    if not results:
        print("   [ERROR] No successful optimizations (all scenarios were infeasible)")
        print()
    else:
      
        assert len(strategies) % 2 == 0, "This analysis assumes exactly two strategies: cache and baseline."
        strategies_pairs = [(strategies[i], strategies[i+1]) for i in range(0, len(strategies), 2)]

        for s1, s2 in strategies_pairs:

            # Separate results by approach
            our_results = [r for r in results if r.get('approach') == s1.name]
            baseline_results = [r for r in results if r.get('approach') == s2.name]

            num_adaptations = len(our_results)
        
            # Our approach statistics
            cache_hits = sum(1 for r in our_results if r['status'] == 'hit')
            extensions = sum(1 for r in our_results if r['status'] == 'extended')
            new_entries = sum(1 for r in our_results if r['status'] == 'new_entry')
            our_avg_time = np.mean([r['time_ms'] for r in our_results])
            our_total_time = sum(r['time_ms'] for r in our_results)
        
            # Baseline statistics
            baseline_avg_time = np.mean([r['time_ms'] for r in baseline_results])
            baseline_total_time = sum(r['time_ms'] for r in baseline_results)
        
            # Speedup calculation
            speedup = baseline_total_time / our_total_time if our_total_time > 0 else 1.0
            time_saved = baseline_total_time - our_total_time
            time_saved_pct = (time_saved / baseline_total_time * 100) if baseline_total_time > 0 else 0
            
            # Coverage progression
            final_coverage = min(100.0, (smart_cache_strategy.get_cache_volume() / cs_volume) * 100.0)
            # save it in a txt file
            with open(output_dir / f"{s1.name}_vs_{s2.name}.txt", "a") as f:
                f.write("="*70 + "\n")
                f.write(f"COMPARISON: {s1.name} vs {s2.name}\n")
                f.write("="*70 + "\n")
                f.write(f"Configuration Space Volume: {cs_volume:.2e}\n")
                f.write(f"Final Cached Coverage: {final_coverage:.1f}%\n")
                f.write(f"Total adaptations: {num_adaptations}\n")
                f.write("\n")

                f.write("OUR APPROACH (Smart Caching):\n")
                f.write(f"  • Cache hits: {cache_hits} ({100*cache_hits/num_adaptations:.1f}%)\n")
                f.write(f"  • Region extensions: {extensions} ({100*extensions/num_adaptations:.1f}%)\n")
                f.write(f"  • New entries: {new_entries}\n")
                f.write(f"  • Cache entries: {len(smart_cache_strategy.cache)}\n")
                f.write(f"  • Average latency: {our_avg_time:.2f} ms\n")
                f.write(f"  • Total time: {our_total_time:.2f} ms\n")
                f.write("\n")

                f.write("BASELINE (Full Reoptimization):\n")
                f.write(f"  • Always full optimization: {len(baseline_results)} times\n")
                f.write(f"  • Average latency: {baseline_avg_time:.2f} ms\n")
                f.write(f"  • Total time: {baseline_total_time:.2f} ms\n")
                f.write("\n")
                
                f.write("PERFORMANCE GAIN:\n")
                f.write(f"   Speedup: {speedup:.2f}x faster\n")
                f.write(f"    Time saved: {time_saved:.2f} ms ({time_saved_pct:.1f}%)\n")
                f.write(f"   Efficiency: {100*(cache_hits + extensions)/num_adaptations:.1f}% of queries avoided full optimization\n")
                f.write(f"   Coverage progression: 0% → {final_coverage:.1f}%\n")
                f.write("="*70)
                f.write("\n")
    
                # 6. Cache statistics
                f.write("6. Final Cache Statistics\n")
                f.write("\n")
                f.write(str(smart_cache_strategy.cache) + "\n")
                f.write("\n")