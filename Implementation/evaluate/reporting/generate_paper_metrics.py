"""
Generate paper-ready metrics from cache effectiveness experiments.

This script analyzes results from gamma V1 (old) and V2 (new) experiments
and produces comprehensive statistics for publication.

Metrics computed:
1. Cache Hit Rate (%)
2. Cache Hit vs Miss Latency (ms)
3. Total Latency: P50, P95, P99 (ms)
4. Speedup Factor
5. Cache Memory Footprint (MB) - when available
6. Cache Size (entries)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

from paths import CACHE_RESULTS_DIR, REPORTS_DIR


def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all scenario results from a directory.
    
    Returns:
        Dict mapping scenario names to DataFrames
    """
    scenarios = {}
    
    for scenario_dir in results_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
            
        json_file = scenario_dir / "drone_6d_large_cs_results.json"
        if not json_file.exists():
            warnings.warn(f"No results found in {scenario_dir.name}")
            continue
        
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data['events'])
        scenarios[scenario_dir.name] = df
    
    return scenarios


def compute_cache_metrics(df: pd.DataFrame, strategy: str) -> Dict[str, float]:
    """
    Compute all cache effectiveness metrics for a single strategy.
    
    Args:
        df: DataFrame with all events
        strategy: Strategy name to analyze
        
    Returns:
        Dict with all metrics
    """
    # Filter to this strategy
    strategy_df = df[df['strategy'] == strategy].copy()
    
    if len(strategy_df) == 0:
        return None
    
    metrics = {}
    
    # === 1. Cache Hit Rate (%) ===
    if 'cache_hit' in strategy_df.columns:
        total_requests = len(strategy_df)
        hits = strategy_df['cache_hit'].sum()
        misses = total_requests - hits
        
        metrics['total_requests'] = total_requests
        metrics['cache_hits'] = int(hits)
        metrics['cache_misses'] = int(misses)
        metrics['hit_rate_pct'] = (hits / total_requests * 100) if total_requests > 0 else 0.0
    else:
        metrics['total_requests'] = len(strategy_df)
        metrics['cache_hits'] = 0
        metrics['cache_misses'] = 0
        metrics['hit_rate_pct'] = 0.0
    
    # === 2. Cache Hit vs Miss Latency (ms) ===
    if 'cache_hit' in strategy_df.columns and 'latency' in strategy_df.columns:
        hit_latencies = strategy_df[strategy_df['cache_hit'] == True]['latency']
        miss_latencies = strategy_df[strategy_df['cache_hit'] == False]['latency']
        
        metrics['hit_latency_mean_ms'] = hit_latencies.mean() if len(hit_latencies) > 0 else 0.0
        metrics['hit_latency_median_ms'] = hit_latencies.median() if len(hit_latencies) > 0 else 0.0
        metrics['hit_latency_std_ms'] = hit_latencies.std() if len(hit_latencies) > 0 else 0.0
        
        metrics['miss_latency_mean_ms'] = miss_latencies.mean() if len(miss_latencies) > 0 else 0.0
        metrics['miss_latency_median_ms'] = miss_latencies.median() if len(miss_latencies) > 0 else 0.0
        metrics['miss_latency_std_ms'] = miss_latencies.std() if len(miss_latencies) > 0 else 0.0
        
        # CvM ratio
        if metrics['miss_latency_mean_ms'] > 0:
            metrics['cvm_ratio'] = metrics['hit_latency_mean_ms'] / metrics['miss_latency_mean_ms']
        else:
            metrics['cvm_ratio'] = 0.0
    
    # === 3. Total Latency: P50, P95, P99 (ms) ===
    if 'latency' in strategy_df.columns:
        latencies = strategy_df['latency']
        
        metrics['latency_p50_ms'] = latencies.quantile(0.50)
        metrics['latency_p95_ms'] = latencies.quantile(0.95)
        metrics['latency_p99_ms'] = latencies.quantile(0.99)
        metrics['latency_mean_ms'] = latencies.mean()
        metrics['latency_std_ms'] = latencies.std()
        metrics['latency_min_ms'] = latencies.min()
        metrics['latency_max_ms'] = latencies.max()
    
    # === 4. Energy metrics ===
    if 'energy' in strategy_df.columns:
        energies = strategy_df['energy']
        
        metrics['energy_mean'] = energies.mean()
        metrics['energy_median'] = energies.median()
        metrics['energy_std'] = energies.std()
        metrics['energy_total'] = energies.sum()
    
    # === 5. Cache Memory Footprint (MB) ===
    if 'memory_mb' in strategy_df.columns:
        metrics['memory_peak_mb'] = strategy_df['memory_mb'].max()
        metrics['memory_mean_mb'] = strategy_df['memory_mb'].mean()
        metrics['memory_final_mb'] = strategy_df['memory_mb'].iloc[-1]
    else:
        metrics['memory_peak_mb'] = None
        metrics['memory_mean_mb'] = None
        metrics['memory_final_mb'] = None
    
    # === 6. Cache Size (entries) ===
    if 'cache_size' in strategy_df.columns:
        metrics['cache_size_peak'] = int(strategy_df['cache_size'].max())
        metrics['cache_size_mean'] = strategy_df['cache_size'].mean()
        metrics['cache_size_final'] = int(strategy_df['cache_size'].iloc[-1])
    else:
        metrics['cache_size_peak'] = None
        metrics['cache_size_mean'] = None
        metrics['cache_size_final'] = None
    
    # === 7. Coverage metrics ===
    if 'coverage_pct' in strategy_df.columns:
        metrics['coverage_mean_pct'] = strategy_df['coverage_pct'].mean()
        metrics['coverage_final_pct'] = strategy_df['coverage_pct'].iloc[-1]
    
    return metrics


def compute_speedup_factor(cache_metrics: Dict, baseline_metrics: Dict) -> float:
    """
    Compute speedup factor: baseline_latency / cache_latency
    """
    if cache_metrics is None or baseline_metrics is None:
        return None
    
    cache_latency = cache_metrics.get('latency_mean_ms', 0)
    baseline_latency = baseline_metrics.get('latency_mean_ms', 0)
    
    if cache_latency > 0:
        return baseline_latency / cache_latency
    return None


def generate_summary_table(results_dir: Path, gamma_version: str) -> pd.DataFrame:
    """
    Generate a summary table for all scenarios in a results directory.
    
    Args:
        results_dir: Path to results directory
        gamma_version: "V1" or "V2"
        
    Returns:
        DataFrame with summary statistics
    """
    scenarios = load_results(results_dir)
    
    summary_rows = []
    
    for scenario_name, df in scenarios.items():
        # Extract scenario info from name
        # Format: weather_scenario_180_temp_25_realistic
        parts = scenario_name.split('_')
        duration = parts[2] if len(parts) > 2 else "unknown"
        temp = parts[4] if len(parts) > 4 else "unknown"
        
        # Get unique strategies
        strategies = df['strategy'].unique()
        
        # Separate cache strategies from baselines
        cache_strategies = [s for s in strategies if 'SMART CACHE' in s]
        baseline_strategies = [s for s in strategies if 'BASELINE' in s]
        
        for cache_strategy in cache_strategies:
            # Get corresponding baseline
            if 'DISCRETE' in cache_strategy:
                baseline = 'BASELINE DISCRETE'
            elif 'LINEAR' in cache_strategy:
                baseline = 'BASELINE LINEAR'
            elif 'GA' in cache_strategy:
                baseline = 'GENETIC ALGORITHM'
            elif 'LNS' in cache_strategy:
                baseline = 'LARGE NEIGHBORHOOD SEARCH'
            else:
                baseline = None
            
            # Compute metrics
            cache_metrics = compute_cache_metrics(df, cache_strategy)
            baseline_metrics = compute_cache_metrics(df, baseline) if baseline else None
            
            if cache_metrics is None:
                continue
            
            # Compute speedup
            speedup = compute_speedup_factor(cache_metrics, baseline_metrics)
            
            row = {
                'gamma_version': gamma_version,
                'scenario': scenario_name,
                'duration_s': duration,
                'temperature_c': temp,
                'strategy': cache_strategy,
                'baseline_strategy': baseline,
                
                # Cache effectiveness
                'total_requests': cache_metrics['total_requests'],
                'cache_hits': cache_metrics['cache_hits'],
                'cache_misses': cache_metrics['cache_misses'],
                'hit_rate_pct': cache_metrics['hit_rate_pct'],
                
                # Latency - Cache
                'hit_latency_mean_ms': cache_metrics.get('hit_latency_mean_ms'),
                'hit_latency_median_ms': cache_metrics.get('hit_latency_median_ms'),
                'miss_latency_mean_ms': cache_metrics.get('miss_latency_mean_ms'),
                'miss_latency_median_ms': cache_metrics.get('miss_latency_median_ms'),
                'cvm_ratio': cache_metrics.get('cvm_ratio'),
                
                # Total latency percentiles
                'latency_p50_ms': cache_metrics.get('latency_p50_ms'),
                'latency_p95_ms': cache_metrics.get('latency_p95_ms'),
                'latency_p99_ms': cache_metrics.get('latency_p99_ms'),
                'latency_mean_ms': cache_metrics.get('latency_mean_ms'),
                'latency_std_ms': cache_metrics.get('latency_std_ms'),
                
                # Baseline latency (for comparison)
                'baseline_latency_mean_ms': baseline_metrics.get('latency_mean_ms') if baseline_metrics else None,
                'baseline_latency_p50_ms': baseline_metrics.get('latency_p50_ms') if baseline_metrics else None,
                
                # Speedup factor
                'speedup_factor': speedup,
                
                # Memory footprint
                'memory_peak_mb': cache_metrics.get('memory_peak_mb'),
                'memory_mean_mb': cache_metrics.get('memory_mean_mb'),
                'memory_final_mb': cache_metrics.get('memory_final_mb'),
                
                # Cache size
                'cache_size_peak': cache_metrics.get('cache_size_peak'),
                'cache_size_mean': cache_metrics.get('cache_size_mean'),
                'cache_size_final': cache_metrics.get('cache_size_final'),
                
                # Energy
                'energy_mean': cache_metrics.get('energy_mean'),
                'energy_total': cache_metrics.get('energy_total'),
                
                # Coverage
                'coverage_mean_pct': cache_metrics.get('coverage_mean_pct'),
                'coverage_final_pct': cache_metrics.get('coverage_final_pct'),
            }
            
            summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def generate_latex_table(df: pd.DataFrame, output_file: Path, caption: str = "Cache Effectiveness Metrics"):
    """
    Generate a LaTeX table from the summary DataFrame.
    """
    # Select key columns for paper
    columns_for_paper = [
        'scenario',
        'strategy',
        'hit_rate_pct',
        'hit_latency_median_ms',
        'miss_latency_median_ms',
        'latency_p50_ms',
        'latency_p95_ms',
        'latency_p99_ms',
        'speedup_factor',
        'memory_peak_mb',
        'cache_size_final',
    ]
    
    # Filter columns that exist
    available_cols = [col for col in columns_for_paper if col in df.columns]
    paper_df = df[available_cols].copy()
    
    # Format numeric columns
    if 'hit_rate_pct' in paper_df.columns:
        paper_df['hit_rate_pct'] = paper_df['hit_rate_pct'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    for col in ['hit_latency_median_ms', 'miss_latency_median_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']:
        if col in paper_df.columns:
            paper_df[col] = paper_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    if 'speedup_factor' in paper_df.columns:
        paper_df['speedup_factor'] = paper_df['speedup_factor'].apply(lambda x: f"{x:.2f}×" if pd.notna(x) else "N/A")
    
    if 'memory_peak_mb' in paper_df.columns:
        paper_df['memory_peak_mb'] = paper_df['memory_peak_mb'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Rename columns for paper
    paper_df.columns = [
        col.replace('_', ' ').title().replace('Pct', '(%)').replace('Ms', '(ms)').replace('Mb', '(MB)')
        for col in paper_df.columns
    ]
    
    # Generate LaTeX
    latex = paper_df.to_latex(
        index=False,
        caption=caption,
        label='tab:cache_metrics',
        escape=False,
        column_format='l' * len(paper_df.columns)
    )
    
    with open(output_file, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to: {output_file}")


def main():
    """Generate all paper metrics."""
    
    # Directories to analyze
    results_dirs = {
        'V1 (Old Gamma)': CACHE_RESULTS_DIR / 'final_old_gamma',
        'V2 (New Gamma)': CACHE_RESULTS_DIR / 'final_new_gamma',
    }
    
    all_summaries = []
    
    for gamma_name, results_dir in results_dirs.items():
        if not results_dir.exists():
            warnings.warn(f"Directory not found: {results_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {gamma_name}")
        print(f"Directory: {results_dir}")
        print(f"{'='*80}\n")
        
        # Generate summary
        summary_df = generate_summary_table(results_dir, gamma_name)
        all_summaries.append(summary_df)
        
        # Save individual CSV
        csv_file = REPORTS_DIR / f"paper_metrics_{gamma_name.replace(' ', '_').lower()}.csv"
        summary_df.to_csv(csv_file, index=False)
        print(f"✓ Saved CSV: {csv_file}")

        # Save individual Excel with formatting
        excel_file = REPORTS_DIR / f"paper_metrics_{gamma_name.replace(' ', '_').lower()}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"✓ Saved Excel: {excel_file}")

        # Generate LaTeX table
        latex_file = REPORTS_DIR / f"paper_table_{gamma_name.replace(' ', '_').lower()}.tex"
        generate_latex_table(summary_df, latex_file, caption=f"Cache Effectiveness Metrics - {gamma_name}")
        print(f"✓ Saved LaTeX: {latex_file}")
        
        # Print summary statistics
        print(f"\nSummary Statistics for {gamma_name}:")
        print(f"  Scenarios analyzed: {summary_df['scenario'].nunique()}")
        print(f"  Strategies analyzed: {summary_df['strategy'].nunique()}")
        
        if 'hit_rate_pct' in summary_df.columns:
            print(f"\n  Cache Hit Rate:")
            print(f"    Mean: {summary_df['hit_rate_pct'].mean():.2f}%")
            print(f"    Median: {summary_df['hit_rate_pct'].median():.2f}%")
            print(f"    Min: {summary_df['hit_rate_pct'].min():.2f}%")
            print(f"    Max: {summary_df['hit_rate_pct'].max():.2f}%")
        
        if 'speedup_factor' in summary_df.columns:
            valid_speedups = summary_df['speedup_factor'].dropna()
            if len(valid_speedups) > 0:
                print(f"\n  Speedup Factor:")
                print(f"    Mean: {valid_speedups.mean():.2f}×")
                print(f"    Median: {valid_speedups.median():.2f}×")
                print(f"    Min: {valid_speedups.min():.2f}×")
                print(f"    Max: {valid_speedups.max():.2f}×")
        
        if 'memory_peak_mb' in summary_df.columns:
            valid_memory = summary_df['memory_peak_mb'].dropna()
            if len(valid_memory) > 0:
                print(f"\n  Memory Footprint:")
                print(f"    Mean Peak: {valid_memory.mean():.2f} MB")
                print(f"    Max Peak: {valid_memory.max():.2f} MB")
    
    # Combine all summaries
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        
        # Save combined results
        combined_csv = REPORTS_DIR / "paper_metrics_combined.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n{'='*80}")
        print(f"✓ Saved combined CSV: {combined_csv}")

        combined_excel = REPORTS_DIR / "paper_metrics_combined.xlsx"
        with pd.ExcelWriter(combined_excel, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Add per-gamma sheets
            for gamma_name in combined_df['gamma_version'].unique():
                gamma_df = combined_df[combined_df['gamma_version'] == gamma_name]
                sheet_name = gamma_name.replace(' ', '_')[:31]  # Excel sheet name limit
                gamma_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✓ Saved combined Excel: {combined_excel}")
        
        # Generate comparison table
        comparison_latex = REPORTS_DIR / "paper_table_comparison.tex"
        generate_latex_table(combined_df, comparison_latex, 
                           caption="Cache Effectiveness Metrics - Gamma V1 vs V2 Comparison")
        print(f"✓ Saved comparison LaTeX: {comparison_latex}")
        
        # Generate summary report
        report_file = REPORTS_DIR / "paper_metrics_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CACHE EFFECTIVENESS METRICS - PAPER REPORT\n")
            f.write("="*80 + "\n\n")
            
            for gamma_name in combined_df['gamma_version'].unique():
                gamma_df = combined_df[combined_df['gamma_version'] == gamma_name]
                
                f.write(f"\n{gamma_name}\n")
                f.write("-" * 80 + "\n\n")
                
                f.write(f"Scenarios: {gamma_df['scenario'].nunique()}\n")
                f.write(f"Strategies: {gamma_df['strategy'].nunique()}\n\n")
                
                if 'hit_rate_pct' in gamma_df.columns:
                    f.write("Cache Hit Rate (%):\n")
                    f.write(f"  Mean:   {gamma_df['hit_rate_pct'].mean():8.2f}%\n")
                    f.write(f"  Median: {gamma_df['hit_rate_pct'].median():8.2f}%\n")
                    f.write(f"  Std:    {gamma_df['hit_rate_pct'].std():8.2f}%\n")
                    f.write(f"  Min:    {gamma_df['hit_rate_pct'].min():8.2f}%\n")
                    f.write(f"  Max:    {gamma_df['hit_rate_pct'].max():8.2f}%\n\n")
                
                if 'speedup_factor' in gamma_df.columns:
                    valid_speedups = gamma_df['speedup_factor'].dropna()
                    if len(valid_speedups) > 0:
                        f.write("Speedup Factor:\n")
                        f.write(f"  Mean:   {valid_speedups.mean():8.2f}×\n")
                        f.write(f"  Median: {valid_speedups.median():8.2f}×\n")
                        f.write(f"  Std:    {valid_speedups.std():8.2f}×\n")
                        f.write(f"  Min:    {valid_speedups.min():8.2f}×\n")
                        f.write(f"  Max:    {valid_speedups.max():8.2f}×\n\n")
                
                if 'latency_p50_ms' in gamma_df.columns:
                    f.write("Latency P50 (ms):\n")
                    f.write(f"  Mean:   {gamma_df['latency_p50_ms'].mean():8.2f}\n")
                    f.write(f"  Median: {gamma_df['latency_p50_ms'].median():8.2f}\n\n")
                
                if 'latency_p95_ms' in gamma_df.columns:
                    f.write("Latency P95 (ms):\n")
                    f.write(f"  Mean:   {gamma_df['latency_p95_ms'].mean():8.2f}\n")
                    f.write(f"  Median: {gamma_df['latency_p95_ms'].median():8.2f}\n\n")
                
                if 'latency_p99_ms' in gamma_df.columns:
                    f.write("Latency P99 (ms):\n")
                    f.write(f"  Mean:   {gamma_df['latency_p99_ms'].mean():8.2f}\n")
                    f.write(f"  Median: {gamma_df['latency_p99_ms'].median():8.2f}\n\n")
        
        print(f"✓ Saved report: {report_file}")
        
        print(f"\n{'='*80}")
        print("All metrics generated successfully!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
