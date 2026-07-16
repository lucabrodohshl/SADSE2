"""
Create publication-ready coverage and cache effectiveness visualization for the paper.
Combines coverage progress with cache hit effectiveness in a clean, professional layout.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd

from paths import CACHE_RESULTS_DIR, FIGURES_DIR

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 4


def load_results(json_file: Path):
    """Load experiment results from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['events'])


def create_combined_figure(results_dir: Path, scenario_name: str, gamma_label: str, output_dir: Path):
    """
    Create a combined figure showing:
    1. Coverage progress over time
    2. Cache hit effectiveness (cumulative hit rate)
    """
    
    json_file = results_dir / scenario_name / "drone_6d_large_cs_results.json"
    
    if not json_file.exists():
        print(f"⚠️  Skipping {scenario_name} - no results file")
        return None
    
    df = load_results(json_file)
    
    # Create figure with 2 rows
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Color scheme for strategies
    colors = {
        'SMART CACHE (DISCRETE)': '#2E86AB',  # Blue
        'SMART CACHE (LINEAR)': '#A23B72',     # Purple
        'SMART CACHE (GA)': '#F18F01',         # Orange
        'SMART CACHE (LNS)': '#C73E1D',        # Red
    }
    
    strategies = [s for s in df['strategy'].unique() if 'SMART CACHE' in s]
    
    # ============ Plot 1: Coverage Progress ============
    ax1 = fig.add_subplot(gs[0])
    
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy].copy()
        strategy_df = strategy_df.sort_values('timestamp')
        
        # Convert timestamp to seconds from start
        start_time = strategy_df['timestamp'].min()
        strategy_df['time_s'] = strategy_df['timestamp'] - start_time
        
        # Get coverage progress
        coverage = strategy_df['coverage_pct'].values
        time_s = strategy_df['time_s'].values
        
        # Plot with label cleanup
        label = strategy.replace('SMART CACHE ', '')
        ax1.plot(time_s, coverage, label=label, color=colors[strategy], linewidth=2, alpha=0.85)
    
    ax1.set_xlabel('Time (s)', fontweight='bold')
    ax1.set_ylabel('Coverage (%)', fontweight='bold')
    ax1.set_title('(a) Coverage Progress Over Time', fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', framealpha=0.95, edgecolor='black')
    ax1.set_ylim([0, 100])
    
    # ============ Plot 2: Cache Hit Effectiveness ============
    ax2 = fig.add_subplot(gs[1])
    
    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy].copy()
        strategy_df = strategy_df.sort_values('timestamp')
        
        # Calculate cumulative hit rate
        if 'cache_hit' in strategy_df.columns:
            strategy_df['cumulative_hits'] = strategy_df['cache_hit'].cumsum()
            strategy_df['request_count'] = range(1, len(strategy_df) + 1)
            strategy_df['hit_rate'] = (strategy_df['cumulative_hits'] / strategy_df['request_count']) * 100
            
            # Convert timestamp to seconds
            start_time = strategy_df['timestamp'].min()
            strategy_df['time_s'] = strategy_df['timestamp'] - start_time
            
            hit_rate = strategy_df['hit_rate'].values
            time_s = strategy_df['time_s'].values
            
            label = strategy.replace('SMART CACHE ', '')
            ax2.plot(time_s, hit_rate, label=label, color=colors[strategy], linewidth=2, alpha=0.85)
    
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
    ax2.set_title('(b) Cache Hit Rate Over Time', fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='lower right', framealpha=0.95, edgecolor='black')
    ax2.set_ylim([0, 100])
    
    # Add overall title
    scenario_parts = scenario_name.split('_')
    duration = scenario_parts[2] if len(scenario_parts) > 2 else "?"
    temp = scenario_parts[4] if len(scenario_parts) > 4 else "?"
    
    fig.suptitle(f'{gamma_label} - Scenario: {duration}s, {temp}°C', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    safe_name = scenario_name.replace('weather_scenario_', '').replace('_realistic', '')
    output_file = output_dir / f"coverage_cache_combined_{gamma_label.replace(' ', '_')}_{safe_name}"
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_file.name}.png/pdf")
    return output_file


def create_comparison_figure(results_dirs: dict, scenario_name: str, output_dir: Path):
    """
    Create a comparison figure showing V1 vs V2 for a specific scenario.
    2x2 grid: Coverage (V1), Coverage (V2), Cache Hit (V1), Cache Hit (V2)
    """
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    
    colors = {
        'SMART CACHE (DISCRETE)': '#2E86AB',
        'SMART CACHE (LINEAR)': '#A23B72',
        'SMART CACHE (GA)': '#F18F01',
        'SMART CACHE (LNS)': '#C73E1D',
    }
    
    gamma_labels = ['V1 (Coarse)', 'V2 (Fine-Grained)']
    
    for idx, (gamma_name, results_dir) in enumerate(results_dirs.items()):
        json_file = results_dir / scenario_name / "drone_6d_large_cs_results.json"
        
        if not json_file.exists():
            print(f"⚠️  Skipping {gamma_name} - no results file")
            continue
        
        df = load_results(json_file)
        strategies = [s for s in df['strategy'].unique() if 'SMART CACHE' in s]
        
        # Coverage plot (top row)
        ax_cov = fig.add_subplot(gs[0, idx])
        
        for strategy in strategies:
            strategy_df = df[df['strategy'] == strategy].copy()
            strategy_df = strategy_df.sort_values('timestamp')
            
            start_time = strategy_df['timestamp'].min()
            strategy_df['time_s'] = strategy_df['timestamp'] - start_time
            
            label = strategy.replace('SMART CACHE ', '')
            ax_cov.plot(strategy_df['time_s'], strategy_df['coverage_pct'], 
                       label=label, color=colors[strategy], linewidth=2, alpha=0.85)
        
        ax_cov.set_xlabel('Time (s)', fontweight='bold')
        ax_cov.set_ylabel('Coverage (%)', fontweight='bold')
        ax_cov.set_title(f'({chr(97+idx)}) Coverage - {gamma_labels[idx]}', 
                        fontweight='bold', loc='left')
        ax_cov.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_cov.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=8)
        ax_cov.set_ylim([0, 100])
        
        # Cache hit rate plot (bottom row)
        ax_cache = fig.add_subplot(gs[1, idx])
        
        for strategy in strategies:
            strategy_df = df[df['strategy'] == strategy].copy()
            strategy_df = strategy_df.sort_values('timestamp')
            
            if 'cache_hit' in strategy_df.columns:
                strategy_df['cumulative_hits'] = strategy_df['cache_hit'].cumsum()
                strategy_df['request_count'] = range(1, len(strategy_df) + 1)
                strategy_df['hit_rate'] = (strategy_df['cumulative_hits'] / strategy_df['request_count']) * 100
                
                start_time = strategy_df['timestamp'].min()
                strategy_df['time_s'] = strategy_df['timestamp'] - start_time
                
                label = strategy.replace('SMART CACHE ', '')
                ax_cache.plot(strategy_df['time_s'], strategy_df['hit_rate'], 
                            label=label, color=colors[strategy], linewidth=2, alpha=0.85)
        
        ax_cache.set_xlabel('Time (s)', fontweight='bold')
        ax_cache.set_ylabel('Cache Hit Rate (%)', fontweight='bold')
        ax_cache.set_title(f'({chr(99+idx)}) Cache Effectiveness - {gamma_labels[idx]}', 
                          fontweight='bold', loc='left')
        ax_cache.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_cache.legend(loc='lower right', framealpha=0.95, edgecolor='black', fontsize=8)
        ax_cache.set_ylim([0, 100])
    
    # Overall title
    scenario_parts = scenario_name.split('_')
    duration = scenario_parts[2] if len(scenario_parts) > 2 else "?"
    temp = scenario_parts[4] if len(scenario_parts) > 4 else "?"
    
    fig.suptitle(f'Coverage and Cache Effectiveness Comparison - Scenario: {duration}s, {temp}°C', 
                 fontsize=14, fontweight='bold')
    
    # Save
    safe_name = scenario_name.replace('weather_scenario_', '').replace('_realistic', '')
    output_file = output_dir / f"v1_v2_comparison_{safe_name}"
    
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created comparison: {output_file.name}.png/pdf")
    return output_file


def main():
    output_dir = FIGURES_DIR
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING COVERAGE & CACHE EFFECTIVENESS FIGURES")
    print("="*80 + "\n")
    
    results_dirs = {
        'V1 (Old Gamma)': CACHE_RESULTS_DIR / 'final_old_gamma',
        'V2 (New Gamma)': CACHE_RESULTS_DIR / 'final_new_gamma',
    }
    
    # Find all scenarios
    scenarios = set()
    for results_dir in results_dirs.values():
        if results_dir.exists():
            for scenario_dir in results_dir.iterdir():
                if scenario_dir.is_dir():
                    scenarios.add(scenario_dir.name)
    
    scenarios = sorted(scenarios)
    
    print(f"Found {len(scenarios)} scenarios\n")
    
    # Create individual figures for each gamma version and scenario
    print("Creating individual figures...")
    for gamma_name, results_dir in results_dirs.items():
        print(f"\n{gamma_name}:")
        for scenario in scenarios:
            create_combined_figure(results_dir, scenario, gamma_name, output_dir)
    
    # Create comparison figures (V1 vs V2 side by side)
    print("\n" + "-"*80)
    print("Creating V1 vs V2 comparison figures...")
    for scenario in scenarios:
        create_comparison_figure(results_dirs, scenario, output_dir)
    
    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated:")
    print("  - Individual figures: coverage_cache_combined_*.png/pdf")
    print("  - Comparison figures: v1_v2_comparison_*.png/pdf")
    print("\nRecommendation: Use comparison figures for main paper (shows V1 vs V2)")
    print("                Use individual figures for supplementary material")
    print()


if __name__ == "__main__":
    main()
