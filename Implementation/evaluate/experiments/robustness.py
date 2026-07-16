"""
Robustness Analysis for SADSE Paper - Experiment A: Sensitivity to Model Mismatch

This script implements the robustness testing described in docs/experiments.md:
- Tests system behavior under sim-to-real energy model mismatch
- Varies degradation factor from 0% to 20%
- Measures violation rates and feasibility
- Generates data for paper Section 6.4

IMPORTANT: This is a NEW implementation that does NOT modify existing code.
All original functions in drone_scenario.py remain unchanged.
"""

from paths import ASSETS_DIR, ROBUSTNESS_RESULTS_DIR

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple, Dict
import argparse
import random

from src.domain import DomainSpec, Parameter, ConfigurationSpace
from src.zonotope_ops import Zonotope
from src.milp_solver import solve_task_assignment_milp, drone_energy_model
from src.milp_solver_v2 import (
    drone_energy_model_v2_perturbed, 
    drone_energy_model_v3_perturbed,
    calculate_real_cost,
    check_feasibility_violation,
    RobustnessMetrics
)
from src.utils import print_cache_summary
from src.smart_cache import SmartZonotopeCache, SmartCacheEntry
from src.visualization import generate_visualizations, visualize_optimal_configs, plot_energy_comparison
from src.zonotope_cache import ZonotopeCache
from src.memory_utils import MemoryComparator, StrategyMemoryTracker

from evaluate.implementation_specifics.domain_specific import DroneODD, TightDroneODD, Drone, Fleet
from evaluate.implementation_specifics.strategies import (
    SmartBayesianStrategy, SmartCacheStrategy, DiscreteStrategy, 
    SmartLinearStrategy, LinearStrategy, BayesianOptimizationStrategy, 
    GAStrategy, SmartGAStrategy, LNSStrategy, SmartLNSStrategy
)
from evaluate.implementation_specifics.vis_utils import print_cache_performance, convert_numpy_types
from evaluate.implementation_specifics.weather import WeatherEnvironmentFixed



from src.visualization import  DEFAULT_COLORS, DEFAULT_LINESTYLES, DEFAULT_LINEWIDTHS, DEFAULT_MARKERS, DEFAULT_MARKERSIZES, get_visual_info


plt.rcParams.update({
    "font.family": "sans-serif",      # or "Arial" / "Helvetica" / "DejaVu Sans"
    "font.size": 8,                   # matches ~9 pt LaTeX text when scaled
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "pgf.preamble":  r"""
\usepackage{amsmath}
\newcommand{\cached}[1]{\ensuremath{\text{#1}_{C}}}
\newcommand{\DMILP}{DMILP}
\newcommand{\LMILP}{LMILP}
\newcommand{\LNS}{LNS}
\newcommand{\GA}{GA}
\newcommand{\BO}{BO}
"""
})


def run_robustness_evaluation(
    strategies: List,
    drone_domain: Fleet,
    env: WeatherEnvironmentFixed,
    duration_minutes: int,
    adaptation_interval: int,
    degradation_factors: List[float],
    output_dir: Path,
    battery_capacity_wh: float = 300.0,
    seed: int = 42
):
    """
    Run robustness analysis with varying degradation factors.
    
    This implements Experiment A from docs/experiments.md:
    Tests sensitivity to objective function mismatch (energy model error).
    
    Args:
        strategies: List of optimization strategies to test
        drone_domain: Fleet/Drone domain specification
        env: Weather environment
        duration_minutes: Simulation duration
        adaptation_interval: Time between adaptations (minutes)
        degradation_factors: List of degradation factors to test (e.g., [0.0, 0.05, 0.10, 0.15, 0.20])
        output_dir: Directory to save results
        battery_capacity_wh: Battery capacity in Wh
        seed: Random seed for reproducibility
    """
    
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS: SENSITIVITY TO ENERGY MODEL MISMATCH")
    print("=" * 80)
    print(f"Testing degradation factors: {degradation_factors}")
    print(f"Battery capacity: {battery_capacity_wh} Wh")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Adaptation interval: {adaptation_interval} minutes")
    print("=" * 80)
    print()
    
    # Results storage for each degradation factor
    all_results = {}
    
    for degradation_factor in degradation_factors:
        print(f"\n{'='*80}")
        print(f"TESTING DEGRADATION FACTOR: {degradation_factor*100:.1f}%")
        print(f"{'='*80}\n")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
        
        # Reset environment
        
        
        # Initialize fresh strategies for this degradation level
        strategy_copies = []
        for s in strategies:
            # Create a new instance of each strategy
            strategy_type = type(s)
            
            if hasattr(s, 'cache') and s.cache is not None:
                # Smart cache strategy - must have file_path
                if not hasattr(s, 'file_path'):
                    raise ValueError(f"Smart cache strategy {s.name} missing file_path attribute")
                new_strategy = strategy_type(
                    name=s.name,
                    cs=drone_domain,
                    file_path=s.file_path
                )
            else:
                # Baseline strategy - no file_path needed
                new_strategy = strategy_type(
                    name=s.name,
                    cs=drone_domain
                )
            
            strategy_copies.append(new_strategy)
        noise_multiplier = random_state.normal(0.0, float(0.2))
        # Run evaluation for this degradation factor
        results = run_single_degradation_test_v2(
            strategy_copies,
            drone_domain,
            env,
            duration_minutes,
            adaptation_interval,
            degradation_factor,
            battery_capacity_wh,
            noise_multiplier
        )
        
        all_results[degradation_factor] = results
        
        # Print summary for this degradation level
        print_degradation_summary(results, degradation_factor)
    
    # Save comprehensive results
    save_robustness_results(all_results, degradation_factors, output_dir)
    
    # Generate plots
    generate_robustness_plots(all_results, degradation_factors, output_dir, strategies)
    
    print("\n" + "="*80)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")

def run_single_degradation_test_v2(
    strategies: List,
    drone_domain: Fleet,
    env: WeatherEnvironmentFixed,
    duration_minutes: int,
    adaptation_interval: int,
    degradation_factor: float,
    battery_capacity_wh: float,
    noise_multiplier: float = 0.02
) -> Dict:
    """
    Run a single test with a specific degradation factor.
    
    This simulates the Digital Twin (DT) / Physical Twin (PT) split:
    - DT uses ideal model to optimize
    - PT applies perturbed model to measure actual energy
    """
    

    

    num_adaptations = duration_minutes // adaptation_interval
    
    results_per_strategy = {}
    
    for strategy in strategies:
        print(f"   Testing {strategy.name}...")
        
        strategy_results = {
            'timestamps': [],
            'predicted_energies': [],
            'real_energies': [],
            'violations': [],
            'margins': [],
            'cache_hits': [],
            'latencies': [],
            'configs': []
        }
        env_copy = WeatherEnvironmentFixed("Field Weather", data_function='realistic')
        env_copy.set_seed(env.seed)
        env_copy.duration_hours = duration_minutes * 12
        env_copy.dt_minutes = 60
        env_copy.lat_deg = env.lat_deg
        env_copy.day_of_year = env.day_of_year
        env_copy.base_temp_c = env.base_temp_c
        env_copy.data = env_copy.generate_realistic_data()
        
        for t in range(num_adaptations):
            timestamp = t * adaptation_interval * 60.0  # Convert to seconds
            


            # Get current ODD
            odd = env_copy.step(timestamp)
            
            # Get constrained domain (DS)
            ds = drone_domain.get_domain_spec().apply_odd_constraints(odd)
            if ds.is_empty():
                continue
            
            # STEP 1: Digital Twin (DT) - Uses ideal model
            # Strategy queries cache or optimizes using ideal model
            config, milp_objective, _, latency = strategy.execute(odd, ds)
            
            if config is None:
                continue
            
            # Get predicted energy for this configuration using ideal model
            predicted_energy = drone_energy_model(config, task={'length': 1000.0})
            
            # STEP 2: Physical Twin (PT) - Apply perturbation
            # Simulate real-world execution with model mismatch
            real_energy = drone_energy_model_v3_perturbed(
                config,
                task={'length': 1000.0},  # Standard task
                degradation_factor=degradation_factor,
                noise_multiplier=noise_multiplier,
                apply_perturbation=True
            )
            
            # STEP 3: Check feasibility violation
            is_violation, margin = check_feasibility_violation(
                real_energy,
                battery_capacity_wh,
                energy_reserve_ratio=0.10,
                energy_budget_safety=1.00
            )
            
            # Record results
            strategy_results['timestamps'].append(timestamp)
            strategy_results['predicted_energies'].append(predicted_energy)
            strategy_results['real_energies'].append(real_energy)
            strategy_results['violations'].append(is_violation)
            strategy_results['margins'].append(margin)
            strategy_results['cache_hits'].append(strategy.has_hit() if hasattr(strategy, 'has_hit') else False)
            strategy_results['latencies'].append(latency)
            strategy_results['configs'].append(config.as_dict() if config else None)
        
        results_per_strategy[strategy.name] = strategy_results
    
    return results_per_strategy



def save_robustness_results(all_results: Dict, degradation_factors: List[float], output_dir: Path):
    """Save comprehensive robustness results to JSON."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for export
    export_data = {
        'degradation_factors': degradation_factors,
        'results_by_degradation': {}
    }
    
    for degradation_factor, results in all_results.items():
        df_key = f"degradation_{degradation_factor:.3f}"
        export_data['results_by_degradation'][df_key] = {}
        
        for strategy_name, strategy_results in results.items():
            # Convert numpy types to native Python types
            export_data['results_by_degradation'][df_key][strategy_name] = {
                'timestamps': [float(x) for x in strategy_results['timestamps']],
                'predicted_energies': [float(x) for x in strategy_results['predicted_energies']],
                'real_energies': [float(x) for x in strategy_results['real_energies']],
                'violations': [bool(x) for x in strategy_results['violations']],
                'margins': [float(x) for x in strategy_results['margins']],
                'cache_hits': [bool(x) for x in strategy_results['cache_hits']] if strategy_results['cache_hits'] else [],
                'latencies': [float(x) for x in strategy_results['latencies']],
                'summary': {
                    'avg_predicted': float(np.mean(strategy_results['predicted_energies'])),
                    'avg_real': float(np.mean(strategy_results['real_energies'])),
                    'violation_rate': float(np.mean(strategy_results['violations']) * 100),
                    'avg_margin': float(np.mean(strategy_results['margins'])),
                    'total_adaptations': len(strategy_results['timestamps'])
                }
            }
    
    # Save to JSON
    json_path = output_dir / 'robustness_analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Robustness results saved to: {json_path}")
    
    # Also create a summary CSV for easy import into LaTeX
    create_summary_table(all_results, degradation_factors, output_dir)


def create_summary_table(all_results: Dict, degradation_factors: List[float], output_dir: Path):
    """Create a LaTeX-friendly summary table."""
    
    csv_path = output_dir / 'robustness_summary.csv'
    tex_path = output_dir / 'robustness_summary.tex'
    
    # Get strategy names (from first degradation factor)
    first_df = degradation_factors[0]
    strategy_names = list(all_results[first_df].keys())
    
    # Create CSV
    with open(csv_path, 'w') as f:
        # Header
        f.write('Degradation Factor')
        for strategy_name in strategy_names:
            f.write(f',{strategy_name} Avg Energy (Wh)')
            f.write(f',{strategy_name} Violation Rate (%)')
        f.write('\n')
        
        # Data rows
        for df in degradation_factors:
            f.write(f'{df*100:.1f}%')
            for strategy_name in strategy_names:
                results = all_results[df][strategy_name]
                if results['timestamps']:
                    avg_real = np.mean(results['real_energies'])
                    violation_rate = np.mean(results['violations']) * 100
                    f.write(f',{avg_real:.2f},{violation_rate:.1f}')
                else:
                    f.write(',N/A,N/A')
            f.write('\n')
    
    print(f"✓ Summary CSV saved to: {csv_path}")
    
    # Create LaTeX table
    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write('\\caption{Robustness Analysis: Impact of Energy Model Mismatch}\n')
        f.write('\\label{tab:robustness}\n')
        f.write('\\begin{tabular}{c' + 'c' * len(strategy_names) + '}\n')
        f.write('\\toprule\n')
        
        # Header
        f.write('Model Error & ')
        f.write(' & '.join([f'\\multicolumn{{1}}{{c}}{{{sn}}}' for sn in strategy_names]))
        f.write(' \\\\\n')
        f.write('(\\%) & ')
        f.write(' & '.join(['Violation Rate (\\%)'] * len(strategy_names)))
        f.write(' \\\\\n')
        f.write('\\midrule\n')
        
        # Data rows
        for df in degradation_factors:
            f.write(f'{df*100:.0f}')
            for strategy_name in strategy_names:
                results = all_results[df][strategy_name]
                if results['timestamps']:
                    violation_rate = np.mean(results['violations']) * 100
                    f.write(f' & {violation_rate:.1f}')
                else:
                    f.write(' & --')
            f.write(' \\\\\n')
        
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    
    print(f"✓ LaTeX table saved to: {tex_path}")


def generate_robustness_plots(all_results: Dict, degradation_factors: List[float], output_dir: Path, strategies: List):
    """Generate visualization plots for robustness analysis matching paper style."""
    
    print("\nGenerating robustness plots...")
    
    # Use exact same colors as paper (from visualization.py DEFAULT_COLORS)
    # These are assigned to strategies in the order they appear


    labels_to_latex = {
        "BASELINE DISCRETE": r"\DMILP",
        "SMART CACHE (DISCRETE)": r"\cached{\DMILP}",
        "BASELINE LINEAR": r"\LMILP",
        "SMART CACHE (LINEAR)": r"\cached{\LMILP}",
        "GENETIC ALGORITHM": r"\GA",
        "SMART CACHE (GA)": r"\cached{\GA}",
        "LARGE NEIGHBORHOOD SEARCH": r"\LNS",
        "SMART CACHE (LNS)": r"\cached{\LNS}",
        "BAYESIAN OPTIMIZATION": r"\BO",
        "SMART CACHE (BAYESIAN)": r"\cached{\BO}"
    }


    fig_size = (3.8, 2.5)
    
    # Get strategy names
    first_df = degradation_factors[0]
    strategy_names = list(all_results[first_df].keys())
    visual_info = get_visual_info(strategies)
    # Prepare data for plotting
    data_by_strategy = {sn: {'energies': [], 'violations': [], 'margins': []} 
                       for sn in strategy_names}
    
    for df in degradation_factors:
        for strategy_name in strategy_names:
            results = all_results[df][strategy_name]
            if results['timestamps']:
                data_by_strategy[strategy_name]['energies'].append(np.mean(results['real_energies']))
                data_by_strategy[strategy_name]['violations'].append(np.mean(results['violations']) * 100)
                data_by_strategy[strategy_name]['margins'].append(np.mean(results['margins']))
            else:
                data_by_strategy[strategy_name]['energies'].append(np.nan)
                data_by_strategy[strategy_name]['violations'].append(np.nan)
                data_by_strategy[strategy_name]['margins'].append(np.nan)
    
    # Assign colors exactly as paper does (sequential from DEFAULT_COLORS)
    #strategy_colors = {}
    #for idx, sn in enumerate(strategy_names):
    #    strategy_colors[sn] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    
    # Plot 1: Energy consumption vs degradation factor
    plt.style.use('seaborn-v0_8-darkgrid')  # Match paper style
    plt.figure(figsize=fig_size)
    for strategy_name in strategy_names:
        plt.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy[strategy_name]['energies'],
            marker=visual_info['markers'][strategy_name],
            label=labels_to_latex[strategy_name],
            linewidth=visual_info['linewidths'][strategy_name],
            color=visual_info['colors'][strategy_name]
        )
    plt.xlabel('Model Error (%)')
    plt.ylabel('Average Energy Consumption (Wh)')
   # plt.title('Impact of Model Mismatch on Energy Consumption', fontsize=14, fontweight='bold')
    plt.legend(loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_energy_vs_error.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_energy_vs_error.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_energy_vs_error.pgf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Violation rate vs degradation factor
    plt.style.use('seaborn-v0_8-darkgrid')  # Match paper style
    plt.figure(figsize=fig_size)
    for strategy_name in strategy_names:
        plt.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy[strategy_name]['violations'],
            marker=visual_info['markers'][strategy_name],
            label=labels_to_latex[strategy_name],
            linewidth=visual_info['linewidths'][strategy_name],
            color=visual_info['colors'][strategy_name]
        )
    plt.xlabel('Model Error (%)')
    plt.ylabel('Violation Rate (%)')
    #plt.title('Impact of Model Mismatch on Feasibility', fontsize=14, fontweight='bold')
    #plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=5.0, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5, label='5% Threshold')
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_violations_vs_error.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_violations_vs_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Combined plot (dual y-axis) - simplified for paper
    plt.style.use('seaborn-v0_8-darkgrid')  # Match paper style
    fig, ax1 = plt.subplots(figsize=fig_size)  # Smaller size to match paper figures
    
    ax1.set_xlabel('Model Error (%)')
    ax1.set_ylabel('Average Energy (Wh)', color='#34495e')
    ax1.tick_params(axis='y', labelcolor='#34495e')
    
    # Plot only key strategies for clarity (avoid clutter)
    key_strategies = [sn for sn in strategy_names ]#if 'DISCRETE' in sn or 'LINEAR' in sn][:4]
    
    # Plot energy lines
    for i, strategy_name in enumerate(key_strategies):
        linestyle = '-' if 'SMART' in strategy_name else '--'
        ax1.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy[strategy_name]['energies'],
            marker=visual_info['markers'][strategy_name],
            label=labels_to_latex[strategy_name],
            linewidth=visual_info['linewidths'][strategy_name],
            linestyle=visual_info['linestyles'][strategy_name],
            color=visual_info['colors'][strategy_name],
            alpha=0.85
        )
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.9, ncol=2)
    
    # Create second y-axis for violations
    ax2 = ax1.twinx()
    ax2.set_ylabel('Violation Rate (%)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(-1, max(10, max([max(data_by_strategy[sn]['violations']) for sn in key_strategies]) + 2))
    
    # Plot violation bars (grouped by degradation)
    bar_width = 1.5
    x_positions = np.array([df * 100 for df in degradation_factors])
    
    for i, strategy_name in enumerate(key_strategies):
        offset = (i - len(key_strategies)/2 + 0.5) * (bar_width / len(key_strategies))
        ax2.bar(
            x_positions + offset,
            data_by_strategy[strategy_name]['violations'],
            width=bar_width/len(key_strategies),
            alpha=0.4,
            color = visual_info['colors'][strategy_name],
            edgecolor=visual_info['colors'][strategy_name],
            linewidth=visual_info['linewidths'][strategy_name],
        )
    plt.legend(ncol=2)
    #plt.title('Robustness to Model Mismatch', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_combined.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_combined.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_combined.pgf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Prediction error (real - predicted energy) vs degradation factor
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=fig_size)
    
    # Calculate prediction errors for each strategy
    data_by_strategy_error = {sn: [] for sn in strategy_names}
    
    for df in degradation_factors:
        for strategy_name in strategy_names:
            results = all_results[df][strategy_name]
            if results['timestamps'] and results['predicted_energies']:
                avg_predicted = np.mean(results['predicted_energies'])
                avg_real = np.mean(results['real_energies'])
                # Calculate absolute difference
                error = avg_real - avg_predicted
                data_by_strategy_error[strategy_name].append(error)
            else:
                data_by_strategy_error[strategy_name].append(np.nan)
    
    # Plot prediction errors
    for strategy_name in strategy_names:
        plt.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy_error[strategy_name],
            marker=visual_info['markers'][strategy_name],
            label=labels_to_latex[strategy_name],
            linewidth=visual_info['linewidths'][strategy_name],
            linestyle=visual_info['linestyles'][strategy_name],
            color=visual_info['colors'][strategy_name],
            #marker_size = visual_info['markersizes'][strategy_name]
        )
    
    plt.xlabel('Model Error (%)')
    plt.ylabel('Prediction Error (Wh)')
    #plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.0, label='Perfect Prediction')
    plt.legend(loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_prediction_error.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_prediction_error.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'robustness_prediction_error.pgf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to: {output_dir}")
    print(f"  - Energy vs error: robustness_energy_vs_error.pdf")
    print(f"  - Violations vs error: robustness_violations_vs_error.pdf")
    print(f"  - Combined plot: robustness_combined.pdf")
    print(f"  - Prediction error: robustness_prediction_error.pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Robustness Analysis for SADSE Paper (Experiment A)"
    )
    parser.add_argument("--duration", type=int, default=180, 
                       help="Duration of evaluation in minutes (default: 180 = 3 hours)")
    parser.add_argument("--adaptation_interval", type=int, default=5, 
                       help="Adaptation interval in minutes (default: 5)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--configuration_file", type=str, default=str(ASSETS_DIR / "fleet.json"),
                       help="Path to the CS configuration file")
    parser.add_argument("--is_drone", action='store_true', 
                       help="Indicates if configuration file is for a single drone")
    parser.add_argument("--output_dir", type=str, default="latest_run",
                       help="Directory to save results")
    parser.add_argument("--weather_scenario", type=int, default=180, 
                       help="Weather scenario seed (day of year)")
    parser.add_argument("--temperature", type=int, default=15, 
                       help="Base temperature in Celsius")
    parser.add_argument("--degradation_factors", type=str, default="0.0,0.05,0.10,0.15,0.20",
                       help="Comma-separated degradation factors to test (e.g., '0.0,0.05,0.10')")
    parser.add_argument("--battery_capacity", type=float, default=300.0,
                       help="Battery capacity in Wh (default: 300)")
    
    args = parser.parse_args()
    
    # Parse degradation factors
    degradation_factors = [float(x.strip()) for x in args.degradation_factors.split(',')]
    
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS CONFIGURATION")
    print("="*80)
    print(f"Configuration file: {args.configuration_file}")
    print(f"Duration: {args.duration} minutes")
    print(f"Adaptation interval: {args.adaptation_interval} minutes")
    print(f"Weather scenario: {args.weather_scenario} (day of year)")
    print(f"Temperature: {args.temperature}°C")
    print(f"Degradation factors: {degradation_factors}")
    print(f"Battery capacity: {args.battery_capacity} Wh")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load domain
    if not args.is_drone:
        drone_domain = Fleet(file_path=args.configuration_file)
    else:
        drone_domain = Drone(file_path=args.configuration_file)
    
    # Setup weather environment
    env = WeatherEnvironmentFixed("Field Weather", data_function='realistic')
    env.set_seed(args.seed + args.temperature)
    env.duration_hours = args.duration * 12
    env.dt_minutes = 60
    env.lat_deg = 52.5  # Berlin-ish
    env.day_of_year = args.weather_scenario
    env.base_temp_c = args.temperature
    env.data = env.generate_realistic_data()
    

    # Define strategies to test
    strategies = [
        SmartCacheStrategy(
            name="SMART CACHE (DISCRETE)",
            cs=drone_domain,
            file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")
        ),
        DiscreteStrategy("BASELINE DISCRETE", cs=drone_domain),
        SmartLinearStrategy(
            name="SMART CACHE (LINEAR)",
            cs=drone_domain,
            file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")
        ),
        LinearStrategy("BASELINE LINEAR", cs=drone_domain),
        SmartGAStrategy(
            name="SMART CACHE (GA)",
            cs=drone_domain,
            file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")
        ),
        GAStrategy("GENETIC ALGORITHM", cs=drone_domain),
        SmartLNSStrategy(
            name="SMART CACHE (LNS)",
            cs=drone_domain,
            file_path=str(ASSETS_DIR / "smart_strategy_no_thresh.json")
        ),
        LNSStrategy("LARGE NEIGHBORHOOD SEARCH", cs=drone_domain)
    ]
    
    # Store file_path as attribute for strategies that need it (for recreation)
    for s in strategies:
        if hasattr(s, 'cache') and s.cache is not None:
            s.file_path = str(ASSETS_DIR / "smart_strategy_no_thresh.json")
        else:
            s.file_path = None
    
    # Setup output directory
    output_dir = ROBUSTNESS_RESULTS_DIR / args.output_dir
    scenario_dir = output_dir / f"weather_{args.weather_scenario}_temp_{args.temperature}_v2"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Run robustness evaluation
    run_robustness_evaluation(
        strategies=strategies,
        drone_domain=drone_domain,
        env=env,
        duration_minutes=args.duration,
        adaptation_interval=args.adaptation_interval,
        degradation_factors=degradation_factors,
        output_dir=scenario_dir,
        battery_capacity_wh=args.battery_capacity,
        seed=args.seed
    )
    
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults available in: {scenario_dir}")
    print("\nFiles generated:")
    print(f"  - robustness_analysis_results.json  (detailed data)")
    print(f"  - robustness_summary.csv            (summary table)")
    print(f"  - robustness_summary.tex            (LaTeX table)")
    print(f"  - robustness_energy_vs_error.pdf    (energy plot)")
    print(f"  - robustness_violations_vs_error.pdf (violations plot)")
    print(f"  - robustness_combined.pdf           (combined plot)")
    print("\nUse these results for paper Section 6.4 (Robustness Analysis)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
