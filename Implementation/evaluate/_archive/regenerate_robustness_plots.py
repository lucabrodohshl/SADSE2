"""
Quick script to regenerate robustness plots with updated styling from existing results.
Uses EXACT same colors and linestyles as paper_graphs.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from paths import ROBUSTNESS_RESULTS_DIR

def regenerate_plots_from_json(results_dir: Path):
    """Load existing results and regenerate plots with new styling."""
    
    # Load results
    json_file = results_dir / 'robustness_analysis_results.json'
    if not json_file.exists():
        print(f"❌ Results file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if 'degradation_factors' in data and 'results_by_degradation' in data:
        degradation_factors = data['degradation_factors']
        all_results = data['results_by_degradation']
    else:
        degradation_factors = sorted([float(k) for k in data.keys()])
        all_results = data
    
    print(f"✓ Loaded results for degradation factors: {[df*100 for df in degradation_factors]}%")
    
    # Use EXACT same color mapping as paper_graphs.py
    # Strategies are paired: (cached, baseline) and share the same color but different linestyle
    # C0, C1, C2, C3 from seaborn-v0_8-darkgrid color cycle
    PAPER_COLORS = {
        'C0': '#1f77b4',  # Blue - DISCRETE
        'C1': '#ff7f0e',  # Orange - LINEAR
        'C2': '#2ca02c',  # Green - GA
        'C3': '#d62728',  # Red - LNS
    }
    
    # Strategy color and linestyle mapping (EXACT match to paper_graphs.py)
    STRATEGY_VISUAL = {
        'SMART CACHE (DISCRETE)': {'color': PAPER_COLORS['C0'], 'linestyle': '-'},
        'BASELINE DISCRETE': {'color': PAPER_COLORS['C0'], 'linestyle': 'dotted'},
        'SMART CACHE (LINEAR)': {'color': PAPER_COLORS['C1'], 'linestyle': '-'},
        'BASELINE LINEAR': {'color': PAPER_COLORS['C1'], 'linestyle': 'dotted'},
        'SMART CACHE (GA)': {'color': PAPER_COLORS['C2'], 'linestyle': '-'},
        'GENETIC ALGORITHM': {'color': PAPER_COLORS['C2'], 'linestyle': 'dotted'},
        'SMART CACHE (LNS)': {'color': PAPER_COLORS['C3'], 'linestyle': '-'},
        'LARGE NEIGHBORHOOD SEARCH': {'color': PAPER_COLORS['C3'], 'linestyle': 'dotted'},
    }
    
    # Map strategy names to paper's LaTeX command naming convention
    # Follows pattern: \DMILP, \LMILP, \GA, \LNS and \cached{NAME} for cached versions
    STRATEGY_LEGEND_NAMES = {
        'SMART CACHE (DISCRETE)': r'$\mathsf{DMILP}_{\mathsf{C}}$',
        'BASELINE DISCRETE': r'$\mathsf{DMILP}$',
        'SMART CACHE (LINEAR)': r'$\mathsf{LMILP}_{\mathsf{C}}$',
        'BASELINE LINEAR': r'$\mathsf{LMILP}$',
        'SMART CACHE (GA)': r'$\mathsf{GA}_{\mathsf{C}}$',
        'GENETIC ALGORITHM': r'$\mathsf{GA}$',
        'SMART CACHE (LNS)': r'$\mathsf{LNS}_{\mathsf{C}}$',
        'LARGE NEIGHBORHOOD SEARCH': r'$\mathsf{LNS}$',
    }
    
    # Get strategy names
    if 'degradation_factors' in data:
        first_key = f"degradation_{degradation_factors[0]:.3f}"
    else:
        first_key = str(degradation_factors[0])
    
    strategy_names = list(all_results[first_key].keys())
    
    # Prepare data for plotting
    data_by_strategy = {sn: {'energies': [], 'violations': [], 'margins': []} 
                       for sn in strategy_names}
    
    for df in degradation_factors:
        if 'degradation_factors' in data:
            df_key = f"degradation_{df:.3f}"
        else:
            df_key = str(df)
        
        for strategy_name in strategy_names:
            results = all_results[df_key][strategy_name]
            if results['timestamps']:
                data_by_strategy[strategy_name]['energies'].append(np.mean(results['real_energies']))
                data_by_strategy[strategy_name]['violations'].append(np.mean(results['violations']) * 100)
                data_by_strategy[strategy_name]['margins'].append(np.mean(results['margins']))
            else:
                data_by_strategy[strategy_name]['energies'].append(np.nan)
                data_by_strategy[strategy_name]['violations'].append(np.nan)
                data_by_strategy[strategy_name]['margins'].append(np.nan)
    
    print("\n📊 Generating plots with paper-matching styling...")
    
    # Plot 1: Energy consumption vs degradation factor
    plt.style.use('seaborn-v0_8-darkgrid')  # Match paper style
    plt.figure(figsize=(10, 6))
    for strategy_name in strategy_names:
        visual = STRATEGY_VISUAL.get(strategy_name, {'color': '#000000', 'linestyle': '-'})
        legend_name = STRATEGY_LEGEND_NAMES.get(strategy_name, strategy_name)
        plt.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy[strategy_name]['energies'],
            marker='o',
            label=legend_name,
            linewidth=2.5 if visual['linestyle'] == '-' else 2.0,
            color=visual['color'],
            linestyle=visual['linestyle']
        )
    plt.xlabel('Model Error (%)', fontsize=12)
    plt.ylabel('Average Energy Consumption (Wh)', fontsize=12)
    plt.title('Impact of Model Mismatch on Energy Consumption', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'robustness_energy_vs_error.pdf', bbox_inches='tight')
    plt.savefig(results_dir / 'robustness_energy_vs_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Energy plot saved")
    
    # Plot 2: Violation rate vs degradation factor
    plt.style.use('seaborn-v0_8-darkgrid')  # Match paper style
    plt.figure(figsize=(10, 6))
    for strategy_name in strategy_names:
        visual = STRATEGY_VISUAL.get(strategy_name, {'color': '#000000', 'linestyle': '-'})
        legend_name = STRATEGY_LEGEND_NAMES.get(strategy_name, strategy_name)
        plt.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy[strategy_name]['violations'],
            marker='s',
            label=legend_name,
            linewidth=2.5 if visual['linestyle'] == '-' else 2.0,
            color=visual['color'],
            linestyle=visual['linestyle']
        )
    plt.xlabel('Model Error (%)', fontsize=12)
    plt.ylabel('Violation Rate (%)', fontsize=12)
    plt.title('Impact of Model Mismatch on Feasibility', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=5.0, color='#d62728', linestyle='--', alpha=0.5, linewidth=1.5, label='5% Threshold')
    plt.tight_layout()
    plt.savefig(results_dir / 'robustness_violations_vs_error.pdf', bbox_inches='tight')
    plt.savefig(results_dir / 'robustness_violations_vs_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Violations plot saved")
    
    # Plot 3: Combined plot (dual y-axis) - paper version
    plt.style.use('seaborn-v0_8-darkgrid')  # Match paper style
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    
    ax1.set_xlabel('Model Error (%)', fontsize=11)
    ax1.set_ylabel('Average Energy (Wh)', fontsize=11, color='#34495e')
    ax1.tick_params(axis='y', labelcolor='#34495e')
    
    # Plot all 8 strategies for completeness
    for strategy_name in strategy_names:
        visual = STRATEGY_VISUAL.get(strategy_name, {'color': '#000000', 'linestyle': '-'})
        legend_name = STRATEGY_LEGEND_NAMES.get(strategy_name, strategy_name)
        ax1.plot(
            [df * 100 for df in degradation_factors],
            data_by_strategy[strategy_name]['energies'],
            marker='o' if visual['linestyle'] == '-' else 's',
            label=legend_name,
            linewidth=2.5 if visual['linestyle'] == '-' else 2.0,
            linestyle=visual['linestyle'],
            color=visual['color'],
            alpha=0.85
        )
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=7, framealpha=0.9)
    
    # Create second y-axis for violations
    ax2 = ax1.twinx()
    ax2.set_ylabel('Violation Rate (%)', fontsize=11, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(-1, max(10, max([max(data_by_strategy[sn]['violations']) for sn in strategy_names]) + 2))
    
    # Plot violation bars (grouped by degradation) - show all strategies
    bar_width = 1.5
    x_positions = np.array([df * 100 for df in degradation_factors])
    
    for i, strategy_name in enumerate(strategy_names):
        visual = STRATEGY_VISUAL.get(strategy_name, {'color': '#000000', 'linestyle': '-'})
        offset = (i - len(strategy_names)/2 + 0.5) * (bar_width / len(strategy_names))
        ax2.bar(
            x_positions + offset,
            data_by_strategy[strategy_name]['violations'],
            width=bar_width/len(strategy_names),
            alpha=0.4,
            color=visual['color'],
            edgecolor=visual['color'],
            linewidth=0.8
        )
    
    plt.title('Robustness to Model Mismatch', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(results_dir / 'robustness_combined.pdf', bbox_inches='tight')
    plt.savefig(results_dir / 'robustness_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Combined plot saved")
    
    print(f"\n✅ All plots regenerated with paper-matching styling in: {results_dir}")


if __name__ == "__main__":
    # Find the most recent results directory
    base_dir = ROBUSTNESS_RESULTS_DIR / "final" / "weather_180_temp_15_v2"

    if not base_dir.exists():
        # Try alternative location
        base_dir = ROBUSTNESS_RESULTS_DIR / "final" / "weather_180_temp_15_v2"

    if not base_dir.exists():
        print(f"❌ Results directory not found: {base_dir}")
        sys.exit(1)
    
    print(f"📂 Using results from: {base_dir}")
    regenerate_plots_from_json(base_dir)
