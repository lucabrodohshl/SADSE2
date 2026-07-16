"""
Create publication-ready visualizations comparing Gamma V1 vs V2 performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from paths import REPORTS_DIR, FIGURES_DIR

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_combined_metrics() -> pd.DataFrame:
    """Load the combined metrics CSV."""
    csv_file = REPORTS_DIR / "paper_metrics_combined.csv"
    return pd.read_csv(csv_file)


def plot_hit_rate_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare cache hit rates between V1 and V2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by gamma version
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    # Plot 1: Box plot by strategy
    ax = axes[0]
    data_to_plot = []
    labels = []
    
    for strategy in sorted(df['strategy'].unique()):
        v1_data = v1_df[v1_df['strategy'] == strategy]['hit_rate_pct']
        v2_data = v2_df[v2_df['strategy'] == strategy]['hit_rate_pct']
        
        if len(v1_data) > 0:
            data_to_plot.append(v1_data)
            labels.append(f"{strategy.replace('SMART CACHE ', '')}\n(V1)")
        
        if len(v2_data) > 0:
            data_to_plot.append(v2_data)
            labels.append(f"{strategy.replace('SMART CACHE ', '')}\n(V2)")
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color V1 and V2 differently
    colors = ['lightblue', 'lightcoral'] * (len(data_to_plot) // 2 + 1)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Cache Hit Rate (%)')
    ax.set_title('Cache Hit Rate: V1 vs V2 by Strategy')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Bar plot of means
    ax = axes[1]
    
    strategies = sorted(df['strategy'].unique())
    x = np.arange(len(strategies))
    width = 0.35
    
    v1_means = [v1_df[v1_df['strategy'] == s]['hit_rate_pct'].mean() for s in strategies]
    v2_means = [v2_df[v2_df['strategy'] == s]['hit_rate_pct'].mean() for s in strategies]
    
    v1_bars = ax.bar(x - width/2, v1_means, width, label='V1 (Old Gamma)', color='lightblue')
    v2_bars = ax.bar(x + width/2, v2_means, width, label='V2 (New Gamma)', color='lightcoral')
    
    ax.set_ylabel('Mean Cache Hit Rate (%)')
    ax.set_title('Average Cache Hit Rate by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [v1_bars, v2_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hit_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hit_rate_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: hit_rate_comparison.png/pdf")
    plt.close()


def plot_speedup_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare speedup factors between V1 and V2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    # Plot 1: Scatter plot
    ax = axes[0]
    
    for strategy in sorted(df['strategy'].unique()):
        v1_data = v1_df[v1_df['strategy'] == strategy]['speedup_factor'].dropna()
        v2_data = v2_df[v2_df['strategy'] == strategy]['speedup_factor'].dropna()
        
        if len(v1_data) > 0:
            ax.scatter([strategy.replace('SMART CACHE ', '')] * len(v1_data), v1_data, 
                      alpha=0.6, s=100, label='V1' if strategy == sorted(df['strategy'].unique())[0] else '', 
                      color='blue', marker='o')
        
        if len(v2_data) > 0:
            ax.scatter([strategy.replace('SMART CACHE ', '')] * len(v2_data), v2_data, 
                      alpha=0.6, s=100, label='V2' if strategy == sorted(df['strategy'].unique())[0] else '', 
                      color='red', marker='s')
    
    ax.set_ylabel('Speedup Factor (×)')
    ax.set_title('Speedup Factor Distribution')
    ax.set_xlabel('Strategy')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Plot 2: Bar comparison
    ax = axes[1]
    
    strategies = sorted(df['strategy'].unique())
    x = np.arange(len(strategies))
    width = 0.35
    
    v1_means = [v1_df[v1_df['strategy'] == s]['speedup_factor'].mean() for s in strategies]
    v2_means = [v2_df[v2_df['strategy'] == s]['speedup_factor'].mean() for s in strategies]
    
    v1_bars = ax.bar(x - width/2, v1_means, width, label='V1 (Old Gamma)', color='lightblue')
    v2_bars = ax.bar(x + width/2, v2_means, width, label='V2 (New Gamma)', color='lightcoral')
    
    ax.set_ylabel('Mean Speedup Factor (×)')
    ax.set_title('Average Speedup: Cache vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [v1_bars, v2_bars]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}×',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'speedup_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: speedup_comparison.png/pdf")
    plt.close()


def plot_latency_percentiles(df: pd.DataFrame, output_dir: Path):
    """Compare latency percentiles between V1 and V2."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    percentiles = ['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']
    percentile_labels = ['P50 (Median)', 'P95', 'P99']
    
    strategies = sorted(df['strategy'].unique())
    
    # Plots for each percentile
    for idx, (percentile, label) in enumerate(zip(percentiles, percentile_labels)):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        v1_values = [v1_df[v1_df['strategy'] == s][percentile].mean() for s in strategies]
        v2_values = [v2_df[v2_df['strategy'] == s][percentile].mean() for s in strategies]
        
        ax.bar(x - width/2, v1_values, width, label='V1 (Old Gamma)', color='lightblue')
        ax.bar(x + width/2, v2_values, width, label='V2 (New Gamma)', color='lightcoral')
        
        ax.set_ylabel(f'Latency {label} (ms)')
        ax.set_title(f'Latency {label} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Overall latency comparison
    ax = axes[1, 1]
    
    # Create grouped data for all percentiles
    width = 0.15
    x = np.arange(len(strategies))
    
    for i, (percentile, label) in enumerate(zip(percentiles, percentile_labels)):
        v1_values = [v1_df[v1_df['strategy'] == s][percentile].mean() for s in strategies]
        v2_values = [v2_df[v2_df['strategy'] == s][percentile].mean() for s in strategies]
        
        offset = (i - 1) * width
        ax.bar(x + offset - width/2, v1_values, width, label=f'V1 {label}', alpha=0.7)
        ax.bar(x + offset + width/2, v2_values, width, label=f'V2 {label}', alpha=0.7)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('All Percentiles Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_percentiles.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'latency_percentiles.pdf', bbox_inches='tight')
    print(f"✓ Saved: latency_percentiles.png/pdf")
    plt.close()


def plot_cache_vs_miss_latency(df: pd.DataFrame, output_dir: Path):
    """Compare cache hit vs miss latency."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    strategies = sorted(df['strategy'].unique())
    
    # Plot 1: Hit vs Miss latency for V1
    ax = axes[0]
    x = np.arange(len(strategies))
    width = 0.35
    
    hit_latencies = [v1_df[v1_df['strategy'] == s]['hit_latency_median_ms'].mean() for s in strategies]
    miss_latencies = [v1_df[v1_df['strategy'] == s]['miss_latency_median_ms'].mean() for s in strategies]
    
    ax.bar(x - width/2, hit_latencies, width, label='Cache Hit', color='green', alpha=0.7)
    ax.bar(x + width/2, miss_latencies, width, label='Cache Miss', color='red', alpha=0.7)
    
    ax.set_ylabel('Median Latency (ms)')
    ax.set_title('V1 (Old Gamma): Hit vs Miss Latency')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Hit vs Miss latency for V2
    ax = axes[1]
    
    hit_latencies = [v2_df[v2_df['strategy'] == s]['hit_latency_median_ms'].mean() for s in strategies]
    miss_latencies = [v2_df[v2_df['strategy'] == s]['miss_latency_median_ms'].mean() for s in strategies]
    
    ax.bar(x - width/2, hit_latencies, width, label='Cache Hit', color='green', alpha=0.7)
    ax.bar(x + width/2, miss_latencies, width, label='Cache Miss', color='red', alpha=0.7)
    
    ax.set_ylabel('Median Latency (ms)')
    ax.set_title('V2 (New Gamma): Hit vs Miss Latency')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hit_vs_miss_latency.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hit_vs_miss_latency.pdf', bbox_inches='tight')
    print(f"✓ Saved: hit_vs_miss_latency.png/pdf")
    plt.close()


def plot_scenario_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare performance across different scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = sorted(df['scenario'].unique())
    
    # Extract scenario info
    scenario_labels = []
    for scenario in scenarios:
        parts = scenario.split('_')
        duration = parts[2] if len(parts) > 2 else "?"
        temp = parts[4] if len(parts) > 4 else "?"
        scenario_labels.append(f"{duration}D\n{temp}°C")
    
    metrics = [
        ('hit_rate_pct', 'Cache Hit Rate (%)', axes[0, 0]),
        ('speedup_factor', 'Speedup Factor (×)', axes[0, 1]),
        ('latency_p50_ms', 'Latency P50 (ms)', axes[1, 0]),
        ('latency_p95_ms', 'Latency P95 (ms)', axes[1, 1]),
    ]
    
    for metric_col, metric_label, ax in metrics:
        x = np.arange(len(scenarios))
        width = 0.35
        
        v1_values = [df[(df['scenario'] == s) & (df['gamma_version'] == 'V1 (Old Gamma)')][metric_col].mean() 
                     for s in scenarios]
        v2_values = [df[(df['scenario'] == s) & (df['gamma_version'] == 'V2 (New Gamma)')][metric_col].mean() 
                     for s in scenarios]
        
        ax.bar(x - width/2, v1_values, width, label='V1 (Old Gamma)', color='lightblue')
        ax.bar(x + width/2, v2_values, width, label='V2 (New Gamma)', color='lightcoral')
        
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scenario_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: scenario_comparison.png/pdf")
    plt.close()


def create_summary_figure(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive summary figure for the paper."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    # 1. Hit Rate
    ax1 = fig.add_subplot(gs[0, 0])
    v1_hit = v1_df['hit_rate_pct'].mean()
    v2_hit = v2_df['hit_rate_pct'].mean()
    ax1.bar(['V1', 'V2'], [v1_hit, v2_hit], color=['lightblue', 'lightcoral'])
    ax1.set_ylabel('Hit Rate (%)')
    ax1.set_title('A. Cache Hit Rate')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate([v1_hit, v2_hit]):
        ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Speedup
    ax2 = fig.add_subplot(gs[0, 1])
    v1_speedup = v1_df['speedup_factor'].mean()
    v2_speedup = v2_df['speedup_factor'].mean()
    ax2.bar(['V1', 'V2'], [v1_speedup, v2_speedup], color=['lightblue', 'lightcoral'])
    ax2.set_ylabel('Speedup Factor (×)')
    ax2.set_title('B. Average Speedup')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate([v1_speedup, v2_speedup]):
        ax2.text(i, v, f'{v:.1f}×', ha='center', va='bottom', fontweight='bold')
    
    # 3. Latency P50
    ax3 = fig.add_subplot(gs[0, 2])
    v1_p50 = v1_df['latency_p50_ms'].mean()
    v2_p50 = v2_df['latency_p50_ms'].mean()
    ax3.bar(['V1', 'V2'], [v1_p50, v2_p50], color=['lightblue', 'lightcoral'])
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('C. Median Latency (P50)')
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate([v1_p50, v2_p50]):
        ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Hit rate by strategy
    ax4 = fig.add_subplot(gs[1, :])
    strategies = sorted(df['strategy'].unique())
    x = np.arange(len(strategies))
    width = 0.35
    
    v1_hits = [v1_df[v1_df['strategy'] == s]['hit_rate_pct'].mean() for s in strategies]
    v2_hits = [v2_df[v2_df['strategy'] == s]['hit_rate_pct'].mean() for s in strategies]
    
    ax4.bar(x - width/2, v1_hits, width, label='V1 (Old Gamma)', color='lightblue')
    ax4.bar(x + width/2, v2_hits, width, label='V2 (New Gamma)', color='lightcoral')
    ax4.set_ylabel('Hit Rate (%)')
    ax4.set_title('D. Hit Rate by Strategy')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Speedup by strategy
    ax5 = fig.add_subplot(gs[2, :])
    
    v1_speedups = [v1_df[v1_df['strategy'] == s]['speedup_factor'].mean() for s in strategies]
    v2_speedups = [v2_df[v2_df['strategy'] == s]['speedup_factor'].mean() for s in strategies]
    
    ax5.bar(x - width/2, v1_speedups, width, label='V1 (Old Gamma)', color='lightblue')
    ax5.bar(x + width/2, v2_speedups, width, label='V2 (New Gamma)', color='lightcoral')
    ax5.set_ylabel('Speedup Factor (×)')
    ax5.set_title('E. Speedup Factor by Strategy')
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.replace('SMART CACHE ', '') for s in strategies], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_dir / 'summary_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_figure.pdf', bbox_inches='tight')
    print(f"✓ Saved: summary_figure.png/pdf")
    plt.close()


def main():
    """Generate all visualizations."""
    output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING PAPER VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load data
    print("Loading metrics...")
    df = load_combined_metrics()
    print(f"✓ Loaded {len(df)} rows\n")
    
    # Generate plots
    print("Generating plots...\n")
    
    plot_hit_rate_comparison(df, output_dir)
    plot_speedup_comparison(df, output_dir)
    plot_latency_percentiles(df, output_dir)
    plot_cache_vs_miss_latency(df, output_dir)
    plot_scenario_comparison(df, output_dir)
    create_summary_figure(df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
