"""
Visualization for 6D design space exploration.

Generates exactly the same visualizations as the 10D implementation:
1. Coverage percentage over time
2. Cache hit rate progression  
3. Cumulative latency (Fig 2)
4. Cache effectiveness (Fig 3)
5. Energy comparison across all approaches
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Any, Optional
import os


from .strategy import Strategy


#define a list of 10 colors for the strategies
DEFAULT_COLORS = [
    '#e74c3c',  
    '#e67e22',  
    '#3498db',  
    '#1abc9c',  
    '#9b59b6',
    '#f1c40f',
    '#2ecc71',
    '#34495e',
    '#7f8c8d',
    '#d35400'
    
]

DEFAULT_LINESTYLES = ['-', '--', '-.', ':'] * 3  # Repeat to cover more strategies
DEFAULT_MARKERS = ['o', 's', 'D', '^'] * 3  # Repeat to cover more strategies
DEFAULT_LINEWIDTHS = [2.0, 2.5, 3.0, 3.5] * 3  # Repeat to cover more strategies
DEFAULT_MARKERSIZES = [4, 5, 6, 7] * 3  # Repeat to cover more strategies


def plot_cumulative_latency(df_events: pd.DataFrame, 
                            strategies: List[Strategy],
                            visual_info: Dict[str, Any],
                            output_dir: Path):
    """
    Figure 2: Cumulative planning cost over time.
    
    Demonstrates total computational savings of our approach.
    """
    output_path = output_dir / 'cumulative_latency.png'
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = visual_info['colors']
    linestyles = visual_info['linestyles']
    done_strategies = set()
    for strategy in [s.name for s in strategies]:
        if strategy in done_strategies:
            continue  # Already processed this strategy
        data = df_events[df_events['strategy'] == strategy].sort_values('timestamp')
        if len(data) == 0:
            continue
            
        cumulative = data['latency'].cumsum()
        
        label = strategy.replace('_', ' ').title()
        linestyle = '--' if 'linear' in strategy else '-'
        
        ax.plot(data['timestamp'], cumulative, 
               label=label, 
               color=colors[strategy], linewidth=2.5, alpha=0.8,
               linestyle=linestyles[strategy])
        
        # Add final value annotation
        if len(cumulative) > 0:
            final_val = cumulative.iloc[-1]
            ax.text(data['timestamp'].iloc[-1], final_val, 
                   f' {final_val:.0f} ms', 
                   fontsize=9, fontweight='bold', 
                   verticalalignment='center',
                   color=colors[strategy])
        done_strategies.add(strategy)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Planning Cost ', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_cache_effectiveness(df_events: pd.DataFrame, 
                             strategies: List[Strategy],
                             visual_info: Dict[str, Any],
                             output_dir: Path):
    """
    Figure 3: Cache hit effectiveness for our approach.
    
    Shows progression from exploration (full optimizations) to 
    exploitation (instant cache hits).
    """
    
    plt.style.use('seaborn-v0_8-darkgrid')
    for i, strategy in enumerate(strategies):
        if i % 2 != 0:
            continue  # Skip every second strategy (assumed baseline)
        fig, ax = plt.subplots(figsize=(10, 6))
        output_path = output_dir / f'cache_effectiveness_{strategy.name}.png'
        # Only analyze our_approach
        our_events_col = 't' if 't' in df_events.columns else 'timestamp'
        data = df_events[df_events['strategy'] == strategy.name].sort_values(our_events_col)
        
        if len(data) == 0:
            print("⚠ No data for strategy:", strategy.name)
            return
        
        # Determine cache hits from boolean field
        cache_hits = data.get('cache_hit', False).astype(int)
        full_solves = (~data.get('cache_hit', False)).astype(int)
        
        # Stacked bar chart
        x = np.arange(len(data))
        width = 1.0
        
        ax.bar(x, full_solves, width, label='Full Optimization', 
            color='#e67e22', alpha=0.8)
        ax.bar(x, cache_hits, width, bottom=full_solves, 
            label='Cache Hit (0ms)', color='#2ecc71', alpha=0.8)
        
        # Add cumulative cache hit rate line
        ax2 = ax.twinx()
        cumulative_rate = cache_hits.cumsum() / np.arange(1, len(cache_hits) + 1) * 100
        ax2.plot(x, cumulative_rate, color='#8e44ad', linewidth=3, 
                marker='o', markersize=5, label='Cache Hit Rate', alpha=0.9)
        ax2.set_ylabel('Cumulative Cache Hit Rate (%)', fontsize=12, 
                    fontweight='bold', color='#8e44ad')
        ax2.tick_params(axis='y', labelcolor='#8e44ad')
        ax2.set_ylim([0, 105])
        
        # Add horizontal line at 100%
        if len(cumulative_rate) > 0 and cumulative_rate.iloc[-1] > 90:
            ax2.axhline(y=100, color='#8e44ad', linestyle='--', 
                    linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Adaptation Event', fontsize=12, fontweight='bold')
        ax.set_ylabel('Event Type', fontsize=12, fontweight='bold')
        ax.set_title('Cache Hit Effectiveness - 6D (Our Approach)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.5])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Full Solve', 'Cache Hit'])
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                fontsize=10, loc='upper left')
        
        # Add statistics box
        total_adaptations = len(data)
        total_cache_hits = cache_hits.sum()
        cache_hit_rate = (total_cache_hits / total_adaptations * 100) if total_adaptations > 0 else 0
        
        stats_text = f'Total Adaptations: {total_adaptations}\n'
        stats_text += f'Cache Hits: {total_cache_hits}\n'
        stats_text += f'Cache Hit Rate: {cache_hit_rate:.1f}%'
        
        ax.text(0.98, 0.40, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def plot_coverage_progress(df_events: pd.DataFrame, cs_volume: float,
                           strategies: List[Strategy],
                                visual_info: Dict[str, Any],
                                  output_dir: Path):
    """
    Visualize exploration progress in 6D space.
    
    Shows the percentage of design space explored over time.

    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    
    # =========================================================================
    # Top: Coverage Percentage Over Time
    # =========================================================================
    
    # Calculate cumulative explored volume

    for i, strategy in enumerate(strategies):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        if i % 2 != 0:
            continue  # Skip every second strategy (assumed baseline)
        output_path = output_dir / f"coverage_progress_{strategy.name}.png"
        our_events = df_events[df_events['strategy'] == strategy.name].sort_values('timestamp')

        if len(our_events) > 0:
            # Calculate coverage from actual simulation data
            times = our_events['t'].values if 't' in our_events.columns else our_events['timestamp'].values
            
            # Use coverage_pct from events if available, otherwise calculate from explored_volume
            if 'coverage_pct' in our_events.columns:
                coverages = our_events['coverage_pct'].values
            elif 'explored_volume' in our_events.columns:
                # Calculate coverage from explored volume relative to CS volume
                coverages = []
                for _, row in our_events.iterrows():
                    explored_vol = row.get('explored_volume', 0)
                    coverage_pct = min(100, (explored_vol / cs_volume) * 100) if cs_volume > 0 else 0
                    coverages.append(coverage_pct)
                coverages = np.array(coverages)
            else:
                # Fallback: estimate coverage from number of events
                coverages = []
                explored_volume = 0
                for i, (idx, row) in enumerate(our_events.iterrows()):
                    # Each optimization explores some portion of space
                    if row.get('cache_hit', False) == False:  # Full optimization
                        explored_volume += cs_volume * 0.05  # Each optimization explores 5% more
                    
                    coverage_pct = min(100, (explored_volume / cs_volume) * 100)
                    coverages.append(coverage_pct)
                coverages = np.array(coverages)
                
            ax1.fill_between(times, 0, coverages, alpha=0.3, color='#2ecc71', label='Explored')
            ax1.plot(times, coverages, color='#2ecc71', linewidth=3, marker='o', markersize=6)
            
            # Add 100% reference line
            ax1.axhline(y=100, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5, label='100% Coverage')
            
            # Mark cache hits
            cache_hits = our_events[our_events.get('cache_hit', False) == True]
            if len(cache_hits) > 0:
                cache_times = cache_hits['t'].values if 't' in cache_hits.columns else cache_hits['timestamp'].values
                cache_coverages = [coverages[list(times).index(t)] if t in times else 0 for t in cache_times]
                ax1.scatter(cache_times, cache_coverages, marker='*', s=300, 
                        color='gold', edgecolors='black', linewidths=2, 
                        zorder=5, label='Cache Hit', alpha=0.9)
            
            ax1.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Cached CS Coverage (%)', fontsize=13, fontweight='bold')
            ax1.set_title('Cached Configuration Space Coverage Over Time', fontsize=15, fontweight='bold')
            ax1.legend(fontsize=11, loc='lower right')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 105])
            
            # Add statistics box
            final_coverage = coverages[-1] if len(coverages) > 0 else 0
            num_adaptations = len(our_events)
            num_cache_hits = len(cache_hits)
            cache_hit_rate = (num_cache_hits / num_adaptations * 100) if num_adaptations > 0 else 0

            stats_text = f'Final Cached Coverage: {final_coverage:.1f}%\n'
            stats_text += f'Total Adaptations: {num_adaptations}\n'
            stats_text += f'Cache Hits: {num_cache_hits} ({cache_hit_rate:.1f}%)'
        
        # =========================================================================
        # Bottom: Cache Hit Rate Progression
        # =========================================================================
        
        if len(our_events) > 0:
            # Calculate cumulative cache hit rate
            cache_hits_cumulative = (our_events.get('cache_hit', False) == True).cumsum()
            total_adaptations = np.arange(1, len(our_events) + 1)
            cache_hit_rates = (cache_hits_cumulative / total_adaptations) * 100
            
            times_col = 't' if 't' in our_events.columns else 'timestamp'
            ax2.plot(our_events[times_col].values, cache_hit_rates.values, 
                    color='#9b59b6', linewidth=3, marker='s', markersize=6)
            ax2.fill_between(our_events[times_col].values, 0, cache_hit_rates.values, 
                            alpha=0.2, color='#9b59b6')
            
            # Add 100% reference
            ax2.axhline(y=100, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5)
            
            ax2.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Cumulative Cache Hit Rate (%)', fontsize=13, fontweight='bold')
            ax2.set_title('Learning Effectiveness: Cache Hit Rate Over Time', fontsize=15, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 105])
            
            # Add annotation for first 100% cache hit rate
            if len(cache_hit_rates) > 0 and cache_hit_rates.max() >= 99:
                first_100_idx = np.where(cache_hit_rates >= 99)[0][0]
                times_col = 't' if 't' in our_events.columns else 'timestamp'
                first_100_time = our_events.iloc[first_100_idx][times_col]
                ax2.annotate(f'100% at t={first_100_time:.0f}s',
                            xy=(first_100_time, 100),
                            xytext=(first_100_time, 80),
                            fontsize=10, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='red', lw=2),
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def plot_coverage_summary(df_events: pd.DataFrame, cs_volume: float, 
                          strategies: list,
                        visual_info: Dict[str, Any],
                          output_dir: Path):
    """
    Summary visualization showing final coverage statistics. 
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    output_path = output_dir / 'coverage_summary.png'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    
    colors = visual_info['colors']
    assert len(strategies) % 2 == 0, "This analysis assumes exactly two strategies: cache and baseline."
    strategies_pairs = [(strategies[i], strategies[i+1]) for i in range(0, len(strategies), 2)]
    data_to_plot = []
    labels = []
    colors_list = []
    hatches = []

    done_strategies = set()

    for s0, s1 in strategies_pairs:
        our_events = df_events[df_events['strategy'] == s0.name]
        exact_events = df_events[df_events['strategy'] == s1.name]
        # =========================================================================
        # 1. Latency Comparison
        # =========================================================================
        cache_hits = our_events[our_events.get('cache_hit', False) == True]
        regular_opt = our_events[our_events.get('cache_hit', False) == False]
        should_add_our = False
        if len(exact_events) > 0 and s1.name not in done_strategies:
            data_to_plot.append(exact_events['latency'].values)
            labels.append(s1.name.replace('_', '\n').title())
            colors_list.append(colors.get(s1.name, '#95a5a6'))
            hatches.append('')
            done_strategies.add(s1.name)
        if len(regular_opt) > 0 and s0.name not in done_strategies:
            data_to_plot.append(regular_opt['latency'].values)
            labels.append(s0.name.replace('_', '\n').title() + '\n(Full Opt)')
            colors_list.append(colors.get(s0.name, '#95a5a6'))
            hatches.append('')
            should_add_our = True
        if len(cache_hits) > 0 and s0.name not in done_strategies:
            data_to_plot.append(cache_hits['latency'].values)
            labels.append(s0.name.replace('_', '\n').title() + '\n(Cache Hit)')
            colors_list.append(colors.get(s0.name, '#95a5a6'))
            hatches.append('**')
            
            #we also sum the cache hits to the regular_opt for better comparison

            data_to_plot.append([a + b for a, b in zip(cache_hits['latency'].values, regular_opt['latency'].values)])
            labels.append(s0.name.replace('_', '\n').title() + '\n(Full + Cache)')
            colors_list.append(colors.get(s0.name, '#95a5a6'))
            hatches.append('//')
            should_add_our = True
        if should_add_our:
            done_strategies.add(s0.name)

    # rotate names 45 % for better visibility
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if data_to_plot:
        bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
        for i, (patch, color) in enumerate(zip(bp['boxes'], colors_list)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            if hatches[i]:
                patch.set_hatch(hatches[i])

            
            

        ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Latency Distribution by Strategy', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # =========================================================================
    # 2. Energy Distribution
    # =========================================================================
    
    
    energy_data = []
    energy_labels = []
    energy_colors = []
    
    for strategy, color in zip(strategies, colors):
        strategy_events = df_events[df_events['strategy'] == strategy.name]
        if len(strategy_events) > 0:
            energies = [e for e in strategy_events['energy'] if e != float('inf')]
            if energies:
                energy_data.append(energies)
                energy_labels.append(strategy.name.replace('_', '\n').title())
                energy_colors.append(colors.get(strategy.name, '#95a5a6'))
    
    if energy_data:
        bp2 = ax2.boxplot(energy_data, labels=energy_labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp2['boxes'], energy_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
        ax2.set_title('Energy Distribution by Strategy', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {output_path}")

# Remove pair assumptions from all visualization functions
# def plot_coverage_summary(df_events: pd.DataFrame, cs_volume: float, 
#                          strategies: list,
#                          visual_info: Dict[str, Any],
#                          output_dir: Path):
#     """Summary visualization showing final coverage statistics."""
#     plt.style.use('seaborn-v0_8-darkgrid')
#     output_path = output_dir / 'coverage_summary.png'
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
#     colors = visual_info['colors']
#     data_to_plot = []
#     labels = []
#     colors_list = []
#     hatches = []
#     done_strategies = set()

#     # Process each strategy individually
#     for strategy in strategies:
#         if strategy.name in done_strategies:
#             continue
            
#         strategy_events = df_events[df_events['strategy'] == strategy.name]
#         if len(strategy_events) == 0:
#             continue

#         # Handle cache-based strategies differently
#         if hasattr(strategy, 'cache'):
#             cache_hits = strategy_events[strategy_events.get('cache_hit', False) == True]
#             regular_opt = strategy_events[strategy_events.get('cache_hit', False) == False]
            
#             if len(regular_opt) > 0:
#                 data_to_plot.append(regular_opt['latency'].values)
#                 labels.append(f"{strategy.name}\n(Full Opt)")
#                 colors_list.append(colors[strategy.name])
#                 hatches.append('')
            
#             if len(cache_hits) > 0:
#                 data_to_plot.append(cache_hits['latency'].values)
#                 labels.append(f"{strategy.name}\n(Cache Hit)")
#                 colors_list.append(colors[strategy.name])
#                 hatches.append('**')
                
#                 # Combined stats
#                 data_to_plot.append(np.concatenate([regular_opt['latency'].values, 
#                                                   cache_hits['latency'].values]))
#                 labels.append(f"{strategy.name}\n(Full + Cache)")
#                 colors_list.append(colors[strategy.name])
#                 hatches.append('//')
#         else:
#             # Non-cache strategies (Linear, Bayesian, etc.)
#             data_to_plot.append(strategy_events['latency'].values)
#             labels.append(strategy.name.replace('_', '\n').title())
#             colors_list.append(colors[strategy.name])
#             hatches.append('')
            
#         done_strategies.add(strategy.name)
    
#     # rotate names 45 % for better visibility
#     plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
#     if data_to_plot:
#         bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
#         for i, (patch, color) in enumerate(zip(bp['boxes'], colors_list)):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.7)
#             if hatches[i]:
#                 patch.set_hatch(hatches[i])

            
            

#         ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
#         ax1.set_title('Latency Distribution by Strategy', fontsize=13, fontweight='bold')
#         ax1.grid(True, alpha=0.3)
#         ax1.set_yscale('log')
    
#     # =========================================================================
#     # 2. Energy Distribution
#     # =========================================================================
    
    
#     energy_data = []
#     energy_labels = []
#     energy_colors = []
    
#     for strategy, color in zip(strategies, colors):
#         strategy_events = df_events[df_events['strategy'] == strategy.name]
#         if len(strategy_events) > 0:
#             energies = [e for e in strategy_events['energy'] if e != float('inf')]
#             if energies:
#                 energy_data.append(energies)
#                 energy_labels.append(strategy.name.replace('_', '\n').title())
#                 energy_colors.append(colors.get(strategy.name, '#95a5a6'))
    
#     if energy_data:
#         bp2 = ax2.boxplot(energy_data, labels=energy_labels, patch_artist=True, showmeans=True)
#         for patch, color in zip(bp2['boxes'], energy_colors):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.7)
        
#         ax2.set_ylabel('Energy (Wh)', fontsize=12, fontweight='bold')
#         ax2.set_title('Energy Distribution by Strategy', fontsize=13, fontweight='bold')
#         ax2.grid(True, alpha=0.3)
    

#     plt.tight_layout()
#     fig.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f" Saved: {output_path}")

def plot_energy_comparison(optimal_configs_path: Path, 
                           strategies: List[Strategy], 
                           visual_info: Dict[str, Dict[str, str]],
                           output_dir: Path):
    """
    Create a dedicated energy comparison plot for all approaches.
    
    """
    
    
    # Load optimal configurations
    with open(optimal_configs_path, 'r') as f:
        optimal_configs = json.load(f)
    


    # Create the energy comparison plot
    plt.style.use('seaborn-v0_8-darkgrid')

    colors = visual_info.get('colors', {})
    linestyles = visual_info.get('linestyles', {})
    markers = visual_info.get('markers', {})
    linewidths = visual_info.get('linewidths', {})
    markersizes = visual_info.get('markersizes', {})
    
    
    strategy_labels = {s.name: s.name.replace('_', ' ').title() for s in strategies}
    
    

    assert len(strategies) % 2 == 0, "This analysis assumes exactly two strategies: cache and baseline."
    strategies_pairs = [(strategies[i], strategies[i+1]) for i in range(0, len(strategies), 2)]

    for s0, s1 in strategies_pairs:
        strategies_with_data = []
        fig, ax = plt.subplots(figsize=(12, 8))
        energy_output_path = output_dir / f"energy_comparison_{s0.name}_{s1.name}.png"

        if s0.name not in optimal_configs or len(optimal_configs[s0.name]) == 0:
            print(f"  Warning: No data for {s0.name}")
            continue
        if s1.name not in optimal_configs or len(optimal_configs[s1.name]) == 0:
            print(f"  Warning: No data for {s1.name}")
            continue

        for strategy in [s0.name, s1.name]:

            # Extract energy values over time
            times = [entry['t'] for entry in optimal_configs[strategy]]
            energies = [entry['energy'] for entry in optimal_configs[strategy]]
            
            if not times or not energies:
                print(f"  Warning: Empty times/energies for {strategy}")
                continue
            
            # Plot energy evolution
            alpha = 0.9

            
            label = strategy_labels[strategy]
            
            ax.plot(times, energies, 
                label=label,
                color=colors[strategy], 
                linewidth=linewidths[strategy], 
                alpha=alpha,
                linestyle=linestyles[strategy],
                marker= markers[strategy],
                markersize=markersizes[strategy],
                markeredgewidth=1,
                markeredgecolor='white')
            
            strategies_with_data.append(strategy)
            
            # Add statistics annotation
            avg_energy = np.mean(energies)
            min_energy = min(energies)
            max_energy = max(energies)
        
            print(f"  {strategy}: {len(energies)} points, avg={avg_energy:.2f} Wh, range=[{min_energy:.2f}, {max_energy:.2f}] Wh")
    
            # Customize the plot
            ax.set_title('Energy Consumption Comparison Across All Approaches', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Energy Consumption (Wh)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(labelsize=12)
            
            # Add legend with better positioning
            legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                            fancybox=True, shadow=True, borderpad=1)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            
            # Add average energy lines for comparison
        for strategy in strategies_with_data:
                energies = [entry['energy'] for entry in optimal_configs[strategy]]
                avg_energy = np.mean(energies)
                ax.axhline(y=avg_energy, color=colors[strategy], 
                        linestyle=':', alpha=0.6, linewidth=2)
                print("avg_energy:" , avg_energy)
                # Add average value text
                ax.text(0.02, avg_energy, f'Avg: {avg_energy:.2f} Wh', 
                    transform=ax.get_yaxis_transform(),
                    color=colors[strategy], fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Add summary statistics box
        if strategies_with_data:
                stats_text = "Summary Statistics:\n"
                for strategy in strategies_with_data:
                    energies = [entry['energy'] for entry in optimal_configs[strategy]]
                    avg_energy = np.mean(energies)
                    std_energy = np.std(energies)
                    label = strategy_labels[strategy]
                    stats_text += f"{label}: {avg_energy:.2f}±{std_energy:.2f} Wh\n"
                
                ax.text(0.98, 0.02, stats_text.strip(), 
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
        plt.tight_layout()
        fig.savefig(energy_output_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.clear()
        print(f" Energy comparison saved: {energy_output_path}")
    return energy_output_path

# def plot_energy_comparison(optimal_configs_path: Path, 
#                          strategies: List[Strategy], 
#                          visual_info: Dict[str, Dict[str, str]],
#                          output_dir: Path):
#     """Create energy comparison plot for all approaches."""
#     with open(optimal_configs_path, 'r') as f:
#         optimal_configs = json.load(f)

#     plt.style.use('seaborn-v0_8-darkgrid')
#     fig, ax = plt.subplots(figsize=(12, 8))
#     energy_output_path = output_dir / "energy_comparison_all.png"

#     colors = visual_info.get('colors', {})
#     linestyles = visual_info.get('linestyles', {})
#     markers = visual_info.get('markers', {})
#     linewidths = visual_info.get('linewidths', {})
#     markersizes = visual_info.get('markersizes', {})

#     strategy_labels = {s.name: s.name.replace('_', ' ').title() for s in strategies}

#     strategies_with_data = []
    
#     for strategy in strategies:
#         if strategy.name not in optimal_configs or len(optimal_configs[strategy.name]) == 0:
#             print(f"  Warning: No data for {strategy.name}")
#             continue

#         # Extract energy values over time
#         times = [entry['t'] for entry in optimal_configs[strategy.name]]
#         energies = [entry['energy'] for entry in optimal_configs[strategy.name]]
        
#         if not times or not energies:
#             continue
        
#         label = strategy.name.replace('_', ' ').title()
#         ax.plot(times, energies,
#                 label=label,
#                 color=colors[strategy.name],
#                 linewidth=linewidths[strategy.name],
#                 linestyle=linestyles[strategy.name],
#                 marker=markers[strategy.name],
#                 markersize=markersizes[strategy.name],
#                 alpha=0.9)
        
#         strategies_with_data.append(strategy.name)

#         # Add statistics annotation
#         avg_energy = np.mean(energies)
#         min_energy = min(energies)
#         max_energy = max(energies)
    
#         print(f"  {strategy}: {len(energies)} points, avg={avg_energy:.2f} Wh, range=[{min_energy:.2f}, {max_energy:.2f}] Wh")

#         # Customize the plot
#         ax.set_title('Energy Consumption Comparison Across All Approaches', 
#                     fontsize=16, fontweight='bold', pad=20)
#         ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
#         ax.set_ylabel('Energy Consumption (Wh)', fontsize=14, fontweight='bold')
#         ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
#         ax.tick_params(labelsize=12)
        
#         # Add legend with better positioning
#         legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
#                         fancybox=True, shadow=True, borderpad=1)
#         legend.get_frame().set_facecolor('white')
#         legend.get_frame().set_alpha(0.95)
            
#             # Add average energy lines for comparison
#     for strategy in strategies_with_data:
#             energies = [entry['energy'] for entry in optimal_configs[strategy]]
#             avg_energy = np.mean(energies)
#             ax.axhline(y=avg_energy, color=colors[strategy], 
#                     linestyle=':', alpha=0.6, linewidth=2)
            
#             # Add average value text
#             ax.text(0.02, avg_energy + 0.1, f'Avg: {avg_energy:.2f} Wh', 
#                 transform=ax.get_yaxis_transform(),
#                 color=colors[strategy], fontsize=10, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
#             # Add summary statistics box
#     if strategies_with_data:
#             stats_text = "Summary Statistics:\n"
#             for strategy in strategies_with_data:
#                 energies = [entry['energy'] for entry in optimal_configs[strategy]]
#                 avg_energy = np.mean(energies)
#                 std_energy = np.std(energies)
#                 label = strategy_labels[strategy]
#                 stats_text += f"{label}: {avg_energy:.2f}±{std_energy:.2f} Wh\n"
            
#             ax.text(0.98, 0.02, stats_text.strip(), 
#                 transform=ax.transAxes, fontsize=10,
#                 verticalalignment='bottom', horizontalalignment='right',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
#     plt.tight_layout()
#     fig.savefig(energy_output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     fig.clear()
#     print(f" Energy comparison saved: {energy_output_path}")
#     return energy_output_path

def visualize_optimal_configs(
        optimal_configs_path: Path, 
        strategies: List[Strategy], 
        visual_info: Dict[str, dict],
        output_dir: Path):
    """
    Visualize optimal configurations found by different approaches (6D version).
    """
    print("Creating optimal configuration comparison...")
    
    # Load optimal configurations
    with open(optimal_configs_path, 'r') as f:
        optimal_configs = json.load(f)
    
    assert len(strategies) % 2 == 0, "This analysis assumes exactly two strategies: cache and baseline."
    strategies_pairs = [(strategies[i], strategies[i+1]) for i in range(0, len(strategies), 2)]

    for (cache_strategy, baseline_strategy) in strategies_pairs:
        output_path = output_dir / f"optimal_configs_{cache_strategy.name}.png"
        #we assume all strategies to have the same parameter names
        params = [p.name for p in cache_strategy.domain_spec.parameters]  # List of 6 parameter names

        # Create visualization (3x2 grid for 6 parameters)
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()
        
        colors = visual_info.get('colors', {})
        linestyles = visual_info.get('linestyles', {})
        markers = visual_info.get('markers', {})
        linewidths = visual_info.get('linewidths', {})
        markersizes = visual_info.get('markersizes', {})
        
        
        for i, param in enumerate(params):
            ax = axes[i]

            for strategy_name in [cache_strategy.name, baseline_strategy.name]:
                if strategy_name not in optimal_configs or len(optimal_configs[strategy_name]) == 0:
                    print(f" Plotting optimal configs: Warning: No data for {strategy_name}")
                    continue
                    
                # Extract parameter values over time
                times = [entry['t'] for entry in optimal_configs[strategy_name]]
                values = [entry['config'].get(param, 0) for entry in optimal_configs[strategy_name]]

                if not times or not values:
                    continue
                
                #linestyle = '--' if 'linear' in strategy else '-'
                #linewidth = 2.5 if 'linear' in strategy else 2.0
                #marker = 'o' if 'linear' in strategy else 's'
                #markersize = 4 if 'linear' in strategy else 3
                
                ax.plot(times, values, 
                    label=strategy_name.replace('_', ' ').title(),
                    color=colors[strategy_name],
                    linewidth=linewidths.get(strategy_name, 2.5),
                    linestyle=linestyles.get(strategy_name, '-'),
                    marker=markers.get(strategy_name, 'o'),
                    markersize=markersizes.get(strategy_name, 4),
                    alpha=0.8)
            
            ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{param.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_title(f'Optimal {param.replace("_", " ").title()} Over Time', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        # Save
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f" Saved: {output_path}")
        
        # Create summary statistics
        print("\nOptimal Configuration Summary:")
        print("="*80)
        
        for strategy in strategies:
            if strategy not in optimal_configs or len(optimal_configs[strategy]) == 0:
                continue
                
            configs = optimal_configs[strategy]
            energies = [entry['energy'] for entry in configs]
            
            print(f"\n{strategy.replace('_', ' ').title()}:")
            print(f"  Configurations found: {len(configs)}")
            print(f"  Average energy: {np.mean(energies):.2f} Wh")
            print(f"  Energy range: {min(energies):.2f} - {max(energies):.2f} Wh")
            
            # Show most recent configuration
            if configs:
                latest = configs[-1]['config']
                print(f"  Latest configuration:")
                for param in params:
                    print(f"    {param}: {latest.get(param, 0):.2f}")
        
        print("="*80)

# def visualize_optimal_configs(optimal_configs_path: Path, 
#                             strategies: List[Strategy], 
#                             visual_info: Dict[str, dict],
#                             output_dir: Path):
#     """Visualize optimal configurations found by different approaches."""
#     with open(optimal_configs_path, 'r') as f:
#         optimal_configs = json.load(f)

#     # Create single plot showing all strategies
#     output_path = output_dir / "optimal_configs_all.png"
#     params = [p.name for p in strategies[0].domain_spec.parameters]
    
#     plt.style.use('seaborn-v0_8-darkgrid')
#     fig, axes = plt.subplots(3, 2, figsize=(16, 18))
#     axes = axes.flatten()
    
#     colors = visual_info.get('colors', {})
#     linestyles = visual_info.get('linestyles', {})
#     markers = visual_info.get('markers', {})
#     linewidths = visual_info.get('linewidths', {})
#     markersizes = visual_info.get('markersizes', {})
    
#     for i, param in enumerate(params):
#         ax = axes[i]
        
#         for strategy in strategies:
#             if strategy.name not in optimal_configs:
#                 continue
                
#             times = [entry['t'] for entry in optimal_configs[strategy.name]]
#             values = [entry['config'].get(param, 0) for entry in optimal_configs[strategy.name]]
            
#             if not times or not values:
#                 continue
                
#             ax.plot(times, values,
#                    label=strategy.name.replace('_', ' ').title(),
#                    color=colors[strategy.name],
#                    linewidth=linewidths[strategy.name],
#                    linestyle=linestyles[strategy.name],
#                    marker=markers[strategy.name],
#                    markersize=markersizes[strategy.name],
#                    alpha=0.8)
#         ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
#         ax.set_ylabel(f'{param.replace("_", " ").title()}', fontsize=12, fontweight='bold')
#         ax.set_title(f'Optimal {param.replace("_", " ").title()} Over Time', fontsize=13, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=9)
        
#     plt.tight_layout()
    
#     # Save
    
#     fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     print(f" Saved: {output_path}")
    
#     # Create summary statistics
#     print("\nOptimal Configuration Summary:")
#     print("="*80)
    
#     for strategy in strategies:
#         if strategy not in optimal_configs or len(optimal_configs[strategy]) == 0:
#             continue
            
#         configs = optimal_configs[strategy]
#         energies = [entry['energy'] for entry in configs]
        
#         print(f"\n{strategy.replace('_', ' ').title()}:")
#         print(f"  Configurations found: {len(configs)}")
#         print(f"  Average energy: {np.mean(energies):.2f} Wh")
#         print(f"  Energy range: {min(energies):.2f} - {max(energies):.2f} Wh")
        
#         # Show most recent configuration
#         if configs:
#             latest = configs[-1]['config']
#             print(f"  Latest configuration:")
#             for param in params:
#                 print(f"    {param}: {latest.get(param, 0):.2f}")
    
#     print("="*80)
    

def __generate_visualizations(df_events: pd.DataFrame, 
                               cs_volume: float, 
                               strategies: List[Strategy],
                                visual_info: Dict[str, Any],
                               output_dir: Path):
    """Generate all 6D-specific visualizations."""
    print("\n" + "="*80)
    print("GENERATING 6D VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all figures
    plot_coverage_progress(df_events, cs_volume, strategies, visual_info, output_dir)
    plot_coverage_summary(df_events, cs_volume, strategies, visual_info, output_dir)
    plot_cumulative_latency(df_events, strategies, visual_info, output_dir)
    plot_cache_effectiveness(df_events, strategies, visual_info, output_dir)

    print("\n" + "="*80)
    print("6D VISUALIZATION COMPLETE!")
    print("="*80)
    print(f" Generated visualizations in: {output_dir}")
    print("   1. coverage_progress.png - Exploration progress over time")
    print("   2. coverage_summary.png - Summary statistics")
    print("   3. cumulative_latency.png - Fig 2: Cumulative planning cost")
    print("   4. cache_effectiveness.png - Fig 3: Cache hit progression")
    print("   5. energy_comparison_all_approaches.png - Energy comparison")
    print("="*80 + "\n")




def get_visual_info(strategies: List[Strategy]) -> Dict[str, Dict[str, str]]:
    """Generate visual info dictionary for strategies."""
    colors = {}
    linestyles = {}
    markers = {}
    linewidths = {}
    markersizes = {}
    for strategy in strategies:
        # Assign colors based on strategy attributes or randomly
        colors[strategy.name] = strategy.color if hasattr(strategy, 'color') else DEFAULT_COLORS[len(colors) % len(DEFAULT_COLORS)]
        linestyles[strategy.name] = strategy.linestyle if hasattr(strategy, 'linestyle') else DEFAULT_LINESTYLES[len(linestyles) % len(DEFAULT_LINESTYLES)]
        linewidths[strategy.name] = strategy.linewidth if hasattr(strategy, 'linewidth') else DEFAULT_LINEWIDTHS[len(linewidths) % len(DEFAULT_LINEWIDTHS)]
        markers[strategy.name] = strategy.marker if hasattr(strategy, 'marker') else DEFAULT_MARKERS[len(markers) % len(DEFAULT_MARKERS)]
        markersizes[strategy.name] = strategy.markersize if hasattr(strategy, 'markersize') else DEFAULT_MARKERSIZES[len(markersizes) % len(DEFAULT_MARKERSIZES)]

    visual_info = {
        'colors': colors,
        'linestyles': linestyles,
        'markers': markers,
        'linewidths': linewidths,
        'markersizes': markersizes
    }
    return visual_info







def generate_visualizations(results: Dict[str, Any],
                            strategies: List[Strategy],
                             output_dir: Path):
    """
    Main entry point for generating all visualizations.
    
    Args:
        results: Dictionary containing:
            - events: List of event dictionaries
            - optimal_configs: Dictionary of optimal configurations by strategy
            - cs_volume: Configuration space volume
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert events to DataFrame
    if 'events' in results and results['events']:
        df_events = pd.DataFrame(results['events'])
    else:
        print("Warning: No events data found in results")
        df_events = pd.DataFrame()
    
    # Get CS volume
    cs_volume = results.get('cs_volume', 1.0)
    
    ## Define colors for strategies names 
    #colors = {}
    #linestyles = {}
    #markers = {}
    #linewidths = {}
    #markersizes = {}
    #for strategy in strategies:
    #    # Assign colors based on strategy attributes or randomly
    #    colors[strategy.name] = strategy.color if hasattr(strategy, 'color') else DEFAULT_COLORS[len(colors) % len(DEFAULT_COLORS)]
    #    linestyles[strategy.name] = strategy.linestyle if hasattr(strategy, 'linestyle') else DEFAULT_LINESTYLES[len(linestyles) % len(DEFAULT_LINESTYLES)]
    #    linewidths[strategy.name] = strategy.linewidth if hasattr(strategy, 'linewidth') else DEFAULT_LINEWIDTHS[len(linewidths) % len(DEFAULT_LINEWIDTHS)]
    #    markers[strategy.name] = strategy.marker if hasattr(strategy, 'marker') else DEFAULT_MARKERS[len(markers) % len(DEFAULT_MARKERS)]
    #    markersizes[strategy.name] = strategy.markersize if hasattr(strategy, 'markersize') else DEFAULT_MARKERSIZES[len(markersizes) % len(DEFAULT_MARKERSIZES)]
#
    #visual_info = {
    #    'colors': colors,
    #    'linestyles': linestyles,
    #    'markers': markers,
    #    'linewidths': linewidths,
    #    'markersizes': markersizes
    #}
    visual_info = get_visual_info(strategies)
    # Generate all visualizations
    if not df_events.empty:
        __generate_visualizations(df_events, 
                                   cs_volume, 
                                   strategies,
                                   visual_info,
                                   output_dir)
    
    # Generate optimal config visualizations if data is available
    if 'optimal_configs' in results:
        # Save optimal configs to temporary file
        optimal_configs_path = output_dir / 'optimal_configs.json'
        with open(optimal_configs_path, 'w') as f:
            json.dump(results['optimal_configs'], f, indent=2)
        
        # Generate visualizations
        visualize_optimal_configs(optimal_configs_path, 
                                     strategies,
                                     visual_info,
                                       output_dir)
        plot_energy_comparison(optimal_configs_path, 
                               strategies, 
                               visual_info,
                                output_dir)


def save_results_csv(results: Dict[str, Any], output_dir: Path):
    """Save results data to CSV for further analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'events' not in results or not results['events']:
        print("No results data to save!")
        return
    
    df = pd.DataFrame(results['events'])
    csv_path = output_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Results CSV saved to: {csv_path}")
