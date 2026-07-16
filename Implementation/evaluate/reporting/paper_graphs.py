import argparse, json, os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from evaluate.implementation_specifics.strategies import *
from evaluate.implementation_specifics.domain_specific import Fleet, Drone

from src.visualization import  get_visual_info

from paths import ASSETS_DIR, CACHE_RESULTS_DIR, REPORTS_DIR, FIGURES_DIR

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



# Set hatch line width globally
# This makes the hatch lines thinner and less obtrusive
# Adjust as needed for visibility

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1  
#mpl.rcParams['hatch.fill_between'] = 0.3
#print(mpl.rcParams.keys())



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



def plot_coverage_progress(df_events: pd.DataFrame, cs_volume: float,
                           strategies: List[Strategy],
                                visual_info: Dict[str, Any],
                                  output_dir: Path,
                                  extension: str = 'png',
                          fig_size: tuple = (12, 10)):
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


        # do it in two diffrent plots
        fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
        output_path = output_dir / f"top_coverage_progress.{extension}"
        if i % 2 != 0:
            continue  # Skip every second strategy (assumed baseline)
        
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
                
            
            ax1.plot(times, coverages, color='#2ecc71', linewidth=1, marker='', markersize=2, label='Coverage')
            ax1.fill_between(times, 0, coverages, alpha=0.3, color='#2ecc71', label='Explored')
            # Add 100% reference line
            
            
            # Mark cache hits
            cache_hits = our_events[our_events.get('cache_hit', False) == True]
            if len(cache_hits) > 0:
                cache_times = cache_hits['t'].values if 't' in cache_hits.columns else cache_hits['timestamp'].values
                cache_coverages = [coverages[list(times).index(t)] if t in times else 0 for t in cache_times]
                ax1.scatter(cache_times, cache_coverages, marker='o', s=5, 
                        color='#2ecc71',linewidths=1, 
                        zorder=5, label='Cache Hit', alpha=0.9)
            ax1.axhline(y=100, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5, label='100% Coverage')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Cached CS Coverage (%)')
            ax1.set_title('Cached Configuration Space Coverage Over Time')
            # Add legend upper left, but a little bit shift below
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
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
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
        fig, ax2 = plt.subplots(1, 1, figsize=fig_size)
        output_path = output_dir / f"bottom_coverage_progress.{extension}"
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
                    color='#9b59b6', linewidth=1, marker='', markersize=2, label='Cache Hit Rate')
            ax2.fill_between(our_events[times_col].values, 0, cache_hit_rates.values, 
                            alpha=0.2, color='#9b59b6')
            
            # Add 100% reference
            ax2.axhline(y=100, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5, label='100% Hit Rate')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Cumulative Cache Hit Rate (%)')
            ax2.set_title('Learning Effectiveness: Cache Hit Rate Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 105])

            ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
            
            # Add annotation for first 100% cache hit rate
            if len(cache_hit_rates) > 0 and cache_hit_rates.max() >= 99:
                first_100_idx = np.where(cache_hit_rates >= 99)[0][0]
                times_col = 't' if 't' in our_events.columns else 'timestamp'
                first_100_time = our_events.iloc[first_100_idx][times_col]
                ax2.annotate(f'100% at t={first_100_time:.0f}s',
                            xy=(first_100_time, 100),
                            xytext=(first_100_time, 80),
    
                            arrowprops=dict(arrowstyle='->', color='red', lw=2),
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")
        break  # Only plot for the first strategy (assumed cache version)










def plot_coverage_summary(df_events: pd.DataFrame, cs_volume: float,
                          strategies: list,
                          visual_info: Dict[str, Any],
                          output_dir: Path,
                          extension: str = 'png',
                          fig_size: tuple = (6.5, 3.5)):
    """
    Generate two separate summary visualizations:
    1. Top (bar plot) -> Mean latency comparison (baseline, cache miss, cache hit, combined)
    2. Bottom (boxplot) -> Energy distribution by strategy
    """

    plt.style.use('seaborn-v0_8-darkgrid')
    colors = visual_info['colors']

    # ======================================================================
    # TOP PLOT: Bar chart of mean latency values
    # ======================================================================
    output_path_top = output_dir / f'luca_top_coverage_summary.{extension}'

    assert len(strategies) % 2 == 0, "Expected pairs: cache and baseline."
    strategies_pairs = [(strategies[i], strategies[i + 1]) for i in range(0, len(strategies), 2)]

    latency_means = []
    latency_labels = []
    latency_colors = []

    for s0, s1 in strategies_pairs:
        # baseline
        baseline = df_events[df_events['strategy'] == s1.name]['latency']
        if len(baseline) > 0:
            latency_means.append(baseline.mean())
            latency_labels.append(s1.name.replace('_', '\n').title())
            latency_colors.append(colors.get(s1.name, '#95a5a6'))

        # cache miss and hit
        our_events = df_events[df_events['strategy'] == s0.name]
        cache_hits = our_events[our_events.get('cache_hit', False) == True]
        cache_miss = our_events[our_events.get('cache_hit', False) == False]

        if len(cache_miss) > 0:
            latency_means.append(cache_miss['latency'].mean())
            latency_labels.append(s0.name.replace('_', '\n').title() + '\n(Cache Miss)')
            latency_colors.append(colors.get(s0.name, '#95a5a6'))

        if len(cache_hits) > 0:
            latency_means.append(cache_hits['latency'].mean())
            latency_labels.append(s0.name.replace('_', '\n').title() + '\n(Cache Hit)')
            latency_colors.append(colors.get(s0.name, '#7f8c8d'))

        if len(cache_hits) > 0 and len(cache_miss) > 0:
            combined = pd.concat([cache_hits['latency'], cache_miss['latency']])
            latency_means.append(combined.mean())
            latency_labels.append(s0.name.replace('_', '\n').title() + '\n(Combined)')
            latency_colors.append(colors.get(s0.name, '#34495e'))

    # Plot mean latency bar chart
    fig_top, ax_top = plt.subplots(figsize=fig_size)
    ax_top.bar(latency_labels, latency_means, color=latency_colors, alpha=0.8, edgecolor='black')
    ax_top.set_ylabel('Mean Latency (ms)')
    ax_top.set_title('Mean Latency by Strategy')
    #ax_top.set_yscale('log')
    plt.setp(ax_top.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    fig_top.savefig(output_path_top, dpi=300, bbox_inches='tight')
    plt.close(fig_top)
    print(f" Saved: {output_path_top}")

    # ======================================================================
    # BOTTOM PLOT: Energy distribution boxplot
    # ======================================================================
    output_path_bottom = output_dir / f'bottom_coverage_summary.{extension}'
    fig_bottom, ax_bottom = plt.subplots(figsize=fig_size)

    bar_means = []
    bar_labels = []
    bar_colors = []
    bar_errors = []


    for p in range(0, len(strategies), 2):
        c = strategies[p].name
        nc = strategies[p + 1].name
        cached = df_events[df_events['strategy'] == c]['energy']
        uncached = df_events[df_events['strategy'] == nc]['energy']
        
        if len(cached) == 0 or len(uncached) == 0:
            continue

        # Boxplot data
        cached_mean = cached.mean()
        uncached_mean = uncached.mean()
        diff = abs(uncached_mean - cached_mean)  # difference to uncached

        bar_means.append(cached_mean)
        bar_labels.append(labels_to_latex[c.replace('_', ' ')])
        bar_colors.append(strategies[p].color)
        bar_errors.append(diff)

    x = np.arange(len(bar_means))
        # --- Plot boxplots ---
    bars = ax_bottom.bar(
        x, bar_means, yerr=bar_errors, capsize=6,
        color=bar_colors, alpha=0.8, edgecolor='black'
    )

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(bar_labels, rotation=45, ha='right')
    #ax_bottom.set_ylabel('Energy (Wh)', fontsize=10, fontweight='bold')
    #ax_bottom.set_title('Cached Energy vs. Uncached Difference', fontsize=11, fontweight='bold')
    ax_bottom.grid(axis='y', alpha=0.3)

    # --- Annotate differences on bars (optional) ---
    for i, (mean, diff) in enumerate(zip(bar_means, bar_errors)):
        ax_bottom.text(i, mean + np.sign(diff)*abs(diff)/2, f"{diff:+.1f}", 
                    ha='center', va='bottom' if diff>0 else 'top', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig_bottom.savefig(output_path_bottom, dpi=300, bbox_inches='tight')
    plt.close(fig_bottom)
    print(f" Saved: {output_path_bottom}")



def plot_latency_summary(df_events, strategies, visual_info, output_dir, extension='png', fig_size=(6.5, 3.5)):
    """
    Creates the top latency summary bar chart with:
    - one color per strategy pair
    - hatch patterns distinguishing cache info
    - cache hit split into zero-latency and >0 latency
    """

    plt.style.use('seaborn-v0_8-darkgrid')
    colors = visual_info['colors']
    output_path_top = output_dir / f'luca_top_coverage_summary.{extension}'

    assert len(strategies) % 2 == 0, "Expected pairs: cache and baseline."
    strategies_pairs = [(strategies[i], strategies[i + 1]) for i in range(0, len(strategies), 2)]

    bar_width = 0.6
    x_positions = np.arange(len(strategies_pairs)*3)
    fig, ax = plt.subplots(figsize=fig_size)

    # define hatch patterns (same across strategies)
    hatch_map = {
        'Miss': '\\\\\\\\',  # cache misses
        'Extension': '-----',  # cache hits with latency > 0
        'Total Performance': '',  # performance overhead (not used here)
        #'cache_hit_zero': ''  # cache hits with zero latency
    }

    

    i=0
    labels = []
    for (s_cache, s_base) in strategies_pairs:

        base_color = colors.get(s_base.name, '#95a5a6')

        # baseline mean
        base_mean = df_events.loc[df_events['strategy'] == s_base.name, 'latency'].mean()

        # cache data
        cache_df = df_events[df_events['strategy'] == s_cache.name]
        cache_miss = cache_df[cache_df.get('cache_hit', False) == False]
        cache_hit = cache_df[cache_df.get('cache_hit', False) == True]

        cache_hit_zero = cache_hit[cache_hit['latency'] == 0]
        cache_hit_ext = cache_hit[cache_hit['latency'] > 0]

        miss_mean = cache_miss['latency'].mean() if not cache_miss.empty else 0
        hit_zero_mean = cache_hit_zero['latency'].mean() if not cache_hit_zero.empty else 0
        hit_ext_mean = cache_hit_ext['latency'].mean() if not cache_hit_ext.empty else 0
        combined_mean = pd.concat([cache_hit['latency'], cache_miss['latency']]).mean() if len(cache_hit) + len(cache_miss) > 0 else 0

        # Plot baseline
        ax.bar(x_positions[i], base_mean, width=bar_width,
               color=base_color, edgecolor='black',
               alpha=0.8, label=None)
        labels += [labels_to_latex[s_base.name.replace('_', ' ')]]
        i+=1

        # Plot cache miss
        ax.bar(x_positions[i], miss_mean, width=bar_width,
               color=base_color, edgecolor='black', hatch=hatch_map['Miss'],
               alpha=0.8)
        i+=1
        labels += [labels_to_latex[s_base.name.replace('_', ' ')] + " (Miss)"]
        # Plot cache hit (split stacked: zero + extension)
        print(hit_ext_mean)
        if hit_zero_mean > 0 or hit_ext_mean > 0:
            #       color=base_color, edgecolor='black', hatch=hatch_map['cache_hit_zero'],
            #ax.bar(x_positions[i], hit_zero_mean, width=bar_width,
            #       bottom=miss_mean, alpha=0.8)
#
            ax.bar(x_positions[i], hit_ext_mean, width=bar_width,
                   color=base_color, edgecolor='black', hatch=hatch_map['Extension'], alpha=0.8)

        ## Plot combined (outline bar)
        if combined_mean > 0:
            ax.bar(x_positions[i], combined_mean, width=bar_width,
                   color='none', edgecolor='black',
                   linewidth=1.3)
        i+=1
        labels += [labels_to_latex[s_cache.name.replace('_', ' ')] + " (Hit)"]
    # x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        rotation=45, ha='right'
    )

    # axis labels and title
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Mean Latency Summary by Strategy")
    #ax.set_yscale('log')
    #set y-axis limit to 500
    ax.set_ylim(0, 1500)
    ax.grid(True, alpha=0.3)

    # build legend (colors + hatches)
    from matplotlib.patches import Patch
    #color_legend = [Patch(facecolor=colors.get(s_base.name, '#95a5a6'), edgecolor='black', label=s_base.name.replace('_', ' ').title())
    #                for _, s_base in strategies_pairs]

    hatch_legend = [Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label.replace('_', ' ').title())
                    for label, hatch in hatch_map.items()]

    #legend1 = ax.legend(handles=color_legend, title="Strategy Color", loc='upper left', frameon=True)
    legend2 = ax.legend(handles=hatch_legend, title=None, loc='upper right', frameon=True)
    ax.add_artist(legend2)


    #----------- Add zoomed inset -----------
    ##use yticks from 0 to 1000 with step of 200
#
    #axins = ax.inset_axes([0.35, 0.25, 0.42, 0.42], 
    #                      xlim=(x_positions[0]-0.5, x_positions[-1]+0.5), ylim=(0, 1000),
    #                    xticklabels=[], yticklabels=np.arange(0, 1200, 500))
#
    ## Replot bars in the inset (same data as main plot)
    #i = 0
    #for (s_cache, s_base) in strategies_pairs:
    #    base_color = colors.get(s_base.name, '#95a5a6')
#
    #    base_mean = df_events.loc[df_events['strategy'] == s_base.name, 'latency'].mean()
    #    cache_df = df_events[df_events['strategy'] == s_cache.name]
    #    cache_miss = cache_df[cache_df.get('cache_hit', False) == False]
    #    cache_hit = cache_df[cache_df.get('cache_hit', False) == True]
    #    cache_hit_zero = cache_hit[cache_hit['latency'] == 0]
    #    cache_hit_ext = cache_hit[cache_hit['latency'] > 0]
#
    #    miss_mean = cache_miss['latency'].mean() if not cache_miss.empty else 0
    #    hit_ext_mean = cache_hit_ext['latency'].mean() if not cache_hit_ext.empty else 0
    #    combined_mean = pd.concat([cache_hit['latency'], cache_miss['latency']]).mean() if len(cache_hit) + len(cache_miss) > 0 else 0
#
    #    # Recreate baseline bar
    #    axins.bar(x_positions[i], base_mean, width=bar_width,
    #            color=base_color, edgecolor='black', alpha=0.8)
    #    i += 1
    #    # Recreate cache miss
    #    axins.bar(x_positions[i], miss_mean, width=bar_width,
    #            color=base_color, edgecolor='black', hatch=hatch_map['Miss'], alpha=0.8)
    #    i += 1
    #    # Recreate cache hit extension
    #    if hit_ext_mean > 0:
    #        axins.bar(x_positions[i], hit_ext_mean, width=bar_width,
    #                color=base_color, edgecolor='black', hatch=hatch_map['Extension'], alpha=0.8)
    #    # Recreate outline bar
    #    if combined_mean > 0:
    #        axins.bar(x_positions[i], combined_mean, width=bar_width,
    #                color='none', edgecolor='black', linewidth=1.3)
    #    i += 1
#
    #ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    fig.savefig(output_path_top, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {output_path_top}")

    if extension == 'pgf':
        with open(output_path_top, 'r') as file:
            content = file.readlines()

        fixed_path = output_dir / f'luca_top_coverage_summary_fixed.{extension}'
        with open(fixed_path, 'w') as file:
            for line in content:
                if 'pgfsys@transformshift' in line:
                    try:
                        parts = line.split('{')
                        second_value = parts[2].split('}')[0]
                        value = float(second_value[:-2])  # strip unit
                        if abs(value) > 400:
                            file.write('%' + line)
                        else:
                            file.write(line)
                    except (IndexError, ValueError):
                        file.write(line)
                else:
                    file.write(line)







def plot_cumulative_latency(df_events: pd.DataFrame, 
                            strategies: List[Strategy],
                            visual_info: Dict[str, Any],
                            scenario: str,
                            output_dir: Path, 
                            extension='png',
                            fig_size=(6.5, 3.5)):
    """
    Figure 2: Cumulative planning cost over time.
    
    Demonstrates total computational savings of our approach.
    """
    output_path = output_dir / f"cumulative_latency_{scenario}.{extension}"
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=fig_size)
    colors = visual_info['colors']
    linestyles = visual_info['linestyles']
    done_strategies = set()
    strategies_names = [s.name for s in strategies][2:]
    first = df_events[df_events['strategy'] == strategies_names[1]]
    
    for strategy in strategies_names:
        if strategy in done_strategies:
            continue  # Already processed this strategy
        print(f"Processing strategy: {strategy}")
        data = df_events[df_events['strategy'] == strategy].sort_values('timestamp')
        if len(data) == 0:
            continue
        data = data[data['timestamp'] <= 60000]
        # Convert the y-axis to seconds and limit to 100 seconds
        cumulative = data['latency'].cumsum() // 1000 

        label = strategy.replace('_', ' ').title()
        linestyle = '--' if 'linear' in strategy else '-'
        
        ax.plot(data['timestamp'] , cumulative, 
               label=labels_to_latex[strategy], 
               color=colors[strategy],
               linestyle=linestyles[strategy])
               
        
        
        # Add final value annotation
        #if len(cumulative) > 0:
        #    final_val = cumulative.iloc[-1]
        #    ax.text(data['timestamp'].iloc[-1], final_val, 
        #           f' {final_val:.0f} ms', 
        #           #fontsize=7, fontweight='bold', 
        #           verticalalignment='center',
        #           color=colors[strategy])
        
        done_strategies.add(strategy)
    ax.legend( loc='upper left', ncol=3, frameon=False,
              fontsize=6, 
              handlelength=1,
              labelspacing=0.2,
              handletextpad=0.1,
              columnspacing = 0.2)
    #ax.set_xscale('log')
    #ax.set_xlabel('Time (s)',  fontweight='bold', labelpad=1)
    #ax.set_ylabel('Latency (s)',fontweight='bold', labelpad=1)

    #create a formatter for x that shows ticks // 3600
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x // 3600)}h' if x >= 3600 else f'{int(x // 60)}m' if x >= 60 else f'{int(x)}s'))

    ax.tick_params(axis='y', which='major', pad=0)
    ax.tick_params(axis='x', which='major', pad=0)
    #ax.set_title('Cumulative Planning Cost ',  fontweight='bold')
    
    # Set Y-axis to logarithmic scale for better visibility of differences
    #ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")








def plot_com_speedup_factor(df_events: pd.DataFrame, 
                            strategies: List[Strategy],
                            visual_info: Dict[str, Any],
                            scenario: str,
                            output_dir: Path, 
                            extension='png',
                            fig_size=(6.5, 3.5)):
    """
    Figure 2: Cumulative planning cost over time.
    
    Demonstrates total computational savings of our approach.
    """
    output_path = output_dir / f"speedup_{scenario}.{extension}"
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=fig_size)
    colors = visual_info['colors']
    linestyles = visual_info['linestyles']
    done_strategies = set()
    
    strategies_names = [s.name for s in strategies][:]
    first = df_events[df_events['strategy'] == strategies_names[1]]
    #iterate every 2
    #create paris of strategies
    for i in range(0, len(strategies_names), 2):
        strategy_c, strategy_n = strategies_names[i], strategies_names[i+1]
        if strategy_c in done_strategies or strategy_n in done_strategies:
            continue  # Already processed this strategy
        print(f"Processing strategy: {strategy_c}, {strategy_n}")
        data_c = df_events[df_events['strategy'] == strategy_c].sort_values('timestamp')
        if len(data_c) == 0:
            continue
        data_n = df_events[df_events['strategy'] == strategy_n].sort_values('timestamp')
        if len(data_n) == 0:
            continue 
    
        merged = pd.merge_asof(
            data_c.sort_values('timestamp'),
            data_n.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            suffixes=('_c', '_n')
        )

        merged['cum_c'] = merged['latency_c'].cumsum()
        merged['cum_n'] = merged['latency_n'].cumsum()

        merged['speedup'] = merged['cum_n'] / merged['cum_c']

        ax.plot(merged['timestamp'] , merged["speedup"], 
               label=labels_to_latex[strategy_c], 
               color=colors[strategy_c],
               linestyle=linestyles[strategy_c])
               
        
        
        # Add final value annotation
        #if len(cumulative) > 0:
        #    final_val = cumulative.iloc[-1]
        #    ax.text(data['timestamp'].iloc[-1], final_val, 
        #           f' {final_val:.0f} ms', 
        #           #fontsize=7, fontweight='bold', 
        #           verticalalignment='center',
        #           color=colors[strategy])
        
        done_strategies.add(strategy_n)
    ax.legend( loc='upper left', ncol=3, frameon=False,
              fontsize=6, 
              handlelength=1,
              labelspacing=0.2,
              handletextpad=0.1,
              columnspacing = 0.2)
    #ax.set_xscale('log')
    #ax.set_xlabel('Time (s)',  fontweight='bold', labelpad=1)
    #ax.set_ylabel('Latency (s)',fontweight='bold', labelpad=1)

    #create a formatter for x that shows ticks // 3600
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x // 3600)}h' if x >= 3600 else f'{int(x // 60)}m' if x >= 60 else f'{int(x)}s'))

    ax.tick_params(axis='y', which='major', pad=0)
    ax.tick_params(axis='x', which='major', pad=0)
    #ax.set_title('Cumulative Planning Cost ',  fontweight='bold')
    
    # Set Y-axis to logarithmic scale for better visibility of differences
    #ax.set_yscale('log')
    ax.set_ylim((0,7))
    
    ax.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")






def plot_scenario_comparison(df: pd.DataFrame, 
                            visual_info: Dict[str, Any],
                            output_dir: Path, 
                            extension='png',
                            fig_size=(6.5, 3.5)):
    """Compare performance across different scenarios."""
    
    
    scenarios = sorted(df['scenario'].unique())
    
    # Extract scenario info
    scenario_labels = []
    for scenario in scenarios:
        parts = scenario.split('_')
        duration = parts[2] if len(parts) > 2 else "?"
        temp = parts[4] if len(parts) > 4 else "?"
        scenario_labels.append(f"{duration}D\n{temp}°C")
    
    metrics = [
        ('hit_rate_pct', r'Cache Hit Rate (\%)'),
        ('speedup_factor', 'Speedup Factor ($\\times$)'),
        ('latency_p50_ms', 'Latency P50 (ms)'),
        ('latency_p95_ms', 'Latency P95 (ms)'),
    ]

    fig_size = (fig_size[0] / 1.9, fig_size[1]/1.9)
    for i, (metric_col, metric_label) in enumerate(metrics):
        fig, ax = plt.subplots(figsize=fig_size)
        x = np.arange(len(scenarios))
        width = 0.35
        
        v1_values = [df[(df['scenario'] == s) & (df['gamma_version'] == 'V1 (Old Gamma)')][metric_col].mean() 
                     for s in scenarios]
        v2_values = [df[(df['scenario'] == s) & (df['gamma_version'] == 'V2 (New Gamma)')][metric_col].mean() 
                     for s in scenarios]
        
        ax.bar(x - width/2, v1_values, width, label=r'$\Gamma_1$', color='lightblue')
        ax.bar(x + width/2, v2_values, width, label=r'$\Gamma_2$', color='lightcoral')
        
        #ax.set_ylabel(metric_label)
        #ax.set_title(f'{metric_label} by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        #ax.legend(ncol=2, frameon=False, loc='upper left')
    
        plt.tight_layout()
        plt.savefig(output_dir / f'scenario_comparison_{i}.{extension}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir / f'scenario_comparison_{i}.{extension}'}")
    #In the same folder, we create a latex file called scenario_comparison.tex that creates a figure with 4 subigures (one for each metric)

    with open(output_dir / f'scenario_comparison_{extension}.tex', 'w') as f:
        f.write(r"""
\definecolor{lightblue}{RGB}{173,216,230}   % LightBlue
\definecolor{lightcoral}{RGB}{240,128,128}  % LightCoral
\begin{figure}
    \caption{Cache Effectiveness and Coverage between $\Gamma_1$ (\textcolor{lightblue}{$\blacksquare$}) and $\Gamma_2$ (\textcolor{lightcoral}{$\blacksquare$}) for Different Scenarios.}
    \centering
""")
        for i, (_,label) in enumerate(metrics[:2]):
            if extension == 'pgf':
                add= rf"""\input{{./assets/figures/scenario_comparison_{i}.pgf}}
"""         
            else:
                add =rf"""\includegraphics[width=\textwidth]{{./assets/figures/scenario_comparison_{i}.{extension}}}
"""
            f.write(rf"""\begin{{subfigure}}{{0.23\textwidth}}
\centering""" + rf"""
\caption{{{label}}}
""" + add + rf"""\end{{subfigure}}
""")
            if i % 2 == 1:
                f.write(r"\\" + "\n")  # New line after evertwo subfigures   
            else:
                f.write("")
        f.write(r"""\end{figure}
""")
    print(f"✓ Saved: {output_dir / f'scenario_comparison_{extension}.tex'}")

def load_combined_metrics(evaluate_dir: Path) -> pd.DataFrame:
    """Load the combined metrics CSV."""
    csv_file = REPORTS_DIR / "paper_metrics_combined.csv"
    return pd.read_csv(csv_file)


def load_data(file_path):
    # Load data from the specified json file path
    if not file_path.endswith('.json'):
        raise ValueError("Input file must be a .json file")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_strategies(args):
    if not args.is_drone:
        drone_domain = Fleet(file_path=args.configuration_file)
    else:
        drone_domain = Drone(file_path=args.configuration_file)

    strategies = [
                    SmartCacheStrategy(name = "SMART CACHE (DISCRETE)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy.json")
                                ),
                    DiscreteStrategy("BASELINE DISCRETE",cs=drone_domain),
                    SmartLinearStrategy("SMART CACHE (LINEAR)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy.json")
                                ),
                    LinearStrategy("BASELINE LINEAR",cs=drone_domain),
                    # SmartBayesianStrategy(name = "SMART CACHE (BAYESIAN)",
                    #                 cs=drone_domain,
                    #                 file_path="./assets/smart_strategy.json"),
                    # BayesianOptimizationStrategy("BAYESIAN OPTIMIZATION",cs=drone_domain),
                    SmartGAStrategy(name = "SMART CACHE (GA)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy.json")),
                    GAStrategy("GENETIC ALGORITHM",cs=drone_domain),
                    SmartLNSStrategy(name = "SMART CACHE (LNS)",
                                    cs=drone_domain,
                                    file_path=str(ASSETS_DIR / "smart_strategy.json")),
                    LNSStrategy("LARGE NEIGHBORHOOD SEARCH",cs=drone_domain)
                ]
    
    #add colors and linestyles to visual_info
    #iterate over strategies in pairs
    for i in range(0, len(strategies), 2):
        base_strategy = strategies[i+1]
        cache_strategy = strategies[i]
        #assign a color
        color = f"C{i//2}"
        base_strategy.color = color
        cache_strategy.color = color
        #assign linestyles
        base_strategy.linestyle = 'dotted'
        cache_strategy.linestyle = '-'


    return strategies


input_to_scenario_label = {
    "weather_scenario_180_temp_25_realistic": "S1",
    "weather_scenario_90_temp_15_realistic": "S2",
    "weather_scenario_330_temp_5_realistic": "S3",
}


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_energy_per_gamma_scenario(inputs, cs_volume, strategies, visual_info, output_dir,
                                   extension='pgf', fig_size=(7, 4)):
    # --- Load data for each gamma ---
    dfs_gamma1, dfs_gamma2 = {}, {}
    for _, input_file in inputs.items():
        data = load_data(input_file)
        if 'events' not in data or not data['events']:
            continue
        df_events = pd.DataFrame(data['events'])
        scenario = input_to_scenario_label[input_file.split('/')[-2]]
        if "new" in input_file.lower():
            dfs_gamma2[scenario] = df_events
        else:
            dfs_gamma1[scenario] = df_events

    output_path = output_dir / f'energy_per_gamma_per_scenario.{extension}'
    fig, ax = plt.subplots(figsize=fig_size)

    # --- Hatches and legend setup ---
    hatch_gamma1 = ''
    hatch_gamma2 = '////'
    legend_handles = [
        Patch(facecolor='white', edgecolor='black', hatch=hatch_gamma1, label=r'$\Gamma_1$'),
        Patch(facecolor='white', edgecolor='black', hatch=hatch_gamma2, label=r'$\Gamma_2$')
    ]

    # --- Prepare grouped data ---
    group_labels = []
    gamma1_means, gamma2_means = [], []
    gamma1_diffs, gamma2_diffs = [], []
    bar_colors = []

    for p in range(0, len(strategies), 2):
        c = strategies[p].name
        nc = strategies[p + 1].name
        strategy_label = labels_to_latex[c.replace('_', ' ')]
        strategy_color = strategies[p].color

        for scenario in sorted(set(dfs_gamma1.keys()) & set(dfs_gamma2.keys())):
            df_g1 = dfs_gamma1[scenario]
            df_g2 = dfs_gamma2[scenario]

            # --- Gamma 1 ---
            cached = df_g1[df_g1['strategy'] == c]['energy']
            uncached = df_g1[df_g1['strategy'] == nc]['energy']
            cached = cached[cached != float('inf')]
            uncached = uncached[uncached != float('inf')]
            if len(cached) == 0 or len(uncached) == 0:
                continue
            cached_mean1 = cached.mean()
            uncached_mean1 = uncached.mean()
            #diff1 = abs(uncached_mean1 - cached_mean1)
            #get the difference in percentage
            diff1 = abs((uncached_mean1 - cached_mean1) / uncached_mean1 * 100)

            # --- Gamma 2 ---
            cached = df_g2[df_g2['strategy'] == c]['energy']
            uncached = df_g2[df_g2['strategy'] == nc]['energy']
            cached = cached[cached != float('inf')]
            uncached = uncached[uncached != float('inf')]
            if len(cached) == 0 or len(uncached) == 0:
                continue
            cached_mean2 = cached.mean()
            uncached_mean2 = uncached.mean()
            #diff2 = abs(uncached_mean2 - cached_mean2)
            diff2 = abs((uncached_mean2 - cached_mean2) / uncached_mean2 * 100)

            # Append results
            group_labels.append(f"{strategy_label}{scenario}")
            gamma1_means.append(cached_mean1)
            gamma2_means.append(cached_mean2)
            gamma1_diffs.append(diff1)
            gamma2_diffs.append(diff2)
            bar_colors.append(strategy_color)

    # --- Create grouped bar plot ---
    x = np.arange(len(group_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, gamma1_means, width,
                   color=bar_colors, edgecolor='black', hatch=hatch_gamma1, alpha=0.6)
    bars2 = ax.bar(x + width/2, gamma2_means, width,
                   color=bar_colors, edgecolor='black', hatch=hatch_gamma2, alpha=0.6)

    # --- Add numeric difference labels on top of bars ---
    for i, (bar, diff) in enumerate(zip(bars1, gamma1_diffs)):

        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2 - 0.1, height + 0.02 * height,
                f"{int(diff)}\\%" if diff > -0.0 else "0\\%", ha='center', va='bottom', fontsize=7)

    for i, (bar, diff) in enumerate(zip(bars2, gamma2_diffs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 * height,
                f"{int(diff)}\\%" if diff > -0.0 else "0\\%", ha='center', va='bottom', fontsize=7)

    # --- Axis and label styling ---
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=40, ha='right')
    for label in ax.get_xticklabels():
        label.set_linespacing(1.5)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(handles=legend_handles,  frameon=False, ncol=2, loc='upper right', columnspacing = 0.2)

    # --- Optional annotations on bars ---
    #for i, (m1, e1, m2, e2) in enumerate(zip(gamma1_means, gamma1_errors, gamma2_means, gamma2_errors)):
    #    ax.text(i - width/2, m1 + e1 + 0.05, f"{e1:+.1f}", ha='center', va='bottom', fontsize=7)
    #    ax.text(i + width/2, m2 + e2 + 0.05, f"{e2:+.1f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved grouped bar plot with hatches: {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate paper graphs for zonotope-based optimization.")
    #parser.add_argument("--input", type=str, default="./evaluate/results_new_gamma_FIXED/weather_scenario_180_temp_25_realistic/drone_6d_large_cs_results.json", help="Path to the input data file.")
    parser.add_argument("--output", type=str, default=str(FIGURES_DIR), help="Path to save the generated graphs.")
    parser.add_argument("--configuration_file", type=str, default=str(ASSETS_DIR / "fleet.json"), help="Path to the cs configuration file")
    parser.add_argument("--is_drone", action='store_true', help="Indicates if the configuration file is for a drone")
    parser.add_argument("--evaluate_dir", type=str, default="./evaluate/", help="Path to the evaluate directory containing metrics.")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        0:str(CACHE_RESULTS_DIR / "final_old_gamma" / "weather_scenario_330_temp_5_realistic" / "drone_6d_large_cs_results.json"),
        1:str(CACHE_RESULTS_DIR / "final_new_gamma" / "weather_scenario_330_temp_5_realistic" / "drone_6d_large_cs_results.json"),
        2:str(CACHE_RESULTS_DIR / "final_old_gamma" / "weather_scenario_90_temp_15_realistic" / "drone_6d_large_cs_results.json"),
        3:str(CACHE_RESULTS_DIR / "final_new_gamma" / "weather_scenario_90_temp_15_realistic" / "drone_6d_large_cs_results.json"),
    }

    for scenario, input in inputs.items():
        data = load_data(input)
        

        strategies = get_strategies(args)

        # Convert events to DataFrame
        if 'events' in data and data['events']:
            df_events = pd.DataFrame(data['events'])
        else:
            print("Warning: No events data found in results")
            df_events = pd.DataFrame()
        
        # Get CS volume
        cs_volume = data.get('cs_volume', 1.0)

        visual_info = get_visual_info(strategies)
        #using the fig_size perfect for half column in latex
        fig_size = (3.45, 2.5)
        ############ PLOTTING #############
        #plot_latency_summary(df_events, strategies, visual_info, output_dir, extension='pgf', fig_size=fig_size)
        #plot_latency_summary(df_events, strategies, visual_info, output_dir, extension='png', fig_size=fig_size)
        #plot_coverage_summary(df_events, cs_volume, strategies, visual_info, output_dir, extension='pgf')
        #plot_coverage_progress(df_events, cs_volume, strategies, visual_info, output_dir, extension='pgf', fig_size=fig_size)
        #plot_coverage_summary(df_events, cs_volume, strategies, visual_info, output_dir, extension='png')
        #plot_coverage_progress(df_events, cs_volume, strategies, visual_info, output_dir, extension='png', fig_size=fig_size)
        if scenario == 0:
            div = 1.6
            
        else:
            div = 1.6
        div_v = 1.8
        plot_cumulative_latency(df_events, 
                                strategies, 
                                visual_info, 
                                scenario=f"scenario_{scenario}",
                                output_dir=output_dir, extension='png', fig_size=(fig_size[0]/div, fig_size[1]/div_v) )
        plot_cumulative_latency(df_events, strategies, visual_info,
                                scenario=scenario,
                                  output_dir=output_dir, extension='pgf', fig_size=(fig_size[0]/div, fig_size[1]/div_v) )
        plot_com_speedup_factor(df_events, 
                                strategies, 
                                visual_info, 
                                scenario=f"scenario_{scenario}",
                                output_dir=output_dir, extension='png', fig_size=(fig_size[0]/div, fig_size[1]/div_v) )
        plot_com_speedup_factor(
            df_events, strategies, visual_info,
                                scenario=scenario,
                                  output_dir=output_dir, extension='pgf', fig_size=(fig_size[0]/div, fig_size[1]/div_v)
        )





    print("Loading metrics...")
    df = load_combined_metrics(args.evaluate_dir)
    print(f"✓ Loaded {len(df)} rows\n")
    plot_scenario_comparison(df, visual_info, output_dir, extension='png', fig_size=fig_size)
    plot_scenario_comparison(df, visual_info, output_dir, extension='pgf', fig_size=fig_size)

    inputs = {
        0:str(CACHE_RESULTS_DIR / "final_old_gamma" / "weather_scenario_330_temp_5_realistic" / "drone_6d_large_cs_results.json"),
        1:str(CACHE_RESULTS_DIR / "final_new_gamma" / "weather_scenario_330_temp_5_realistic" / "drone_6d_large_cs_results.json"),
        2:str(CACHE_RESULTS_DIR / "final_old_gamma" / "weather_scenario_180_temp_25_realistic" / "drone_6d_large_cs_results.json"),
        3:str(CACHE_RESULTS_DIR / "final_new_gamma" / "weather_scenario_180_temp_25_realistic" / "drone_6d_large_cs_results.json"),
        4:str(CACHE_RESULTS_DIR / "final_old_gamma" / "weather_scenario_90_temp_15_realistic" / "drone_6d_large_cs_results.json"),
        5:str(CACHE_RESULTS_DIR / "final_new_gamma" / "weather_scenario_90_temp_15_realistic" / "drone_6d_large_cs_results.json"),
    }


    plot_energy_per_gamma_scenario(inputs, cs_volume, strategies, visual_info, output_dir, extension='pgf', fig_size=(3.8, 2.5))
    plot_energy_per_gamma_scenario(inputs, cs_volume, strategies, visual_info, output_dir, extension='png', fig_size=(3.8, 2.5))


if __name__ == "__main__":
    main()