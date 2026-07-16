"""
Utility functions for zonotope-based caching system.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import seaborn as sns

from .zonotope_ops import Zonotope
from .domain import DomainSpec, ConstrainedDomain, Configuration
from .zonotope_cache import ZonotopeCache


def visualize_2d_cache(
    cache: ZonotopeCache,
    param1_idx: int = 0,
    param2_idx: int = 1,
    show_explored: bool = True,
    show_entries: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize 2D projection of cache entries and explored regions.
    
    Useful for understanding cache behavior and coverage.
    
    Args:
        cache: ZonotopeCache to visualize
        param1_idx: Index of first parameter to plot (x-axis)
        param2_idx: Index of second parameter to plot (y-axis)
        show_explored: Whether to show explored regions
        show_entries: Whether to show cache entries
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    param_names = cache.domain_spec.parameter_names
    param1_name = param_names[param1_idx]
    param2_name = param_names[param2_idx]
    
    # Plot explored regions
    if show_explored:
        for region in cache.explored_regions:
            bounds = region.to_box_bounds()
            rect = Rectangle(
                (bounds[param1_idx, 0], bounds[param2_idx, 0]),
                bounds[param1_idx, 1] - bounds[param1_idx, 0],
                bounds[param2_idx, 1] - bounds[param2_idx, 0],
                fill=True,
                facecolor='lightblue',
                edgecolor='blue',
                alpha=0.3,
                linewidth=1
            )
            ax.add_patch(rect)
    
    # Plot cache entries
    if show_entries:
        for entry in cache.entries:
            bounds = entry.ds_zonotope.to_box_bounds()
            rect = Rectangle(
                (bounds[param1_idx, 0], bounds[param2_idx, 0]),
                bounds[param1_idx, 1] - bounds[param1_idx, 0],
                bounds[param2_idx, 1] - bounds[param2_idx, 0],
                fill=True,
                facecolor='lightgreen',
                edgecolor='darkgreen',
                alpha=0.5,
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Plot configuration point
            config_values = entry.optimal_config.as_array()
            ax.plot(config_values[param1_idx], config_values[param2_idx],
                   'ro', markersize=8, label='Config')
    
    # Set limits to full domain
    domain_bounds = cache.domain_spec.get_bounds_array()
    ax.set_xlim(domain_bounds[param1_idx])
    ax.set_ylim(domain_bounds[param2_idx])
    
    ax.set_xlabel(f"{param1_name} ({cache.domain_spec.parameters[param1_idx].unit})")
    ax.set_ylabel(f"{param2_name} ({cache.domain_spec.parameters[param2_idx].unit})")
    ax.set_title(f"Cache Visualization: {len(cache.entries)} entries, "
                f"{cache.cache_hit_rate():.1f}% hit rate")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_cache_statistics(
    cache: ZonotopeCache,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot comprehensive cache statistics.
    
    Shows:
    - Cache hit rate over time
    - Number of entries over time
    - Query time distribution
    - Merge time distribution
    
    Args:
        cache: ZonotopeCache to analyze
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    stats = cache.get_statistics()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Cache hit rate
    ax = axes[0, 0]
    hit_rate = stats['hit_rate']
    ax.bar(['Hit Rate'], [hit_rate], color='green', alpha=0.7)
    ax.bar(['Miss Rate'], [100 - hit_rate], bottom=[hit_rate], color='red', alpha=0.7)
    ax.set_ylabel('Percentage')
    ax.set_title(f"Cache Hit Rate: {hit_rate:.1f}%")
    ax.set_ylim([0, 100])
    ax.legend(['Hits', 'Misses'])
    
    # 2. Cache size over time (simulated)
    ax = axes[0, 1]
    ax.bar(['Entries', 'Explored\nRegions'], 
           [stats['num_entries'], stats['num_explored_regions']],
           color=['blue', 'orange'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Cache Size')
    
    # 3. Query time distribution
    ax = axes[1, 0]
    if cache.query_times:
        query_times_ms = np.array(cache.query_times) * 1000
        ax.hist(query_times_ms, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Query Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Query Times (avg: {stats['avg_query_time_ms']:.2f}ms)")
        ax.axvline(stats['avg_query_time_ms'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No query data', ha='center', va='center', transform=ax.transAxes)
    
    # 4. Merge efficiency
    ax = axes[1, 1]
    if stats['total_additions'] > 0:
        merge_rate = (stats['merge_count'] / stats['total_additions']) * 100
        ax.bar(['Additions', 'Merges'], 
               [stats['total_additions'], stats['merge_count']],
               color=['blue', 'green'], alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title(f"Merge Efficiency: {merge_rate:.1f}% merge rate")
    else:
        ax.text(0.5, 0.5, 'No merge data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


def compare_configurations(
    configs: List[Configuration],
    param_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare multiple configurations visually.
    
    Args:
        configs: List of configurations to compare
        param_names: Parameter names to include (None = all)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not configs:
        raise ValueError("No configurations to compare")
    
    domain_spec = configs[0].domain_spec
    if param_names is None:
        param_names = domain_spec.parameter_names
    
    # Extract values
    config_data = []
    for config in configs:
        config_dict = config.as_dict()
        values = [config_dict[name] for name in param_names]
        config_data.append(values)
    
    config_data = np.array(config_data)
    
    # Create radar/spider plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart comparison
    x = np.arange(len(param_names))
    width = 0.8 / len(configs)
    
    for i, config in enumerate(configs):
        offset = width * i - width * len(configs) / 2
        ax1.bar(x + offset, config_data[i], width, label=f'Config {i+1}', alpha=0.7)
    
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Values')
    ax1.set_title('Configuration Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Normalized heatmap
    # Normalize each parameter to [0, 1]
    normalized_data = config_data.copy()
    for j in range(len(param_names)):
        param = domain_spec.get_parameter(param_names[j])
        normalized_data[:, j] = param.normalize(config_data[:, j])
    
    sns.heatmap(normalized_data, annot=True, fmt='.2f', cmap='RdYlGn',
               xticklabels=param_names, yticklabels=[f'Config {i+1}' for i in range(len(configs))],
               ax=ax2, cbar_kws={'label': 'Normalized Value'})
    ax2.set_title('Normalized Configuration Heatmap')
    
    plt.tight_layout()
    return fig


def export_cache_to_json(cache: ZonotopeCache, filepath: str):
    """
    Export cache to JSON for analysis or persistence.
    
    Args:
        cache: Cache to export
        filepath: Output file path
    """
    import json
    
    data = {
        'domain_spec': {
            'dimension': cache.dimension,
            'parameters': [
                {
                    'name': p.name,
                    'bounds': p.bounds,
                    'unit': p.unit,
                    'type': p.param_type.value
                }
                for p in cache.domain_spec.parameters
            ]
        },
        'statistics': cache.get_statistics(),
        'entries': [
            {
                'odd_hash': entry.odd_hash,
                'ds_bounds': entry.ds_zonotope.to_box_bounds().tolist(),
                'optimal_config': entry.optimal_config.as_dict(),
                'optimal_objective': entry.optimal_objective,
                'timestamp': entry.timestamp
            }
            for entry in cache.entries
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Cache exported to {filepath}")


def print_cache_summary(cache: ZonotopeCache):
    """
    Print human-readable cache summary.
    
    Args:
        cache: Cache to summarize
    """
    stats = cache.get_statistics()
    
    print("=" * 70)
    print(f"ZONOTOPE CACHE SUMMARY ({cache.dimension}D)")
    print("=" * 70)
    print(f"Domain: {', '.join(cache.domain_spec.parameter_names)}")
    print()
    print(f"Cache Entries:        {stats['num_entries']}")
    print(f"Explored Regions:     {stats['num_explored_regions']}")
    print(f"Total Queries:        {stats['total_queries']}")
    print(f"Cache Hits:           {stats['cache_hits']}")
    print(f"Hit Rate:             {stats['hit_rate']:.1f}%")
    print(f"Total Additions:      {stats['total_additions']}")
    print(f"Successful Merges:    {stats['merge_count']}")
    print()
    print(f"Avg Query Time:       {stats['avg_query_time_ms']:.2f} ms")
    print(f"Avg Merge Time:       {stats['avg_merge_time_ms']:.2f} ms")
    print("=" * 70)
