"""
Generate a concise comparison table for quick reference.
"""

import pandas as pd

from paths import REPORTS_DIR


def main():
    df = pd.read_csv(REPORTS_DIR / "paper_metrics_combined.csv")
    
    print("\n" + "="*100)
    print("GAMMA V1 vs V2 QUICK COMPARISON TABLE")
    print("="*100 + "\n")
    
    # Group by gamma version
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    # Create comparison table
    comparison = []
    
    metrics = [
        ('hit_rate_pct', 'Cache Hit Rate (%)', 2),
        ('hit_latency_median_ms', 'Hit Latency Median (ms)', 2),
        ('miss_latency_median_ms', 'Miss Latency Median (ms)', 2),
        ('latency_p50_ms', 'Latency P50 (ms)', 2),
        ('latency_p95_ms', 'Latency P95 (ms)', 2),
        ('latency_p99_ms', 'Latency P99 (ms)', 2),
        ('speedup_factor', 'Speedup Factor (×)', 2),
        ('total_requests', 'Total Requests', 0),
        ('cache_hits', 'Cache Hits', 0),
        ('cache_misses', 'Cache Misses', 0),
    ]
    
    for metric_col, metric_label, decimals in metrics:
        v1_mean = v1_df[metric_col].mean()
        v2_mean = v2_df[metric_col].mean()
        v1_std = v1_df[metric_col].std()
        v2_std = v2_df[metric_col].std()
        
        if decimals == 0:
            comparison.append({
                'Metric': metric_label,
                'V1 (Old Gamma)': f'{v1_mean:.0f}',
                'V2 (New Gamma)': f'{v2_mean:.0f}',
                'Difference': f'{v2_mean - v1_mean:.0f}',
                'Change (%)': f'{((v2_mean - v1_mean) / v1_mean * 100):.1f}%' if v1_mean != 0 else 'N/A'
            })
        else:
            comparison.append({
                'Metric': metric_label,
                'V1 (Old Gamma)': f'{v1_mean:.{decimals}f} ± {v1_std:.{decimals}f}',
                'V2 (New Gamma)': f'{v2_mean:.{decimals}f} ± {v2_std:.{decimals}f}',
                'Difference': f'{v2_mean - v1_mean:+.{decimals}f}',
                'Change (%)': f'{((v2_mean - v1_mean) / v1_mean * 100):+.1f}%' if v1_mean != 0 else 'N/A'
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Print table
    print(comparison_df.to_string(index=False))
    
    # Save to file
    output_file = REPORTS_DIR / "quick_comparison.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("GAMMA V1 vs V2 QUICK COMPARISON TABLE\n")
        f.write("="*100 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "="*100 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("="*100 + "\n\n")
        
        # Calculate key insights
        hit_rate_v1 = v1_df['hit_rate_pct'].mean()
        hit_rate_v2 = v2_df['hit_rate_pct'].mean()
        speedup_v1 = v1_df['speedup_factor'].mean()
        speedup_v2 = v2_df['speedup_factor'].mean()
        p50_v1 = v1_df['latency_p50_ms'].mean()
        p50_v2 = v2_df['latency_p50_ms'].mean()
        
        f.write(f"1. CACHE EFFECTIVENESS:\n")
        f.write(f"   - V1 achieves {hit_rate_v1:.1f}% hit rate (coarse discretization)\n")
        f.write(f"   - V2 achieves {hit_rate_v2:.1f}% hit rate (fine-grained discretization)\n")
        f.write(f"   - Trade-off: V1 has {hit_rate_v1 - hit_rate_v2:.1f}% higher hit rate\n\n")
        
        f.write(f"2. PERFORMANCE SPEEDUP:\n")
        f.write(f"   - V1 provides {speedup_v1:.1f}× speedup over baseline\n")
        f.write(f"   - V2 provides {speedup_v2:.1f}× speedup over baseline\n")
        f.write(f"   - Trade-off: V1 has {speedup_v1 / speedup_v2:.1f}× higher speedup\n\n")
        
        f.write(f"3. LATENCY:\n")
        f.write(f"   - V1 median latency: {p50_v1:.2f} ms\n")
        f.write(f"   - V2 median latency: {p50_v2:.2f} ms\n")
        f.write(f"   - Trade-off: V2 has {((p50_v2 - p50_v1) / p50_v1 * 100):+.1f}% higher latency\n\n")
        
        f.write(f"4. INTERPRETATION:\n")
        f.write(f"   - V1 (coarse): Fewer unique configurations → more cache hits → higher speedup\n")
        f.write(f"   - V2 (fine-grained): More unique configurations → fewer cache hits → lower speedup\n")
        f.write(f"   - V2 (fine-grained): More precise environmental adaptation → better optimality\n")
        f.write(f"   - Design choice: Cache effectiveness vs. solution optimality\n\n")
        
        f.write(f"5. RECOMMENDATION:\n")
        if hit_rate_v2 > 70 and speedup_v2 > 3:
            f.write(f"   - V2 still achieves good cache performance (>70% hit rate, >3× speedup)\n")
            f.write(f"   - V2 provides more accurate environmental constraint modeling\n")
            f.write(f"   - **Recommend V2** for production systems prioritizing accuracy\n")
            f.write(f"   - Consider V1 for systems prioritizing speed over precision\n")
        else:
            f.write(f"   - V1 provides significantly better cache performance\n")
            f.write(f"   - **Recommend V1** unless fine-grained precision is critical\n")
    
    print(f"\n✓ Saved to: {output_file}")
    print("="*100 + "\n")
    
    # Also save as CSV
    csv_file = REPORTS_DIR / "quick_comparison.csv"
    comparison_df.to_csv(csv_file, index=False)
    print(f"✓ Also saved as CSV: {csv_file}\n")
    
    # Print by strategy
    print("="*100)
    print("PER-STRATEGY BREAKDOWN")
    print("="*100 + "\n")
    
    for strategy in sorted(df['strategy'].unique()):
        v1_strategy = v1_df[v1_df['strategy'] == strategy]
        v2_strategy = v2_df[v2_df['strategy'] == strategy]
        
        if len(v1_strategy) == 0 or len(v2_strategy) == 0:
            continue
        
        print(f"\n{strategy.replace('SMART CACHE ', '')}:")
        print(f"  V1: Hit Rate = {v1_strategy['hit_rate_pct'].mean():.1f}%, Speedup = {v1_strategy['speedup_factor'].mean():.1f}×, P50 = {v1_strategy['latency_p50_ms'].mean():.2f}ms")
        print(f"  V2: Hit Rate = {v2_strategy['hit_rate_pct'].mean():.1f}%, Speedup = {v2_strategy['speedup_factor'].mean():.1f}×, P50 = {v2_strategy['latency_p50_ms'].mean():.2f}ms")


if __name__ == "__main__":
    main()
