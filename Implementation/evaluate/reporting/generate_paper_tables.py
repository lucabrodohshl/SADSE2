"""
Generate two clear, well-structured tables for the paper:
1. Cache System Characteristics (Global/Strategy-Independent)
2. Performance Metrics by Scenario (Scenario-Dependent)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from paths import REPORTS_DIR, CACHE_RESULTS_DIR
gamma_map = {
        'V1 (Coarse)': r'$\Gamma_1$',
        'V2 (Fine-Grained)': r'$\Gamma_2$',
    }
algo_map = {
        'MILP (Discrete)': 'DMILP',
        'LP Relaxation': 'LP',
        'Genetic Algorithm': 'GA',
        'Large Neighborhood Search': 'LNS',
    }

scenario_map = {
    "330D, 5°C":"S1",
    "180D, 25°C":"S2",
    "90D, 15°C":"S3"
}

scenario_map_inv = {v:k for k,v in scenario_map.items()}

def load_memory_data(results_dir: Path, scenario: str) -> dict:
    """Load memory data from memory_efficiency_report.txt"""
    memory_file = results_dir / scenario / "memory_efficiency_report.txt"
    
    if not memory_file.exists():
        return {}
    
    memory_data = {}
    with open(memory_file, 'r') as f:
        content = f.read()
        
    # Parse cache entries and cache data size for each strategy
    strategies = ['SMART CACHE (DISCRETE)', 'SMART CACHE (LINEAR)', 
                  'SMART CACHE (GA)', 'SMART CACHE (LNS)']
    
    for strategy in strategies:
        # Find the section for this strategy
        start_idx = content.find(f"{strategy}:")
        if start_idx == -1:
            continue
            
        section = content[start_idx:start_idx+500]
        
        # Extract cache entries
        if "Cache entries:" in section:
            entries_line = [l for l in section.split('\n') if 'Cache entries:' in l][0]
            entries = int(entries_line.split(':')[1].strip())
            memory_data[strategy] = {'cache_entries': entries}
        
        # Extract cache data size
        if "Cache data:" in section:
            data_line = [l for l in section.split('\n') if 'Cache data:' in l][0]
            cache_kb = float(data_line.split(':')[1].strip().replace(' KB', ''))
            if strategy in memory_data:
                memory_data[strategy]['cache_size_kb'] = cache_kb
    
    return memory_data


def generate_table1_cache_characteristics(df: pd.DataFrame, results_dirs: dict) -> pd.DataFrame:
    """
    Table 1: Cache System Characteristics (Global, Strategy-Independent)
    
    This table shows the cache infrastructure characteristics that are independent
    of the specific scenario being executed.
    """
    
    rows = []
    
    for gamma_name, results_dir in results_dirs.items():
        # Get memory data from all scenarios to compute averages
        all_memory_data = []
        
        for scenario_dir in results_dir.iterdir():
            if scenario_dir.is_dir():
                memory_data = load_memory_data(results_dir, scenario_dir.name)
                if memory_data:
                    all_memory_data.append(memory_data)
        
        # Get data for this gamma version
        gamma_df = df[df['gamma_version'] == gamma_name]
        
        if len(gamma_df) == 0:
            continue
        
        # Compute averages across all scenarios for each strategy
        strategies = sorted(gamma_df['strategy'].unique())
        
        for strategy in strategies:
            strategy_df = gamma_df[gamma_df['strategy'] == strategy]
            
            # Average cache entries across scenarios
            cache_entries = []
            cache_sizes = []
            
            for mem_data in all_memory_data:
                if strategy in mem_data:
                    cache_entries.append(mem_data[strategy].get('cache_entries', 0))
                    cache_sizes.append(mem_data[strategy].get('cache_size_kb', 0))
            
            avg_entries = np.mean(cache_entries) if cache_entries else 0
            avg_size_kb = np.mean(cache_sizes) if cache_sizes else 0
            
            # Average entry size
            avg_entry_size_kb = avg_size_kb / avg_entries if avg_entries > 0 else 0
            
            # Extract algorithm type
            if 'DISCRETE' in strategy:
                algorithm = 'MILP (Discrete)'
            elif 'LINEAR' in strategy:
                algorithm = 'LP Relaxation'
            elif 'GA' in strategy:
                algorithm = 'Genetic Algorithm'
            elif 'LNS' in strategy:
                algorithm = 'Large Neighborhood Search'
            else:
                algorithm = strategy
            
            row = {
                'Gamma Version': 'V1 (Coarse)' if 'Old' in gamma_name else 'V2 (Fine-Grained)',
                'Optimization Algorithm': algorithm,
                'Avg Cache Entries': f'{avg_entries:.1f}',
                'Avg Cache Size (KB)': f'{avg_size_kb:.1f}',
                'Avg Entry Size (KB)': f'{avg_entry_size_kb:.2f}',
                'Cache Overhead': 'Negligible (<20 KB)',
            }
            
            rows.append(row)
    
    table1 = pd.DataFrame(rows)
    
    # Sort by gamma version and algorithm
    table1 = table1.sort_values(['Gamma Version', 'Optimization Algorithm'])
    
    return table1


def generate_table2_performance_metrics(df: pd.DataFrame, results_dirs: dict) -> pd.DataFrame:
    """
    Table 2: Performance Metrics by Scenario (Scenario-Dependent)
    
    This table shows performance metrics that vary based on the specific
    environmental scenario and strategy combination.
    """
    
    rows = []
    
    # Extract scenario information
    for _, row in df.iterrows():
        gamma = 'V1 (Coarse)' if 'Old' in row['gamma_version'] else 'V2 (Fine-Grained)'
        
        # Parse scenario
        scenario_parts = row['scenario'].split('_')
        duration = scenario_parts[2] if len(scenario_parts) > 2 else "?"
        temp = scenario_parts[4] if len(scenario_parts) > 4 else "?"
        
        scenario_label = f"{duration}D, {temp}°C"
        
        # Extract algorithm
        strategy = row['strategy']
        if 'DISCRETE' in strategy:
            algorithm = 'MILP'
        elif 'LINEAR' in strategy:
            algorithm = 'LP'
        elif 'GA' in strategy:
            algorithm = 'GA'
        elif 'LNS' in strategy:
            algorithm = 'LNS'
        else:
            algorithm = strategy
        
        table_row = {
            'Gamma': gamma,
            'Scenario': scenario_label,
            'Algorithm': algorithm,
            'Hit Rate (%)': f'{int(row["hit_rate_pct"])}',
            'Speedup (×)': f'{row["speedup_factor"]:.1f}' if pd.notna(row['speedup_factor']) else 'N/A',
            'P50 Latency (ms)': f'{row["latency_p50_ms"]:.2f}',
            'P95 Latency (ms)': f'{row["latency_p95_ms"]:.1f}',
            'P99 Latency (ms)': f'{row["latency_p99_ms"]:.1f}',
            'Mean Latency (ms)': f'{row["latency_mean_ms"]:.2f}' if pd.notna(row['latency_mean_ms']) else 'N/A',
            'Avg Energy': f'{row["energy_mean"]:.1f}' if pd.notna(row['energy_mean']) else 'N/A',
            'Baseline Latency (ms)': f'{row["baseline_latency_mean_ms"]:.2f}' if pd.notna(row['baseline_latency_mean_ms']) else 'N/A',
            'Coverage (%)': f'{int(row["coverage_final_pct"])}',
            
        }
        
        rows.append(table_row)
    
    table2 = pd.DataFrame(rows)
    
    # Sort by gamma, scenario, and algorithm
    table2 = table2.sort_values(['Gamma', 'Scenario', 'Algorithm'])
    
    return table2


def generate_latex_table1(table1: pd.DataFrame, output_file: Path):
    """Generate LaTeX code for Table 1"""
    
    latex = r"""\begin{table}[t]
\centering
\caption{Cache System Characteristics (Strategy-Independent)}
\label{tab:cache_characteristics}
\small
\begin{tabular}{p{1.5cm}c c c c c c c c c }
\toprule
\multirow{2}{*}{\textbf{Algorithm}} & \multicolumn{2}{c}{\textbf{Avg Entries }}  & \multicolumn{2}{c}{\textbf{Avg Size (KB)}} & \multicolumn{2}{c}{\textbf{Entry Size (KB)}}   \\
\cmidrule(lr){2-7} 
"""

    gamma_line = r""" """
    gammas = table1['Gamma Version'].unique()

    
    # forget last column 'Cache Overhead' for header
    for  row in table1.columns[1:-2]:
        for gamma in gammas:
            # we only add v1, v2 etc, no other parts of the name
            gamma_line += f"""& {gamma_map[gamma]} """

    latex += gamma_line + r"""\\"""+" \n"
    latex += r"""\midrule
    """ 
    
    current_gamma = None
    
    for algo in list(table1['Optimization Algorithm'].unique()):
        latex += f"{algo_map.get(algo)} "
        for column in table1.columns[2:-1]:
                for gam in gammas:
                    #get the value for this algo and gamma
                    print("Algo:", algo, "Gamma:", gam, "Column:", column)
                    value = table1[(table1['Optimization Algorithm'] == algo) & (table1['Gamma Version']==gam)][column].values
                    print("Value:", value)
                    if len(value) > 0:
                        latex += f"& {value[0]} "
                    else:
                        latex += "& N/A "

        latex += r" \\" + "\n"

   
    
    latex += r"""\bottomrule
\end{tabular}

\end{table}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex)


def generate_latex_table2(table2: pd.DataFrame, output_file: Path):
    """Generate LaTeX code for Table 2"""
    
    latex = r"""\begin{table*}[t]
\centering
\caption{Overview of the simulation results. """ +fr""" Scenarios are denoted as S1({scenario_map_inv['S1']}), S2({scenario_map_inv['S2']}), and S3({scenario_map_inv['S3']}).""" +r"""}
\label{tab:performance_metrics}
\small
\begin{tabular}{l| cc|cc|cc|cc|cc|cc|cc|cc}
\toprule
 
\multirow{1}{*}{\textbf{Strategy}} &\multicolumn{2}{c|}{\textbf{Hit Rate}}  & \multicolumn{2}{c|}{\textbf{Speedup}} & \multicolumn{8}{c|}{\textbf{Latency (ms)}} &  \multicolumn{2}{c|}{\textbf{Baseline}}  & \multicolumn{2}{c}{\textbf{Coverage}} \\
\cmidrule(lr){6-13} 
\multicolumn{1}{l|}{\textbf{(Scenario)}}& \multicolumn{2}{c|}{\textbf{ (\%)}}&\multicolumn{2}{c|}{}& \multicolumn{2}{c|}{P50} & \multicolumn{2}{c|}{P95} & \multicolumn{2}{c|}{P99} &\multicolumn{2}{c|}{Avg.} &\multicolumn{2}{c|}{\textbf{Latency (ms)}}&\multicolumn{2}{c}{\textbf{ (\%)}}  \\

\cmidrule(lr){2-17} 
"""

    gamma_line = r""" """
    gammas = table2['Gamma'].unique()
    # forget last column 'Cache Overhead' for header
    for  row in table2.columns[1:-2]:
        if row == "Avg Energy":
            continue
        for gamma in gammas:
            # we only add v1, v2 etc, no other parts of the name
            gamma_line += f"""& {gamma_map[gamma]} """

    latex += gamma_line + r"""\\"""+" \n"
    latex += r"""\midrule
    """ 

    print(list(table2['Algorithm'].unique()))
        

    #since scenario is 3 and algo is 4, we can do nested loops
    columns = [
        "Gamma","Scenario", "Algorithm","Hit Rate (%)","Speedup (×)","P50 Latency (ms)","P95 Latency (ms)","P99 Latency (ms)","Mean Latency (ms)","Avg Energy","Baseline Latency (ms)","Coverage (%)"
    ]
    for scenario in list(sorted(table2['Scenario'].unique(), key=lambda x: int(x.split('D')[0]), reverse=True)):
        for algo in list(table2['Algorithm'].unique()):
            latex += f" {algo} ({scenario_map[scenario]}) "
            for column in columns[3:]:
                if column == "Avg Energy":
                    continue
                for gam in gammas:
                    #get the value for this algo and gamma
                    value = table2[(table2['Scenario'] == scenario) &(table2['Algorithm'] == algo) & (table2['Gamma']==gam)][column].values
                    if len(value) > 0:
                        latex += f"& {value[0]} "
                    else:
                        latex += "& N/A "

            latex += r" \\" + "\n"

    latex += r"""\bottomrule

\end{tabular}
\end{table*}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex)


def main():
    print("\n" + "="*100)
    print("GENERATING PAPER TABLES")
    print("="*100 + "\n")

    # Load data
    print("Loading metrics...")
    df = pd.read_csv(REPORTS_DIR / "paper_metrics_combined.csv")
    print(f"✓ Loaded {len(df)} rows\n")

    # Results directories
    results_dirs = {
        'V1 (Old Gamma)': CACHE_RESULTS_DIR / "final_old_gamma",
        'V2 (New Gamma)': CACHE_RESULTS_DIR / "final_new_gamma",
    }
    
    # Generate Table 1: Cache System Characteristics
    print("Generating Table 1: Cache System Characteristics (Global)...")
    table1 = generate_table1_cache_characteristics(df, results_dirs)
    
    # Save Table 1
    table1_csv = REPORTS_DIR / "table1_cache_characteristics.csv"
    table1.to_csv(table1_csv, index=False)
    print(f"✓ Saved CSV: {table1_csv}")

    table1_latex = REPORTS_DIR / "table1_cache_characteristics.tex"
    generate_latex_table1(table1, table1_latex)
    print(f"✓ Saved LaTeX: {table1_latex}")
    
    print("\nTable 1 Preview:")
    print(table1.to_string(index=False))
    print()
    
    # Generate Table 2: Performance Metrics by Scenario
    print("\nGenerating Table 2: Performance Metrics by Scenario...")
    table2 = generate_table2_performance_metrics(df, results_dirs)
    
    # Save Table 2
    table2_csv = REPORTS_DIR / "table2_performance_metrics.csv"
    table2.to_csv(table2_csv, index=False)
    print(f"✓ Saved CSV: {table2_csv}")

    table2_latex = REPORTS_DIR / "table2_performance_metrics.tex"
    generate_latex_table2(table2, table2_latex)
    print(f"✓ Saved LaTeX: {table2_latex}")
    
    print("\nTable 2 Preview (first 15 rows):")
    print(table2.head(15).to_string(index=False))
    print(f"... ({len(table2)} total rows)")
    
    # Generate summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100 + "\n")
    
    print("TABLE 1: Cache System Characteristics")
    print("-" * 100)
    print("This table shows GLOBAL metrics that are independent of specific scenarios:")
    print("  - Average cache entries (how many unique configurations cached)")
    print("  - Average cache size in KB (total memory footprint)")
    print("  - Average entry size (memory per configuration)")
    print("  - Cache overhead (negligible, <20 KB total)")
    print()
    print("Key Finding: Cache infrastructure has minimal memory overhead regardless")
    print("             of gamma version or optimization algorithm.\n")
    
    print("TABLE 2: Performance Metrics by Scenario")
    print("-" * 100)
    print("This table shows SCENARIO-DEPENDENT metrics that vary based on:")
    print("  - Environmental conditions (temperature, duration)")
    print("  - Gamma version (V1 coarse vs V2 fine-grained)")
    print("  - Optimization algorithm (MILP, LP, GA, LNS)")
    print()
    
    # Compute summary stats
    v1_df = df[df['gamma_version'] == 'V1 (Old Gamma)']
    v2_df = df[df['gamma_version'] == 'V2 (New Gamma)']
    
    print("V1 (Coarse) Average Performance:")
    print(f"  Hit Rate: {v1_df['hit_rate_pct'].mean():.1f}%")
    print(f"  Speedup: {v1_df['speedup_factor'].mean():.1f}×")
    print(f"  P50 Latency: {v1_df['latency_p50_ms'].mean():.2f} ms")
    print()
    
    print("V2 (Fine-Grained) Average Performance:")
    print(f"  Hit Rate: {v2_df['hit_rate_pct'].mean():.1f}%")
    print(f"  Speedup: {v2_df['speedup_factor'].mean():.1f}×")
    print(f"  P50 Latency: {v2_df['latency_p50_ms'].mean():.2f} ms")
    print()
    
    print("="*100)
    print("TABLES GENERATED SUCCESSFULLY!")
    print("="*100 + "\n")
    
    print("Files created:")
    print(f"  - {table1_csv.name} (CSV)")
    print(f"  - {table1_latex.name} (LaTeX)")
    print(f"  - {table2_csv.name} (CSV)")
    print(f"  - {table2_latex.name} (LaTeX)")
    print()


if __name__ == "__main__":
    main()
