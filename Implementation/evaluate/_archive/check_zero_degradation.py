import json

from paths import ROBUSTNESS_RESULTS_DIR

results_file = ROBUSTNESS_RESULTS_DIR / "final" / "weather_180_temp_15_v2" / "robustness_analysis_results.json"
data = json.load(open(results_file))

print("=== 0% DEGRADATION - ENERGY VALUES ===\n")
df0 = data['results_by_degradation']['degradation_0.000']

for strategy_name, results in df0.items():
    if results['timestamps']:
        avg_energy = sum(results['real_energies']) / len(results['real_energies'])
        print(f"{strategy_name:35s}: {avg_energy:.2f} Wh")

print("\n=== EXPLANATION ===")
print("At 0% degradation (perfect model), different strategies still produce")
print("different energy values because they use different optimization methods:")
print("  - DMILP: Exact MILP solver over discretized space")
print("  - LMILP: Linear/relaxed approximation")
print("  - GA: Genetic Algorithm (heuristic)")
print("  - LNS: Large Neighborhood Search (heuristic)")
print("\nHeuristics find different (possibly local) optima.")
print("Even exact methods may differ due to discretization granularity.")
