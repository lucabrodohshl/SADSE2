# Metrics, Tables & Figures Reporting Pipeline

This document is the single reference for the reporting pipeline that turns raw
cache-effectiveness / robustness experiment results into publication-ready
metrics, LaTeX tables, and figures. It consolidates what previously lived in
`METRICS_README.md`, `TABLES_README.md`, `TABLE_SUMMARY.md`, and
`TABLE_QUICK_REFERENCE.txt`.

## Overview

The reporting scripts extract metrics from the experiment result JSONs and emit:

- **Metrics** as CSV / Excel (per-gamma-version and combined) plus a
  human-readable text report.
- **LaTeX tables** ready for `\input` into the paper.
- **Figures** (PNG & PDF) comparing the two gamma versions and covering
  hit-rate, speedup, latency, and coverage.

Everything is driven off two experiment result sets, one per **gamma version**:

- **V1 (old gamma, coarse discrete)** — broad environmental thresholds, fewer
  unique configurations.
- **V2 (new gamma, fine-grained discrete)** — precise environmental thresholds,
  more unique configurations.

All commands below are run **from the `Implementation/` root** via Python module
execution (`python -m ...`).

## Data Sources

Experiment result JSONs are read from:

- `results/cache/final_old_gamma/<scenario>/` — V1 (coarse discrete) results.
- `results/cache/final_new_gamma/<scenario>/` — V2 (fine-grained discrete) results.

Related experiment outputs:

- Robustness results: `results/robustness/final/`
- Scalability results: `results/scalability/`

Each gamma directory contains the three scenarios:

- `weather_scenario_90_temp_15_realistic/` — 90 s duration, 15 °C
- `weather_scenario_180_temp_25_realistic/` — 180 s duration, 25 °C
- `weather_scenario_330_temp_5_realistic/` — 330 s duration, 5 °C

Each scenario contains `drone_6d_large_cs_results.json` with the event-based
data consumed by the reporting scripts. Shared assets (`fleet.json`, etc.) live
in the single top-level `assets/` directory.

All generated metrics, tables, and figures are written under
`results/reports/` (figures in `results/reports/figures/`).

## Pipeline

Run the reporting scripts **in this order**. `generate_paper_metrics` must run
first because every downstream script consumes the metrics it produces
(`paper_metrics_combined.csv` and friends).

If experiments themselves need to be (re)run first, the experiment drivers are:

```bash
# Cache-effectiveness experiments (V1/V2 gamma sweeps)
python -m evaluate.experiments.cache_effectiveness ...

# Robustness experiments
python -m evaluate.experiments.robustness ...
python -m evaluate.experiments.run_robustness_tests <opt>
python -m evaluate.experiments.run_paper_robustness
```

Then run the reporting pipeline, in order:

```bash
# 1. Extract metrics -> CSV / Excel / LaTeX / text report (RUN FIRST)
python -m evaluate.reporting.generate_paper_metrics

# 2. Build the two LaTeX/CSV paper tables (Table 1 & Table 2)
python -m evaluate.reporting.generate_paper_tables

# 3. Quick console comparison of V1 vs V2
python -m evaluate.reporting.quick_comparison

# 4. Publication-ready comparison visualizations
python -m evaluate.reporting.visualize_metrics

# 5. Coverage & cache figures
python -m evaluate.reporting.create_coverage_cache_figures

# 6. Table structure / data-flow explanatory diagrams
python -m evaluate.reporting.create_table_diagrams

# 7. Full paper graph set
python -m evaluate.reporting.paper_graphs
```

`generate_paper_tables` in particular:

1. Loads the latest metrics from `paper_metrics_combined.csv`.
2. Extracts memory data from the `memory_efficiency_report.txt` files.
3. Generates fresh LaTeX and CSV files for both tables.

## Generated Files

Written under `results/reports/` (figures under `results/reports/figures/`).

### Metrics (CSV / Excel)

- `paper_metrics_combined.csv` — all metrics from both gamma versions
- `paper_metrics_combined.xlsx` — Excel version with multiple sheets
- `paper_metrics_v1_(old_gamma).csv` — V1 (coarse discrete) metrics only
- `paper_metrics_v1_(old_gamma).xlsx` — V1 Excel version
- `paper_metrics_v2_(new_gamma).csv` — V2 (fine-grained discrete) metrics only
- `paper_metrics_v2_(new_gamma).xlsx` — V2 Excel version

### LaTeX Tables

- `paper_table_comparison.tex` — combined V1 vs V2 comparison table
- `paper_table_v1_(old_gamma).tex` — V1 results table
- `paper_table_v2_(new_gamma).tex` — V2 results table
- `table1_cache_characteristics.tex` / `.csv` — Table 1 (global cache characteristics)
- `table2_performance_metrics.tex` / `.csv` — Table 2 (scenario-dependent performance)

### Figures (PNG & PDF)

In `results/reports/figures/`:

- `summary_figure.png/pdf` — comprehensive overview figure (recommended for paper)
- `hit_rate_comparison.png/pdf` — cache hit rate analysis
- `speedup_comparison.png/pdf` — speedup factor comparison
- `latency_percentiles.png/pdf` — P50/P95/P99 latency analysis
- `hit_vs_miss_latency.png/pdf` — cache hit vs miss latency
- `scenario_comparison.png/pdf` — performance across scenarios
- `table_comparison.png` — visual explanation of the two-table structure
- `data_organization.png` — data-flow diagram

### Summary Report

- `paper_metrics_report.txt` — human-readable summary of all metrics

## Metrics Computed

`generate_paper_metrics` computes the following.

### 1. Cache Hit Rate (%)

- Formula: `(hits / total_requests) × 100`
- Measures cache effectiveness.
- **V1 Average:** 95.07% (Range: 92.50% – 98.33%)
- **V2 Average:** 80.90% (Range: 68.33% – 89.17%)

### 2. Cache Hit vs Miss Latency (CvM)

- Separate analysis of hit and miss latencies.
- Median and mean values computed.
- CvM ratio: `hit_latency / miss_latency`.

### 3. Total Latency Percentiles

- **P50 (Median):** 50th percentile latency
- **P95:** 95th percentile latency (captures most outliers)
- **P99:** 99th percentile latency (worst-case scenarios)

### 4. Speedup Factor

- Formula: `baseline_latency / cache_latency`
- Compares cache strategy to its corresponding baseline.
- **V1 Average:** 26.38× (Range: 11.97× – 59.42×)
- **V2 Average:** 5.65× (Range: 3.13× – 8.62×)

### 5. Cache Memory Footprint

- Peak, mean, and final memory usage (MB).
- **Note:** Not currently tracked in results (rerun needed to populate).

### 6. Cache Size

- Peak, mean, and final cache entries.
- **Note:** Not currently tracked in results (rerun needed to populate).

## Strategies Analyzed

### Cache Strategies (Smart Cache)

1. **DISCRETE** — Discrete optimization (MILP)
2. **LINEAR** — Linear programming relaxation (LP)
3. **GA** — Genetic Algorithm
4. **LNS** — Large Neighborhood Search

### Baseline Strategies (No Cache)

1. **BASELINE DISCRETE** — Discrete optimization without caching
2. **BASELINE LINEAR** — Linear programming without caching
3. **GENETIC ALGORITHM** — GA without caching
4. **LARGE NEIGHBORHOOD SEARCH** — LNS without caching

## The V1 vs V2 Gamma Comparison

The core result is a trade-off between **cache effectiveness** and
**environmental adaptation precision**, driven by the discretization
granularity of the gamma function.

### V1 (Old Gamma — Coarse Discrete)

**Philosophy:** use broader environmental thresholds.

- **Higher hit rate:** 95.07% average
- **Much higher speedup:** 26.38× average
- **Lower latency P50:** 0.54 ms average
- **Pros:** higher cache hit rate, better speedup, minimal latency.
- **Cons:** less precise environmental adaptation; configurations may not be
  optimal for exact conditions.
- **Interpretation:** coarse discretization creates fewer unique configurations,
  leading to more cache hits but potentially less optimal task assignments.
- **Best for:** systems prioritizing **speed over precision**.

### V2 (New Gamma — Fine-Grained Discrete)

**Philosophy:** use precise environmental thresholds.

- **Lower hit rate:** 80.90% average (but still good!)
- **Moderate speedup:** 5.65× average
- **Slightly higher latency P50:** 1.06 ms average
- **Pros:** more accurate environmental modeling, better task assignment quality,
  still-good cache performance (81% hit rate).
- **Cons:** lower cache hit rate, moderate speedup, slightly higher latency.
- **Interpretation:** fine-grained discretization creates more unique
  configurations, leading to more precise environmental adaptation but fewer
  cache hits.
- **Best for:** systems prioritizing **precision over speed**.

**Design choice:** cache effectiveness ↔ solution optimality.

## Tables

Two clear, well-structured tables separate **global / strategy-independent**
cache properties from **scenario-dependent** performance outcomes. Mixing these
would confuse infrastructure properties with performance outcomes.

- **Table 1** answers: *"What does the cache COST?"* → Nothing (~20 KB).
- **Table 2** answers: *"How well does the cache PERFORM?"* → Depends on scenario.

| Aspect | Table 1 (Global) | Table 2 (Scenario-Dependent) |
|--------|------------------|------------------------------|
| **Focus** | Cache infrastructure | Performance outcomes |
| **Variability** | Minimal (consistent across scenarios) | High (varies by scenario) |
| **Use Case** | Show cache overhead is negligible | Show cache effectiveness trade-offs |
| **Message** | "Cache is lightweight" | "V1 = high cache hit, V2 = high precision" |
| **Grouping** | By gamma version and algorithm | By gamma, scenario, and algorithm |

### Table 1: Cache System Characteristics (Global)

**Purpose:** cache infrastructure properties that are **independent of specific
scenarios**. Describes the cache infrastructure itself, not how well it performs
("how much does the cache weigh?", always ~10–13 KB).

**Files:** `table1_cache_characteristics.tex` (LaTeX) and `.csv`
**Shape:** 8 rows (4 algorithms × 2 gamma versions), 6 columns.

#### Columns

| Column | Description | Why It's Global |
|--------|-------------|-----------------|
| **Gamma** | V1 (Coarse) or V2 (Fine-Grained) | Grouping key |
| **Algorithm** | GA, LP, LNS, or MILP | Grouping key |
| **Avg Entries** | Average number of cached configurations | Relatively stable across scenarios (~11–12 entries) |
| **Avg Size (KB)** | Total cache memory footprint | Depends only on configuration representation |
| **Entry Size (KB)** | Memory per cached configuration | Fixed by data structure (~0.85 KB for most, 1.13 KB for LP) |
| **Overhead** | Total cache memory cost | Always negligible (<20 KB) regardless of scenario |

#### Key Insights

1. **Minimal memory overhead:** cache uses <20 KB total, negligible compared to
   overall system memory.
2. **Consistent across gamma versions:** V1 and V2 have similar cache sizes
   despite different discretization granularities (~11.7 entries, ~10–13 KB
   total each).
3. **Algorithm impact:** only LP Relaxation has slightly larger entries
   (1.13 KB vs 0.86 KB) due to continuous variables.
4. **Scalability:** cache overhead doesn't grow with scenario complexity.

#### Structure

```latex
V1 (Coarse)
  - Genetic Algorithm:          11.7 entries, 10.0 KB, 0.86 KB/entry
  - LP Relaxation:              11.7 entries, 13.2 KB, 1.13 KB/entry
  - Large Neighborhood Search:  11.7 entries, 10.0 KB, 0.86 KB/entry
  - MILP (Discrete):            11.7 entries, 10.0 KB, 0.86 KB/entry

V2 (Fine-Grained)
  - [Same structure]
```

### Table 2: Performance Metrics by Scenario (Scenario-Dependent)

**Purpose:** performance metrics that **vary based on environmental conditions,
gamma version, and optimization algorithm** ("how fast can you run with it?",
depends on terrain).

**Files:** `table2_performance_metrics.tex` (LaTeX) and `.csv`
**Shape:** 24 rows (2 gamma × 3 scenarios × 4 algorithms).

#### Columns

| Column | Description | Why It's Scenario-Dependent |
|--------|-------------|-----------------------------|
| **Gamma** | V1 (Coarse) or V2 (Fine-Grained) | Grouping key |
| **Scenario** | Duration (s) and Temperature (°C) | Environmental conditions affect task complexity |
| **Algorithm** | MILP, LP, GA, or LNS | Different optimization approaches |
| **Hit Rate (%)** | Cache effectiveness | Depends on environmental variability and gamma granularity |
| **Speedup (×)** | Performance vs. baseline | Varies with hit rate and task complexity |
| **P50/P95/P99 (ms)** | Latency percentiles | Depends on cache hits, misses, and recomputation cost |
| **Coverage (%)** | Area coverage achieved | Depends on scenario duration and conditions |

#### Key Insights — By Gamma Version

- **V1 (Coarse):**
  - Hit Rate: 92.5% – 98.3% (excellent)
  - Speedup: 12.0× – 59.4× (very high)
  - P50 Latency: 0.0 – 1.0 ms (very fast)
  - Trade-off: higher cache effectiveness, but less precise environmental adaptation.
- **V2 (Fine-Grained):**
  - Hit Rate: 68.3% – 89.2% (good)
  - Speedup: 3.1× – 8.6× (moderate)
  - P50 Latency: 1.0 – 1.4 ms (fast)
  - Trade-off: more precise environmental adaptation, but lower cache effectiveness.

#### Key Insights — By Scenario

- **90 s, 15 °C** (short, moderate temp):
  - Highest hit rates (82.5% – 98.3%)
  - Best speedups (5.2× – 59.4×)
  - Best coverage (58.3% – 72.7%)
  - **V1:** 98.3% hit rate, 43–59× speedup, 72.7% coverage.
  - **V2:** 82.5–84.2% hit rate, 5–6× speedup, 58.3% coverage.
- **180 s, 25 °C** (medium, warm):
  - Moderate hit rates (89.2% – 97.5%)
  - Good speedups (7.5× – 15.0×)
  - Moderate coverage (50.9% – 54.5%)
  - **V1:** 93.3–97.5% hit rate, 12–15× speedup, 54.5% coverage.
  - **V2:** 89.2% hit rate, 7–9× speedup, 50.9% coverage.
- **330 s, 5 °C** (long, cold — most challenging):
  - Hit rate 68.3% – 92.5%
  - Lower speedups (3.1× – 13.6×)
  - Lowest coverage (18.3% – 29.4%)
  - **V1:** 92.5% hit rate, 12–14× speedup, 29.4% coverage.
  - **V2:** 68.3–70.8% hit rate, 3–4× speedup, 18.3% coverage.
  - Cold temperature limits drone performance.

#### Key Insights — By Algorithm

- **MILP (Discrete):** highest hit rates, best for cache.
- **LP Relaxation:** good balance, moderate hit rates.
- **GA (Genetic Algorithm):** lower hit rates, highest P99 latencies.
- **LNS (Large Neighborhood Search):** good hit rates, balanced performance.

#### Structure

```latex
V1 (Coarse)
  180s, 25°C
    - GA:   93.3% hit, 14.5× speedup, 0.00ms P50, ...
    - LNS:  93.3% hit, 15.0× speedup, 1.00ms P50, ...
    - LP:   ...
    - MILP: ...
  330s, 5°C
    - [4 algorithms]
  90s, 15°C
    - [4 algorithms]

V2 (Fine-Grained)
  [Same scenario structure]
```

## Recommended for Paper

### Main Figure

Use `summary_figure.png/pdf` as the primary figure, showing:

- Overall hit rate comparison (panel A)
- Overall speedup comparison (panel B)
- Overall latency comparison (panel C)
- Per-strategy hit rates (panel D)
- Per-strategy speedup factors (panel E)

### Main Table

Use `paper_table_comparison.tex` for the comprehensive metrics table (all
scenarios and strategies; hit rates, latencies P50/P95/P99, speedup factors;
side-by-side V1 vs V2). Alternatively, use the cleaner two-table split
(`table1_cache_characteristics.tex` + `table2_performance_metrics.tex`).

### Supplementary Figures

- `hit_rate_comparison.png/pdf` — detailed hit rate analysis
- `speedup_comparison.png/pdf` — detailed speedup analysis
- `latency_percentiles.png/pdf` — comprehensive latency breakdown

### Recommended Placement

1. **Table 1 → "Cache Implementation" / "System Design":**
   > "As shown in Table 1, the cache infrastructure has negligible memory
   > overhead (<20 KB) regardless of the gamma function version or optimization
   > algorithm."
2. **Table 2 → "Experimental Results" / "Performance Evaluation":**
   > "Table 2 demonstrates that V1 achieves higher cache hit rates (95.1%
   > average) but V2 provides more precise environmental adaptation while
   > maintaining good performance (80.9% average hit rate)."

### Example Text Snippets

**For Table 1:**
> The cache system maintains a small memory footprint (Table 1), with an average
> of 11.7 cached configurations consuming less than 20 KB of memory. This
> overhead remains constant regardless of scenario complexity, demonstrating the
> scalability of our approach. The LP relaxation algorithm requires slightly more
> memory per entry (1.13 KB) due to continuous variable representations, compared
> to 0.86 KB for discrete algorithms.

**For Table 2:**
> Table 2 presents performance metrics across three environmental scenarios. V1
> (coarse discretization) achieves cache hit rates between 92.5% and 98.3%,
> resulting in speedup factors of 12× to 59×. V2 (fine-grained discretization)
> trades cache effectiveness for environmental precision, achieving 68.3% to
> 89.2% hit rates with 3× to 9× speedups. The 90-second, 15 °C scenario shows the
> best performance for both versions, while the 330-second, 5 °C scenario is most
> challenging due to cold temperature constraints on drone operation.

**For Discussion:**
> Our results demonstrate a clear trade-off between cache effectiveness and
> environmental adaptation precision (Tables 1–2). While both gamma versions
> maintain negligible memory overhead (<20 KB, Table 1), V1's coarser
> discretization creates fewer unique configurations, leading to higher cache hit
> rates but potentially suboptimal task assignments. V2's fine-grained approach
> reduces cache effectiveness but improves solution quality for specific
> environmental conditions.

## Citation-Ready Statistics

**For abstract/introduction:**

- "Our cache achieves hit rates of 95.07% (V1) and 80.90% (V2)."
- "Speedup factors range from 5.65× to 26.38× depending on discretization granularity."
- "Median latency reduced to <2 ms in both configurations."

**For discussion:**

- "Trade-off between cache effectiveness (hit rate) and solution optimality (fine-grained constraints)."
- "V1's coarser discretization yields 95% hit rate but less environmental precision."
- "V2's fine-grained approach reduces hit rate to 81% but provides more accurate environmental adaptation."

## LaTeX Styling Notes

Both tables use:

- `\begin{table*}` for two-column wide format
- `booktabs` for professional rules (`\toprule`, `\midrule`, `\bottomrule`)
- `\small` font for compact presentation
- `tablenotes` (via `threeparttable`) for detailed explanations
- Proper alignment: left for text, right for numbers

Paper preamble:

```latex
\usepackage{booktabs}
\usepackage{threeparttable}
```

In the document body:

```latex
\input{table1_cache_characteristics.tex}
\input{table2_performance_metrics.tex}
```

## Reviewer Soundbites & FAQ

### Key messages

**From Table 1:**

- Cache overhead is negligible (<20 KB total).
- Implementation is lightweight and scalable.
- Memory cost doesn't depend on scenario complexity.

**From Table 2:**

- V1 achieves 95% hit rate with 26× average speedup (cache-optimized).
- V2 achieves 81% hit rate with 6× average speedup (precision-optimized).
- Both versions maintain sub-2 ms median latency.
- Performance varies with environmental conditions (temperature, duration).
- Design choice: cache effectiveness vs. environmental precision.

### One-sentence summaries

- **Table 1:** "The cache infrastructure has a constant, negligible memory
  footprint (<20 KB) regardless of scenario complexity."
- **Table 2:** "Cache effectiveness varies with environmental conditions, with V1
  achieving 95% hit rate (coarse) and V2 achieving 81% hit rate (precise)."

### Common reviewer questions

- **"Why two tables instead of one?"** Table 1 shows cache infrastructure costs
  (global), Table 2 shows cache performance (scenario-dependent). Mixing these
  would confuse infrastructure properties with performance outcomes.
- **"Is the cache overhead acceptable?"** Yes — Table 1 shows <20 KB total
  overhead, negligible compared to overall system memory.
- **"Which gamma version is better?"** Table 2 shows V1 is better for cache
  effectiveness (95% hit rate), V2 is better for environmental precision (81% hit
  rate but more accurate). It's a design trade-off, not a clear winner.
- **"How does performance vary with scenarios?"** Best at 90 s/15 °C (moderate
  conditions), worst at 330 s/5 °C (cold temperature limits drones).

## Quick-Reference Cheat Sheet

```
================================================================================
                        QUICK REFERENCE CARD
                   TWO-TABLE STRUCTURE FOR PAPER
================================================================================

TABLE 1: Cache System Characteristics (GLOBAL)
  File:  table1_cache_characteristics.tex
  What:  Cache infrastructure metrics (independent of scenario)
  Rows:  8 (4 algorithms x 2 gamma versions)
  Cols:  Gamma | Algorithm | Entries | Size | Entry Size | Overhead
  Key Findings:
    - ~11.7 cached configurations on average
    - ~10-13 KB total cache size
    - <20 KB overhead (negligible!)
    - Same for V1 and V2 (independent of gamma)
  Message: "Cache is lightweight and scalable"

TABLE 2: Performance Metrics by Scenario (SCENARIO-DEPENDENT)
  File:  table2_performance_metrics.tex
  What:  Performance outcomes (varies by scenario & gamma)
  Rows:  24 (2 gamma x 3 scenarios x 4 algorithms)
  Cols:  Gamma | Scenario | Algo | Hit Rate | Speedup | P50/P95/P99 | Cov
  Key Findings:
    V1 (Coarse):       95.1% avg hit, 26.4x avg speedup, 0.54ms median latency
    V2 (Fine-Grained): 80.9% avg hit,  5.6x avg speedup, 1.06ms median latency
  Message: "V1 = high cache effectiveness, V2 = high precision"

--------------------------------------------------------------------------------
                          KEY STATISTICS
--------------------------------------------------------------------------------
                        V1 (Coarse)         V2 (Fine-Grained)
Cache Hit Rate:        95.1% +/- 2.8%       80.9% +/- 8.3%
Speedup Factor:        26.4x +/- 19.5x      5.6x +/- 2.1x
Median Latency:        0.54 +/- 0.50 ms     1.06 +/- 0.11 ms
Cache Overhead:        <20 KB               <20 KB
Cache Entries:         ~11.7                ~11.7

Best Scenario:         90s, 15C             90s, 15C
Worst Scenario:        330s, 5C             330s, 5C
================================================================================
```

Data-flow of the two-table split:

```
+-------------------------------------------------------------+
|                    RAW EXPERIMENT DATA                      |
+--------------------+----------------------------------------+
                     |
            +--------+--------+
            |                 |
      +-----v------+    +-----v----+
      |   GLOBAL   |    | SCENARIO |
      |  METRICS   |    | METRICS  |
      +-----+------+    +-----+----+
            |                 |
      +-----v------+    +-----v----+
      |  TABLE 1   |    | TABLE 2  |
      |   Cache    |    |   Perf   |
      |   Chars    |    |  Metrics |
      +------------+    +----------+
```

## Next Steps / Extending

### To add memory tracking

1. Modify simulation code to track cache size and memory usage.
2. Rerun experiments with tracking enabled.
3. Rerun `python -m evaluate.reporting.generate_paper_metrics` to include memory
   metrics; the memory columns in the CSV will then be populated.

### To add more scenarios

1. Run simulations with new weather conditions
   (`python -m evaluate.experiments.cache_effectiveness ...`).
2. Place results in `results/cache/final_old_gamma/` or
   `results/cache/final_new_gamma/`.
3. Rerun the reporting pipeline (starting with `generate_paper_metrics`) to
   regenerate all outputs.
