# Zonotope-Based MILP Caching for Adaptive Drone Mission Planning

A research codebase for caching MILP optimization results using **zonotopes** as the
region representation, applied to adaptive multi-drone mission planning under changing
weather / operating conditions (ODDs).

Instead of re-solving the mission-planning MILP every time the operating conditions
change, the system caches solutions over regions of the design space and reuses them
when a new situation falls inside an already-solved region.

## The two caching approaches

| Module | Approach | Role |
|--------|----------|------|
| `src/zonotope_cache.py` | **Baseline** — stores a solution per region, no re-validation of cached solutions | comparison baseline |
| `src/smart_cache.py` | **Smart cache** — region *extension* + selective re-optimization + intelligent merging | main method |

The smart cache algorithm, given a new design space `A` and existing cache `C`:

1. Compute the unexplored part `A \ C`.
2. Optimize **only** in the unexplored region (not the whole design space).
3. If the unexplored objective is no better than the cached one, **extend** the cached
   region to cover it; otherwise create a **new** entry with the better solution.
4. Try to **merge** the new entry with neighbours, keeping the best objective.

## Project layout

```
Implementation/                     # <- run everything from here
├── paths.py                        # PROJECT_ROOT / ASSETS_DIR / RESULTS_DIR ... constants
├── requirements.txt
├── assets/                         # input configs: fleet.json, drone.json, smart_strategy*.json
├── src/                            # core library  (import as `from src....`)
│   ├── domain.py  zonotope_ops.py  zonotope_cache.py  smart_cache.py
│   ├── milp_solver.py  milp_solver_v2.py  strategy.py  environment.py
│   ├── memory_utils.py  utils.py  visualization.py
├── evaluate/                       # experiments & analysis (import as `from evaluate....`)
│   ├── implementation_specifics/   # domain models, strategies, weather model, viz helpers
│   ├── experiments/                # the runnable experiment drivers + wrappers
│   ├── reporting/                  # results -> metrics / tables / figures
│   ├── tests/                      # smoke + verification checks
│   └── _archive/                   # one-off debug/diagnostic scripts (not part of the pipeline)
├── examples/                       # standalone tutorials (basic_2d, drone_4d)
├── docs/                           # documentation (see below)
└── results/                        # all experiment outputs
    ├── cache/{final_old_gamma,final_new_gamma}/   # cache-effectiveness runs (per scenario)
    ├── robustness/final/                          # model-mismatch robustness run
    ├── scalability/                               # 100-drone scalability run
    └── reports/                                   # metrics csv/xlsx, LaTeX tables, figures/
```

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running

Everything is run **from the `Implementation/` directory** using module execution
(`python -m ...`). Paths are anchored to the project root, so the working directory
does not matter as long as you launch from here.

### Experiments

```bash
# Cache-effectiveness study (SmartCache vs baselines over weather scenarios)
python -m evaluate.experiments.cache_effectiveness

# Robustness study (sensitivity to energy-model mismatch, 0-N% degradation)
python -m evaluate.experiments.robustness
python -m evaluate.experiments.run_robustness_tests quick     # convenience presets: quick | paper | multi
python -m evaluate.experiments.run_paper_robustness           # full paper run + enhanced figures
```

### Reporting (run after the experiments, in this order)

```bash
python -m evaluate.reporting.generate_paper_metrics      # -> results/reports/paper_metrics_combined.csv
python -m evaluate.reporting.generate_paper_tables       # LaTeX tables
python -m evaluate.reporting.quick_comparison            # V1 vs V2 summary
python -m evaluate.reporting.visualize_metrics           # comparison figures
python -m evaluate.reporting.create_coverage_cache_figures
python -m evaluate.reporting.create_table_diagrams
python -m evaluate.reporting.paper_graphs                # paper figures from raw results
```

### Examples & tests

```bash
python -m examples.basic_2d
python -m examples.drone_4d
python -m evaluate.tests.smoke_smart_cache
python -m evaluate.tests.verify_robustness_implementation
```

## Results & version control

Result **data** (JSON / CSV / TXT / TEX / XLSX) is kept under version control; heavy
**regenerable** figures (PNG / PGF / PDF under `results/`) are git-ignored and produced
by the reporting scripts. The canonical inputs the paper is built from are:

- `results/cache/final_old_gamma/<scenario>/drone_6d_large_cs_results.json` (and `final_new_gamma`)
- `results/robustness/final/weather_180_temp_15_v2/robustness_analysis_results.json`
- `results/reports/paper_metrics_combined.csv`

## Documentation

- [`docs/evaluation.md`](docs/evaluation.md) — **CRR evaluation** (RQ1–3): Certified Refinement Revalidation on model evolution (`src/crr/`, `evaluate/experiments/crr_*.py`)
- [`docs/specs/crr-evaluation-design.md`](docs/specs/crr-evaluation-design.md) — the CRR evaluation spec/design
- [`docs/experiments.md`](docs/experiments.md) — experiment definitions and methodology
- [`docs/robustness.md`](docs/robustness.md) — the model-mismatch robustness study
- [`docs/reporting.md`](docs/reporting.md) — the metrics / tables / figures pipeline
- [`docs/REORGANIZATION.md`](docs/REORGANIZATION.md) — how this repo was reorganized (change log)

## Dependencies

numpy, scipy, pulp (MILP), matplotlib, seaborn, pandas, openpyxl, pyyaml
(optional, imported lazily: psutil, scikit-learn). See `requirements.txt`.
