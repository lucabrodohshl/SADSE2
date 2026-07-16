# Reorganization plan & change log

This document records the structural cleanup of the `Implementation/` project
(the zonotope-based MILP caching codebase). It is both the migration plan and a
"what moved where" reference.

Branch: `chore/reorg-implementation`. All moves use `git mv` so history is preserved.

## Goals

1. **One import convention, run-from-root.** Kill the three inconsistent import
   styles and the cwd-dependent relative paths that caused duplicate `assets/`
   and a stray nested `evaluate/evaluate/`.
2. **Remove duplication.** One `implementation_specifics/`, one `assets/`.
3. **Separate source from artifacts.** All experiment outputs live under
   `results/`; heavy binaries (PNG/PGF) are gitignored, canonical JSON/CSV kept.
4. **Group scripts by role** (experiments / reporting / tests / archive).
5. **Remove dead code** and rewrite the stale docs.

## Target layout

```
Implementation/                      # run everything from here (python -m ...)
├── paths.py                         # PROJECT_ROOT / ASSETS_DIR / RESULTS_DIR
├── README.md  requirements.txt
├── assets/                          # the single authoritative asset dir
├── docs/                            # consolidated documentation
├── src/                             # core library  (import: from src....)
│   ├── zonotope_cache.py            # OLD cache: no revalidation (baseline)
│   ├── smart_cache.py               # smarter query_and_optimize cache
│   └── domain / zonotope_ops / milp_solver(+_v2) / strategy / environment /
│       memory_utils / utils / visualization
├── evaluate/                        # experiments & analysis (import: from evaluate....)
│   ├── implementation_specifics/    # THE single copy (domain_specific, strategies, vis_utils, weather)
│   ├── experiments/                 # cache_effectiveness.py, robustness.py, run_*.py
│   ├── reporting/                   # generate_paper_*, visualize_metrics, create_*, paper_graphs
│   ├── tests/                       # verify_robustness_implementation, smoke_smart_cache
│   └── _archive/                    # one-off debug scripts (kept, out of the way)
├── examples/                        # standalone tutorials (basic_2d, drone_4d)
└── results/
    ├── cache/{final_new_gamma, final_old_gamma}/
    ├── robustness/final/
    ├── scalability/revision_100_drones/
    └── reports/                     # metrics csv/xlsx, tables, figures/
```

## Key decisions

- **Not a pip package.** Run from the `Implementation/` root, e.g.
  `python -m evaluate.experiments.robustness`. `paths.py` anchors all
  filesystem access to the project root so cwd no longer matters.
- **`zonotope_cache.py` (old, no-revalidation) is kept** as the baseline the
  cache experiment compares against; `smart_cache.py` is the current approach.
- **`examples/` stays self-contained** (keeps its own simplified
  `implementation_specifics`) as a readable tutorial.
- **`setup.py` removed** (contradicted the not-a-package decision);
  `requirements.txt` is the single dependency source.

## Moves (old path -> new path)

Experiments:
- `evaluate/drone_scenario.py`            -> `evaluate/experiments/cache_effectiveness.py`
- `evaluate/drone_scenario_robustness.py` -> `evaluate/experiments/robustness.py`
- `evaluate/run_robustness_tests.py`      -> `evaluate/experiments/run_robustness_tests.py`
- `evaluate/run_paper_robustness.py`      -> `evaluate/experiments/run_paper_robustness.py`

Reporting:
- `evaluate/generate_paper_metrics.py`       -> `evaluate/reporting/`
- `evaluate/generate_paper_tables.py`        -> `evaluate/reporting/`
- `evaluate/quick_comparison.py`             -> `evaluate/reporting/`
- `evaluate/visualize_metrics.py`            -> `evaluate/reporting/`
- `evaluate/create_coverage_cache_figures.py`-> `evaluate/reporting/`
- `evaluate/create_table_diagrams.py`        -> `evaluate/reporting/`
- `paper_graphs.py` (root)                   -> `evaluate/reporting/paper_graphs.py`

Tests / archive:
- `evaluate/verify_robustness_implementation.py` -> `evaluate/tests/`
- `evaluate/test_smart_cache_tightdrone.py`      -> `evaluate/tests/smoke_smart_cache.py`
- `evaluate/check_zero_degradation.py`           -> `evaluate/_archive/`
- `evaluate/compare_scalability.py`              -> `evaluate/_archive/`
- `evaluate/debug_gamma_comparison.py`           -> `evaluate/_archive/`
- `evaluate/regenerate_robustness_plots.py`      -> `evaluate/_archive/`
- `summarize_robustness_demo.py` (root)          -> `evaluate/_archive/`

Results (canonical, kept):
- `results_FINAL_NEW_GAMMA`            -> `results/cache/final_new_gamma`
- `results_FINAL_OLD_GAMMA`            -> `results/cache/final_old_gamma`
- `results_robustness_paper_final_v2`  -> `results/robustness/final`
- `Results_for_revision`               -> `results/scalability/revision_100_drones`
- `paper_metrics_*`, `table*`, `quick_comparison.*`, `paper_figures/` -> `results/reports/`

## Removed from the working tree (still in git history)

- `evaluate/results_new_gamma_FIXED`, `evaluate/results_old_gamma_FIXED`
  (redundant reruns; `create_coverage_cache_figures.py` repointed to the FINAL dirs)
- `evaluate/evaluate/` (stray nested dir: orphan `results_100_drones` +
  `results_robustness_quick_demo`)
- `evaluate/results_robustness` (scratch default output, never read back)
- `evaluate/debug_cache_check`, `output_test/` (generated scratch)
- `evaluate/assets/` (byte-identical subset of root `assets/`, plus a stray "fleet copy.json")
- `src/memory_tracker.py` (dead: nothing imports it; `StrategyMemoryTracker`
  lives in `memory_utils.py`)

## Verification

Import-smoke every entrypoint via `python -m ...`, run the two test scripts, and
re-run the reporting scripts against the kept JSON to confirm outputs reproduce.
The expensive full MILP experiments are not re-run; imports and paths are verified.
