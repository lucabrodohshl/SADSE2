"""Project paths and import bootstrap.

Run everything from the ``Implementation/`` root using module execution, e.g.::

    python -m evaluate.experiments.cache_effectiveness
    python -m evaluate.experiments.robustness
    python -m evaluate.reporting.generate_paper_metrics

Import this module at the top of any runnable script to get project-anchored
paths that do not depend on the current working directory::

    from paths import PROJECT_ROOT, ASSETS_DIR, RESULTS_DIR

Importing it also guarantees that ``src`` and ``evaluate`` are importable as
top-level packages.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Directory that contains this file == the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

# Canonical input assets (fleet/drone/strategy configs).
ASSETS_DIR = PROJECT_ROOT / "assets"

# All experiment/reporting outputs live under here.
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_RESULTS_DIR = RESULTS_DIR / "cache"           # cache-effectiveness runs
ROBUSTNESS_RESULTS_DIR = RESULTS_DIR / "robustness"  # model-mismatch runs
SCALABILITY_RESULTS_DIR = RESULTS_DIR / "scalability"
REPORTS_DIR = RESULTS_DIR / "reports"                # metrics csv/xlsx, tables, figures
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure the project is importable (src.*, evaluate.*) even when a script is run
# directly rather than via ``python -m``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = [
    "PROJECT_ROOT",
    "ASSETS_DIR",
    "RESULTS_DIR",
    "CACHE_RESULTS_DIR",
    "ROBUSTNESS_RESULTS_DIR",
    "SCALABILITY_RESULTS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
]
