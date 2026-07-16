"""Metrics for CRR revalidation and the baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CRRMetrics:
    """Per-revalidation accounting: stage distribution and solver-call types."""

    total_entries: int = 0
    stage_counts: Dict[str, int] = field(
        default_factory=lambda: {"S1": 0, "S2": 0, "S3": 0, "S4": 0})
    n_milp: int = 0    # expensive full MILP re-solves (Stage 4)
    n_lp: int = 0      # cheap warm dual-simplex repairs (Stage 3)
    n_arith: int = 0   # certificate checks (Stage 2)
    n_none: int = 0    # reused untouched (Stage 1)
    warm_pivots: int = 0   # dual-simplex pivots used by warm Stage-3 repairs
    cold_pivots: int = 0   # pivots a cold LP solve of the same repairs would take
    wall_s: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "total_entries": self.total_entries,
            "stage_counts": dict(self.stage_counts),
            "n_milp": self.n_milp,
            "n_lp": self.n_lp,
            "n_arith": self.n_arith,
            "n_none": self.n_none,
            "warm_pivots": self.warm_pivots,
            "cold_pivots": self.cold_pivots,
            "wall_s": self.wall_s,
        }


@dataclass
class StaleMetrics:
    """Cost of keeping a stale cache (the prior no-revalidation approach)."""

    total_entries: int = 0
    violations: int = 0            # entries now infeasible under M1
    suboptimal: int = 0            # entries now suboptimal under M1
    mean_suboptimality: float = 0.0
    max_suboptimality: float = 0.0

    @property
    def violation_rate(self) -> float:
        return self.violations / max(1, self.total_entries)

    def as_dict(self) -> Dict:
        return {
            "total_entries": self.total_entries,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "suboptimal": self.suboptimal,
            "mean_suboptimality": self.mean_suboptimality,
            "max_suboptimality": self.max_suboptimality,
        }
