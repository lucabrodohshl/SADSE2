"""Baselines: full revalidation (ground truth) and no revalidation (stale cache)."""
from __future__ import annotations

import time
from typing import Dict, Tuple

from .cache import Cache
from .certificate import Entry
from .metrics import CRRMetrics, StaleMetrics
from .refinement import Refinement
from .revalidation import RevalResult


def full_revalidation(cache: Cache, ref: Refinement) -> Tuple[Dict[Entry, RevalResult], CRRMetrics]:
    """Re-solve the MILP for every entry under M1 -- the naive upper bound = N solves.

    Also serves as the ground truth for CRR's soundness/optimality check.
    """
    t0 = time.perf_counter()
    m = CRRMetrics(total_entries=len(cache.entries))
    results: Dict[Entry, RevalResult] = {}
    for e in cache.entries:
        M1 = ref.apply(cache.models[e]) if ref.applies_to(e) else cache.models[e]
        r = M1.solve()
        results[e] = RevalResult(r.config_idx, r.value, r.feasible, "S4")
        m.n_milp += 1
        m.stage_counts["S4"] += 1
    m.wall_s = time.perf_counter() - t0
    return results, m


def no_revalidation(cache: Cache, ref: Refinement) -> StaleMetrics:
    """Keep the M0 cache unchanged and quantify how wrong it is under M1.

    The stale approach itself performs no solve; the re-solve here is *instrumentation*
    to measure the feasibility violations and optimality gap it incurs.
    """
    m = StaleMetrics(total_entries=len(cache.entries))
    subopt = []
    for e in cache.entries:
        M1 = ref.apply(cache.models[e]) if ref.applies_to(e) else cache.models[e]
        if not M1.plan_feasible(e.config_idx, e.assignment):
            m.violations += 1
        truth = M1.solve()
        if truth.feasible:
            stale_value = M1.config_cost(e.config_idx)
            gap = (stale_value - truth.value) / max(1e-9, abs(truth.value))
            if gap > 1e-6:
                m.suboptimal += 1
                subopt.append(gap)
    if subopt:
        m.mean_suboptimality = float(sum(subopt) / len(subopt))
        m.max_suboptimality = float(max(subopt))
    return m
