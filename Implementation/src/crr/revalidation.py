"""Algorithm 1 -- the 4-stage Certified Refinement Revalidation pipeline.

Each entry either certifies exactly (dashed exits) or escalates to the next,
costlier stage, so the population narrows from all entries to only those whose
integer optimum genuinely changed -- the sole entries that reach a full re-solve.
Stage 3 is a genuine warm dual-simplex repair of the fixed-config bin-packing LP
(warm-started from the entry's stored basis); Stage 4 is the self-contained
branch-and-bound. No external solver is used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .cache import Cache
from .certificate import Entry
from .metrics import CRRMetrics
from .model import OptModel
from .refinement import Refinement


@dataclass
class RevalResult:
    config_idx: Optional[int]
    value: float
    feasible: bool
    stage: str


def _cheapest_config(model: OptModel) -> int:
    costs = [(model.config_cost(k), k) for k in range(len(model.scenario.configs))]
    costs.sort()
    return costs[0][1]


def crr_revalidate(cache: Cache, ref: Refinement, tau_req: float = 0.0,
                   collect_pivot_stats: bool = False
                   ) -> Tuple[Dict[Entry, RevalResult], CRRMetrics]:
    """Revalidate ``cache`` under refinement ``ref`` via the 4-stage pipeline.

    ``collect_pivot_stats`` additionally cold-solves each Stage-3 repair to report
    the warm-vs-cold pivot comparison. That solve is instrumentation, not part of
    the algorithm, so its cost is excluded from ``wall_s``.
    """
    t0 = time.perf_counter()
    instrumentation_s = 0.0
    m = CRRMetrics(total_entries=len(cache.entries))
    results: Dict[Entry, RevalResult] = {}

    # ---- Stage 1: directory lookup at the footprint --------------------------
    affected = cache.index.query(ref.footprint)
    if ref.kind == "I":
        affected = affected | cache.index.adjacent(affected)

    for e in cache.entries:
        if e not in affected:
            results[e] = RevalResult(e.config_idx, e.value, True, "S1")
            m.stage_counts["S1"] += 1
            m.n_none += 1
            continue

        M1 = ref.apply(cache.models[e]) if ref.applies_to(e) else cache.models[e]

        # ---- Stage 2: certificate survival (no solver) -----------------------
        m.n_arith += 1
        cheapest = _cheapest_config(M1)
        ranking_preserved = (
            cheapest == e.config_idx
            or abs(M1.config_cost(cheapest) - M1.config_cost(e.config_idx)) <= tau_req
        )
        handled = False
        if ranking_preserved and M1.plan_feasible(e.config_idx, e.assignment):
            results[e] = RevalResult(e.config_idx, M1.config_cost(e.config_idx), True, "S2")
            m.stage_counts["S2"] += 1
            handled = True

        # ---- Stage 3: warm simplex repair (same optimum, re-assign) ----------
        elif ranking_preserved and "binpack_basis" in e.cert:
            m.n_lp += 1
            basic, at_upper = e.cert["binpack_basis"]
            if ref.kind == "II":
                # Type II changes only ``usable`` -> the battery RHS. A and c are
                # untouched, so the stored basis stays dual-feasible and the dual
                # simplex is the valid warm start.
                lp = M1.binpack_warm(e.config_idx, basic, at_upper)
                m.warm_pivots += lp.n_pivots
            else:
                # Type I/III change the energy matrix E, which in this model enters
                # the battery CONSTRAINT rows (A_ub), not the objective -- the LP's
                # objective is a pure tie-break. A changed A preserves neither primal
                # nor dual feasibility of the stored basis, so no warm start is valid
                # and the repair must cold-solve.
                lp = M1.binpack_lp(e.config_idx)
                m.cold_pivots += lp.n_pivots
            if collect_pivot_stats and ref.kind == "II":
                # Instrumentation only: what a cold solve of the same warm-started
                # repair would cost. Timed separately so it never inflates wall_s.
                # Only meaningful for Type II, the sole warm-startable case here.
                t_i = time.perf_counter()
                m.cold_pivots += M1.binpack_lp(e.config_idx).n_pivots
                instrumentation_s += time.perf_counter() - t_i
            # LP feasible => the cheapest config may still be packable. Try a cheap
            # rounding of the warm LP first; only confirm with the engine if needed.
            if lp.status == "optimal":
                assignment = M1.round_assignment(lp.x)
                if not M1.plan_feasible(e.config_idx, assignment):
                    feasible, assignment = M1.binpack_feasible(e.config_idx)
                else:
                    feasible = True
                if feasible:
                    results[e] = RevalResult(e.config_idx, M1.config_cost(e.config_idx), True, "S3")
                    m.stage_counts["S3"] += 1
                    handled = True

        # ---- Stage 4: full re-solve (own branch-and-bound) -------------------
        if not handled:
            m.n_milp += 1
            r = M1.solve()
            results[e] = RevalResult(r.config_idx, r.value, r.feasible, "S4")
            m.stage_counts["S4"] += 1

    m.wall_s = (time.perf_counter() - t0) - instrumentation_s
    return results, m
