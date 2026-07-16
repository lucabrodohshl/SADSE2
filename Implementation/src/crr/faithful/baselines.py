"""Revalidation strategies compared on equal terms, on the paper's actual model.

Comparator set, chosen so that CRR is scored against the strongest thing that exists
rather than against a strawman:

``no_reval``
    Keep the stale cache. Not a competitor -- it measures what doing nothing costs.

``full_resolve``
    Re-solve every entry from scratch. The paper's naive upper bound (N solver calls),
    and the optimality ground truth.

``footprint_resolve``
    Re-solve only entries the refinement's footprint reaches; reuse the rest. The
    competent-engineer baseline: it uses the reverse index but no certificates.

``warm_resolve``   <-- THE COMPARATOR THAT MATTERS
    Re-solve the footprint, but warm-started from each entry's stored basis. This is
    the *prior art* the paper positions itself against (Section IV: "Incremental
    re-optimization reuses solver state: reoptimizing branch-and-bound [37], frontier
    time-shifting [38] warm-start from a previous search while preserving optimality").
    The paper's claim against it is precise: "Each, however, still runs a
    branch-and-bound *search* for one instance, merely from a better start; on the
    common path we run no search at all, re-checking a stored certificate in polynomial
    time." So CRR's contribution is only real if it beats a warm-started re-solve.

``crr``
    Algorithm 1: directory lookup -> certificate survival -> repair -> rationed re-solve.

Note on what is NOT here: there is no cheap "naive recheck" in this model, and its
absence is structural rather than an omission. ``x*_e`` is an argmin over a continuous
*region*, not the best of a finite candidate list, so there is no way to check
optimality by recomputing a handful of numbers -- you either reuse the stored proof or
you search. An earlier discretised implementation of this study sampled K candidate
configurations, which manufactured exactly such a cheap alternative and made CRR look
redundant. That was an artifact of the discretisation, not a property of CRR.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .certificate import Entry
from .refinement import Refinement
from .revalidation import Metrics, RevalResult, crr_revalidate
from .solve import solve_model


@dataclass
class Outcome:
    results: Dict[Entry, RevalResult]
    metrics: Metrics


def _blank(cache) -> Metrics:
    return Metrics(total_entries=len(cache.entries))


def no_reval(cache, ref: Refinement, backend: str = "engine") -> Dict[str, float]:
    """Keep the stale M0 cache; measure how wrong it is under M1."""
    violations = 0
    suboptimal = 0
    for e in cache.entries:
        m1 = ref.apply(cache.models[e])
        truth = solve_model(m1, backend=backend)
        if truth.status != "optimal":
            violations += 1
            continue
        # is the stored x* still feasible under M1?
        if ref.kind == "II" and ref.cut_normal is not None:
            if float(ref.cut_normal @ e.x_star) > ref.cut_bound + 1e-9:
                violations += 1
                continue
        if abs(truth.obj - e.v_star) > 1e-6 * max(1.0, abs(truth.obj)):
            suboptimal += 1
    n = max(1, len(cache.entries))
    return {"violation_rate": violations / n, "suboptimal_rate": suboptimal / n}


def full_resolve(cache, ref: Refinement, backend: str = "engine") -> Outcome:
    """Re-solve every entry (N solver calls). Also the ground truth."""
    t0 = time.perf_counter()
    m = _blank(cache)
    out: Dict[Entry, RevalResult] = {}
    for e in cache.entries:
        m1 = ref.apply(cache.models[e])
        r = solve_model(m1, backend=backend)
        m.n_milp += 1
        m.n_energy_evals += m1.J * m1.d
        m.stage_counts["S4"] += 1
        out[e] = RevalResult(r.x_full[: m1.d] if r.status == "optimal" else None,
                             r.obj, r.status == "optimal", "S4")
    m.wall_s = time.perf_counter() - t0
    return Outcome(out, m)


def footprint_resolve(cache, ref: Refinement, backend: str = "engine") -> Outcome:
    """Re-solve only entries the footprint reaches; reuse the rest. No certificates."""
    t0 = time.perf_counter()
    m = _blank(cache)
    out: Dict[Entry, RevalResult] = {}
    affected = cache.index.query(ref.footprint)
    if ref.kind == "I":
        affected = affected | cache.index.adjacent(affected)
    for e in cache.entries:
        if e not in affected:
            out[e] = RevalResult(e.x_star, e.v_star, True, "S1")
            m.stage_counts["S1"] += 1
            continue
        m1 = ref.apply(cache.models[e])
        r = solve_model(m1, backend=backend)
        m.n_milp += 1
        m.n_energy_evals += m1.J * m1.d
        m.stage_counts["S4"] += 1
        out[e] = RevalResult(r.x_full[: m1.d] if r.status == "optimal" else None,
                             r.obj, r.status == "optimal", "S4")
    m.wall_s = time.perf_counter() - t0
    return Outcome(out, m)


def warm_resolve(cache, ref: Refinement, backend: str = "engine") -> Outcome:
    """Footprint-scoped re-solve, warm-started from each entry's stored basis.

    The prior-art baseline (Section IV, [37], [38]): it reuses solver *state* rather
    than the optimality *proof*. It still runs a search per instance; it just starts
    from a better point.
    """
    t0 = time.perf_counter()
    m = _blank(cache)
    out: Dict[Entry, RevalResult] = {}
    affected = cache.index.query(ref.footprint)
    if ref.kind == "I":
        affected = affected | cache.index.adjacent(affected)

    for e in cache.entries:
        if e not in affected:
            out[e] = RevalResult(e.x_star, e.v_star, True, "S1")
            m.stage_counts["S1"] += 1
            continue
        m1 = ref.apply(cache.models[e])
        r = solve_model(m1, backend=backend, warm_basis=(e.cert.basis, e.cert.at_upper))
        m.n_milp += 1
        m.n_lp += 1
        m.n_energy_evals += m1.J * m1.d
        m.stage_counts["S4"] += 1
        out[e] = RevalResult(r.x_full[: m1.d] if r.status == "optimal" else None,
                             r.obj, r.status == "optimal", "S4")
    m.wall_s = time.perf_counter() - t0
    return Outcome(out, m)


def crr(cache, ref: Refinement, backend: str = "engine", tau_req: float = 0.0) -> Outcome:
    res, m = crr_revalidate(cache, ref, tau_req=tau_req, backend=backend)
    return Outcome(res, m)


STRATEGIES = {
    "crr": crr,
    "warm_resolve": warm_resolve,
    "footprint_resolve": footprint_resolve,
    "full_resolve": full_resolve,
}
