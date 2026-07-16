"""Revalidation strategies, compared on equal terms.

The point of this module is the *comparator set*. The original evaluation scores CRR
against exactly one opponent -- ``full_revalidation``, which re-solves every entry
including entries it has already determined the refinement does not touch
(``applies_to(e)`` is False and it still calls ``solve()``). No engineer would do that,
so beating it is not evidence.

The strategies here, cheapest-opponent first:

``no_reval``
    Keep the stale cache. Not a competitor -- it exists only to quantify the safety
    cost of doing nothing, which is the prior approach's actual position.

``naive_recheck``  <-- THE PRIMARY COMPARATOR
    The obvious algorithm, with no CRR machinery at all: no certificates, no reverse
    index, no zonotopes. For each entry, recompute the objective under M1, check
    whether the cached choice is still the best and still feasible, and re-solve only
    if not. This is what CRR's Stage 2 *actually does* once you strip the vocabulary
    away. **CRR's entire claim is that its machinery buys something over this.** If it
    does not, that is the finding.

``footprint_resolve``
    Re-solve only entries the refinement genuinely touches (scope + dependency
    footprint), reuse the rest. The competent-engineer baseline. Note this is only
    distinguishable from ``full_resolve`` when refinements are ODD-scoped -- with
    global refinements the footprint is the whole cache and the two coincide, which
    is reported rather than hidden.

``warm_resolve``
    Footprint scoping plus a warm-started re-solve, no certificates. Isolates what the
    certificate machinery adds over plain warm-starting -- a question a reviewer will
    ask, and one classical LP basis ranging already partly answers.

``full_resolve``
    Re-solve everything. The paper's baseline, kept for comparability.

Cost accounting
---------------
Every strategy reports **energy-model evaluations**, solver calls, and wall-clock.
Counting only "expensive solver calls" is what lets a Stage-2 certificate check that
rebuilds the entire objective matrix be billed as free: measured on the original
evaluation, ``_cheapest_config`` and ``solve()`` perform *the same 35* energy
evaluations, yet one is counted as ``n_arith`` ("no solver") and the other as
``n_milp`` ("expensive"). That is the whole "100% call reduction, 0.9x wall-clock"
paradox. Here, work is work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import (FleetScenario, SolveOutcome, TrackAModel, TrackBModel,
                     solve_track_a, solve_track_b)
from .physics import ODDPoint
from .refinements import Refinement


@dataclass
class CacheEntry:
    """A proven-optimal entry for one ODD cell, plus what its certificate reads."""

    odd: ODDPoint
    weight: float                       # visit frequency of this ODD cell
    config_idx: Optional[int]
    assignment: Dict[int, List[int]]
    value: float
    deps: set = field(default_factory=set)
    obj_margin: float = 0.0             # gap to the runner-up configuration
    battery_slack: float = 0.0          # min over agents of (usable - load)
    # Per-agent slack (usable_i - load_i). The scalar min above is not enough to decide
    # a Type-II tightening: usable differs per agent via state-of-health, so knowing
    # only the worst agent's slack forces a conservative "might be broken" verdict on
    # entries that are in fact fine. Storing the vector makes the certificate exact.
    per_agent_slack: List[float] = field(default_factory=list)


@dataclass
class Cost:
    """Honest cost accounting: all work, not just the work we chose to call expensive."""

    energy_evals: int = 0
    solver_calls: int = 0
    resolves: int = 0                   # full re-solves specifically
    wall_s: float = 0.0
    entries_examined: int = 0
    stage_counts: Dict[str, int] = field(default_factory=lambda: {"reuse": 0, "cert": 0, "resolve": 0})

    def add(self, other: "Cost") -> None:
        self.energy_evals += other.energy_evals
        self.solver_calls += other.solver_calls
        self.resolves += other.resolves
        self.wall_s += other.wall_s


@dataclass
class RevalOutcome:
    results: Dict[int, Tuple[Optional[int], float, bool]]   # entry idx -> (config, value, feasible)
    cost: Cost


def _model_for(track: str, scn: FleetScenario, odd: ODDPoint, ref: Optional[Refinement]):
    """Build the model at ``odd``, applying ``ref`` if it is in scope there."""
    fidelity, reserve, payload = "M0", None, 0.0
    if ref is not None and ref.applies_at(odd):
        if ref.kind == "III":
            fidelity = ref.fidelity or "M1"
        elif ref.kind == "II":
            reserve = ref.reserve
        elif ref.kind == "I":
            payload = ref.payload_kg
    cls = TrackAModel if track == "A" else TrackBModel
    m = cls(scn, odd, fidelity=fidelity, reserve=reserve)
    m.payload_kg = payload
    return m


def _solve(track: str, model, backend: str) -> SolveOutcome:
    return solve_track_a(model, backend) if track == "A" else solve_track_b(model, backend)


def _plan_ok(track: str, model, k: Optional[int], assignment) -> bool:
    if k is None:
        return False
    return model.plan_feasible(k, assignment)


def _config_costs(track: str, model) -> np.ndarray:
    if track == "A":
        E = model.energy_matrix()
        return E.sum(axis=1)
    # Track B: cost depends on the assignment, so a "config cost" needs one.
    # The cheapest *lower bound* per config: each task done by its cheapest agent.
    E = model.energy_tensor()
    return E.min(axis=0).sum(axis=1)


# ---------------------------------------------------------------------------
# 0. no revalidation — the prior approach's actual position
# ---------------------------------------------------------------------------
def no_reval(scn, entries: List[CacheEntry], ref: Refinement, track: str,
             backend: str) -> Dict[str, float]:
    """Keep the stale cache; measure how unsafe that is under M1."""
    violations = 0
    for e in entries:
        m1 = _model_for(track, scn, e.odd, ref)
        if not _plan_ok(track, m1, e.config_idx, e.assignment):
            violations += 1
    return {"violation_rate": violations / max(1, len(entries)), "violations": violations}


# ---------------------------------------------------------------------------
# 1. naive_recheck — THE PRIMARY COMPARATOR (no CRR machinery whatsoever)
# ---------------------------------------------------------------------------
def naive_recheck(scn, entries: List[CacheEntry], ref: Refinement, track: str,
                  backend: str) -> RevalOutcome:
    """Recompute the objective, check the cached choice, re-solve only if it moved.

    No certificates, no reverse index, no validity ranges, no zonotopes. This is the
    algorithm a competent engineer writes in an afternoon, and it is the bar CRR has
    to clear.
    """
    t0 = time.perf_counter()
    cost = Cost()
    out: Dict[int, Tuple[Optional[int], float, bool]] = {}

    for idx, e in enumerate(entries):
        cost.entries_examined += 1
        m1 = _model_for(track, scn, e.odd, ref)
        costs = _config_costs(track, m1)                 # rebuilds the objective
        cheapest = int(np.argmin(costs))
        still_best = (cheapest == e.config_idx)
        still_feasible = _plan_ok(track, m1, e.config_idx, e.assignment)

        if still_best and still_feasible:
            value = (m1.config_cost(e.config_idx) if track == "A"
                     else m1.plan_value(e.config_idx, e.assignment))
            out[idx] = (e.config_idx, float(value), True)
            cost.stage_counts["cert"] += 1
        else:
            r = _solve(track, m1, backend)
            cost.resolves += 1
            cost.solver_calls += r.solver_calls
            cost.stage_counts["resolve"] += 1
            out[idx] = (r.config_idx, r.value, r.status == "optimal")
        cost.energy_evals += m1.energy_evals

    cost.wall_s = time.perf_counter() - t0
    return RevalOutcome(out, cost)


# ---------------------------------------------------------------------------
# 2. footprint_resolve — the competent-engineer baseline
# ---------------------------------------------------------------------------
def footprint_resolve(scn, entries: List[CacheEntry], ref: Refinement, track: str,
                      backend: str) -> RevalOutcome:
    """Re-solve only entries the refinement actually touches; reuse the rest."""
    t0 = time.perf_counter()
    cost = Cost()
    out: Dict[int, Tuple[Optional[int], float, bool]] = {}

    for idx, e in enumerate(entries):
        touched = ref.applies_at(e.odd) and bool(ref.footprint & e.deps)
        if not touched:
            out[idx] = (e.config_idx, e.value, True)
            cost.stage_counts["reuse"] += 1
            continue
        cost.entries_examined += 1
        m1 = _model_for(track, scn, e.odd, ref)
        r = _solve(track, m1, backend)
        cost.resolves += 1
        cost.solver_calls += r.solver_calls
        cost.energy_evals += m1.energy_evals
        cost.stage_counts["resolve"] += 1
        out[idx] = (r.config_idx, r.value, r.status == "optimal")

    cost.wall_s = time.perf_counter() - t0
    return RevalOutcome(out, cost)


# ---------------------------------------------------------------------------
# 3. full_resolve — the paper's baseline
# ---------------------------------------------------------------------------
def full_resolve(scn, entries: List[CacheEntry], ref: Refinement, track: str,
                 backend: str) -> RevalOutcome:
    """Re-solve every entry. Also the optimality ground truth."""
    t0 = time.perf_counter()
    cost = Cost()
    out: Dict[int, Tuple[Optional[int], float, bool]] = {}
    for idx, e in enumerate(entries):
        cost.entries_examined += 1
        m1 = _model_for(track, scn, e.odd, ref)
        r = _solve(track, m1, backend)
        cost.resolves += 1
        cost.solver_calls += r.solver_calls
        cost.energy_evals += m1.energy_evals
        cost.stage_counts["resolve"] += 1
        out[idx] = (r.config_idx, r.value, r.status == "optimal")
    cost.wall_s = time.perf_counter() - t0
    return RevalOutcome(out, cost)


# ---------------------------------------------------------------------------
# 4. crr — the four-stage pipeline
# ---------------------------------------------------------------------------
def crr(scn, entries: List[CacheEntry], ref: Refinement, track: str,
        backend: str, tau_req: float = 0.0, use_stored_margins: bool = True) -> RevalOutcome:
    """CRR: reverse-index lookup -> certificate -> repair -> re-solve.

    ``use_stored_margins`` enables the certificate the paper describes: conclude from
    the entry's STORED optimality margin and a bound on how much the refinement can
    perturb the objective, WITHOUT rebuilding the objective matrix. This is the only
    version that can beat ``naive_recheck`` on energy evaluations -- with it off, Stage
    2 recomputes every configuration's cost and is bit-identical in work to the naive
    check, which is precisely the situation in the current implementation.
    """
    t0 = time.perf_counter()
    cost = Cost()
    out: Dict[int, Tuple[Optional[int], float, bool]] = {}

    for idx, e in enumerate(entries):
        # ---- Stage 1: reverse-index lookup at the footprint -----------------
        touched = ref.applies_at(e.odd) and bool(ref.footprint & e.deps)
        if not touched:
            out[idx] = (e.config_idx, e.value, True)
            cost.stage_counts["reuse"] += 1
            continue

        cost.entries_examined += 1

        # ---- Stage 2: certificate survival ---------------------------------
        if use_stored_margins:
            cert = certify_type_II(scn, e, ref, track)
            if cert is not None:
                survives, value = cert
                if survives:
                    # Certified from STORED quantities only: no objective rebuild, no
                    # solver. This is the one case where the paper's cr(e) idea pays
                    # off, and it is sound rather than heuristic.
                    out[idx] = (e.config_idx, value, True)
                    cost.stage_counts["cert"] += 1
                    continue
                # Certificate says the entry is definitely broken -> straight to a
                # re-solve, skipping the pointless recheck.
                m1 = _model_for(track, scn, e.odd, ref)
                r = _solve(track, m1, backend)
                cost.resolves += 1
                cost.solver_calls += r.solver_calls
                cost.energy_evals += m1.energy_evals
                cost.stage_counts["resolve"] += 1
                out[idx] = (r.config_idx, r.value, r.status == "optimal")
                continue

        m1 = _model_for(track, scn, e.odd, ref)
        costs = _config_costs(track, m1)
        cheapest = int(np.argmin(costs))
        ranking_preserved = (cheapest == e.config_idx
                             or abs(costs[cheapest] - costs[e.config_idx]) <= tau_req)
        if ranking_preserved and _plan_ok(track, m1, e.config_idx, e.assignment):
            value = (m1.config_cost(e.config_idx) if track == "A"
                     else m1.plan_value(e.config_idx, e.assignment))
            out[idx] = (e.config_idx, float(value), True)
            cost.stage_counts["cert"] += 1
            cost.energy_evals += m1.energy_evals
            continue

        # ---- Stage 4: re-solve ---------------------------------------------
        r = _solve(track, m1, backend)
        cost.resolves += 1
        cost.solver_calls += r.solver_calls
        cost.energy_evals += m1.energy_evals
        cost.stage_counts["resolve"] += 1
        out[idx] = (r.config_idx, r.value, r.status == "optimal")

    cost.wall_s = time.perf_counter() - t0
    return RevalOutcome(out, cost)


def certify_type_II(scn, e: CacheEntry, ref: Refinement, track: str
                    ) -> Optional[Tuple[bool, float]]:
    """Decide a Type-II refinement from STORED quantities alone -- no objective rebuild.

    This is the one place the paper's ``cr(e)`` idea genuinely pays off, and it is
    sound rather than heuristic:

    * A reserve change is a pure RHS change. The objective is untouched, so the
      configuration ranking provably cannot move and ``e.config_idx`` stays optimal
      *provided it stays feasible*.
    * Feasibility is decidable from the stored ``battery_slack`` and the ratio of the
      new to the old usable budget: the per-agent loads are unchanged, so the entry
      survives iff ``max_i load_i <= usable_1``, i.e. iff
      ``usable_0 - battery_slack <= usable_1``. That is O(1) arithmetic on stored
      numbers -- no energy-model evaluations, no solver.

    Returns ``(survives, value)``, or ``None`` if this refinement is not a Type-II in
    scope (the caller must then fall through to recomputing).

    Contrast with the shipped ``cr(e)``: ``_residual_radius`` uses a hardcoded
    curvature over the shared design-space box, so it is the SAME 11.649074 for every
    entry -- independent of the ODD *and* of the refinement severity -- and ~18x larger
    than the ``obj_margin`` it must beat. It could never fire. A bound that never fires
    is not a certificate.

    No equivalent exists here for Types I/III: they change the energy model itself, and
    a sound a-priori bound on ``|E1_k - E0_k|`` for the rotary-wing fidelity step is not
    available in closed form without evaluating the model -- i.e. without doing the very
    work the certificate exists to avoid. Being unable to certify is a legitimate
    answer; certifying unsoundly is not.
    """
    if ref.kind != "II" or not ref.applies_at(e.odd):
        return None
    if e.config_idx is None:
        return None

    if not e.per_agent_slack:
        return None                      # no stored vector -> cannot certify soundly

    from .physics import usable_budget
    odd = e.odd
    surv = True
    for i, a in enumerate(scn.agents):
        if i >= len(e.per_agent_slack):
            return None
        u0 = usable_budget(scn.capacity_wh, scn.reserve, scn.safety, odd, soh=a.soh)
        u1 = usable_budget(scn.capacity_wh, ref.reserve, scn.safety, odd, soh=a.soh)
        load_i = u0 - e.per_agent_slack[i]        # loads are unchanged by a Type II
        if load_i > u1 + 1e-9:
            surv = False
            break
    # A Type-II change leaves the objective untouched, so the stored value still holds.
    return (surv, e.value) if surv else (False, float("inf"))
