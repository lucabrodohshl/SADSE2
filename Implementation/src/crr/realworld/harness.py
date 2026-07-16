"""Experiment harness: cache population, validity gates, and the statistics.

Design commitments, made here rather than after seeing results
--------------------------------------------------------------
* **Unit of analysis is the (seed, refinement) cache, not the entry.** Entries within
  one cache share a scenario and are strongly dependent, so pooling entry-level rows
  would inflate the effective sample size and shrink confidence intervals to fiction.
  Each cache contributes ONE number per strategy; the bootstrap resamples *seeds*.
* **One primary endpoint**: paired log-ratio of **energy-model evaluations**,
  ``crr`` vs ``naive_recheck``, on **Track B**. Everything else is exploratory.
  With five metrics x six strategies x three types x two tracks there are enough
  comparisons that an uncorrected sweep would find something by chance; the original
  evaluation already shows metric choice alone flipping the verdict on identical data
  ("100% call reduction" vs "0.9x wall-clock").
* **Practical-significance margin**: CRR must win by >= 1.2x to be worth its
  complexity. A statistically detectable 3% win is not a reason to carry certificates,
  a reverse index and zonotopes.
* **Severity is reported as a curve**, never marginalized into one number: every
  metric is monotone in severity, so a single number is really a statement about the
  chosen severity distribution.
* **Validity gates run BEFORE any headline.** A run that fails one is void, not slow.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import (CacheEntry, Cost, RevalOutcome, crr, footprint_resolve,
                        full_resolve, naive_recheck, no_reval)
from .models import (FleetScenario, TrackAModel, TrackBModel, make_fleet,
                     solve_track_a, solve_track_b, track_a_coupling_report,
                     assert_track_b_is_coupled)
from .odd import ODDCell, generate_trace, stratified_cells
from .physics import ODDPoint
from .refinements import Refinement

# Pinned operating point. 6 agents / 8 tasks / 9 configs is where a single
# assignment-coupled re-solve costs ~3.2 s on the self-contained engine -- i.e. the
# smallest fleet at which "avoid the expensive optimizer" describes something real.
# (At the originally evaluated 4/5/7 a re-solve is 4.89 ms and the entire cache
# revalidates in 78 ms.)
FLEET_M, FLEET_J, FLEET_K = 6, 8, 9
CACHE_N = 12


@dataclass
class ValidityReport:
    """Gates that must pass before a run may produce a headline."""

    distinct_argmins: int = 0
    argmin_migration_rate: float = 0.0
    rail_pinned_fraction: float = 0.0
    track_b_coupled: bool = False
    track_b_max_rank: int = 0
    ground_truth_node_limits: int = 0
    false_certifications: int = 0
    unsafe_certifications: int = 0

    def failures(self) -> List[str]:
        out = []
        if self.distinct_argmins <= 1:
            out.append("cache is argmin-homogeneous (every entry picks the same config): "
                       "the ODD population cannot exercise revalidation at all")
        if self.ground_truth_node_limits > 0:
            out.append(f"{self.ground_truth_node_limits} ground-truth solves hit the node "
                       "limit: optimality unproven, so the reference is not a ground truth")
        if self.unsafe_certifications > 0:
            out.append(f"{self.unsafe_certifications} UNSAFE certifications: a strategy "
                       "declared an entry feasible that the ground truth says is not "
                       "(a cached plan violating the battery reserve)")
        if self.false_certifications > 0:
            out.append(f"{self.false_certifications} false certifications: a strategy "
                       "concluded differently from ground truth")
        if not self.track_b_coupled:
            out.append("Track B regressed to an agent-independent scan")
        return out

    @property
    def ok(self) -> bool:
        return not self.failures()


def build_cache(scn: FleetScenario, cells: Sequence[ODDCell], track: str,
                backend: str) -> Tuple[List[CacheEntry], int]:
    """Solve M0 at each ODD cell and enrich into cache entries."""
    entries: List[CacheEntry] = []
    node_limits = 0
    for cell in cells:
        model = (TrackAModel(scn, cell.point, "M0") if track == "A"
                 else TrackBModel(scn, cell.point, "M0"))
        r = solve_track_a(model, backend) if track == "A" else solve_track_b(model, backend)
        if r.status == "node_limit":
            node_limits += 1
        if r.status != "optimal":
            continue

        # cert / dep / cr, built from what the certificate actually reads
        if track == "A":
            costs = model.energy_matrix().sum(axis=1)
        else:
            costs = model.energy_tensor().min(axis=0).sum(axis=1)
        order = np.sort(costs)
        margin = float(order[1] - order[0]) if len(order) > 1 else float("inf")

        if track == "A":
            E = model.energy_matrix()
            loads = [sum(E[r.config_idx, j] for j in r.assignment.get(i, []))
                     for i in range(len(scn.agents))]
            per_slack = [model.usable(a) - l for a, l in zip(scn.agents, loads)]
        else:
            E = model.energy_tensor()
            loads = [sum(E[i, r.config_idx, j] for j in r.assignment.get(i, []))
                     for i in range(len(scn.agents))]
            per_slack = [model.usable(i) - loads[i] for i in range(len(scn.agents))]
        slack = min(per_slack)

        # Entry dependencies. These must be a SOUND over-approximation of what the
        # certificate reads: an entry omitted from a footprint is reused with no check
        # at all, so omitting one that the refinement could actually break is an
        # unsound skip, not an optimisation.
        #
        # A tempting-but-wrong version of this tagged `battery_row` only when the
        # entry's battery was near-binding (slack < 15% of usable). That is a heuristic,
        # not a bound: a reserve hike from 0.20 to 0.55 cuts usable by ~44%, so an entry
        # with 20% slack becomes infeasible and would have been silently reused. Every
        # entry depends on the battery row, because any Type-II tightening can reach it.
        # The stored slack is still valuable -- but at Stage 2, as a *sound certificate*
        # (see `crr()`), not as a Stage-1 skip.
        deps = {"obj_coeffs", "energy_model", "battery_row",
                f"wind_band:{int(cell.point.wind_speed)}"}

        entries.append(CacheEntry(odd=cell.point, weight=cell.weight,
                                  config_idx=r.config_idx, assignment=r.assignment,
                                  value=r.value, deps=deps, obj_margin=margin,
                                  battery_slack=float(slack),
                                  per_agent_slack=[float(s) for s in per_slack]))
    return entries, node_limits


def gate_cache(scn: FleetScenario, entries: List[CacheEntry], track: str,
               node_limits: int) -> ValidityReport:
    rep = ValidityReport(ground_truth_node_limits=node_limits)
    argmins = [e.config_idx for e in entries]
    rep.distinct_argmins = len(set(argmins))
    rep.argmin_migration_rate = (rep.distinct_argmins - 1) / max(1, len(entries) - 1)

    speeds = [scn.configs[e.config_idx].as_dict()["speed"] for e in entries if e.config_idx is not None]
    lo, hi = 3.0, 22.0
    rep.rail_pinned_fraction = float(np.mean([(s <= lo + 0.5 or s >= hi - 0.5) for s in speeds])) if speeds else 0.0

    if entries:
        mb = TrackBModel(scn, entries[0].odd, "M0")
        c = assert_track_b_is_coupled(mb)
        rep.track_b_coupled = bool(c["agent_dependent"])
        rep.track_b_max_rank = int(c["max_rank"])
    return rep


STRATEGIES: Dict[str, Callable] = {
    "naive_recheck": naive_recheck,
    "footprint_resolve": footprint_resolve,
    "full_resolve": full_resolve,
    "crr": crr,
}


def run_point(seed: int, ref: Refinement, track: str, backend: str,
              fleet: Tuple[int, int, int] = (FLEET_M, FLEET_J, FLEET_K),
              cache_n: int = CACHE_N) -> Optional[dict]:
    """One (seed, refinement) cache: run every strategy on the SAME cache (paired)."""
    M, J, K = fleet
    scn = make_fleet(M, J, K, seed=seed, capacity_wh=110.0)
    trace, clim = generate_trace(seed, hours=240.0)
    cells = stratified_cells(trace, cache_n, seed)

    entries, node_limits = build_cache(scn, cells, track, backend)
    if len(entries) < 3:
        return None
    gate = gate_cache(scn, entries, track, node_limits)

    truth = full_resolve(scn, entries, ref, track, backend)
    stale = no_reval(scn, entries, ref, track, backend)

    rows = {}
    false_certs = 0
    unsafe_certs = 0
    for name, fn in STRATEGIES.items():
        out = fn(scn, entries, ref, track, backend)
        # Soundness: every strategy must agree with the ground truth.
        #
        # The safety-critical case is a strategy claiming an entry is FEASIBLE when
        # the ground truth says it is not -- a cached plan that now violates the
        # battery reserve. An earlier version of this check required `tfeas and feas`
        # before comparing, which skipped exactly that case and therefore reported
        # zero violations by construction.
        for idx, (k, v, feas) in out.results.items():
            tk, tv, tfeas = truth.results[idx]
            if feas and not tfeas:
                unsafe_certs += 1                      # claimed feasible; truth: not
            elif tfeas and not feas:
                false_certs += 1                       # claimed infeasible; truth: feasible
            elif tfeas and feas:
                if tk is not None and tk != k and abs(v - tv) > 1e-6 * max(1.0, abs(tv)):
                    false_certs += 1                   # wrong optimum
                elif abs(v - tv) > 1e-6 * max(1.0, abs(tv)):
                    false_certs += 1                   # right config, wrong value
        rows[name] = {
            "energy_evals": out.cost.energy_evals,
            "solver_calls": out.cost.solver_calls,
            "resolves": out.cost.resolves,
            "wall_s": out.cost.wall_s,
            "stages": dict(out.cost.stage_counts),
        }
    gate.false_certifications = false_certs
    gate.unsafe_certifications = unsafe_certs

    touched = sum(1 for e in entries if ref.applies_at(e.odd) and (ref.footprint & e.deps))
    return {
        "seed": seed, "climatology": clim.name, "track": track, "backend": backend,
        "kind": ref.kind, "label": ref.label, "severity": ref.severity,
        "scoped": not ref.is_global,
        "N": len(entries),
        "footprint_fraction": touched / max(1, len(entries)),
        "strategies": rows,
        "stale_violation_rate": stale["violation_rate"],
        "validity": asdict(gate),
        "validity_ok": gate.ok,
        "validity_failures": gate.failures(),
    }


# ---------------------------------------------------------------------------
# statistics: cluster bootstrap over SEEDS, paired within cache
# ---------------------------------------------------------------------------
def paired_ratio(rows: List[dict], metric: str, a: str, b: str) -> np.ndarray:
    """``a/b`` per cache. < 1 means ``a`` is CHEAPER (better).

    A zero numerator is meaningful and must be kept: ``crr`` legitimately performs
    **zero** energy-model evaluations on a Type-II refinement, because a pure RHS
    change cannot move the ranking and the certificate concludes without touching the
    objective. An earlier version of this used ``log(a/b)`` and dropped any row where
    either side was zero -- which silently discarded exactly the caches where CRR wins
    outright, and made a decisive win look like an exact 1.000 tie. Only a zero
    *denominator* is genuinely undefined, and is dropped (with the count reported).
    """
    out = []
    for r in rows:
        va = r["strategies"][a][metric]
        vb = r["strategies"][b][metric]
        if vb > 0:
            out.append(va / vb)
    return np.array(out, dtype=float)


def n_dropped(rows: List[dict], metric: str, b: str) -> int:
    """How many caches were excluded because the denominator was zero."""
    return sum(1 for r in rows if r["strategies"][b][metric] <= 0)


def paired_log_ratio(rows: List[dict], metric: str, a: str, b: str) -> np.ndarray:
    """Deprecated: use :func:`paired_ratio`. Retained only for the older call sites.

    Drops zero-numerator rows, which biases against whichever strategy is capable of
    doing no work at all. Do not use for a headline.
    """
    out = []
    for r in rows:
        va = r["strategies"][a][metric]
        vb = r["strategies"][b][metric]
        if va > 0 and vb > 0:
            out.append(math.log(va / vb))
    return np.array(out)


def cluster_bootstrap_ci(values: np.ndarray, seeds: np.ndarray, n_boot: int = 5000,
                         alpha: float = 0.05, rng_seed: int = 0) -> Tuple[float, float, float]:
    """Median + percentile CI, resampling SEEDS (clusters), not observations."""
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.RandomState(rng_seed)
    uniq = np.unique(seeds)
    med = float(np.median(values))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([values[seeds == s] for s in pick])
        boots[i] = np.median(vals) if len(vals) else np.nan
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return med, float(lo), float(hi)


def wilcoxon_signed_rank(values: np.ndarray) -> float:
    """Two-sided p-value that the paired differences are centred on zero."""
    from scipy.stats import wilcoxon
    v = values[np.abs(values) > 1e-12]
    if len(v) < 6:
        return float("nan")
    try:
        return float(wilcoxon(v).pvalue)
    except Exception:
        return float("nan")


def loss_rate(values: np.ndarray, band: float = math.log(1.05)) -> Dict[str, float]:
    """Fraction of caches where the strategy LOSES by more than a 5% tie band.

    Reported with a Clopper-Pearson interval, because "0 losses in 30 caches" bounds
    the true rate at roughly <=10%, not at 0.
    """
    if len(values) == 0:
        return {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    losses = int(np.sum(values > band))
    n = len(values)
    from scipy.stats import beta
    lo = 0.0 if losses == 0 else float(beta.ppf(0.025, losses, n - losses + 1))
    hi = 1.0 if losses == n else float(beta.ppf(0.975, losses + 1, n - losses))
    return {"rate": losses / n, "lo": lo, "hi": hi, "n": n, "losses": losses}
