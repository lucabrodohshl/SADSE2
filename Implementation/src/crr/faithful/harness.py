"""Experiment harness: ODD-trace-driven caches, validity gates, and statistics.

The operating regimes are derived from the simulated weather process in
``src/crr/realworld/odd.py`` (AR(1) persistence + diurnal cycles + fronts, over a
declared grid of climatologies), rather than from a hand-picked "calm / light / strong"
ladder. Regimes are drawn **stratified over the wind quantiles the trace actually
visits** -- wind is the channel the energy is nonlinear in -- and each carries its visit
frequency as a weight, so results can be read deployment-weighted or unweighted.

Design commitments, fixed before the run:

* **Unit of analysis is the (seed, refinement) cache**, not the entry. Entries in a
  cache share tasks and fleet and are strongly dependent; pooling entry rows would
  inflate the effective sample size. The bootstrap resamples **seeds**.
* **Validity gates run before any headline.** A run is *void*, not slow, if any
  ground-truth solve is unproven, or if any strategy certifies something the ground
  truth contradicts -- including the safety-critical direction (claimed feasible,
  actually infeasible).
* **Severity is reported as a curve**, not marginalised: every metric is monotone in
  how hard the refinement bites, so a single number is a statement about the chosen
  severity distribution rather than about CRR.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import STRATEGIES, no_reval
from .cache import build_cache
from .model import ODDRegime, Task
from .refinement import Refinement, refine_type_I, refine_type_II, refine_type_III
from .solve import solve_model

# Pinned configuration.
N_REGIMES = 4          # ODD regimes drawn from the trace
N_REGIONS = 6          # design-space slices per regime  -> N = 24 entries
NUM_AGENTS = 3
CAPACITY_WH = 95.0


@dataclass
class Validity:
    ground_truth_unproven: int = 0
    unsafe_certifications: int = 0      # claimed feasible; ground truth says infeasible
    false_certifications: int = 0       # any other disagreement with ground truth
    distinct_optima: int = 0
    eps_distinct: int = 0

    def failures(self) -> List[str]:
        out = []
        if self.ground_truth_unproven:
            out.append(f"{self.ground_truth_unproven} ground-truth solves unproven")
        if self.unsafe_certifications:
            out.append(f"{self.unsafe_certifications} UNSAFE certifications "
                       "(claimed feasible, ground truth infeasible)")
        if self.false_certifications:
            out.append(f"{self.false_certifications} false certifications")
        if self.eps_distinct <= 1:
            out.append("eps(Z_e) identical across entries: the regions do not "
                       "discriminate, so cr(e) carries no information")
        return out

    @property
    def ok(self) -> bool:
        return not self.failures()


def regimes_from_trace(seed: int, n: int = N_REGIMES) -> Tuple[List[ODDRegime], str]:
    """Draw ODD regimes stratified over the wind quantiles the trace actually visits."""
    from src.crr.realworld.odd import generate_trace, stratified_cells

    trace, clim = generate_trace(seed, hours=240.0)
    cells = stratified_cells(trace, n, seed)
    regimes = []
    for i, c in enumerate(cells):
        regimes.append(ODDRegime(name=f"w{int(round(c.point.wind_speed)):02d}t{int(round(c.point.temperature)):+03d}",
                                 wind=float(c.point.wind_speed),
                                 temperature=float(c.point.temperature),
                                 weight=float(c.weight)))
    return regimes, clim.name


def make_tasks(seed: int, n: int = 5) -> List[Task]:
    rng = np.random.RandomState(seed * 7717 + 5)
    return [Task(length=float(rng.uniform(300.0, 1500.0))) for _ in range(n)]


def refinement_suite() -> List[Tuple[str, Refinement, float]]:
    """(label, refinement, severity) -- the declared refinement set.

    Severity is expressed in a physical unit per form so the curves are readable:
    form II by how far the cap cuts into the speed box, form I by the new column's
    reduced cost, form III as a single fidelity step (it has no free magnitude -- that
    is the point of it being a model-fidelity change rather than a dial).
    """
    out: List[Tuple[str, Refinement, float]] = []
    for cap in [22.0, 19.0, 16.0, 13.0, 10.0, 7.0]:
        out.append((f"II speed<={cap:.0f}", refine_type_II(np.array([1.0, 0.0, 0.0]), cap),
                    (22.0 - cap) / 19.0))
    for alt_cap in [120.0, 90.0, 60.0]:
        out.append((f"II alt<={alt_cap:.0f}", refine_type_II(np.array([0.0, 1.0, 0.0]), alt_cap),
                    (120.0 - alt_cap) / 110.0))
    out.append(("III secant", refine_type_III(), 1.0))
    for c in [5.0, 1.0, 0.2, -2.0]:
        out.append((f"I new-col c={c:+.1f}", refine_type_I(c, np.zeros(400)),
                    float(np.clip((5.0 - c) / 7.0, 0, 1))))
    return out


def run_point(seed: int, label: str, ref: Refinement, severity: float,
              backend: str = "engine") -> Optional[dict]:
    """One (seed, refinement) cache: every strategy on the SAME cache (paired)."""
    regimes, clim = regimes_from_trace(seed)
    tasks = make_tasks(seed)
    cache = build_cache(tasks, regimes, num_agents=NUM_AGENTS, n_regions=N_REGIONS,
                        capacity_wh=CAPACITY_WH, seed=seed, backend=backend)
    if len(cache.entries) < 4:
        return None

    v = Validity()
    v.eps_distinct = len({round(e.cr.eps_region, 9) for e in cache.entries})
    v.distinct_optima = len({round(float(e.x_star[0]), 6) for e in cache.entries})

    # ground truth
    truth: Dict[object, Tuple[Optional[float], bool]] = {}
    for e in cache.entries:
        m1 = ref.apply(cache.models[e])
        r = solve_model(m1, backend=backend)
        if r.status == "node_limit":
            v.ground_truth_unproven += 1
        truth[e] = (r.obj, r.status == "optimal")

    rows: Dict[str, dict] = {}
    for name, fn in STRATEGIES.items():
        o = fn(cache, ref, backend=backend)
        for e in cache.entries:
            got = o.results[e]
            t_obj, t_feas = truth[e]
            if got.feasible and not t_feas:
                v.unsafe_certifications += 1
            elif t_feas and not got.feasible:
                v.false_certifications += 1
            elif t_feas and got.feasible:
                if abs(got.value - t_obj) > 1e-6 * max(1.0, abs(t_obj)):
                    v.false_certifications += 1
        m = o.metrics
        rows[name] = {
            "milp": m.n_milp, "lp": m.n_lp,
            "support_queries": m.n_support_queries,
            "inner_products": m.n_inner_products,
            "energy_evals": m.n_energy_evals,
            "wall_s": m.wall_s,
            "stages": dict(m.stage_counts),
        }

    stale = no_reval(cache, ref, backend=backend)

    return {
        "seed": seed, "climatology": clim, "label": label, "kind": ref.kind,
        "severity": severity, "N": len(cache.entries),
        "strategies": rows,
        "stale_violation_rate": stale["violation_rate"],
        "stale_suboptimal_rate": stale["suboptimal_rate"],
        "validity": asdict(v),
        "validity_ok": v.ok,
        "validity_failures": v.failures(),
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def paired_ratio(rows: List[dict], metric: str, a: str, b: str) -> np.ndarray:
    """a/b per cache. < 1 means ``a`` is cheaper. Zero numerators are KEPT.

    CRR legitimately performs zero solver calls on a certified refinement; a log-ratio
    would drop exactly those caches and turn a decisive win into an apparent tie.
    """
    out = []
    for r in rows:
        va, vb = r["strategies"][a][metric], r["strategies"][b][metric]
        if vb > 0:
            out.append(va / vb)
    return np.array(out, float)


def cluster_bootstrap_ci(values: np.ndarray, seeds: np.ndarray, n_boot: int = 4000,
                         rng_seed: int = 0) -> Tuple[float, float, float]:
    if len(values) == 0:
        return (float("nan"),) * 3
    rng = np.random.RandomState(rng_seed)
    uniq = np.unique(seeds)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([values[seeds == s] for s in pick]) if len(uniq) else values
        boots[i] = np.median(vals) if len(vals) else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(np.median(values)), float(lo), float(hi)
