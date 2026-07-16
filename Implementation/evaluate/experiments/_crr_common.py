"""Shared harness for the CRR evaluation experiments (RQ1-3).

Run experiments from the Implementation/ root, e.g.::

    python -m evaluate.experiments.crr_efficiency
"""
from __future__ import annotations

import json

import numpy as np

from paths import RESULTS_DIR
from src.crr.scenario import make_scenario
from src.crr.cache import build_cache
from src.crr.revalidation import crr_revalidate
from src.crr.baselines import full_revalidation, no_revalidation
from src.crr.refinement import refine_type_I, refine_type_II, refine_type_III

CRR_DIR = RESULTS_DIR / "crr"


def make_cache(n_regimes: int = 12, na: int = 4, nt: int = 5, nc: int = 7,
               capacity: float = 34.0, reserve: float = 0.20, seed: int = 11,
               wind_max: float = 1.6):
    """A cache of ``n_regimes`` wind regimes (calm -> strong), solved under M0."""
    winds = np.linspace(0.0, wind_max, n_regimes)
    regimes = [(f"w{int(round(w * 100)):03d}", float(w)) for w in winds]
    scn = make_scenario(num_agents=na, num_tasks=nt, num_configs=nc, seed=seed)
    return build_cache(scn, regimes, capacity=capacity, reserve=reserve)


def severity_grid(kind: str):
    """Increasing-severity refinements of a given type."""
    if kind == "II":
        return [(f"reserve={r:.2f}", refine_type_II(reserve=r))
                for r in [0.30, 0.42, 0.54, 0.66, 0.78]]
    if kind == "III":
        return [(f"delta={d:.2f}", refine_type_III(delta=d))
                for d in [0.05, 0.12, 0.22, 0.38, 0.60]]
    if kind == "I":
        return [(f"strength={s:.2f}", refine_type_I(strength=s))
                for s in [0.05, 0.12, 0.22, 0.38, 0.60]]
    raise ValueError(kind)


def eval_point(cache, ref, with_stale: bool = False, pivot_stats: bool = True) -> dict:
    """Run CRR + full-revalidation (ground truth) for one refinement and summarise.

    ``pivot_stats`` enables the warm-vs-cold pivot comparison, which costs an extra
    cold solve per Stage-3 repair. That solve is instrumentation and is excluded
    from the reported wall-clock.
    """
    cres, mc = crr_revalidate(cache, ref, collect_pivot_stats=pivot_stats)
    gt, mf = full_revalidation(cache, ref)
    gaps = [abs(cres[e].value - gt[e].value) for e in cache.entries if gt[e].feasible]
    max_gap = max(gaps) if gaps else 0.0
    sound = all(cres[e].feasible == gt[e].feasible for e in cache.entries) and max_gap < 1e-6
    out = {
        "N": len(cache.entries), "kind": ref.kind,
        "stages": dict(mc.stage_counts),
        "crr_milp": mc.n_milp, "crr_lp": mc.n_lp, "crr_arith": mc.n_arith, "crr_none": mc.n_none,
        "full_milp": mf.n_milp,
        "milp_reduction": 1.0 - mc.n_milp / max(1, mf.n_milp),
        "crr_wall": mc.wall_s, "full_wall": mf.wall_s,
        "speedup": mf.wall_s / max(1e-9, mc.wall_s),
        "warm_pivots": mc.warm_pivots, "cold_pivots": mc.cold_pivots,
        "max_gap": max_gap, "sound": bool(sound),
    }
    if with_stale:
        stale = no_revalidation(cache, ref)
        out.update({
            "stale_violation_rate": stale.violation_rate,
            "stale_mean_subopt": stale.mean_suboptimality,
            "stale_max_subopt": stale.max_suboptimality,
        })
    return out


def save_json(name: str, obj: dict):
    CRR_DIR.mkdir(parents=True, exist_ok=True)
    path = CRR_DIR / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
