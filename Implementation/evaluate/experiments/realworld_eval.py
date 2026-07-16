"""Simulation-based CRR evaluation: CRR vs a fair comparator set, over ODD traces.

Run from the Implementation/ root::

    python -m evaluate.experiments.realworld_eval

What this measures, and why it is built this way, is documented in
``src/crr/realworld/{physics,odd,models,refinements,baselines,harness}.py``. In short:

* the ODD is a simulated weather process (AR(1) + diurnal + fronts) over a grid of
  climatologies, not a linspace on one scalar at one seed;
* wind is applied as a vector per leg, which is the only channel that can re-rank
  configurations -- and it does, in ~90% of draws (the original uniform-scale model
  re-ranks in 0%, by construction);
* the primary comparator is ``naive_recheck`` -- CRR with all its machinery removed;
* cost is counted as energy-model evaluations, solver calls and wall-clock, so a
  certificate check that rebuilds the objective cannot be billed as free;
* the unit of analysis is the cache, and the bootstrap resamples seeds.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from paths import RESULTS_DIR
from src.crr.realworld.harness import (CACHE_N, FLEET_M, FLEET_J, FLEET_K,
                                       cluster_bootstrap_ci, loss_rate,
                                       paired_log_ratio, run_point,
                                       wilcoxon_signed_rank)
from src.crr.realworld.refinements import (refine_type_I, refine_type_II,
                                           refine_type_III, scope_cold_below,
                                           scope_wind_above)

OUT_DIR = RESULTS_DIR / "crr" / "realworld"

# Pre-registered: the primary endpoint and the margin below which CRR is not worth
# its complexity. Declared here, in the runner, so it cannot be chosen after the fact.
PRIMARY = dict(metric="energy_evals", a="crr", b="naive_recheck", track="B",
               practical_margin=1.2)


def _refinements(n_sev: int = 4) -> List:
    """The declared refinement set: types x severities x scopes."""
    refs = []
    for r in np.linspace(0.25, 0.55, n_sev):                     # Type II, global
        refs.append(refine_type_II(float(r)))
    for r in np.linspace(0.30, 0.55, 2):                         # Type II, ODD-scoped
        refs.append(refine_type_II(float(r), scope=scope_wind_above(8.0)))
    refs.append(refine_type_III())                               # Type III, global
    refs.append(refine_type_III(scope=scope_wind_above(7.0)))    # Type III, scoped
    for p in np.linspace(0.5, 2.0, n_sev):                       # Type I, global
        refs.append(refine_type_I(float(p)))
    refs.append(refine_type_I(1.5, scope=scope_cold_below(10.0)))
    return refs


def _job(args):
    seed, ridx, track, backend, fleet, cache_n = args
    ref = _refinements()[ridx]
    try:
        return run_point(seed, ref, track, backend, fleet=fleet, cache_n=cache_n)
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}", "seed": seed, "ridx": ridx,
                "track": track}


def main():
    n_seeds = int(os.environ.get("CRR_SEEDS", "24"))
    backend = os.environ.get("CRR_BACKEND", "engine")
    fleet = (int(os.environ.get("CRR_M", FLEET_M)),
             int(os.environ.get("CRR_J", FLEET_J)),
             int(os.environ.get("CRR_K", FLEET_K)))
    cache_n = int(os.environ.get("CRR_N", CACHE_N))
    tracks = os.environ.get("CRR_TRACKS", "B,A").split(",")

    refs = _refinements()
    jobs = [(s, i, tr, backend, fleet, cache_n)
            for s in range(n_seeds) for i in range(len(refs)) for tr in tracks]
    print(f"[realworld] fleet={fleet} cache_n={cache_n} seeds={n_seeds} "
          f"refinements={len(refs)} tracks={tracks} backend={backend} "
          f"-> {len(jobs)} points")

    t0 = time.perf_counter()
    rows: List[dict] = []
    workers = max(1, min(20, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r is not None:
                rows.append(r)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r and "error" not in r]
    errs = [r for r in rows if r and "error" in r]
    print(f"[realworld] {len(ok)} points in {wall:.0f}s ({len(errs)} errors)")
    if errs:
        print("  first error:", errs[0]["error"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "points.json", "w") as fh:
        json.dump({"config": {"fleet": fleet, "cache_n": cache_n, "seeds": n_seeds,
                              "backend": backend, "primary": PRIMARY},
                   "points": ok}, fh, indent=1)

    report = analyse(ok)
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print_report(report)
    return report


def analyse(rows: List[dict]) -> dict:
    void = [r for r in rows if not r["validity_ok"]]
    live = [r for r in rows if r["validity_ok"]]
    out = {"n_points": len(rows), "n_void": len(void), "n_live": len(live),
           "void_reasons": sorted({f for r in void for f in r["validity_failures"]})}

    if not live:
        out["verdict"] = "NO VALID POINTS"
        return out

    # ---- the pre-registered primary endpoint -----------------------------
    prim = [r for r in live if r["track"] == PRIMARY["track"]]
    res = {}
    if prim:
        seeds = np.array([r["seed"] for r in prim])
        vals = paired_log_ratio(prim, PRIMARY["metric"], PRIMARY["a"], PRIMARY["b"])
        keep = np.array([i for i, r in enumerate(prim)
                         if r["strategies"][PRIMARY["a"]][PRIMARY["metric"]] > 0
                         and r["strategies"][PRIMARY["b"]][PRIMARY["metric"]] > 0])
        s2 = seeds[keep] if len(keep) else seeds
        med, lo, hi = cluster_bootstrap_ci(vals, s2)
        res = {
            "definition": f"median paired log({PRIMARY['a']}/{PRIMARY['b']}) of "
                          f"{PRIMARY['metric']} on Track {PRIMARY['track']}",
            "n_caches": len(vals),
            "median_log_ratio": med,
            "median_ratio": math.exp(med) if np.isfinite(med) else float("nan"),
            "ci95_ratio": [math.exp(lo), math.exp(hi)] if np.isfinite(lo) else [float("nan")] * 2,
            "wilcoxon_p": wilcoxon_signed_rank(vals),
            "practical_margin": PRIMARY["practical_margin"],
        }
        r_ = res["median_ratio"]
        if np.isfinite(r_):
            res["crr_cheaper_by"] = 1.0 / r_ if r_ > 0 else float("nan")
            res["meets_practical_margin"] = bool(r_ > 0 and (1.0 / r_) >= PRIMARY["practical_margin"])
    out["primary_endpoint"] = res

    # ---- exploratory: every strategy pair, every metric, per track -------
    expl = {}
    for track in sorted({r["track"] for r in live}):
        sub = [r for r in live if r["track"] == track]
        seeds = np.array([r["seed"] for r in sub])
        per = {}
        for metric in ("energy_evals", "resolves", "solver_calls", "wall_s"):
            per[metric] = {}
            for opp in ("naive_recheck", "footprint_resolve", "full_resolve"):
                vals = paired_log_ratio(sub, metric, "crr", opp)
                if not len(vals):
                    continue
                med, lo, hi = cluster_bootstrap_ci(vals, seeds[:len(vals)])
                per[metric][f"crr_vs_{opp}"] = {
                    "median_ratio": math.exp(med) if np.isfinite(med) else None,
                    "ci95_ratio": [math.exp(lo), math.exp(hi)] if np.isfinite(lo) else None,
                    "crr_loses_rate": loss_rate(vals),
                }
        expl[f"track_{track}"] = per
    out["exploratory"] = expl

    # ---- severity curves --------------------------------------------------
    curves = {}
    for track in sorted({r["track"] for r in live}):
        for kind in ("II", "III", "I"):
            pts = [r for r in live if r["track"] == track and r["kind"] == kind]
            if not pts:
                continue
            by_sev: Dict[float, List[dict]] = {}
            for r in pts:
                by_sev.setdefault(round(r["severity"], 3), []).append(r)
            curve = []
            for sev in sorted(by_sev):
                g = by_sev[sev]
                curve.append({
                    "severity": sev,
                    "n": len(g),
                    "crr_resolves": float(np.mean([x["strategies"]["crr"]["resolves"] for x in g])),
                    "naive_resolves": float(np.mean([x["strategies"]["naive_recheck"]["resolves"] for x in g])),
                    "full_resolves": float(np.mean([x["strategies"]["full_resolve"]["resolves"] for x in g])),
                    "crr_energy": float(np.mean([x["strategies"]["crr"]["energy_evals"] for x in g])),
                    "naive_energy": float(np.mean([x["strategies"]["naive_recheck"]["energy_evals"] for x in g])),
                    "stale_violation_rate": float(np.mean([x["stale_violation_rate"] for x in g])),
                })
            curves[f"track_{track}_type_{kind}"] = curve
    out["severity_curves"] = curves

    # ---- is footprint_resolve distinguishable from full_resolve? ---------
    ff = []
    for r in live:
        a = r["strategies"]["footprint_resolve"]["resolves"]
        b = r["strategies"]["full_resolve"]["resolves"]
        ff.append(1.0 if a == b else 0.0)
    out["footprint_equals_full_fraction"] = float(np.mean(ff)) if ff else float("nan")
    out["mean_footprint_fraction"] = float(np.mean([r["footprint_fraction"] for r in live]))

    # ---- safety: what does doing nothing cost? --------------------------
    out["stale_violation_rate_mean"] = float(np.mean([r["stale_violation_rate"] for r in live]))
    return out


def print_report(rep: dict) -> None:
    print()
    print("=" * 88)
    print("SIMULATION-BASED CRR EVALUATION")
    print("=" * 88)
    print(f"  points: {rep['n_points']}   valid: {rep['n_live']}   void: {rep['n_void']}")
    for r in rep.get("void_reasons", []):
        print(f"    void reason: {r}")

    p = rep.get("primary_endpoint") or {}
    if p:
        print()
        print("PRIMARY ENDPOINT (pre-registered)")
        print(f"  {p.get('definition')}")
        mr = p.get("median_ratio")
        if mr:
            print(f"  median ratio = {mr:.4f}   => CRR is {1/mr:.3f}x the cost of naive_recheck")
            ci = p.get("ci95_ratio") or [float('nan')]*2
            print(f"  95% CI (cluster bootstrap over seeds): [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"  Wilcoxon p = {p.get('wilcoxon_p')}")
            print(f"  practical margin >= {p['practical_margin']}x : "
                  f"{'MET' if p.get('meets_practical_margin') else 'NOT MET'}")

    print()
    print("EXPLORATORY (all labelled exploratory; not corrected for multiplicity)")
    for track, per in (rep.get("exploratory") or {}).items():
        print(f"  {track}:")
        for metric, pairs in per.items():
            for pair, d in pairs.items():
                mr = d.get("median_ratio")
                if mr is None:
                    continue
                lr = d["crr_loses_rate"]
                print(f"    {metric:<14} {pair:<28} ratio={mr:6.3f}  "
                      f"CRR loses {lr['losses']}/{lr['n']} caches "
                      f"({100*lr['rate']:.0f}%, CP95 {100*lr['lo']:.0f}-{100*lr['hi']:.0f}%)")

    print()
    print(f"  footprint_resolve identical to full_resolve in "
          f"{100*rep.get('footprint_equals_full_fraction', float('nan')):.0f}% of caches "
          f"(mean footprint fraction {rep.get('mean_footprint_fraction', float('nan')):.2f})")
    print(f"  stale cache (no revalidation) violation rate: "
          f"{100*rep.get('stale_violation_rate_mean', float('nan')):.1f}%")


if __name__ == "__main__":
    main()
