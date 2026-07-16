"""CRR evaluated as the paper specifies it, over simulated ODD traces.

Run from the Implementation/ root::

    python -m evaluate.experiments.faithful_eval

Design rationale lives in ``src/crr/faithful/*.py``. In short: entries are
``(Z_e, x*_e, v*_e)`` over per-entry zonotope regions of a **continuous** design space;
``cert(e)`` retains the dual/KKT multipliers, active set, reduced-cost signs and bound
witness; Stage 2 is the three algebraic tests of Section VII.B (support function /
reduced-cost inner product / reduced-cost sign survival) and touches no solver.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

from paths import RESULTS_DIR
from src.crr.faithful.harness import (cluster_bootstrap_ci, paired_ratio,
                                      refinement_suite, run_point)

OUT_DIR = RESULTS_DIR / "crr" / "faithful"
OPPONENTS = ["warm_resolve", "footprint_resolve", "full_resolve"]


def _job(args):
    seed, idx, backend = args
    label, ref, sev = refinement_suite()[idx]
    try:
        return run_point(seed, label, ref, sev, backend=backend)
    except Exception as ex:
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "tb": traceback.format_exc()[-600:],
                "seed": seed, "idx": idx}


def main():
    n_seeds = int(os.environ.get("CRR_SEEDS", "16"))
    backend = os.environ.get("CRR_BACKEND", "engine")
    refs = refinement_suite()
    jobs = [(s, i, backend) for s in range(n_seeds) for i in range(len(refs))]
    print(f"[faithful] seeds={n_seeds} refinements={len(refs)} -> {len(jobs)} points "
          f"(backend={backend})", flush=True)

    t0 = time.perf_counter()
    rows: List[dict] = []
    workers = max(1, min(20, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            done += 1
            if r:
                rows.append(r)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    ok = [r for r in rows if r and "error" not in r]
    errs = [r for r in rows if r and "error" in r]
    print(f"[faithful] {len(ok)} points in {time.perf_counter()-t0:.0f}s "
          f"({len(errs)} errors)")
    if errs:
        print("  first error:", errs[0]["error"])
        print(errs[0].get("tb", ""))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "points.json", "w") as fh:
        json.dump({"seeds": n_seeds, "backend": backend, "points": ok}, fh, indent=1)

    rep = analyse(ok)
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(rep, fh, indent=1)
    print_tables(rep)
    return rep


def analyse(rows: List[dict]) -> dict:
    live = [r for r in rows if r["validity_ok"]]
    void = [r for r in rows if not r["validity_ok"]]
    out = {"n": len(rows), "n_live": len(live), "n_void": len(void),
           "void_reasons": sorted({f for r in void for f in r["validity_failures"]})}
    if not live:
        out["verdict"] = "NO VALID POINTS"
        return out

    seeds = np.array([r["seed"] for r in live])

    # headline: solver calls and wall-clock, CRR vs each opponent
    head = {}
    for metric in ("milp", "wall_s"):
        head[metric] = {}
        for opp in OPPONENTS:
            v = paired_ratio(live, metric, "crr", opp)
            if not len(v):
                continue
            med, lo, hi = cluster_bootstrap_ci(v, seeds[: len(v)])
            head[metric][opp] = {
                "median_ratio": med, "ci95": [lo, hi],
                "speedup": (1.0 / med) if med > 0 else float("inf"),
                "crr_zero_fraction": float(np.mean(v == 0.0)),
                "crr_worse_fraction": float(np.mean(v > 1.05)),
                "n": len(v),
            }
    out["headline"] = head

    # per-form breakdown -- the table that answers "does it work, and where"
    per_form = {}
    for kind in ("II", "III", "I"):
        sub = [r for r in live if r["kind"] == kind]
        if not sub:
            continue
        f = lambda s, k: float(np.mean([x["strategies"][s][k] for x in sub]))
        per_form[kind] = {
            "n_caches": len(sub),
            "N_entries": float(np.mean([x["N"] for x in sub])),
            "crr_milp": f("crr", "milp"),
            "warm_milp": f("warm_resolve", "milp"),
            "footprint_milp": f("footprint_resolve", "milp"),
            "full_milp": f("full_resolve", "milp"),
            "crr_wall": f("crr", "wall_s"),
            "warm_wall": f("warm_resolve", "wall_s"),
            "full_wall": f("full_resolve", "wall_s"),
            "crr_energy_rebuilds": f("crr", "energy_evals"),
            "full_energy_rebuilds": f("full_resolve", "energy_evals"),
            "stage_S1": float(np.mean([x["strategies"]["crr"]["stages"]["S1"] for x in sub])),
            "stage_S2": float(np.mean([x["strategies"]["crr"]["stages"]["S2"] for x in sub])),
            "stage_S4": float(np.mean([x["strategies"]["crr"]["stages"]["S4"] for x in sub])),
            "stale_violation_rate": float(np.mean([x["stale_violation_rate"] for x in sub])),
        }
    out["per_form"] = per_form

    # severity curve for form II (the only form with a magnitude dial)
    curve = []
    for sev in sorted({round(r["severity"], 3) for r in live if r["kind"] == "II"}):
        g = [r for r in live if r["kind"] == "II" and round(r["severity"], 3) == sev]
        curve.append({
            "severity": sev, "n": len(g), "label": g[0]["label"],
            "crr_milp": float(np.mean([x["strategies"]["crr"]["milp"] for x in g])),
            "full_milp": float(np.mean([x["strategies"]["full_resolve"]["milp"] for x in g])),
            "certified_S2": float(np.mean([x["strategies"]["crr"]["stages"]["S2"] for x in g])),
            "stale_violation_rate": float(np.mean([x["stale_violation_rate"] for x in g])),
        })
    out["severity_curve_II"] = curve
    out["stale_violation_rate_mean"] = float(np.mean([r["stale_violation_rate"] for r in live]))
    return out


def print_tables(rep: dict) -> None:
    print()
    print("=" * 94)
    print("CRR EVALUATED AS SPECIFIED  —  results at a glance")
    print("=" * 94)
    print(f"  caches: {rep['n']}   valid: {rep['n_live']}   void: {rep['n_void']}")
    for r in rep.get("void_reasons", []):
        print(f"    void: {r}")
    if not rep.get("per_form"):
        return

    print()
    print("TABLE 1 — expensive solver calls per cache, by refinement form")
    print("  (CRR must beat warm_resolve: that is the prior art, not full_resolve)")
    print("-" * 94)
    print(f"  {'form':<26} {'N':>4} {'CRR':>7} {'warm':>7} {'footpr':>7} {'full':>7}   {'vs warm':>9} {'vs full':>9}")
    names = {"II": "II  tightened constraint", "III": "III tighter objective",
             "I": "I   new decision factor"}
    for k in ("II", "III", "I"):
        d = rep["per_form"].get(k)
        if not d:
            continue
        vw = (d["warm_milp"] / d["crr_milp"]) if d["crr_milp"] > 0 else float("inf")
        vf = (d["full_milp"] / d["crr_milp"]) if d["crr_milp"] > 0 else float("inf")
        sw = "∞" if vw == float("inf") else f"{vw:.1f}x"
        sf = "∞" if vf == float("inf") else f"{vf:.1f}x"
        print(f"  {names[k]:<26} {d['N_entries']:>4.0f} {d['crr_milp']:>7.2f} {d['warm_milp']:>7.2f} "
              f"{d['footprint_milp']:>7.2f} {d['full_milp']:>7.2f}   {sw:>9} {sf:>9}")

    print()
    print("TABLE 2 — where entries are discharged (CRR stage distribution), and wall-clock")
    print("-" * 94)
    print(f"  {'form':<26} {'S1':>6} {'S2':>6} {'S4':>6}   {'CRR ms':>8} {'full ms':>9} {'speedup':>9}"
          f" {'obj rebuilds':>13}")
    for k in ("II", "III", "I"):
        d = rep["per_form"].get(k)
        if not d:
            continue
        sp = d["full_wall"] / max(1e-9, d["crr_wall"])
        print(f"  {names[k]:<26} {d['stage_S1']:>6.1f} {d['stage_S2']:>6.1f} {d['stage_S4']:>6.1f}   "
              f"{d['crr_wall']*1000:>8.1f} {d['full_wall']*1000:>9.1f} {sp:>8.1f}x"
              f" {d['crr_energy_rebuilds']:>7.0f}/{d['full_energy_rebuilds']:.0f}")

    print()
    print("TABLE 3 — headline ratios (CRR / opponent; < 1 means CRR is cheaper)")
    print("  cluster bootstrap over seeds; unit of analysis = the cache")
    print("-" * 94)
    for metric, label in (("milp", "solver calls"), ("wall_s", "wall-clock")):
        d = rep["headline"].get(metric, {})
        for opp in OPPONENTS:
            e = d.get(opp)
            if not e:
                continue
            ci = e["ci95"]
            sp = "∞" if e["speedup"] == float("inf") else f"{e['speedup']:.1f}x"
            print(f"  {label:<13} crr vs {opp:<18} ratio={e['median_ratio']:6.3f} "
                  f"[{ci[0]:.3f},{ci[1]:.3f}]  speedup={sp:>7}"
                  f"  CRR does zero work in {100*e['crr_zero_fraction']:3.0f}% of caches"
                  f"  worse in {100*e['crr_worse_fraction']:3.0f}%")

    print()
    print("TABLE 4 — form II severity curve (does it degrade gracefully?)")
    print("-" * 94)
    print(f"  {'refinement':<18} {'severity':>9} {'certified S2':>13} {'CRR MILP':>9} {'full MILP':>10}"
          f" {'stale violations':>17}")
    for c in rep.get("severity_curve_II", []):
        print(f"  {c['label']:<18} {c['severity']:>9.2f} {c['certified_S2']:>13.1f} "
              f"{c['crr_milp']:>9.2f} {c['full_milp']:>10.2f} {100*c['stale_violation_rate']:>16.0f}%")

    print()
    print(f"  Doing nothing (stale cache) leaves {100*rep['stale_violation_rate_mean']:.1f}% "
          f"of entries violating the evolved model.")


if __name__ == "__main__":
    main()
