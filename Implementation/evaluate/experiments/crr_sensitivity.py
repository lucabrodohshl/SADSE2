"""RQ3 -- Sensitivity & scalability: footprint size and problem scale.

- Footprint sweep: a regime-scoped Type-II refinement touching k of N regimes;
  the reverse index reuses the rest at Stage 1, so the affected set (and the
  MILP work) tracks |Delta|.
- Scale sweeps: growing fleet size M and cache size N -- CRR's expensive-call
  reduction and wall-clock speedup should hold (or grow) with scale.

Run:  python -m evaluate.experiments.crr_sensitivity
"""
from evaluate.experiments._crr_common import make_cache, eval_point, save_json
from src.crr.revalidation import crr_revalidate
from src.crr.baselines import full_revalidation
from src.crr.refinement import refine_type_II, refine_type_III


def main():
    out = {"footprint": [], "scale_fleet": [], "scale_cache": []}

    # ---- footprint size: scoped Type-II touching the k highest-wind regimes ----
    cache = make_cache(n_regimes=16, seed=11)
    regime_names = [e.regime_name for e in cache.entries]
    for k in (1, 2, 4, 8, 12, 16):
        ref = refine_type_II(reserve=0.62, scope_regimes=regime_names[-k:])
        _, mc = crr_revalidate(cache, ref)
        _, mf = full_revalidation(cache, ref)
        affected = mc.total_entries - mc.stage_counts["S1"]
        out["footprint"].append({
            "footprint_regimes": k, "affected": affected,
            "crr_milp": mc.n_milp, "full_milp": mf.n_milp,
            "milp_reduction": 1.0 - mc.n_milp / max(1, mf.n_milp),
        })

    # ---- scale: fleet size M (call reduction should hold as instances grow) ----
    for M in (3, 5, 8):
        c = make_cache(n_regimes=12, na=M, nt=6, nc=6, capacity=44.0, seed=3)
        pt = eval_point(c, refine_type_II(reserve=0.5))
        out["scale_fleet"].append({"M": M, **{k: pt[k] for k in
                                   ("N", "crr_milp", "full_milp", "milp_reduction",
                                    "speedup", "warm_pivots", "cold_pivots")}})

    # ---- scale: cache size N ----
    for N in (8, 16, 24):
        c = make_cache(n_regimes=N, seed=7)
        pt = eval_point(c, refine_type_III(delta=0.15))
        out["scale_cache"].append({"N": N, **{k: pt[k] for k in
                                  ("crr_milp", "full_milp", "milp_reduction", "speedup")}})

    path = save_json("sensitivity.json", out)
    print(f"[E3] wrote {path}")
    print("  footprint |Delta|->affected/CRR_MILP:",
          ", ".join(f"{r['footprint_regimes']}->{r['affected']}/{r['crr_milp']}" for r in out["footprint"]))
    print("  fleet speedup:", ", ".join(f"M{r['M']}:{r['speedup']:.1f}x" for r in out["scale_fleet"]))


if __name__ == "__main__":
    main()
