"""RQ2 -- Correctness/Soundness: CRR preserves exact optimality; cost of staleness.

For every refinement and severity we (a) compare CRR's revalidated value against
the full-revalidation ground truth (the optimality gap must be 0 at tau_req=0 --
an empirical check of Theorem 1) and (b) measure what the prior no-revalidation
approach costs: how many stored entries become infeasible or suboptimal under M1.

Run:  python -m evaluate.experiments.crr_correctness
"""
from evaluate.experiments._crr_common import make_cache, severity_grid, eval_point, save_json


def main():
    cache = make_cache(n_regimes=16, seed=11)
    out = {"N": len(cache.entries), "crr": [], "stale": []}
    all_sound = True

    for kind in ("II", "III", "I"):
        for label, ref in severity_grid(kind):
            pt = eval_point(cache, ref, with_stale=True)
            all_sound = all_sound and pt["sound"]
            out["crr"].append({"kind": kind, "label": label,
                               "max_gap": pt["max_gap"], "sound": pt["sound"]})
            out["stale"].append({"kind": kind, "label": label,
                                 "violation_rate": pt["stale_violation_rate"],
                                 "mean_subopt": pt["stale_mean_subopt"],
                                 "max_subopt": pt["stale_max_subopt"]})

    out["all_sound"] = all_sound
    out["max_gap_overall"] = max(r["max_gap"] for r in out["crr"])
    out["stale_worst_violation_rate"] = max(r["violation_rate"] for r in out["stale"])
    out["stale_worst_max_subopt"] = max(r["max_subopt"] for r in out["stale"])

    path = save_json("correctness.json", out)
    print(f"[E2] wrote {path}")
    print(f"  CRR sound on every point: {all_sound}   max optimality gap overall = {out['max_gap_overall']:.2e}")
    print(f"  Stale cache worst-case: violation_rate={out['stale_worst_violation_rate']:.2f}  "
          f"max suboptimality={100*out['stale_worst_max_subopt']:.1f}%")


if __name__ == "__main__":
    main()
