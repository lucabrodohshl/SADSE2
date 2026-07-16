"""RQ1 -- Efficiency: CRR's expensive-solver-call reduction and stage distribution.

For each refinement type we report where entries settle across the four stages
(reuse / certificate / repair / re-solve) and how many expensive MILP re-solves
CRR performs versus full cache revalidation (which re-solves every entry).

Run:  python -m evaluate.experiments.crr_efficiency
"""
from evaluate.experiments._crr_common import make_cache, severity_grid, eval_point, save_json


def main():
    cache = make_cache(n_regimes=16, seed=11)
    out = {"N": len(cache.entries), "by_type": {}, "severity_sweep": {}}

    # A representative moderate refinement of each type.
    reps = {k: severity_grid(k)[2] for k in ("II", "III", "I")}
    for kind, (label, ref) in reps.items():
        out["by_type"][kind] = {"label": label, **eval_point(cache, ref)}

    # Severity sweeps: escalation from certificate (S2) toward re-solve (S4).
    for kind in ("II", "III", "I"):
        out["severity_sweep"][kind] = [
            {"label": label, **eval_point(cache, ref)} for label, ref in severity_grid(kind)
        ]

    path = save_json("efficiency.json", out)
    print(f"[E1] wrote {path}   (N={out['N']} entries)")
    for kind, v in out["by_type"].items():
        s = v["stages"]
        print(f"  Type {kind:>3} ({v['label']:>12}): "
              f"S1/S2/S3/S4={s['S1']}/{s['S2']}/{s['S3']}/{s['S4']}  "
              f"MILP {v['crr_milp']:>2}/{v['full_milp']:<2}  "
              f"reduction={100*v['milp_reduction']:5.1f}%  speedup={v['speedup']:4.1f}x  sound={v['sound']}")


if __name__ == "__main__":
    main()
