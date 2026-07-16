"""Correctness + performance harness for the self-contained LP/MILP engine.

Every engine optimisation must keep the oracle agreement at 100% while improving
the timings. Run from the Implementation/ root::

    python -m evaluate.tests.bench_engine

Reports, per instance family:
  * oracle agreement vs scipy/HiGHS (linprog / milp) -- MUST stay 1.00
  * median wall-clock of the engine vs HiGHS
  * pivot / node counts
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from src.crr.simplex import linprog_mine
from src.crr.branch_and_bound import solve_milp


# ---------------------------------------------------------------------------
# instance generators
# ---------------------------------------------------------------------------
def random_lp(seed: int, m: int, n: int):
    rng = np.random.RandomState(seed)
    A_ub = rng.uniform(-1.0, 3.0, size=(m, n))
    b_ub = rng.uniform(2.0, 8.0, size=m)
    c = rng.uniform(-2.0, 2.0, size=n)
    bounds = [(0.0, 1.0)] * n
    return dict(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)


def fleet_assignment(seed: int, M: int, J: int, K: int, slack: float = 1.25):
    """Assignment-coupled fleet MILP: x[i,j,k] + y[k], aggregated linking.

    Agent-dependent energy E[i,k,j] makes the objective genuinely couple to the
    assignment (unlike the config-determined model in src/crr/model.py).
    """
    rng = np.random.RandomState(seed)
    base_E = rng.uniform(2.0, 9.0, size=(K, J))
    agent_mult = rng.uniform(0.85, 1.35, size=M)
    E = np.einsum('kj,i->ikj', base_E, agent_mult)

    nx = M * J * K
    n = nx + K
    xi = lambda i, j, k: (i * J + j) * K + k
    yi = lambda k: nx + k

    c = np.zeros(n)
    for i in range(M):
        for j in range(J):
            for k in range(K):
                c[xi(i, j, k)] = E[i, k, j]

    rows_eq, b_eq = [], []
    for j in range(J):
        r = np.zeros(n)
        for i in range(M):
            for k in range(K):
                r[xi(i, j, k)] = 1.0
        rows_eq.append(r); b_eq.append(1.0)
    r = np.zeros(n)
    for k in range(K):
        r[yi(k)] = 1.0
    rows_eq.append(r); b_eq.append(1.0)

    rows_ub, b_ub = [], []
    soh = rng.uniform(0.75, 1.0, size=M)
    budget = float(base_E.mean() * J / M) * slack
    for i in range(M):
        r = np.zeros(n)
        for j in range(J):
            for k in range(K):
                r[xi(i, j, k)] = E[i, k, j]
        rows_ub.append(r); b_ub.append(budget * soh[i])
    for k in range(K):
        r = np.zeros(n)
        for i in range(M):
            for j in range(J):
                r[xi(i, j, k)] = 1.0
        r[yi(k)] = -float(M * J)
        rows_ub.append(r); b_ub.append(0.0)

    return dict(c=c, A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                A_eq=np.array(rows_eq), b_eq=np.array(b_eq),
                bounds=[(0.0, 1.0)] * n, integer_mask=[1] * n)


def binpack(seed: int, M: int, J: int):
    """The fixed-config bin-packing the current CRR evaluation actually solves."""
    rng = np.random.RandomState(seed)
    E = rng.uniform(2.0, 9.0, size=J)
    n = M * J
    A_eq = np.zeros((J, n))
    for j in range(J):
        for i in range(M):
            A_eq[j, i * J + j] = 1.0
    A_ub = np.zeros((M, n))
    for i in range(M):
        for j in range(J):
            A_ub[i, i * J + j] = E[j]
    b_ub = np.full(M, float(E.sum()) / M * 1.25)
    c = np.array([i for i in range(M) for _ in range(J)], float) * 1e-3
    return dict(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(J),
                bounds=[(0.0, 1.0)] * n, integer_mask=[1] * n)


# ---------------------------------------------------------------------------
# oracles
# ---------------------------------------------------------------------------
def highs_lp(inst) -> Tuple[str, float]:
    r = linprog(inst["c"], A_ub=inst.get("A_ub"), b_ub=inst.get("b_ub"),
                A_eq=inst.get("A_eq"), b_eq=inst.get("b_eq"),
                bounds=inst["bounds"], method="highs")
    return ("optimal" if r.status == 0 else "infeasible"), (r.fun if r.status == 0 else np.inf)


def highs_milp(inst) -> Tuple[str, float]:
    cons = []
    if inst.get("A_ub") is not None and len(inst["A_ub"]):
        cons.append(LinearConstraint(inst["A_ub"], -np.inf, inst["b_ub"]))
    if inst.get("A_eq") is not None and len(inst["A_eq"]):
        cons.append(LinearConstraint(inst["A_eq"], inst["b_eq"], inst["b_eq"]))
    lo = np.array([b[0] for b in inst["bounds"]], float)
    hi = np.array([b[1] for b in inst["bounds"]], float)
    r = milp(c=inst["c"], constraints=cons,
             integrality=np.asarray(inst["integer_mask"], float),
             bounds=Bounds(lo, hi))
    return ("optimal" if r.status == 0 else "infeasible"), (r.fun if r.status == 0 else np.inf)


def _median_time(fn, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0


def compare(name: str, instances: List[dict], is_milp: bool, reps: int = 3,
            verbose: bool = True) -> Dict:
    agree, tested = 0, 0
    t_mine, t_highs = [], []
    work = []
    for inst in instances:
        if is_milp:
            ref_status, ref_obj = highs_milp(inst)
            res = solve_milp(**inst)
            mine_status, mine_obj = res.status, res.obj
            work.append(res.n_nodes)
        else:
            ref_status, ref_obj = highs_lp(inst)
            res = linprog_mine(inst["c"], A_ub=inst.get("A_ub"), b_ub=inst.get("b_ub"),
                               A_eq=inst.get("A_eq"), b_eq=inst.get("b_eq"),
                               bounds=inst["bounds"])
            mine_status, mine_obj = res.status, res.obj
            work.append(res.n_pivots)

        if ref_status == "optimal":
            tested += 1
            if mine_status == "optimal" and abs(mine_obj - ref_obj) <= 1e-6 * max(1.0, abs(ref_obj)):
                agree += 1
            elif verbose:
                print(f"    MISMATCH: mine={mine_status}/{mine_obj:.6f} ref={ref_status}/{ref_obj:.6f}")

        if is_milp:
            t_mine.append(_median_time(lambda: solve_milp(**inst), reps))
            t_highs.append(_median_time(lambda: highs_milp(inst), reps))
        else:
            t_mine.append(_median_time(lambda: linprog_mine(
                inst["c"], A_ub=inst.get("A_ub"), b_ub=inst.get("b_ub"),
                A_eq=inst.get("A_eq"), b_eq=inst.get("b_eq"), bounds=inst["bounds"]), reps))
            t_highs.append(_median_time(lambda: highs_lp(inst), reps))

    out = {
        "name": name,
        "agreement": agree / max(1, tested),
        "tested": tested,
        "engine_ms": float(np.median(t_mine)),
        "highs_ms": float(np.median(t_highs)),
        "ratio": float(np.median(t_mine) / max(1e-9, np.median(t_highs))),
        "work": float(np.median(work)),
    }
    if verbose:
        flag = "OK " if out["agreement"] == 1.0 else "FAIL"
        print(f"  [{flag}] {name:<28} agree={out['agreement']*100:5.1f}% ({tested:>2})  "
              f"engine={out['engine_ms']:8.2f}ms  highs={out['highs_ms']:7.2f}ms  "
              f"ratio={out['ratio']:6.2f}x  work={out['work']:.0f}")
    return out


def main():
    print("=" * 100)
    print("ENGINE BENCHMARK — correctness vs scipy/HiGHS, and cost")
    print("=" * 100)
    results = []

    print("\nLP (random dense):")
    for (m, n) in [(5, 8), (10, 20), (20, 40), (40, 80)]:
        insts = [random_lp(s, m, n) for s in range(8)]
        results.append(compare(f"random_lp m={m} n={n}", insts, is_milp=False))

    print("\nMILP (fixed-config bin-packing — what CRR solves today):")
    for (M, J) in [(4, 5), (5, 8), (6, 10)]:
        insts = [binpack(s, M, J) for s in range(5)]
        results.append(compare(f"binpack M={M} J={J}", insts, is_milp=True))

    print("\nMILP (assignment-coupled fleet — the realistic formulation):")
    for (M, J, K) in [(4, 5, 7), (5, 8, 8)]:
        insts = [fleet_assignment(s, M, J, K) for s in range(3)]
        results.append(compare(f"fleet M={M} J={J} K={K}", insts, is_milp=True, reps=1))

    print()
    bad = [r for r in results if r["agreement"] < 1.0]
    if bad:
        print(f"CORRECTNESS FAILURES in {len(bad)} families: {[r['name'] for r in bad]}")
    else:
        print("All families agree with scipy/HiGHS at 100%.")
    return results


if __name__ == "__main__":
    main()
