"""Tests for the from-scratch LP/MILP engine, validated against scipy.

Run from the Implementation/ root:  python -m pytest evaluate/tests/test_simplex.py -q
"""
import numpy as np
import pytest
from scipy.optimize import linprog

from src.crr.simplex import linprog_mine


def _rand_feasible_lp(rng, m, n, with_eq=False):
    A = rng.uniform(-1.0, 2.0, size=(m, n))
    x0 = rng.uniform(0.0, 3.0, size=n)          # a feasible interior point
    c = rng.uniform(-2.0, 2.0, size=n)
    if with_eq:
        return dict(c=c, A_eq=A, b_eq=A @ x0, bounds=[(0.0, 5.0)] * n)
    return dict(c=c, A_ub=A, b_ub=A @ x0 + rng.uniform(0.0, 2.0, size=m),
                bounds=[(0.0, 5.0)] * n)


@pytest.mark.parametrize("seed", range(30))
def test_lp_inequality_matches_scipy(seed):
    rng = np.random.RandomState(seed)
    m, n = rng.randint(2, 6), rng.randint(2, 6)
    lp = _rand_feasible_lp(rng, m, n)
    ref = linprog(**lp, method="highs")
    res = linprog_mine(**lp)
    assert ref.status == 0
    assert res.status == "optimal"
    assert abs(res.obj - ref.fun) < 1e-6 * (1 + abs(ref.fun))


@pytest.mark.parametrize("seed", range(30))
def test_lp_equality_matches_scipy(seed):
    rng = np.random.RandomState(seed + 500)
    m, n = rng.randint(2, 5), rng.randint(3, 7)
    lp = _rand_feasible_lp(rng, m, n, with_eq=True)
    ref = linprog(**lp, method="highs")
    if ref.status != 0:
        pytest.skip("scipy did not solve to optimality")
    res = linprog_mine(**lp)
    assert res.status == "optimal"
    assert abs(res.obj - ref.fun) < 1e-5 * (1 + abs(ref.fun))


def test_lp_infeasible():
    # 0 <= x <= 1 and x >= 2  (via -x <= -2)  -> infeasible
    res = linprog_mine(c=[1.0], A_ub=[[-1.0]], b_ub=[-2.0], bounds=[(0.0, 1.0)])
    assert res.status == "infeasible"


def test_lp_unbounded():
    # min -x s.t. x >= 0  -> unbounded below
    res = linprog_mine(c=[-1.0], A_ub=[[-1.0]], b_ub=[0.0], bounds=[(0.0, None)])
    assert res.status == "unbounded"


# ---------------------------------------------------------------------------
# Dual simplex warm-start: after a constraint tightening, re-optimising from the
# stored (dual-feasible) basis must reach the same optimum with no more pivots
# than a cold solve.
# ---------------------------------------------------------------------------
from src.crr.simplex import solve_standard, dual_simplex, _standardize


@pytest.mark.parametrize("seed", range(25))
def test_dual_simplex_warm_start_matches_cold(seed):
    rng = np.random.RandomState(seed + 900)
    m, n = rng.randint(3, 6), rng.randint(3, 6)
    A = rng.uniform(-1.0, 2.0, size=(m, n))
    x0 = rng.uniform(0.0, 3.0, size=n)
    c = rng.uniform(-2.0, 2.0, size=n)
    bounds = [(0.0, 5.0)] * n
    b0 = A @ x0 + rng.uniform(0.5, 2.0, size=m)

    cs, As, bs0, lo, hi, n_ = _standardize(c, A, b0, None, None, bounds)
    r0 = solve_standard(cs, As, bs0, lo, hi)
    assert r0.status == "optimal"

    # tighten the constraints: b1 <= b0 (feasible set shrinks)
    b1 = b0 - rng.uniform(0.0, 0.4, size=m)
    cs, As, bs1, lo, hi, n_ = _standardize(c, A, b1, None, None, bounds)

    cold = solve_standard(cs, As, bs1, lo, hi)
    warm = dual_simplex(cs, As, bs1, lo, hi, r0.basic, r0.at_upper)

    assert warm.status == cold.status
    if cold.status == "optimal":
        assert abs(warm.obj - cold.obj) < 1e-6 * (1 + abs(cold.obj))
        # warm dual simplex skips phase 1 -> never more pivots than a cold solve
        assert warm.n_pivots <= cold.n_pivots


def test_dual_simplex_detects_infeasible_after_tightening():
    # x <= 1, then tighten to x <= 0.5 while also requiring x >= 0.8 -> infeasible
    c, bounds = [1.0], [(0.0, None)]
    cs, As, bs, lo, hi, n = _standardize(c, [[1.0], [-1.0]], [1.0, -0.8], None, None, bounds)
    r0 = solve_standard(cs, As, bs, lo, hi)
    cs, As, bs2, lo, hi, n = _standardize(c, [[1.0], [-1.0]], [0.5, -0.8], None, None, bounds)
    warm = dual_simplex(cs, As, bs2, lo, hi, r0.basic, r0.at_upper)
    assert warm.status == "infeasible"


# ---------------------------------------------------------------------------
# Branch-and-bound MILP (warm-started dual-simplex children) vs scipy.milp
# ---------------------------------------------------------------------------
from scipy.optimize import milp, LinearConstraint, Bounds
from src.crr.branch_and_bound import solve_milp


def _oracle(c, A_ub, b_ub, bounds, integrality):
    cons = ([LinearConstraint(np.asarray(A_ub, float), -np.inf, np.asarray(b_ub, float))]
            if A_ub is not None and len(A_ub) else [])
    lb = np.array([b[0] for b in bounds], float)
    ub = np.array([np.inf if b[1] is None else b[1] for b in bounds], float)
    return milp(c=np.asarray(c, float), constraints=cons,
                integrality=np.array(integrality, int), bounds=Bounds(lb, ub))


@pytest.mark.parametrize("seed", range(30))
def test_milp_binary_matches_scipy(seed):
    rng = np.random.RandomState(seed + 7000)
    n, m = rng.randint(2, 6), rng.randint(1, 4)
    A = rng.uniform(0.0, 2.0, size=(m, n))
    b = rng.uniform(0.5 * n, 1.5 * n, size=m)
    c = rng.uniform(-2.0, 2.0, size=n)
    bounds, integ = [(0.0, 1.0)] * n, [1] * n
    ref = _oracle(c, A, b, bounds, integ)
    res = solve_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integer_mask=integ)
    if ref.status == 0:
        assert res.status == "optimal"
        assert abs(res.obj - ref.fun) < 1e-6 * (1 + abs(ref.fun))


def test_milp_knapsack_optimal():
    # max value {6,10,12}, weights {1,2,3}, capacity 5  ->  pick items 2&3 (value 22, weight 5)
    res = solve_milp(c=[-6.0, -10.0, -12.0], A_ub=[[1.0, 2.0, 3.0]], b_ub=[5.0],
                     bounds=[(0.0, 1.0)] * 3, integer_mask=[1, 1, 1])
    assert res.status == "optimal"
    assert abs(res.obj - (-22.0)) < 1e-6


def test_milp_integer_bounds_matches_scipy():
    # small general-integer program
    c, A, b = [-1.0, -1.0], [[3.0, 2.0], [1.0, 4.0]], [12.0, 10.0]
    bounds, integ = [(0.0, 5.0), (0.0, 5.0)], [1, 1]
    ref = _oracle(c, A, b, bounds, integ)
    res = solve_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integer_mask=integ)
    assert res.status == "optimal"
    assert abs(res.obj - ref.fun) < 1e-6 * (1 + abs(ref.fun))
