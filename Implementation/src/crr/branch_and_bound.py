"""A small correct MILP solver: LP-relaxation branch-and-bound with warm-started
dual-simplex children, built on :mod:`src.crr.simplex`.

Each child node tightens one variable bound and re-optimises from the parent's
optimal (hence dual-feasible) basis via the dual simplex -- the same warm-start
mechanism CRR's Stage 3 uses. Not competitive with Gurobi/CBC, but exact and
entirely self-contained (validated against scipy.optimize.milp).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .simplex import INF, _standardize, dual_simplex, solve_standard

_INT_TOL = 1e-6


@dataclass
class MILPResult:
    status: str            # "optimal" | "infeasible" | "unbounded"
    x: np.ndarray          # structural solution (length n)
    obj: float
    n_nodes: int


def solve_milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
               integer_mask=None, max_nodes=200000):
    cs, A, b, lo0, hi0, n = _standardize(c, A_ub, b_ub, A_eq, b_eq, bounds)
    integer_mask = (np.zeros(n, bool) if integer_mask is None
                    else np.asarray(integer_mask, bool))
    int_idx = np.where(integer_mask[:n])[0]

    root = solve_standard(cs, A, b, lo0, hi0)
    if root.status == "infeasible":
        return MILPResult("infeasible", np.zeros(n), INF, 1)
    if root.status == "unbounded":
        return MILPResult("unbounded", root.x[:n], -INF, 1)

    incumbent_obj, incumbent_x = INF, None
    # DFS stack of (lo, hi, basic, at_upper, lp_obj, lp_x)
    stack = [(lo0.copy(), hi0.copy(), root.basic, root.at_upper, root.obj, root.x)]
    nodes = 0

    while stack:
        lo, hi, basic, at_upper, lp_obj, lp_x = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            break
        if lp_obj >= incumbent_obj - 1e-9:           # bound prune
            continue

        # most-fractional integer variable
        frac_j, frac = -1, _INT_TOL
        for j in int_idx:
            f = abs(lp_x[j] - round(lp_x[j]))
            if f > frac:
                frac_j, frac = j, f

        if frac_j == -1:                              # integral -> candidate incumbent
            if lp_obj < incumbent_obj - 1e-12:
                incumbent_obj, incumbent_x = lp_obj, lp_x[:n].copy()
            continue

        val = lp_x[frac_j]
        h_floor = hi.copy(); h_floor[frac_j] = math.floor(val + 1e-9)   # x_j <= floor
        l_ceil = lo.copy();  l_ceil[frac_j] = math.ceil(val - 1e-9)     # x_j >= ceil
        for nl, nh in ((lo.copy(), h_floor), (l_ceil, hi.copy())):
            if np.any(nl > nh + 1e-9):                # empty domain
                continue
            child = dual_simplex(cs, A, b, nl, nh, basic, at_upper)
            if child.status != "optimal":
                continue
            if child.obj >= incumbent_obj - 1e-9:     # bound prune
                continue
            stack.append((nl, nh, child.basic, child.at_upper, child.obj, child.x))

    if incumbent_x is None:
        return MILPResult("infeasible", np.zeros(n), INF, nodes)
    return MILPResult("optimal", incumbent_x, float(incumbent_obj), nodes)
