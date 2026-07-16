"""Solving the fleet model, retaining the artifacts the certificate needs."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .model import FleetModel


@dataclass
class SolveResult:
    status: str
    x_full: np.ndarray
    obj: float
    assignment: Dict[int, List[int]]
    basis: np.ndarray
    at_upper: np.ndarray
    lp_bound: float
    build: dict
    n_solver_calls: int = 1
    wall_s: float = 0.0


def _decode(model: FleetModel, x: np.ndarray) -> Dict[int, List[int]]:
    assign: Dict[int, List[int]] = {i: [] for i in range(model.num_agents)}
    for i in range(model.num_agents):
        for j in range(model.J):
            if x[model.zi(i, j)] > 0.5:
                assign[i].append(j)
    return assign


def solve_model(model: FleetModel, backend: str = "engine",
                warm_basis: Optional[tuple] = None) -> SolveResult:
    """Solve the MILP and keep the LP relaxation's basis + bound witness.

    The LP relaxation value is the global bound witness of Section VII.A: it proves no
    integer solution beats the incumbent by more than the gap.

    ``warm_basis`` warm-starts the root relaxation from a stored basis (the prior-art
    ``warm_resolve`` baseline). Which warm start is *valid* depends on what the
    refinement changed: a form-II cut adds a row and changes the RHS, leaving the basis
    dual-feasible (dual simplex); a form-III objective change leaves it primal-feasible
    (primal simplex). Using the wrong one terminates at a suboptimal point while
    reporting "optimal", so we dispatch rather than guess, and fall back to a cold solve
    if the stored basis does not fit the new column count.
    """
    t0 = time.perf_counter()
    build = model.build()

    from src.crr.simplex import (_standardize, dual_feasibility_violation, dual_simplex,
                                 primal_warm, solve_standard)

    cs, A, b, lo, hi, n = _standardize(build["c"], build["A_ub"], build["b_ub"],
                                       build["A_eq"], build["b_eq"], build["bounds"])
    root = None
    if warm_basis is not None:
        basis, at_up = warm_basis
        basis = np.asarray(basis, int)
        at_up = np.asarray(at_up, bool)
        if len(at_up) == A.shape[1] and basis.size == A.shape[0] and basis.max(initial=-1) < A.shape[1]:
            try:
                if dual_feasibility_violation(cs, A, basis, at_up) <= 1e-7:
                    root = dual_simplex(cs, A, b, lo, hi, basis, at_up, validate=False)
                else:
                    root = primal_warm(cs, A, b, lo, hi, basis, at_up)
            except Exception:
                root = None
        if root is not None and root.status != "optimal":
            root = None                      # warm start failed -> honest cold fallback
    if root is None:
        root = solve_standard(cs, A, b, lo, hi)
    if root.status != "optimal":
        return SolveResult("infeasible", np.zeros(build["n"]), float("inf"), {},
                           np.array([], int), np.array([], bool), float("inf"), build,
                           wall_s=time.perf_counter() - t0)
    lp_bound = float(root.obj) + build["obj_const"]

    if backend == "highs":
        from scipy.optimize import Bounds, LinearConstraint, milp
        cons = []
        if len(build["A_ub"]):
            cons.append(LinearConstraint(build["A_ub"], -np.inf, build["b_ub"]))
        if len(build["A_eq"]):
            cons.append(LinearConstraint(build["A_eq"], build["b_eq"], build["b_eq"]))
        blo = np.array([x[0] for x in build["bounds"]], float)
        bhi = np.array([x[1] for x in build["bounds"]], float)
        r = milp(c=build["c"], constraints=cons,
                 integrality=np.asarray(build["integer_mask"], float),
                 bounds=Bounds(blo, bhi), options={"time_limit": 60.0})
        if r.status != 0:
            return SolveResult("infeasible" if r.status == 2 else "node_limit",
                               np.zeros(build["n"]), float("inf"), {},
                               root.basic, root.at_upper, lp_bound, build,
                               wall_s=time.perf_counter() - t0)
        x = r.x
        obj = float(r.fun) + build["obj_const"]
    else:
        from src.crr.branch_and_bound import solve_milp
        r = solve_milp(c=build["c"], A_ub=build["A_ub"], b_ub=build["b_ub"],
                       A_eq=build["A_eq"], b_eq=build["b_eq"],
                       bounds=build["bounds"], integer_mask=build["integer_mask"])
        if r.status != "optimal":
            return SolveResult(r.status, np.zeros(build["n"]), float("inf"), {},
                               root.basic, root.at_upper, lp_bound, build,
                               wall_s=time.perf_counter() - t0)
        x = r.x
        obj = float(r.obj) + build["obj_const"]

    return SolveResult("optimal", np.asarray(x, float), obj, _decode(model, x),
                       root.basic, root.at_upper, lp_bound, build,
                       wall_s=time.perf_counter() - t0)
