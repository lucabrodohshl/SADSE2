"""A correct, warm-startable bounded-variable revised simplex LP solver.

Standard form solved internally:  min c^T x  s.t.  A x = b,  lo <= x <= hi.
Inequalities are converted to equalities with slack variables. The basis is
refactorized each iteration (np.linalg.solve) -- O(m^3) per pivot, which is fine
at the scale we use and keeps the implementation transparently correct.

Public entry points:
  * ``linprog_mine`` -- scipy.linprog-style convenience wrapper (validated in tests).
  * ``solve_standard`` -- solve a standard-form LP, returning the optimal basis.
  * ``dual_simplex`` -- re-optimize from a dual-feasible (warm) basis after the RHS
    or variable bounds changed (used by CRR Stage 3 and by branch-and-bound).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INF = np.inf
TOL = 1e-9


@dataclass
class LPResult:
    status: str            # "optimal" | "infeasible" | "unbounded"
    x: np.ndarray          # full standard-form solution
    obj: float
    basic: np.ndarray      # indices of basic variables (len m)
    at_upper: np.ndarray   # bool per variable: nonbasic sitting at its upper bound
    n_pivots: int


# ---------------------------------------------------------------------------
# core primal simplex on  min c^T x  s.t.  A x = b,  lo <= x <= hi
# ---------------------------------------------------------------------------
def _nonbasic_values(ntot, basic, at_upper, lo, hi):
    x = np.empty(ntot)
    mask = np.ones(ntot, dtype=bool)
    mask[basic] = False
    x[mask & at_upper] = hi[mask & at_upper]
    x[mask & ~at_upper] = lo[mask & ~at_upper]
    return x, mask


def _primal(c, A, b, lo, hi, basic, at_upper, max_iter=20000):
    m, ntot = A.shape
    basic = np.array(basic, dtype=int)
    at_upper = np.array(at_upper, dtype=bool)
    pivots = 0

    for _ in range(max_iter):
        B = A[:, basic]
        x, nb_mask = _nonbasic_values(ntot, basic, at_upper, lo, hi)
        rhs = b - A[:, nb_mask] @ x[nb_mask]
        try:
            xB = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            xB = np.linalg.lstsq(B, rhs, rcond=None)[0]
        x[basic] = xB

        y = np.linalg.solve(B.T, c[basic])
        d = c - A.T @ y                     # reduced costs

        # entering variable (Bland's rule: smallest index that violates optimality)
        entering, direction = -1, 0
        nb_idx = np.where(nb_mask)[0]
        for j in nb_idx:
            if not at_upper[j] and d[j] < -TOL:
                entering, direction = j, +1
                break
            if at_upper[j] and d[j] > TOL:
                entering, direction = j, -1
                break
        if entering == -1:
            return "optimal", basic, at_upper, x, pivots

        col = np.linalg.solve(B, A[:, entering])       # B^{-1} A_entering
        dxB = -direction * col                          # d xB / dt

        # ratio test
        limit = hi[entering] - lo[entering]             # entering bound-flip limit
        leaving_row, leave_at_upper = -1, False
        for i in range(m):
            bi = basic[i]
            if dxB[i] > TOL and hi[bi] < INF:
                r = (hi[bi] - xB[i]) / dxB[i]
                if r < limit - TOL:
                    limit, leaving_row, leave_at_upper = r, i, True
            elif dxB[i] < -TOL and lo[bi] > -INF:
                r = (xB[i] - lo[bi]) / (-dxB[i])
                if r < limit - TOL:
                    limit, leaving_row, leave_at_upper = r, i, False

        if limit >= INF:
            return "unbounded", basic, at_upper, x, pivots

        pivots += 1
        if leaving_row == -1:
            # bound flip: entering hits its opposite bound, basis unchanged
            at_upper[entering] = (direction > 0)
        else:
            leaving_var = basic[leaving_row]
            basic[leaving_row] = entering
            at_upper[entering] = False
            at_upper[leaving_var] = leave_at_upper

    raise RuntimeError("primal simplex did not converge")


def dual_simplex(c, A, b, lo, hi, basic, at_upper, max_iter=20000):
    """Re-optimise from a DUAL-feasible (warm) basis, restoring primal feasibility.

    The basis passed in must be optimal for some earlier problem with the same
    ``c`` and ``A`` (hence dual-feasible); only ``b`` or the bounds changed. Used
    to warm-start after a constraint tightening (CRR Stage 3) or a branch bound.
    Returns "infeasible" if the dual is unbounded (no primal solution).
    """
    c = np.asarray(c, float); A = np.asarray(A, float); b = np.asarray(b, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    m, ntot = A.shape
    basic = np.array(basic, dtype=int)
    at_upper = np.array(at_upper, dtype=bool)
    pivots = 0

    for _ in range(max_iter):
        B = A[:, basic]
        x, nb_mask = _nonbasic_values(ntot, basic, at_upper, lo, hi)
        xB = np.linalg.solve(B, b - A[:, nb_mask] @ x[nb_mask])
        x[basic] = xB
        y = np.linalg.solve(B.T, c[basic])
        d = c - A.T @ y

        # leaving variable: most bound-violating basic
        r, below, worst = -1, True, TOL
        for i in range(m):
            bi = basic[i]
            if xB[i] < lo[bi] - TOL and (lo[bi] - xB[i]) > worst:
                r, below, worst = i, True, lo[bi] - xB[i]
            elif xB[i] > hi[bi] + TOL and (xB[i] - hi[bi]) > worst:
                r, below, worst = i, False, xB[i] - hi[bi]
        if r == -1:
            return LPResult("optimal", x, float(c @ x), basic, at_upper, pivots)

        rho = np.linalg.solve(B.T, np.eye(m)[r])
        alpha = A.T @ rho                      # row r of B^{-1}A over all columns

        entering, best = -1, INF
        for j in np.where(nb_mask)[0]:
            aj = alpha[j]
            if below:                          # need xB_r to increase
                if (not at_upper[j]) and aj < -TOL:
                    ratio = d[j] / (-aj)
                elif at_upper[j] and aj > TOL:
                    ratio = (-d[j]) / aj
                else:
                    continue
            else:                              # need xB_r to decrease
                if (not at_upper[j]) and aj > TOL:
                    ratio = d[j] / aj
                elif at_upper[j] and aj < -TOL:
                    ratio = (-d[j]) / (-aj)
                else:
                    continue
            if ratio < best - TOL:
                best, entering = ratio, j
        if entering == -1:
            return LPResult("infeasible", x, INF, basic, at_upper, pivots)

        pivots += 1
        leaving = basic[r]
        at_upper[leaving] = (not below)        # below -> leaves at lower; above -> upper
        basic[r] = entering
        at_upper[entering] = False

    raise RuntimeError("dual simplex did not converge")


def solve_standard(c, A, b, lo, hi):
    """Solve min c^T x s.t. A x = b, lo <= x <= hi via two-phase primal simplex."""
    c = np.asarray(c, float)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m, n = A.shape

    # --- Phase 1: artificials give an identity basis; minimise their sum ---
    at_upper = np.zeros(n, dtype=bool)
    x_nb = np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0))
    resid = b - A @ x_nb
    signs = np.where(resid >= 0, 1.0, -1.0)
    A1 = np.hstack([A, np.diag(signs)])
    lo1 = np.concatenate([lo, np.zeros(m)])
    hi1 = np.concatenate([hi, np.full(m, INF)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basic = np.arange(n, n + m)
    at_upper1 = np.concatenate([at_upper, np.zeros(m, dtype=bool)])

    status, basic, at_upper1, x1, piv1 = _primal(c1, A1, b, lo1, hi1, basic, at_upper1)
    art_val = float(np.sum(np.abs(x1[n:])))
    if status != "optimal" or art_val > 1e-7:
        return LPResult("infeasible", np.zeros(n), INF, basic, at_upper1[:n], piv1)

    # --- Phase 2: pin artificials to zero, optimise the real objective ---
    hi1[n:] = 0.0
    c2 = np.concatenate([c, np.zeros(m)])
    status, basic, at_upper1, x2, piv2 = _primal(c2, A1, b, lo1, hi1, basic, at_upper1)
    if status == "unbounded":
        return LPResult("unbounded", x2[:n], -INF, basic, at_upper1[:n], piv1 + piv2)

    x = x2[:n]
    return LPResult("optimal", x, float(c @ x), basic, at_upper1[:n], piv1 + piv2)


# ---------------------------------------------------------------------------
# scipy.linprog-style wrapper (adds slacks, then solves standard form)
# ---------------------------------------------------------------------------
def _standardize(c, A_ub, b_ub, A_eq, b_eq, bounds):
    c = np.asarray(c, float)
    n = len(c)

    def _mat(A, bvec):
        if A is None or len(A) == 0:
            return np.zeros((0, n)), np.zeros(0)
        return np.asarray(A, float).reshape(-1, n), np.asarray(bvec, float).ravel()

    A_ub, b_ub = _mat(A_ub, b_ub)
    A_eq, b_eq = _mat(A_eq, b_eq)
    m_ub = A_ub.shape[0]

    A = np.zeros((m_ub + A_eq.shape[0], n + m_ub))
    A[:m_ub, :n] = A_ub
    if m_ub:
        A[:m_ub, n:] = np.eye(m_ub)
    A[m_ub:, :n] = A_eq
    b = np.concatenate([b_ub, b_eq])

    c_s = np.concatenate([c, np.zeros(m_ub)])
    lo = np.zeros(n + m_ub)
    hi = np.full(n + m_ub, INF)
    if bounds is not None:
        for i, (l, u) in enumerate(bounds):
            lo[i] = -INF if l is None else l
            hi[i] = INF if u is None else u
    return c_s, A, b, lo, hi, n


def linprog_mine(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Solve an LP in scipy.linprog form. Returns an :class:`LPResult` (structural x)."""
    c_s, A, b, lo, hi, n = _standardize(c, A_ub, b_ub, A_eq, b_eq, bounds)
    res = solve_standard(c_s, A, b, lo, hi)
    return LPResult(res.status, res.x[:n], res.obj, res.basic, res.at_upper[:n], res.n_pivots)
