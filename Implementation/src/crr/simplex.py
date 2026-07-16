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
  * ``primal_warm`` -- re-optimize from a primal-feasible (warm) basis after the
    OBJECTIVE changed.

Choosing a warm start
---------------------
Which routine is valid depends on *what changed*:

  RHS / variable bounds  -> basis stays dual-feasible   -> ``dual_simplex``
  objective coefficients -> basis stays primal-feasible -> ``primal_warm``
  constraint matrix      -> neither                     -> ``solve_standard`` (cold)

Using ``dual_simplex`` after an objective change terminates at a primal-feasible
but suboptimal point while reporting ``"optimal"``; ``dual_simplex`` validates its
incoming basis and refuses that case rather than returning a wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import lu_factor, lu_solve

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


def _factorize(B):
    """LU-factorize the basis once per iteration; reused for all three solves."""
    try:
        return lu_factor(B), True
    except (ValueError, LinAlgError):
        return None, False


def _price_primal(d, nb_mask, at_upper, bland):
    """Choose the entering variable.

    Dantzig (most-negative reduced cost) converges in far fewer pivots than
    Bland's rule, but can cycle on degenerate vertices. We use Dantzig by default
    and fall back to Bland (smallest eligible index, provably finite) once the
    objective has stalled -- the standard hybrid: Bland's guarantee, Dantzig's speed.
    """
    elig_up = nb_mask & (~at_upper) & (d < -TOL)     # at lower bound, want to increase
    elig_dn = nb_mask & at_upper & (d > TOL)         # at upper bound, want to decrease
    if not (elig_up.any() or elig_dn.any()):
        return -1, 0

    if bland:
        i_up = np.argmax(elig_up) if elig_up.any() else ntot_max
        i_dn = np.argmax(elig_dn) if elig_dn.any() else ntot_max
        if not elig_up.any():
            return int(i_dn), -1
        if not elig_dn.any():
            return int(i_up), +1
        return (int(i_up), +1) if i_up < i_dn else (int(i_dn), -1)

    best_up = float(np.min(d[elig_up])) if elig_up.any() else INF
    best_dn = float(-np.max(d[elig_dn])) if elig_dn.any() else INF
    if best_up <= best_dn:
        return int(np.argmin(np.where(elig_up, d, INF))), +1
    return int(np.argmax(np.where(elig_dn, d, -INF))), -1


ntot_max = np.iinfo(np.int64).max


def _primal(c, A, b, lo, hi, basic, at_upper, max_iter=20000):
    m, ntot = A.shape
    basic = np.array(basic, dtype=int)
    at_upper = np.array(at_upper, dtype=bool)
    pivots = 0

    bland = False
    stall = 0
    last_obj = INF
    rows = np.arange(m)

    for _ in range(max_iter):
        B = A[:, basic]
        lu, ok = _factorize(B)

        x, nb_mask = _nonbasic_values(ntot, basic, at_upper, lo, hi)
        rhs = b - A[:, nb_mask] @ x[nb_mask]
        if ok:
            xB = lu_solve(lu, rhs)
        else:
            xB = np.linalg.lstsq(B, rhs, rcond=None)[0]
        x[basic] = xB

        # Reuse the SAME factorization for the dual solve (B^T y = c_B).
        if ok:
            y = lu_solve(lu, c[basic], trans=1)
        else:
            y = np.linalg.lstsq(B.T, c[basic], rcond=None)[0]
        d = c - A.T @ y                     # reduced costs

        # Anti-cycling: degenerate pivots leave the objective unchanged. After a
        # run of them, switch to Bland's rule until progress resumes.
        obj = float(c @ x)
        if obj > last_obj - TOL:
            stall += 1
            if stall > 2 * m + 20:
                bland = True
        else:
            stall = 0
            bland = False
        last_obj = obj

        entering, direction = _price_primal(d, nb_mask, at_upper, bland)
        if entering == -1:
            return "optimal", basic, at_upper, x, pivots

        col = lu_solve(lu, A[:, entering]) if ok else np.linalg.lstsq(
            B, A[:, entering], rcond=None)[0]
        dxB = -direction * col                          # d xB / dt

        # ratio test (vectorized over the basis)
        limit = hi[entering] - lo[entering]             # entering bound-flip limit
        bl, bh = lo[basic], hi[basic]
        ratios = np.full(m, INF)
        up = (dxB > TOL) & np.isfinite(bh)
        dn = (dxB < -TOL) & np.isfinite(bl)
        ratios[up] = (bh[up] - xB[up]) / dxB[up]
        ratios[dn] = (xB[dn] - bl[dn]) / (-dxB[dn])

        leaving_row, leave_at_upper = -1, False
        if np.isfinite(ratios).any():
            cand = rows[ratios < limit - TOL]
            if cand.size:
                if bland:
                    # tie-break on smallest variable index for finiteness
                    best_r = float(np.min(ratios[cand]))
                    tied = cand[ratios[cand] <= best_r + TOL]
                    leaving_row = int(tied[np.argmin(basic[tied])])
                else:
                    leaving_row = int(cand[np.argmin(ratios[cand])])
                limit = float(ratios[leaving_row])
                leave_at_upper = bool(up[leaving_row])

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


def dual_feasibility_violation(c, A, basic, at_upper) -> float:
    """Largest reduced-cost sign violation of ``basic`` -- 0.0 iff dual-feasible.

    Dual feasibility for a bounded-variable basis requires ``d_j >= 0`` for every
    nonbasic ``j`` at its lower bound and ``d_j <= 0`` for every nonbasic ``j`` at
    its upper bound. :func:`dual_simplex` is only valid from such a basis.
    """
    c = np.asarray(c, float); A = np.asarray(A, float)
    basic = np.asarray(basic, int); at_upper = np.asarray(at_upper, bool)
    B = A[:, basic]
    try:
        y = np.linalg.solve(B.T, c[basic])
    except np.linalg.LinAlgError:
        return INF
    d = c - A.T @ y
    nb = np.ones(A.shape[1], dtype=bool)
    nb[basic] = False
    viol = 0.0
    at_lo = nb & ~at_upper
    at_hi = nb & at_upper
    if np.any(at_lo):
        viol = max(viol, float(np.max(np.maximum(0.0, -d[at_lo],))))
    if np.any(at_hi):
        viol = max(viol, float(np.max(np.maximum(0.0, d[at_hi]))))
    return viol


def primal_warm(c, A, b, lo, hi, basic, at_upper, max_iter=20000) -> LPResult:
    """Re-optimise from a PRIMAL-feasible (warm) basis after the OBJECTIVE changed.

    When only ``c`` changes, ``A`` and ``b`` are untouched, so the stored basis is
    still *primal* feasible but generally **not** dual-feasible -- exactly the case
    :func:`dual_simplex` may not be used for. This warm-starts the primal simplex
    instead, which restores optimality in a few pivots.

    Use :func:`dual_simplex` when the RHS or bounds changed (basis stays dual-feasible),
    and this when the objective changed.
    """
    c = np.asarray(c, float); A = np.asarray(A, float); b = np.asarray(b, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    status, basic, at_upper, x, pivots = _primal(
        c, A, b, lo, hi, np.asarray(basic, int).copy(), np.asarray(at_upper, bool).copy(),
        max_iter=max_iter)
    if status == "unbounded":
        return LPResult("unbounded", x, -INF, basic, at_upper, pivots)
    return LPResult(status, x, float(c @ x), basic, at_upper, pivots)


def dual_simplex(c, A, b, lo, hi, basic, at_upper, max_iter=20000, validate=True):
    """Re-optimise from a DUAL-feasible (warm) basis, restoring primal feasibility.

    The basis passed in must be optimal for some earlier problem with the same
    ``c`` and ``A`` (hence dual-feasible); only ``b`` or the bounds changed. Used
    to warm-start after a constraint tightening (CRR Stage 3) or a branch bound.
    Returns "infeasible" if the dual is unbounded (no primal solution).

    If the objective ``c`` changed, the incoming basis is *not* dual-feasible and
    this routine would terminate at a primal-feasible but **suboptimal** point while
    reporting ``"optimal"``. ``validate=True`` refuses that case instead of
    returning a wrong answer; use :func:`primal_warm` for objective changes.
    """
    c = np.asarray(c, float); A = np.asarray(A, float); b = np.asarray(b, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    m, ntot = A.shape
    basic = np.array(basic, dtype=int)
    at_upper = np.array(at_upper, dtype=bool)
    pivots = 0

    if validate:
        viol = dual_feasibility_violation(c, A, basic, at_upper)
        if viol > 1e-7:
            raise ValueError(
                f"dual_simplex requires a dual-feasible warm basis (max reduced-cost "
                f"violation {viol:.3e}). The objective almost certainly changed; use "
                f"primal_warm() for objective-only changes."
            )

    eye = np.eye(m)
    for _ in range(max_iter):
        B = A[:, basic]
        lu, ok = _factorize(B)
        x, nb_mask = _nonbasic_values(ntot, basic, at_upper, lo, hi)
        rhs = b - A[:, nb_mask] @ x[nb_mask]
        xB = lu_solve(lu, rhs) if ok else np.linalg.lstsq(B, rhs, rcond=None)[0]
        x[basic] = xB
        # Reuse the same factorization for the dual solve.
        y = (lu_solve(lu, c[basic], trans=1) if ok
             else np.linalg.lstsq(B.T, c[basic], rcond=None)[0])
        d = c - A.T @ y

        # leaving variable: most bound-violating basic (vectorized)
        bl, bh = lo[basic], hi[basic]
        below_viol = np.where(xB < bl - TOL, bl - xB, 0.0)
        above_viol = np.where(xB > bh + TOL, xB - bh, 0.0)
        worst_below = float(below_viol.max()) if m else 0.0
        worst_above = float(above_viol.max()) if m else 0.0
        if max(worst_below, worst_above) <= TOL:
            return LPResult("optimal", x, float(c @ x), basic, at_upper, pivots)
        if worst_below >= worst_above:
            r, below = int(np.argmax(below_viol)), True
        else:
            r, below = int(np.argmax(above_viol)), False

        # row r of B^{-1}A -- reuse the factorization again
        rho = (lu_solve(lu, eye[r], trans=1) if ok
               else np.linalg.lstsq(B.T, eye[r], rcond=None)[0])
        alpha = A.T @ rho                      # row r of B^{-1}A over all columns

        # dual ratio test (vectorized)
        with np.errstate(divide="ignore", invalid="ignore"):
            if below:                          # need xB_r to increase
                m_lo = nb_mask & (~at_upper) & (alpha < -TOL)
                m_hi = nb_mask & at_upper & (alpha > TOL)
                ratios = np.full(ntot, INF)
                ratios[m_lo] = d[m_lo] / (-alpha[m_lo])
                ratios[m_hi] = (-d[m_hi]) / alpha[m_hi]
            else:                              # need xB_r to decrease
                m_lo = nb_mask & (~at_upper) & (alpha > TOL)
                m_hi = nb_mask & at_upper & (alpha < -TOL)
                ratios = np.full(ntot, INF)
                ratios[m_lo] = d[m_lo] / alpha[m_lo]
                ratios[m_hi] = (-d[m_hi]) / (-alpha[m_hi])
        if not np.isfinite(ratios).any():
            return LPResult("infeasible", x, INF, basic, at_upper, pivots)
        entering = int(np.argmin(ratios))

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
