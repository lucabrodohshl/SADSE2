"""cert(e), dep(e), cr(e) -- as Section VII.A defines them.

Section VII.A, verbatim on what an entry carries:

    * a certificate cert(e): the dual/KKT multipliers at x*_e, its binding constraints
      and reduced-cost signs, and a valid global bound witness, proving v*_e optimal;
    * a dependency dep(e): the model features (variables, constraints, linearization
      segments) that this certificate actually uses, read from its active set and dual
      support;
    * a validity range cr(e): a conservative inner approximation of the set of
      coefficient perturbations over which x*_e provably stays optimal -- the critical
      region of the parametric program, whose extent for form III is exactly the
      residual radius ε(Z_e).

and on the principle:

    we seek to reuse the proof, not only the answer. [...] Prior cache-based
    self-optimization discard this knowledge. CRR seeks to revalidate the proof
    arithmetically rather than re-deriving optimality, which would require a solver call.

The point of this module is that last sentence. The duals ARE the certificate; without
them Stage 2 has nothing to price a refinement against and can only recompute the
objective -- which is re-deriving optimality, the thing the paper says not to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.zonotope_ops import Zonotope
from .model import FleetModel, ODDRegime, Task, residual_radius


@dataclass
class Certificate:
    """The optimality proof, retained from the solver call that produced x*_e."""

    y_eq: np.ndarray             # duals of the equality rows (task-coverage)
    y_ub: np.ndarray             # duals of the inequality rows (>= 0)
    reduced_costs: np.ndarray    # d = c - y^T A over all columns
    basis: np.ndarray            # basic column indices of the final LP relaxation
    at_upper: np.ndarray         # nonbasic-at-upper flags
    active_ub: np.ndarray        # bool per inequality row: binding at x*
    bound_witness: float         # the B&B global lower bound proving integer optimality
    lp_value: float              # LP-relaxation optimum (the bound witness's source)

    def dual_vector(self) -> np.ndarray:
        return np.concatenate([self.y_ub, self.y_eq])


@dataclass
class ValidityRange:
    """cr(e): coefficient perturbations over which x*_e provably stays optimal.

    Two complementary pieces, both required by Section VII.B:

    * ``eps_region`` -- ε(Z_e), the a-priori Lagrange-remainder bound of Section VI,
      computed over THIS entry's zonotope (not a shared box, and not a hardcoded
      curvature).
    * ``rc_slack`` -- the critical region proper: how far each reduced cost can move
      before its sign flips. A coefficient change stays inside cr(e) iff every reduced
      cost keeps its sign, which is exactly "the stored reduced-cost signs survive the
      new coefficients".
    """

    eps_region: float
    rc_slack: np.ndarray          # per-column slack before a sign flip (>= 0)
    obj_margin: float             # integer bound gap: lp_value vs incumbent


@dataclass(eq=False)
class Entry:
    """e = (Z_e, x*_e, v*_e) enriched with cert / dep / cr (Definition 2 + VII.A)."""

    name: str
    regime: ODDRegime
    region: Zonotope              # Z_e ⊆ DS -- this entry's OWN region
    x_star: np.ndarray            # x*_e, proven optimal over Z_e
    v_star: float
    assignment: Dict[int, List[int]]
    cert: Certificate
    dep: Set[str]
    cr: ValidityRange


def extract_duals(build: dict, x_full: np.ndarray, basis: np.ndarray,
                  at_upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover y and reduced costs from the final basis of the LP relaxation.

    ``y_B = c_B B^{-1}``, ``d = c - Aᵀy``. This is the information the solver already
    computed and the previous implementation discarded.
    """
    from src.crr.simplex import _standardize

    cs, A, b, lo, hi, n = _standardize(build["c"], build["A_ub"], build["b_ub"],
                                       build["A_eq"], build["b_eq"], build["bounds"])
    B = A[:, basis]
    try:
        y = np.linalg.solve(B.T, cs[basis])
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(B.T, cs[basis], rcond=None)[0]
    d = cs - A.T @ y

    m_ub = build["A_ub"].shape[0]
    y_ub = y[:m_ub]
    y_eq = y[m_ub:]
    return y_ub, y_eq, d, A


def build_entry(model: FleetModel, name: str, backend: str = "engine") -> Optional[Entry]:
    """Solve M0 over ``model.region`` and retain the optimality PROOF, not just x*.

    ``x*_e`` is optimal over the entry's own region ``Z_e`` -- the region is part of the
    problem (it bounds the configuration variables), which is what makes ``x*_e``
    "proven optimal over Z_e" in the sense of Definition 2.
    """
    from .solve import solve_model

    res = solve_model(model, backend=backend)
    if res.status != "optimal":
        return None

    build = res.build
    y_ub, y_eq, rc, A = extract_duals(build, res.x_full, res.basis, res.at_upper)

    # binding inequality rows at x* (the active set)
    slack = build["b_ub"] - build["A_ub"] @ res.x_full[: build["n"]]
    active_ub = np.abs(slack) <= 1e-7

    cert = Certificate(
        y_eq=y_eq, y_ub=y_ub, reduced_costs=rc,
        basis=np.asarray(res.basis, int), at_upper=np.asarray(res.at_upper, bool),
        active_ub=active_ub,
        bound_witness=float(res.lp_bound),
        lp_value=float(res.lp_bound),
    )

    # dep(e): the features the certificate actually reads -- its active set and dual
    # support (Section VII.A), NOT a fixed global set handed to every entry.
    #
    # Soundness note. Theorem 1 requires D to return a **superset** of the entries whose
    # certificate uses a changed feature: an entry omitted from the footprint is reused
    # at Stage 1 with no check at all, so omitting one the refinement can reach is an
    # unsound skip rather than an optimisation ("superset-safe" in Algorithm 1, line 1).
    # The configuration variables are read by every certificate -- they carry the
    # objective and are what the optimum is expressed in -- so every entry depends on
    # ``config_box``. It is Stage 2's support-function query, not Stage 1, that decides
    # whether a cut on those variables actually reaches this entry's region.
    dep: Set[str] = {"obj_coeffs", "linearization", "config_box"}
    dep.add(f"regime:{model.odd.name}")
    for name in ("speed", "altitude", "camera_res"):
        dep.add(f"dim:{name}")
    for i in range(model.num_agents):
        # the battery row for agent i sits at the end of the ub block
        row = build["A_ub"].shape[0] - len(model.extra_constraints) - model.num_agents + i
        if 0 <= row < len(active_ub) and (active_ub[row] or abs(y_ub[row]) > 1e-9):
            dep.add(f"battery_row:{i}")
    if np.any(np.abs(y_eq) > 1e-9):
        dep.add("task_rows")

    eps = residual_radius(model.region, model.tasks, model.odd)
    # critical region: distance of each reduced cost from a sign flip
    rc_slack = np.abs(rc)
    cr = ValidityRange(eps_region=float(eps), rc_slack=rc_slack,
                       obj_margin=float(res.obj - res.lp_bound))

    return Entry(name=name, regime=model.odd, region=model.region,
                 x_star=res.x_full[: build["d"]].copy(), v_star=float(res.obj),
                 assignment=res.assignment, cert=cert, dep=dep, cr=cr)
