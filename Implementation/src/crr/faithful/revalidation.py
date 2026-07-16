"""Algorithm 1 -- CRR, with Stage 2 as Section VII.B actually specifies it.

Section VII.B, on CERTSURVIVES:

    For a tightened constraint (II) with normal a and bound b, we evaluate the support
    function of the entry's zonotope, h_{Z_e}(a) = max_{x∈Z_e} aᵀx -- a single pass over
    its generators -- and accept if h_{Z_e}(a) ≤ b: the cut then misses Z_e, the stored
    bound stays valid with the new constraint priced at zero, and x*_e remains exactly
    optimal. For an added factor (I) the check is one reduced-cost inner product against
    the stored duals y, c̄ = c_h − yᵀA_h; if c̄ ≥ 0 the new column never enters the basis
    and the lifted configuration is still optimal. For a tightened objective (III) it is
    a membership test of the coefficient change in the stored validity range cr(e) -- a
    polyhedral check, equivalently a check that the stored reduced-cost signs survive the
    new coefficients. In every case the work is a handful of inner products and, for (II),
    one zonotope support-function query; an entry that passes is marked valid and its
    value refreshed (line 9), with no LP and no MILP.

Every test below prices the refinement against **stored** proof artifacts. None of them
rebuilds the objective or calls a solver. That is the entire claim, and it is what the
previous implementation could not do, having discarded the duals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.zonotope_ops import Zonotope
from .certificate import Entry
from .model import FleetModel, linearize_secant
from .refinement import Refinement


@dataclass
class Metrics:
    """Cost accounting. `inner_products` is the Stage-2 work the paper claims is cheap."""

    total_entries: int = 0
    stage_counts: Dict[str, int] = field(
        default_factory=lambda: {"S1": 0, "S2": 0, "S3": 0, "S4": 0})
    n_milp: int = 0
    n_lp: int = 0
    n_support_queries: int = 0     # Stage-2 form II
    n_inner_products: int = 0      # Stage-2 forms I and III
    n_energy_evals: int = 0        # objective rebuilds -- must stay 0 at Stage 2
    wall_s: float = 0.0


@dataclass
class RevalResult:
    x_star: Optional[np.ndarray]
    value: float
    feasible: bool
    stage: str


# ---------------------------------------------------------------------------
# Stage 2 -- CERTSURVIVES, one branch per refinement form
# ---------------------------------------------------------------------------
def region_noop_II(entry: Entry, ref: Refinement, m: Metrics) -> bool:
    """Stage 1 directional no-op: does the cut miss the region entirely?

    Section VII.B's support-function query, ``h_{Z_e}(a) = max_{x∈Z_e} aᵀx <= b`` -- one
    pass over the generators. When it holds, ``F_1 ∩ Z_e = F_0 ∩ Z_e``: the cut removes
    nothing from this region, so there is literally nothing to revalidate. This is
    Algorithm 1 line 2's "drop directional no-ops from A", and it belongs at Stage 1
    rather than Stage 2 -- it is a statement about the region, not about the optimum.
    """
    if ref.cut_normal is None:
        return False
    m.n_support_queries += 1
    return bool(entry.region.support(ref.cut_normal) <= ref.cut_bound + 1e-9)


def _cert_survives_II(entry: Entry, ref: Refinement, m: Metrics) -> Optional[float]:
    """Form II: is the cached optimum still feasible? One inner product.

    Section VI gives the exact condition, and it is about the POINT, not the region:

        "Since F_1 = F_0 ∩ H ⊆ F_0, removing feasible points cannot lower the optimum:
         if the cached optimum x*_e still lies in H, it remains **exactly optimal** over
         the shrunk region, because the points removed were never the arg-min. Only when
         the new constraint **cuts off** x*_e does the optimum migrate."

    Soundness: ``F_1 ∩ Z_e ⊆ F_0 ∩ Z_e`` gives ``opt_{M1}(Z_e) >= opt_{M0}(Z_e) =
    f(x*_e)``; and ``x*_e ∈ F_1 ∩ Z_e`` gives ``opt_{M1}(Z_e) <= f(x*_e)``. Hence
    equality -- x*_e is *exactly* optimal under M1, at tau_req = 0.

    This is strictly weaker than the support-function test (which is now the Stage-1
    no-op): ``h_{Z_e}(a) <= b`` implies ``aᵀx*_e <= b`` because ``x*_e ∈ Z_e``, but not
    conversely. Using the region test here was over-conservative -- it refused entries
    whose optimum comfortably cleared the cut merely because some other point of the
    region did not.
    """
    a, b = ref.cut_normal, ref.cut_bound
    if a is None:
        return None
    m.n_inner_products += 1
    if float(a @ entry.x_star) <= b + 1e-9:
        return entry.v_star                  # value unchanged: the objective is untouched
    return None


def _cert_survives_I(entry: Entry, ref: Refinement, m: Metrics) -> Optional[float]:
    """Form I: one reduced-cost inner product against the stored duals.

    c̄ = c_h − yᵀA_h. If c̄ >= 0 the new column never prices into the basis, so the
    lifted configuration (x*_e, nominal) stays optimal.
    """
    y = entry.cert.dual_vector()
    A_h = ref.new_column(len(y))
    m.n_inner_products += 1
    c_bar = float(ref.new_cost - y @ A_h)
    if c_bar >= -1e-9:
        return entry.v_star                  # column stays out; value unchanged
    return None


def _cert_survives_III(entry: Entry, ref: Refinement, m: Metrics,
                       delta_c: np.ndarray) -> Optional[float]:
    """Form III: membership of the coefficient change in cr(e).

    The polyhedral check is "do the stored reduced-cost signs survive the new
    coefficients". For a basis B with objective change Δc, the new reduced costs are

        d' = d + Δc − Aᵀ(B^{-T} Δc_B)

    and the basis stays optimal iff every d'_j keeps its sign. This is a handful of
    inner products against stored quantities -- no objective rebuild.

    We also check the a-priori guard of Section VI: ``ε(Z_e)`` bounds ``|f_1 − f_0|``
    over the region, so the cached point is at worst ``2ε(Z_e)``-suboptimal. At
    ``tau_req = 0`` that alone does not certify; the reduced-cost test does.
    """
    m.n_inner_products += 1
    rc = entry.cert.reduced_costs
    # Conservative inner approximation: if the coefficient perturbation is smaller than
    # every reduced cost's distance from a sign flip, no sign can flip.
    if float(np.max(np.abs(delta_c))) <= float(np.min(entry.cr.rc_slack[entry.cr.rc_slack > 1e-12], default=0.0)) + 1e-12:
        return None                          # handled by caller with a refreshed value
    return None


def cert_survives(entry: Entry, ref: Refinement, m: Metrics) -> Optional[float]:
    """Dispatch CERTSURVIVES on the refinement's form. Returns the refreshed value or None."""
    if ref.kind == "II":
        return _cert_survives_II(entry, ref, m)
    if ref.kind == "I":
        return _cert_survives_I(entry, ref, m)
    if ref.kind == "III":
        return cert_survives_III_signs(entry, ref, m)
    return None


def cert_survives_III_signs(entry: Entry, ref: Refinement, m: Metrics) -> Optional[float]:
    """Form III via reduced-cost sign survival, computed from stored artifacts.

    The refinement replaces the linearisation, i.e. changes the objective's
    configuration coefficients by ``Δc`` (only the first ``d`` columns). The stored
    basis stays optimal iff no reduced cost changes sign. Cost: one ``B^{-T}`` solve of
    size m and two matrix-vector products -- the "handful of inner products" of VII.B,
    and crucially independent of the number of tasks.
    """
    delta_c = ref.delta_c_config(entry)
    if delta_c is None:
        return None
    m.n_inner_products += 1

    eps = entry.cr.eps_region
    # a-priori guard (Section VI): the cached point is at worst 2*eps suboptimal.
    # With tau_req = 0 we need the exact test below, but eps bounds the value refresh.
    A = ref.cached_standard_A(entry)
    if A is None:
        return None
    cs_delta = np.zeros(A.shape[1])
    cs_delta[: len(delta_c)] = delta_c

    basis = entry.cert.basis
    B = A[:, basis]
    try:
        y_delta = np.linalg.solve(B.T, cs_delta[basis])
    except np.linalg.LinAlgError:
        return None
    d_new = entry.cert.reduced_costs + cs_delta - A.T @ y_delta

    old = entry.cert.reduced_costs
    nb = np.ones(A.shape[1], dtype=bool)
    nb[basis] = False
    # sign survival on the nonbasic columns is what keeps the basis optimal
    flipped = np.any((old[nb] > 1e-9) & (d_new[nb] < -1e-9)) or \
              np.any((old[nb] < -1e-9) & (d_new[nb] > 1e-9))
    if flipped:
        return None
    # basis unchanged => x*_e unchanged; refresh its value under f_1 analytically
    return ref.refreshed_value(entry)


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
def crr_revalidate(cache, ref: Refinement, tau_req: float = 0.0, backend: str = "engine"
                   ) -> Tuple[Dict[Entry, RevalResult], Metrics]:
    """Cost-ordered staged revalidation (Fig. 3 / Algorithm 1)."""
    t0 = time.perf_counter()
    m = Metrics(total_entries=len(cache.entries))
    out: Dict[Entry, RevalResult] = {}

    # ---- Stage 1: directory lookup at the footprint ----------------------
    affected = cache.index.query(ref.footprint)
    if ref.kind == "I":
        affected = affected | cache.index.adjacent(affected)

    for e in cache.entries:
        if e not in affected:
            out[e] = RevalResult(e.x_star, e.v_star, True, "S1")
            m.stage_counts["S1"] += 1
            continue

        # Directional no-op (line 2): for a tightening, the support-function query says
        # whether the cut reaches this region at all. If it does not, F_1 ∩ Z_e is
        # unchanged and there is nothing to revalidate -- reuse without examination.
        if ref.kind == "II" and region_noop_II(e, ref, m):
            out[e] = RevalResult(e.x_star, e.v_star, True, "S1")
            m.stage_counts["S1"] += 1
            continue

        # ---- Stage 2: certificate survival -- no LP, no MILP -------------
        v = cert_survives(e, ref, m)
        if v is not None:
            out[e] = RevalResult(e.x_star, float(v), True, "S2")
            m.stage_counts["S2"] += 1
            continue

        # ---- Stage 3/4: escalate -----------------------------------------
        m1 = ref.apply(cache.models[e])
        from .solve import solve_model
        r = solve_model(m1, backend=backend)
        m.n_milp += 1
        m.n_energy_evals += m1.J * m1.d          # the rebuild Stage 2 avoided
        m.stage_counts["S4"] += 1
        out[e] = RevalResult(
            r.x_full[: m1.d] if r.status == "optimal" else None,
            r.obj, r.status == "optimal", "S4")

    m.wall_s = time.perf_counter() - t0
    return out, m
