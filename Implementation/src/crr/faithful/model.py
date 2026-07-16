"""The paper's model, implemented as specified.

Why this module exists
----------------------
The pre-existing implementation (``src/crr/model.py``) is not the model the paper
defines. The differences are structural, not cosmetic, and they are exactly the
reason the certificates could not be implemented:

* **The paper's decision space is continuous.** Definition 2: an entry is
  ``e = (Z_e, x*_e, v*_e)`` where ``Z_e ⊆ DS`` is a *region* stored as a zonotope and
  ``x*_e`` is "a configuration proven optimal **over Z_e**". The old code samples ``K``
  random configurations and takes the best of ``K``. With a finite candidate set there
  is no region to be optimal over, no parametric program, and hence no critical region
  ``cr(e)`` and no support-function test.
* **Every old entry shared one zonotope** (``Zonotope.from_box`` over the whole design
  box, handed to every entry), which is why ``_residual_radius`` evaluated to the same
  11.649074 for all 16 entries: the region genuinely was identical. Section VII.B's
  Type-II test ``h_{Z_e}(a) ≤ b`` carries no information when every ``Z_e`` is the same
  set.
* **No duals were stored.** Section VII.A defines ``cert(e)`` as "the dual/KKT
  multipliers at ``x*_e``, its binding constraints and reduced-cost signs, and a valid
  global bound witness". The old ``cert`` holds an assignment, per-agent loads and a
  scalar margin. The proof itself was discarded -- so Stage 2 had nothing to price
  against and fell back to recomputing the objective, i.e. "re-deriving optimality",
  which Section VII.A explicitly identifies as what prior work does wrong.

The model here follows Section II. A fleet of ``M`` drones covers ``J`` tasks. The
configuration ``x = (speed, altitude, camera_res)`` is a **continuous decision
variable** over the design space. True energy grows quadratically with speed, so -- as
Section II states -- the program is made tractable by a **first-order Taylor
linearisation around a reference configuration** ``x_ref``, giving an affine
``E_j(x) = a_j + g_jᵀx``. The result is a genuine MILP: continuous ``x`` plus binary
assignment ``z[i,j]``, with the bilinear per-agent battery load ``z[i,j]·E_j(x)``
linearised by McCormick/big-M into ``w[i,j]``.

Model forms (Section VI):
  I   -- a new decision factor lifts DS into a new dimension (a new column).
  II  -- a constraint is added/tightened: ``F_1 = F_0 ∩ H``.
  III -- the objective linearisation is replaced by a tighter one, with
         ``|f_1(x) - f_0(x)| ≤ ε(x)`` on ``F``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.zonotope_ops import Zonotope

# Design space DS: the continuous configuration parameters.
DS_DIMS: Tuple[str, ...] = ("speed", "altitude", "camera_res")
DS_BOX: List[Tuple[float, float]] = [(3.0, 22.0), (10.0, 120.0), (4.0, 20.0)]

# Platform constants (aligned with src/milp_solver.py::drone_energy_model).
P0_W = 120.0
V_REF = 10.0
K_DRAG_FRAC = 0.6
K_ALT_PER_M = 0.001
P_CAM_PER_MP = 0.25
P_SENSOR_W = 1.0


@dataclass(frozen=True)
class ODDRegime:
    """A discretised operating regime ("Calm", "Light", "Strong" in Section II).

    ``weight`` is how often the operating envelope actually visits this regime, carried
    so results can be reported deployment-weighted as well as unweighted.
    """

    name: str
    wind: float               # [m/s] headwind component
    temperature: float = 15.0  # [degC] -- derates usable battery capacity
    weight: float = 1.0


def battery_derate(temperature_c: float) -> float:
    """Usable-capacity fraction: bounded and non-monotone (peaks ~25-45 degC).

    Temperature reaches the model only through the battery right-hand side, so it can
    flip feasibility but never re-rank -- it is a form-II channel by construction.
    """
    loss = (0.006 * max(0.0, 25.0 - temperature_c) ** 1.1
            + 0.004 * max(0.0, temperature_c - 45.0))
    return max(0.55, 1.0 - loss)


@dataclass
class Task:
    length: float        # [m]


def true_energy(x: np.ndarray, task: Task, odd: ODDRegime) -> float:
    """The TRUE (nonlinear) energy for one task: quadratic in airspeed.

    ``f`` in the paper. Propulsion is driven by airspeed ``v + wind``; the time on task
    is driven by ground speed ``v``. This is the nonlinear quantity that the model
    linearises, and the thing a Type-III refinement approximates more tightly.
    """
    v, alt, res = float(x[0]), float(x[1]), float(x[2])
    v = max(0.1, v)
    A = P0_W + P0_W * K_ALT_PER_M * alt + P_CAM_PER_MP * res + P_SENSOR_W
    B = P0_W * K_DRAG_FRAC / (V_REF ** 2)
    v_air = max(0.0, v + odd.wind)
    return (A + B * v_air ** 2) * task.length / (v * 3600.0)


def energy_grad(x: np.ndarray, task: Task, odd: ODDRegime) -> np.ndarray:
    """∇_x of :func:`true_energy` (analytic)."""
    v, alt, res = float(max(0.1, x[0])), float(x[1]), float(x[2])
    A = P0_W + P0_W * K_ALT_PER_M * alt + P_CAM_PER_MP * res + P_SENSOR_W
    B = P0_W * K_DRAG_FRAC / (V_REF ** 2)
    w = odd.wind
    va = max(0.0, v + w)
    L = task.length / 3600.0

    # d/dv [ (A + B(v+w)^2) / v ] = (2B(v+w)v - (A + B(v+w)^2)) / v^2
    d_v = L * (2.0 * B * va * v - (A + B * va ** 2)) / (v ** 2)
    d_alt = L * (P0_W * K_ALT_PER_M) / v
    d_res = L * P_CAM_PER_MP / v
    return np.array([d_v, d_alt, d_res])


def energy_curvature_bound(region: Zonotope, task: Task, odd: ODDRegime) -> float:
    """``sup_{Z} |f''|`` along the speed axis -- the constant in ε(Z_e).

    Section VI: for a Taylor linearisation with Lagrange remainder,
    ``ε(Z_e) ≤ ½ · sup_{Z_e}|f''| · diam(Z_e)²``. The old code hardcoded this constant
    (``0.6 / 18**2``) and evaluated diam over the shared whole-design box, which is why
    it produced one entry-independent number. Here it is computed over the entry's OWN
    region, from the actual second derivative.

    ``f(v) = (A + B(v+w)^2)·L/v``  ⇒  ``f''(v) = 2L(A + Bw^2)/v^3``  (the ``2Bv``-linear
    part has zero second derivative), maximised at the region's smallest speed.
    """
    box = region.to_box_bounds()
    v_lo = max(0.1, float(box[0, 0]))
    alt_hi = float(box[1, 1])
    res_hi = float(box[2, 1])
    A = P0_W + P0_W * K_ALT_PER_M * alt_hi + P_CAM_PER_MP * res_hi + P_SENSOR_W
    B = P0_W * K_DRAG_FRAC / (V_REF ** 2)
    L = task.length / 3600.0
    return 2.0 * L * (A + B * odd.wind ** 2) / (v_lo ** 3)


def residual_radius(region: Zonotope, tasks: Sequence[Task], odd: ODDRegime) -> float:
    """ε(Z_e): an a-priori bound on |f_1 - f_0| over the region (Section VI).

    Section VI gives the isotropic form ``ε(Z_e) ≤ ½ sup|f''| · diam(Z_e)²``. Applied
    literally here that is wildly loose, because ``diam(Z_e)`` is dominated by the
    altitude axis (a 110 m span) while the curvature lives almost entirely in speed --
    it produced ε ≈ 63000 against optima of ≈ 17. The energy is *affine* in altitude and
    camera resolution at fixed speed, so those axes contribute no second-order term of
    their own; they enter only through cross terms with speed.

    The Hessian is therefore
        [[h_vv, h_va, h_vr],
         [h_va, 0,    0   ],
         [h_vr, 0,    0   ]]
    and the quadratic remainder ``½ δᵀHδ = ½ δ_v (h_vv δ_v + 2 h_va δ_a + 2 h_vr δ_r)``
    scales with the **speed** deviation. Bounding it that way is still a-priori and
    model-derived -- it is the same Lagrange remainder, just not thrown away by an
    isotropic norm. Slicing the speed axis then shrinks ε quadratically, which is what
    makes ε(Z_e) discriminate between entries at all.
    """
    box = region.to_box_bounds()
    half = 0.5 * (box[:, 1] - box[:, 0])
    dv, da, dr = float(half[0]), float(half[1]), float(half[2])
    v_lo = max(0.1, float(box[0, 0]))
    B = P0_W * K_DRAG_FRAC / (V_REF ** 2)

    total = 0.0
    for t in tasks:
        L = t.length / 3600.0
        A_hi = P0_W + P0_W * K_ALT_PER_M * float(box[1, 1]) + P_CAM_PER_MP * float(box[2, 1]) + P_SENSOR_W
        h_vv = 2.0 * L * (A_hi + B * odd.wind ** 2) / (v_lo ** 3)
        h_va = L * (P0_W * K_ALT_PER_M) / (v_lo ** 2)
        h_vr = L * P_CAM_PER_MP / (v_lo ** 2)
        total += 0.5 * dv * (h_vv * dv + 2.0 * h_va * da + 2.0 * h_vr * dr)
    return float(total)


def exact_linearization_gap(f0: "LinearizedEnergy", f1: "LinearizedEnergy",
                            region: Zonotope) -> float:
    """max_{x∈Z} |f_1(x) - f_0(x)| for two AFFINE models -- exact, one support query.

    Once the refinement is known, both models are affine, so their difference is affine
    and its maximum over a zonotope is exactly the support function:
        max_{x∈Z} (Δa + Δgᵀx) = Δa + h_Z(Δg).
    No Lagrange bound is needed. This is the "handful of inner products and one
    zonotope support-function query" of Section VII.B, and it is exact rather than
    conservative.
    """
    da = float(f1.a.sum() - f0.a.sum())
    dg = f1.G.sum(axis=0) - f0.G.sum(axis=0)
    hi = da + region.support(dg)
    lo = da - region.support(-dg)
    return float(max(abs(hi), abs(lo)))


@dataclass
class LinearizedEnergy:
    """``E_j(x) ≈ a_j + g_jᵀx`` -- the first-order Taylor model of Section II."""

    a: np.ndarray                  # (J,) constants
    G: np.ndarray                  # (J, d) gradients
    x_ref: np.ndarray

    def value(self, x: np.ndarray) -> np.ndarray:
        return self.a + self.G @ np.asarray(x, float)

    def total(self, x: np.ndarray) -> float:
        return float(np.sum(self.value(x)))


def linearize(tasks: Sequence[Task], odd: ODDRegime, x_ref: np.ndarray) -> LinearizedEnergy:
    """First-order Taylor linearisation of the true energy about ``x_ref`` (f_0)."""
    a = np.empty(len(tasks))
    G = np.empty((len(tasks), len(x_ref)))
    for j, t in enumerate(tasks):
        g = energy_grad(x_ref, t, odd)
        a[j] = true_energy(x_ref, t, odd) - g @ x_ref
        G[j] = g
    return LinearizedEnergy(a=a, G=G, x_ref=np.asarray(x_ref, float))


def linearize_secant(tasks: Sequence[Task], odd: ODDRegime, region: Zonotope
                     ) -> LinearizedEnergy:
    """A TIGHTER linearisation over the region (f_1) -- the Type-III refinement.

    Section VI, form III: "the linearisation of a non-linear quantity is too coarse and
    must be replaced by a tighter one". A chord/secant fit over the entry's actual
    region tracks a convex ``f`` far better than a tangent taken at a distant reference
    point, and -- unlike a uniform rescale -- it moves each configuration's coefficients
    by a *different* amount, so it can genuinely re-rank.
    """
    box = region.to_box_bounds()
    lo, hi = box[:, 0], box[:, 1]
    mid = 0.5 * (lo + hi)
    a = np.empty(len(tasks))
    G = np.empty((len(tasks), len(lo)))
    for j, t in enumerate(tasks):
        g = np.empty(len(lo))
        for d in range(len(lo)):
            p_lo, p_hi = mid.copy(), mid.copy()
            p_lo[d], p_hi[d] = lo[d], hi[d]
            span = max(1e-9, hi[d] - lo[d])
            g[d] = (true_energy(p_hi, t, odd) - true_energy(p_lo, t, odd)) / span
        a[j] = true_energy(mid, t, odd) - g @ mid
        G[j] = g
    return LinearizedEnergy(a=a, G=G, x_ref=mid)


@dataclass
class FleetModel:
    """M = (X, f, F): the task-assignment MILP over a continuous design space.

    Decision variables, in this column order:
        x[0:d]                      continuous configuration (DS)
        z[i,j]  (M*J binaries)      agent i performs task j
        w[i,j]  (M*J continuous)    linearised product z[i,j] * E_j(x)
    """

    tasks: List[Task]
    odd: ODDRegime
    num_agents: int
    energy: LinearizedEnergy
    region: Zonotope
    capacity_wh: float = 300.0
    reserve: float = 0.20
    safety: float = 1.05
    soh: Optional[np.ndarray] = None
    extra_constraints: List[Tuple[np.ndarray, float]] = field(default_factory=list)

    @property
    def d(self) -> int:
        return self.energy.G.shape[1]

    @property
    def J(self) -> int:
        return len(self.tasks)

    def usable(self, i: int) -> float:
        s = 1.0 if self.soh is None else float(self.soh[i])
        return ((1.0 - self.reserve) * self.capacity_wh / self.safety * s
                * battery_derate(self.odd.temperature))

    # -- variable layout ---------------------------------------------------
    def n_vars(self) -> int:
        return self.d + 2 * self.num_agents * self.J

    def zi(self, i: int, j: int) -> int:
        return self.d + i * self.J + j

    def wi(self, i: int, j: int) -> int:
        return self.d + self.num_agents * self.J + i * self.J + j

    def build(self) -> dict:
        """Assemble the MILP in scipy.linprog form."""
        d, M, J = self.d, self.num_agents, self.J
        n = self.n_vars()
        box = self.region.to_box_bounds()

        # objective: total fleet energy = sum_j (a_j + g_j^T x) -- affine in x.
        c = np.zeros(n)
        c[:d] = self.energy.G.sum(axis=0)
        obj_const = float(self.energy.a.sum())

        rows_eq, b_eq = [], []
        for j in range(J):                        # each task assigned exactly once
            r = np.zeros(n)
            for i in range(M):
                r[self.zi(i, j)] = 1.0
            rows_eq.append(r); b_eq.append(1.0)

        rows_ub, b_ub = [], []
        # Big-M for the McCormick envelope: an upper bound on any single task's energy
        # over the region.
        e_hi = float(max(self.energy.a[j] + max(self.energy.G[j] @ box[:, 0],
                                                self.energy.G[j] @ box[:, 1])
                         for j in range(J)))
        BIG_M = max(1.0, abs(e_hi) * 2.0 + 1.0)

        for i in range(M):
            for j in range(J):
                # w >= E_j(x) - M(1-z)   ->   -w + g^T x - M z <= -a_j + M
                r = np.zeros(n)
                r[:d] = self.energy.G[j]
                r[self.wi(i, j)] = -1.0
                r[self.zi(i, j)] = -BIG_M
                rows_ub.append(r); b_ub.append(-self.energy.a[j] + BIG_M)
                # w <= M z
                r = np.zeros(n)
                r[self.wi(i, j)] = 1.0
                r[self.zi(i, j)] = -BIG_M
                rows_ub.append(r); b_ub.append(0.0)
                # w <= E_j(x)   ->   w - g^T x <= a_j
                r = np.zeros(n)
                r[self.wi(i, j)] = 1.0
                r[:d] = -self.energy.G[j]
                rows_ub.append(r); b_ub.append(self.energy.a[j])

        for i in range(M):                        # per-agent battery budget
            r = np.zeros(n)
            for j in range(J):
                r[self.wi(i, j)] = 1.0
            rows_ub.append(r); b_ub.append(self.usable(i))

        for (a_vec, b_val) in self.extra_constraints:   # form II: F_1 = F_0 ∩ H
            r = np.zeros(n)
            r[:d] = a_vec
            rows_ub.append(r); b_ub.append(b_val)

        lo = np.concatenate([box[:, 0], np.zeros(M * J), np.zeros(M * J)])
        hi = np.concatenate([box[:, 1], np.ones(M * J), np.full(M * J, BIG_M)])
        integrality = np.zeros(n)
        integrality[d:d + M * J] = 1.0

        return dict(c=c, obj_const=obj_const,
                    A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                    A_eq=np.array(rows_eq), b_eq=np.array(b_eq),
                    bounds=list(zip(lo, hi)), integer_mask=integrality.astype(int).tolist(),
                    big_m=BIG_M, n=n, d=d)
