"""The three refinement forms of Section VI, exposed the way Stage 2 needs them.

Section VI distinguishes three *syntactic* forms by how they act on ``(X, f, F)``:

  I   -- a new factor enters: ``X_1 = X_0 × X_new``; the M1-optimal region lives in a
         higher-dimensional space of which the M0 region is the projection.
  II  -- a constraint is added/tightened: ``F_1 = F_0 ∩ H``; "removing feasible points
         cannot lower the optimum: if the cached optimum x*_e still lies in H, it
         remains **exactly optimal** over the shrunk region... Only when the new
         constraint **cuts off** x*_e does the optimum migrate."
  III -- the objective linearisation is replaced by a tighter one with
         ``|f_1(x) − f_0(x)| ≤ ε(x)`` on F; "With F fixed, the arg-min moves by an
         amount governed by the objective gap: the cached x*_e is provably
         g-suboptimal for M_1 with g ≤ 2ε(Z_e)".

Each form therefore exposes exactly what its Stage-2 test prices:
  II  -> ``cut_normal`` / ``cut_bound``      (support-function query)
  I   -> ``new_column`` / ``new_cost``       (reduced-cost inner product)
  III -> ``delta_c_config``                  (reduced-cost sign survival)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

import numpy as np

from .certificate import Entry
from .model import FleetModel, LinearizedEnergy, linearize_secant


@dataclass
class Refinement:
    kind: str                      # "I" | "II" | "III"
    label: str
    footprint: Set[str]
    description: str = ""

    # form II
    cut_normal: Optional[np.ndarray] = None
    cut_bound: float = 0.0
    # form I
    new_cost: float = 0.0
    _new_col: Optional[np.ndarray] = None
    # form III
    _secant: bool = False

    # -- form II -----------------------------------------------------------
    def is_directional_noop(self, e: Entry) -> bool:
        """Line 2 of Algorithm 1: a tightening where the entry is already slack.

        For form II this is subsumed by the support-function test; kept explicit
        because Algorithm 1 lists it as a separate, cheaper trim.
        """
        return False

    # -- form I ------------------------------------------------------------
    def new_column(self, m_rows: int) -> np.ndarray:
        if self._new_col is None:
            return np.zeros(m_rows)
        col = np.zeros(m_rows)
        k = min(len(self._new_col), m_rows)
        col[:k] = self._new_col[:k]
        return col

    # -- form III ----------------------------------------------------------
    def delta_c_config(self, e: Entry) -> Optional[np.ndarray]:
        """Δc on the configuration columns: (tighter linearisation) − (stored one).

        Computed over the entry's OWN region: a secant fit over Z_e is a tighter model
        of a convex f than a tangent taken at a distant reference point, and it moves
        each coefficient by a different amount -- so unlike a uniform rescale it can
        genuinely re-rank. This is d floats of work, independent of the task count.
        """
        if not self._secant or e is None:
            return None
        f0: LinearizedEnergy = e_energy(e)
        if f0 is None:
            return None
        f1 = linearize_secant(_tasks(e), e.regime, e.region)
        return f1.G.sum(axis=0) - f0.G.sum(axis=0)

    def refreshed_value(self, e: Entry) -> float:
        """Value of the unchanged x*_e under f_1 (line 9: "its value refreshed")."""
        f1 = linearize_secant(_tasks(e), e.regime, e.region)
        return float(f1.total(e.x_star))

    def cached_standard_A(self, e: Entry) -> Optional[np.ndarray]:
        return _stdA(e)

    def apply(self, model: FleetModel) -> FleetModel:
        """Produce M1 (only used once an entry has escalated past Stage 2)."""
        import copy
        m1 = copy.copy(model)
        m1.extra_constraints = list(model.extra_constraints)
        if self.kind == "II":
            m1.extra_constraints.append((self.cut_normal, self.cut_bound))
        elif self.kind == "III":
            m1.energy = linearize_secant(model.tasks, model.odd, model.region)
        elif self.kind == "I":
            m1.reserve = min(0.95, model.reserve + 0.05)
        return m1


# The model artifacts an entry was built from, stashed for Stage-2 reuse.
_ENTRY_ENERGY = {}
_ENTRY_TASKS = {}
_ENTRY_STDA = {}


def register_entry_model(e: Entry, model: FleetModel, A_std: np.ndarray) -> None:
    """Retain the stored linearisation, tasks and standard-form matrix for an entry.

    These are artifacts of the ORIGINAL solve, not recomputation: ``A`` is unchanged by
    a form-III refinement (only ``c`` moves), so reusing it is exactly the "reuse the
    proof" discipline rather than re-deriving anything.
    """
    _ENTRY_ENERGY[id(e)] = model.energy
    _ENTRY_TASKS[id(e)] = model.tasks
    _ENTRY_STDA[id(e)] = A_std


def e_energy(e: Entry):
    return _ENTRY_ENERGY.get(id(e))


def _tasks(e: Entry):
    return _ENTRY_TASKS.get(id(e), [])


def _stdA(e: Entry):
    return _ENTRY_STDA.get(id(e))


# --- constructors -----------------------------------------------------------
def refine_type_II(cut_normal: np.ndarray, cut_bound: float, label: str = "") -> Refinement:
    """Form II: F_1 = F_0 ∩ {x : aᵀx <= b} -- e.g. a new regulatory speed/altitude cap."""
    # Footprint must be superset-safe: a cut on the configuration variables can reach
    # any entry whose certificate reads them, and every certificate does. Stage 2's
    # support-function query is what decides whether it reaches THIS region.
    return Refinement(kind="II", label=label or f"cut<={cut_bound:.2f}",
                      footprint={"config_box", "dim:speed", "dim:altitude", "dim:camera_res"},
                      cut_normal=np.asarray(cut_normal, float), cut_bound=float(cut_bound),
                      description="tightened constraint")


def refine_type_III(label: str = "secant") -> Refinement:
    """Form III: replace the tangent linearisation with a tighter secant over Z_e."""
    return Refinement(kind="III", label=label,
                      footprint={"obj_coeffs", "linearization", "config_box"},
                      _secant=True, description="tighter objective linearization")


def refine_type_I(new_cost: float, col: np.ndarray, label: str = "new-factor") -> Refinement:
    """Form I: a new decision factor enters -- one new column priced against stored duals."""
    return Refinement(kind="I", label=label,
                      footprint={"dim:new", "obj_coeffs", "config_box"},
                      new_cost=float(new_cost), _new_col=np.asarray(col, float),
                      description="new decision factor")
