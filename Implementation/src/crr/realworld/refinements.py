"""Model refinements drawn from engineering practice, and their footprints.

Contrast with the original operators
------------------------------------
``src/crr/refinement.py`` implements Type III as ``base(cfg,task) * (1 + delta)`` --
a *uniform* scalar multiply. Since ``argmin_k sum_j E[k,j]`` is scale-invariant, that
operator provably cannot change the optimum for **any** delta (verified: identical
ranking from delta=0.05 to delta=100). Its own docstring concedes "the ranking of
configurations is preserved". So the reported "Type III: 100% re-solve reduction
across the whole severity range" restates the operator's definition rather than
measuring CRR. It is also numerically identical to the wind factor the cache is keyed
on (``_wind_scaled`` is the same uniform multiply), so the refinement moves along the
very axis the cache indexes.

The operators here are chosen so that each *can* move the optimum, and whether it
does is then an empirical question:

* **Type III (objective fidelity).** M0 -> M1 on the energy model: the existing
  parasitic-drag model is replaced by the rotary-wing model (Zeng et al. 2019), which
  *adds* the induced-power branch. Nothing is deleted to manufacture the step, and the
  two differ non-uniformly across configurations (measured cost-ratio spread ~0.40 vs
  4e-16 for the uniform operator), so it re-ranks in ~77% of draws.
* **Type II (constraint tightening).** A regulatory reserve increase. Note this is a
  pure RHS change, so it can only ever flip *feasibility*, never the ranking -- in
  Track A its "success" is therefore definitional, not evidence, and it is reported
  as such.
* **Type I (new decision factor).** Payload mass, which enters both the objective
  (extra lift draw and drag) *and* the feasible set (agents have payload capacity,
  tasks have payload requirements), so it is a genuine new dimension rather than a
  reweight of existing columns.

Scoping
-------
Refinements carry a **scope**: a predicate over the ODD. A regulator may raise the
reserve only above a wind threshold; a drag correction may only be validated in an
airspeed band. Scope is sampled as part of the refinement draw, because if every
refinement is global then the footprint is the whole cache and a "re-solve only the
footprint" baseline is bit-identical to "re-solve everything" -- which would make the
fair baseline vacuous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from .physics import ODDPoint


@dataclass
class Refinement:
    """A model refinement: what changes, where it applies, and its footprint."""

    kind: str                              # "I" | "II" | "III"
    label: str
    footprint: Set[str]                    # feature tags this refinement touches
    scope: Optional[Callable[[ODDPoint], bool]] = None
    fidelity: Optional[str] = None         # Type III: target fidelity
    reserve: Optional[float] = None        # Type II: new reserve
    payload_kg: float = 0.0                # Type I: new factor magnitude
    severity: float = 0.0                  # normalized severity, for reporting curves

    def applies_at(self, odd: ODDPoint) -> bool:
        return True if self.scope is None else bool(self.scope(odd))

    @property
    def is_global(self) -> bool:
        return self.scope is None


# --- scope predicates -------------------------------------------------------
def scope_global() -> Optional[Callable]:
    return None


def scope_wind_above(threshold: float) -> Callable[[ODDPoint], bool]:
    def pred(odd: ODDPoint) -> bool:
        return odd.wind_speed >= threshold
    return pred


def scope_cold_below(threshold: float) -> Callable[[ODDPoint], bool]:
    def pred(odd: ODDPoint) -> bool:
        return odd.temperature <= threshold
    return pred


# --- operators --------------------------------------------------------------
def refine_type_III(scope: Optional[Callable] = None, label: str = "M0->M1") -> Refinement:
    """Objective fidelity: parasitic-drag model -> rotary-wing model (adds induced power)."""
    fp = {"obj_coeffs", "energy_model"}
    return Refinement(kind="III", label=label, footprint=fp, scope=scope,
                      fidelity="M1", severity=1.0)


def refine_type_II(new_reserve: float, scope: Optional[Callable] = None) -> Refinement:
    """Constraint tightening: a regulatory battery-reserve increase (RHS only)."""
    return Refinement(kind="II", label=f"reserve={new_reserve:.2f}",
                      footprint={"battery_row"}, scope=scope,
                      reserve=new_reserve, severity=new_reserve)


def refine_type_I(payload_kg: float, scope: Optional[Callable] = None) -> Refinement:
    """New decision factor: payload mass (enters both objective and feasible set)."""
    return Refinement(kind="I", label=f"payload={payload_kg:.2f}kg",
                      footprint={"obj_coeffs", "battery_row", "dim:payload"}, scope=scope,
                      payload_kg=payload_kg, severity=payload_kg)


# --- severity as a sampled distribution, reported as a curve -----------------
#
# The original evaluation hand-picks a grid and calls severity_grid(k)[2] the
# "representative moderate" refinement. Sampling from an unnamed distribution would be
# no better -- it would just launder the same author choice, since every metric is
# monotone in severity. So: the support is declared here, results are reported as a
# CURVE over severity, and the sampled range is required to STRADDLE the point where
# the optimum starts to move, so the crossover is measured rather than chosen.

RESERVE_SUPPORT = (0.20, 0.60)     # regulatory reserve: today's 20% up to a severe 60%
PAYLOAD_SUPPORT = (0.0, 2.0)       # kg


def sample_refinements(rng: np.random.RandomState, n_per_type: int = 5,
                       scoped_fraction: float = 0.5) -> List[Refinement]:
    """Draw refinements across types, severities and scopes.

    ``scoped_fraction`` of draws are ODD-scoped rather than global, so the footprint
    is genuinely a subset for some draws and ``footprint_resolve`` is a real baseline.
    """
    out: List[Refinement] = []

    def maybe_scope(kind: str):
        if rng.rand() >= scoped_fraction:
            return None
        if kind == "II":
            return scope_wind_above(float(rng.uniform(6.0, 12.0)))
        if kind == "III":
            return scope_wind_above(float(rng.uniform(4.0, 10.0)))
        return scope_cold_below(float(rng.uniform(5.0, 15.0)))

    for _ in range(n_per_type):
        r = float(rng.uniform(*RESERVE_SUPPORT))
        out.append(refine_type_II(r, scope=maybe_scope("II")))
    for _ in range(n_per_type):
        out.append(refine_type_III(scope=maybe_scope("III")))
    for _ in range(n_per_type):
        p = float(rng.uniform(*PAYLOAD_SUPPORT))
        out.append(refine_type_I(p, scope=maybe_scope("I")))
    return out


def severity_curve_points(kind: str, n: int = 6) -> List[Refinement]:
    """A declared severity ladder for reporting metric-vs-severity curves."""
    if kind == "II":
        return [refine_type_II(r) for r in np.linspace(*RESERVE_SUPPORT, n)]
    if kind == "I":
        return [refine_type_I(p) for p in np.linspace(*PAYLOAD_SUPPORT, n)]
    if kind == "III":
        # Type III has no free magnitude: it is a fidelity step, not a dial. Its
        # "severity" is whatever the physics says it is -- which is the point.
        return [refine_type_III()]
    raise ValueError(kind)
