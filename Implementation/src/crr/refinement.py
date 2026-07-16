"""Model refinement operators (Types I, II, III) and their footprints.

A :class:`Refinement` transforms a model M0 into a refined M1 and reports the
footprint Delta (the set of feature tags it changes). Refinements may be global
or *scoped* to a subset of regimes; a scoped refinement only changes the model
for entries in that scope, so entries outside it are soundly reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Set

from .model import OptModel


@dataclass
class Refinement:
    kind: str                       # "I" | "II" | "III"
    footprint: Set[str]
    _apply: Callable                # OptModel -> OptModel
    scope_regimes: Optional[List[str]] = None
    description: str = ""

    def applies_to(self, entry) -> bool:
        return self.scope_regimes is None or entry.regime_name in self.scope_regimes

    def apply(self, model: OptModel) -> OptModel:
        return self._apply(model)


def _clone(model: OptModel, **overrides) -> OptModel:
    kw = dict(scenario=model.scenario, energy_fn=model.energy_fn, capacity=model.capacity,
              reserve=model.reserve, safety=model.safety, wind=model.wind)
    kw.update(overrides)
    return OptModel(**kw)


def refine_type_II(reserve: Optional[float] = None, capacity: Optional[float] = None,
                   scope_regimes: Optional[List[str]] = None) -> Refinement:
    """Type II: tighten the battery-reserve constraint (feasible set shrinks)."""
    def apply(m: OptModel) -> OptModel:
        ov = {}
        if reserve is not None:
            ov["reserve"] = reserve
        if capacity is not None:
            ov["capacity"] = capacity
        return _clone(m, **ov)

    footprint = {"battery_row"} if scope_regimes is None else {f"regime:{r}" for r in scope_regimes}
    return Refinement("II", footprint, apply, scope_regimes, "tighten battery reserve")


def refine_type_III(delta: float = 0.1, scope_regimes: Optional[List[str]] = None) -> Refinement:
    """Type III: tighten the objective linearization.

    A tighter linearization uniformly raises the energy estimate (the residual
    added back). The feasible set is unchanged and the ranking of configurations
    is preserved, so the cached optimum stays optimal -- its value is corrected
    by an a-priori certificate (Stage 2) unless the higher estimate reveals the
    stored assignment now exceeds the battery budget (Stage 3 repair).
    """
    def apply(m: OptModel) -> OptModel:
        base = m.energy_fn

        def tighter(config, task):
            return base(config, task) * (1.0 + delta)

        return _clone(m, energy_fn=tighter)

    footprint = {"obj_coeffs"}
    if scope_regimes is not None:
        footprint = {f"regime:{r}" for r in scope_regimes} | {"obj_coeffs"}
    return Refinement("III", footprint, apply, scope_regimes, "tighten objective linearization")


def refine_type_I(factor: str = "humidity", strength: float = 0.15,
                  scope_regimes: Optional[List[str]] = None) -> Refinement:
    """Type I: a new factor enters the model (adds a decision-dependent energy term).

    A new environmental factor (e.g. humidity) penalises configurations
    unequally -- here proportionally to altitude (denser moist air aloft) -- so
    it can re-rank configurations and genuinely change the integer optimum,
    forcing a Stage-4 re-solve, while also enlarging the space (adjacent
    regions are pulled in at Stage 1).
    """
    def apply(m: OptModel) -> OptModel:
        base = m.energy_fn

        def with_factor(config, task):
            alt = float(config.as_dict().get("altitude", 60.0))
            return base(config, task) * (1.0 + strength * (0.25 + alt / 120.0))

        return _clone(m, energy_fn=with_factor)

    footprint = {f"dim:{factor}", "obj_coeffs"}
    if scope_regimes is not None:
        footprint = {f"regime:{r}" for r in scope_regimes} | {f"dim:{factor}"}
    return Refinement("I", footprint, apply, scope_regimes, f"add {factor} factor")
