"""Cache-entry enrichment: cert(e), dep(e), cr(e).

Each entry stores its proven-optimal plan plus three artifacts derived from the
solve: a certificate (assignment, per-agent loads, optimality margin), a
dependency set (the model features the certificate reads), and validity ranges
(battery slack, objective margin, and the Type-III residual radius epsilon(Z_e)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from src.zonotope_ops import Zonotope
from .model import OptModel


def _dim_names(model: OptModel) -> List[str]:
    return [p.name for p in model.scenario.domain_spec.parameters]


def entry_dependencies(model: OptModel, regime_name: str) -> Set[str]:
    """Feature tags the entry's certificate reads."""
    dep = {"obj_coeffs", "battery_row", f"regime:{regime_name}"}
    dep.update(f"dim:{name}" for name in _dim_names(model))
    return dep


def _residual_radius(model: OptModel, region: Zonotope) -> float:
    """A-priori Type-III residual epsilon(Z_e) = 1/2 * sup|f''| * diam(Z_e)^2.

    Reporting artifact (the actual certificate checks are exact); ``sup|f''|`` is
    proxied by the energy model's speed curvature scale.
    """
    bounds = region.to_box_bounds()
    diam = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
    curvature = 0.6 / (18.0 ** 2)   # d2/dv2 scale of the quadratic drag term
    return 0.5 * curvature * diam ** 2


@dataclass(eq=False)
class Entry:
    """A proven-optimal cache entry enriched with cert / dep / cr."""

    name: str
    regime_name: str
    region: Zonotope
    config_idx: int
    value: float
    assignment: Dict[int, List[int]]
    per_agent_loads: List[float]
    cert: Dict
    dep: Set[str]
    cr: Dict
    wind: float = 0.0


def build_entry(model: OptModel, name: str, regime_name: str,
                region: Optional[Zonotope] = None) -> Entry:
    """Solve ``model`` under M0 and enrich the result into a cache entry."""
    r = model.solve()
    region = region if region is not None else model.scenario.region
    cert = {
        "assignment": r.assignment,
        "per_agent_loads": r.per_agent_loads,
        "config_cost": r.value,
        "optimality_margin": r.optimality_margin,
        "config_idx": r.config_idx,
    }
    # Warm basis of the fixed-config bin-packing LP relaxation, retained for the
    # Stage-3 warm dual-simplex repair after a battery (Type-II) tightening.
    if r.config_idx is not None:
        lp = model.binpack_lp(r.config_idx)
        cert["binpack_basis"] = (lp.basic, lp.at_upper)
    dep = entry_dependencies(model, regime_name)
    max_load = max(r.per_agent_loads) if r.per_agent_loads else 0.0
    cr = {
        "battery_slack": model.usable_budget() - max_load,
        "obj_margin": r.optimality_margin,
        "residual_radius": _residual_radius(model, region),
    }
    return Entry(name=name, regime_name=regime_name, region=region,
                 config_idx=r.config_idx, value=r.value, assignment=r.assignment,
                 per_agent_loads=r.per_agent_loads, cert=cert, dep=dep, cr=cr,
                 wind=model.wind)
