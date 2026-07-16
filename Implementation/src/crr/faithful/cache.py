"""The cache: per-entry zonotope regions of DS, keyed by ODD regime.

Section II: the fleet "accumulates a cache mapping regions of DS -- stored as zonotopes
and keyed by discretized weather regimes ('Calm', 'Light', 'Strong') -- to
proven-optimal configurations".

So an entry is (regime, region) -> proven-optimal configuration. The previous
implementation kept the regime keying but gave **every entry the same region** (the
whole design box), which is why the region-based machinery could not discriminate:
``h_{Z_e}(a)`` is identical for every entry when every ``Z_e`` is the same set, and
``ε(Z_e)`` is identical for every entry when every ``diam(Z_e)`` is the same number
(the observed 11.649074).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from src.zonotope_ops import Zonotope
from .certificate import Entry, build_entry
from .model import DS_BOX, FleetModel, ODDRegime, Task, linearize
from .refinement import register_entry_model


class ReverseIndex:
    """D: model feature -> entries whose certificate reads it (Section VII.A)."""

    def __init__(self) -> None:
        self._d: Dict[str, Set[Entry]] = {}
        self._entries: List[Entry] = []

    def insert(self, e: Entry) -> None:
        self._entries.append(e)
        for f in e.dep:
            self._d.setdefault(f, set()).add(e)

    def query(self, footprint: Iterable[str]) -> Set[Entry]:
        out: Set[Entry] = set()
        for f in footprint:
            out |= self._d.get(f, set())
        return out

    def adjacent(self, entries: Iterable[Entry]) -> Set[Entry]:
        """Zonotope-adjacent entries -- for the form-I space enlargement (line 4)."""
        entries = set(entries)
        boxes = [e.region.to_box_bounds() for e in entries]
        out: Set[Entry] = set()
        for e in self._entries:
            if e in entries:
                continue
            eb = e.region.to_box_bounds()
            for b in boxes:
                if np.all(eb[:, 0] <= b[:, 1] + 1e-9) and np.all(b[:, 0] <= eb[:, 1] + 1e-9):
                    out.add(e)
                    break
        return out

    def all(self) -> List[Entry]:
        return list(self._entries)


@dataclass
class Cache:
    entries: List[Entry]
    models: Dict[Entry, FleetModel]
    index: ReverseIndex


def partition_ds(n_regions: int, seed: int = 0) -> List[Zonotope]:
    """Carve DS into ``n_regions`` disjoint boxes along the speed axis.

    Speed is the axis the energy is nonlinear in, so slicing it is what makes the
    per-region linearisation residual ε(Z_e) differ across entries -- a narrow
    high-speed slice has a much smaller residual than a wide one. With a single shared
    region (the old code) every ε(Z_e) is necessarily identical.
    """
    lo_v, hi_v = DS_BOX[0]
    edges = np.linspace(lo_v, hi_v, n_regions + 1)
    out = []
    for i in range(n_regions):
        bounds = [(float(edges[i]), float(edges[i + 1])), DS_BOX[1], DS_BOX[2]]
        out.append(Zonotope.from_box(bounds))
    return out


def build_cache(tasks: Sequence[Task], regimes: Sequence[ODDRegime], num_agents: int,
                n_regions: int = 4, capacity_wh: float = 300.0, reserve: float = 0.20,
                seed: int = 0, backend: str = "engine") -> Cache:
    """Solve M0 per (regime, region) and retain each entry's optimality proof."""
    from src.crr.simplex import _standardize

    entries: List[Entry] = []
    models: Dict[Entry, FleetModel] = {}
    index = ReverseIndex()
    regions = partition_ds(n_regions, seed)

    for regime in regimes:
        for ri, region in enumerate(regions):
            box = region.to_box_bounds()
            x_ref = 0.5 * (box[:, 0] + box[:, 1])          # linearise about the region
            energy = linearize(tasks, regime, x_ref)
            model = FleetModel(tasks=list(tasks), odd=regime, num_agents=num_agents,
                               energy=energy, region=region,
                               capacity_wh=capacity_wh, reserve=reserve)
            e = build_entry(model, name=f"{regime.name}:R{ri}", backend=backend)
            if e is None:
                continue
            build = model.build()
            cs, A, b, lo, hi, n = _standardize(build["c"], build["A_ub"], build["b_ub"],
                                               build["A_eq"], build["b_eq"], build["bounds"])
            register_entry_model(e, model, A)
            e.dep.add(f"region:{ri}")
            entries.append(e)
            models[e] = model
            index.insert(e)

    return Cache(entries=entries, models=models, index=index)
