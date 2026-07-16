"""The CRR cache: enriched entries + reverse index + the M0 model per entry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .certificate import Entry, build_entry
from .model import OptModel
from .reverse_index import ReverseIndex
from .scenario import Scenario


def default_regimes() -> List[Tuple[str, float]]:
    """(name, wind factor) operating regimes -- calm ... strong crosswinds."""
    return [("calm", 0.0), ("breeze", 0.35), ("gust", 0.8), ("strong", 1.3)]


@dataclass
class Cache:
    entries: List[Entry]
    models: Dict[Entry, OptModel]      # M0 model each entry was built under
    index: ReverseIndex


def build_cache(scenario: Scenario, regimes, capacity: float = 300.0,
                reserve: float = 0.20) -> Cache:
    """Solve the M0 model over each regime and store enriched, indexed entries."""
    entries: List[Entry] = []
    models: Dict[Entry, OptModel] = {}
    index = ReverseIndex()
    for name, wind in regimes:
        m0 = OptModel.from_scenario(scenario, wind=wind, capacity=capacity, reserve=reserve)
        entry = build_entry(m0, name=name, regime_name=name)
        entries.append(entry)
        models[entry] = m0
        index.insert(entry)
    return Cache(entries=entries, models=models, index=index)
