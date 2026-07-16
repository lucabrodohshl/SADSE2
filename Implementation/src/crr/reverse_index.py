"""Dependency reverse index D: feature tag -> entries whose certificate reads it.

Querying at a refinement's footprint returns exactly the entries the refinement
can affect; the rest of the cache is never examined (Stage 1 of CRR).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set

import numpy as np

from .certificate import Entry


def _boxes_overlap(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.all(a[:, 0] <= b[:, 1] + tol) and np.all(b[:, 0] <= a[:, 1] + tol))


class ReverseIndex:
    """Maps each model feature to the set of entries whose cert(e) depends on it."""

    def __init__(self) -> None:
        self._d: Dict[str, Set[Entry]] = defaultdict(set)
        self._entries: List[Entry] = []

    def insert(self, entry: Entry) -> None:
        self._entries.append(entry)
        for feature in entry.dep:
            self._d[feature].add(entry)

    def query(self, footprint: Iterable[str]) -> Set[Entry]:
        """Return the union over the footprint of the entries depending on each feature."""
        out: Set[Entry] = set()
        for feature in footprint:
            out |= self._d.get(feature, set())
        return out

    def adjacent(self, entries: Iterable[Entry]) -> Set[Entry]:
        """Zonotope-adjacent entries (box-overlap) to those given -- for Type-I enlargement."""
        entries = set(entries)
        boxes = [e.region.to_box_bounds() for e in entries]
        out: Set[Entry] = set()
        for e in self._entries:
            if e in entries:
                continue
            eb = e.region.to_box_bounds()
            if any(_boxes_overlap(eb, b) for b in boxes):
                out.add(e)
        return out

    def all(self) -> List[Entry]:
        return list(self._entries)
