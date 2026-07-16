"""Concrete scenario builder for the CRR evaluation.

A :class:`Scenario` fixes the candidate configurations, the tasks, and the fleet
size for the fleet task-assignment MILP. Operating regimes (ODDs) are modelled
by a *wind* factor that scales energy (calm -> strong), and each cache entry
owns a zonotope region in design space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.domain import DomainSpec, Parameter, Configuration
from src.zonotope_ops import Zonotope

# Design-space parameters consumed by ``drone_energy_model``.
_PARAM_BOUNDS: List[Tuple[str, Tuple[float, float]]] = [
    ("speed", (3.0, 18.0)),
    ("altitude", (10.0, 120.0)),
    ("camera_res", (4.0, 20.0)),
]


@dataclass
class Scenario:
    """A fixed task-assignment instance the model is defined over."""

    domain_spec: DomainSpec
    configs: List[Configuration]
    tasks: List[Dict[str, float]]
    num_agents: int
    region: Zonotope


def make_scenario(num_agents: int, num_tasks: int, num_configs: int, seed: int = 0) -> Scenario:
    """Build a deterministic task-assignment scenario.

    ``num_configs`` candidate configurations are sampled from the design-space
    box; ``num_tasks`` tasks get random lengths. The region is the design-space
    box (individual cache entries carve out sub-regions).
    """
    rng = np.random.RandomState(seed)
    params = [Parameter(name, bounds) for name, bounds in _PARAM_BOUNDS]
    spec = DomainSpec(params)

    lo = np.array([b[0] for _, b in _PARAM_BOUNDS], dtype=float)
    hi = np.array([b[1] for _, b in _PARAM_BOUNDS], dtype=float)
    configs = [
        Configuration(spec, lo + rng.rand(len(_PARAM_BOUNDS)) * (hi - lo))
        for _ in range(num_configs)
    ]
    tasks = [{"length": float(rng.uniform(200.0, 1200.0))} for _ in range(num_tasks)]
    region = Zonotope.from_box([(float(l), float(h)) for l, h in zip(lo, hi)])
    return Scenario(domain_spec=spec, configs=configs, tasks=tasks,
                    num_agents=num_agents, region=region)
