"""OptModel -- the fleet task-assignment MILP under a given (possibly refined) model.

The objective is *config-determined*: total energy for a chosen configuration k is
``sum_j E[k, j]`` (energy is agent-independent), so the optimum value is
``min_k sum_j E[k, j]`` over configs that admit a battery-feasible task assignment.
Solving therefore decomposes into: try configurations in increasing cost order and
return the cheapest one whose per-agent battery bin-packing is feasible.

All optimization is done by the self-contained engine in :mod:`src.crr.simplex`
and :mod:`src.crr.branch_and_bound` -- no external MILP solver.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.milp_solver import drone_energy_model
from .branch_and_bound import solve_milp
from .scenario import Scenario
from .simplex import INF, LPResult, _standardize, dual_simplex, solve_standard


@dataclass
class SolveResult:
    config_idx: Optional[int]
    assignment: Dict[int, List[int]]
    value: float
    per_agent_loads: List[float]
    optimality_margin: float = 0.0
    feasible: bool = True
    n_milp: int = 0
    n_lp: int = 0
    wall_s: float = 0.0


def _wind_scaled(base_fn: Callable, wind: float) -> Callable:
    def fn(config, task):
        return base_fn(config, task) * (1.0 + wind)
    return fn


@dataclass
class OptModel:
    scenario: Scenario
    energy_fn: Callable
    capacity: float = 300.0
    reserve: float = 0.20
    safety: float = 1.05
    wind: float = 0.0
    _E: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_scenario(cls, scn: Scenario, wind: float = 0.0, capacity: float = 300.0,
                      reserve: float = 0.20, safety: float = 1.05,
                      energy_fn: Optional[Callable] = None) -> "OptModel":
        base = energy_fn or drone_energy_model
        fn = base if wind == 0.0 else _wind_scaled(base, wind)
        return cls(scenario=scn, energy_fn=fn, capacity=capacity,
                   reserve=reserve, safety=safety, wind=wind)

    # ---- model quantities -------------------------------------------------
    def usable_budget(self) -> float:
        return max(0.0, (1.0 - self.reserve) * self.capacity / self.safety)

    def energy_matrix(self) -> np.ndarray:
        if self._E is None:
            scn = self.scenario
            E = np.zeros((len(scn.configs), len(scn.tasks)), dtype=float)
            for k, cfg in enumerate(scn.configs):
                for j, task in enumerate(scn.tasks):
                    E[k, j] = float(self.energy_fn(cfg, task))
            self._E = E
        return self._E

    def config_cost(self, k: int) -> float:
        return float(np.sum(self.energy_matrix()[k]))

    def plan_value(self, config_idx: int) -> float:
        return self.config_cost(config_idx)

    def plan_per_agent_loads(self, config_idx: int, assignment: Dict[int, List[int]]) -> List[float]:
        E = self.energy_matrix()
        return [float(sum(E[config_idx, j] for j in assignment.get(i, [])))
                for i in range(self.scenario.num_agents)]

    def plan_feasible(self, config_idx: int, assignment: Dict[int, List[int]]) -> bool:
        u = self.usable_budget()
        return all(load <= u + 1e-9 for load in self.plan_per_agent_loads(config_idx, assignment))

    def config_margin(self, config_idx: Optional[int], allowed_configs: Optional[List[int]] = None) -> float:
        if config_idx is None:
            return 0.0
        ks = list(allowed_configs) if allowed_configs is not None else list(range(len(self.scenario.configs)))
        costs = sorted(self.config_cost(k) for k in ks)
        if len(costs) < 2:
            return float("inf")
        return float(costs[1] - costs[0])

    # ---- fixed-config battery bin-packing (assign J tasks to M agents) ----
    def _binpack_linprog(self, config_idx: int, usable: Optional[float] = None) -> dict:
        E = self.energy_matrix()
        M, J = self.scenario.num_agents, len(self.scenario.tasks)
        u = self.usable_budget() if usable is None else usable
        n = M * J

        def idx(i, j):
            return i * J + j

        A_eq = np.zeros((J, n))
        for j in range(J):
            for i in range(M):
                A_eq[j, idx(i, j)] = 1.0            # each task assigned once
        b_eq = np.ones(J)

        A_ub = np.zeros((M, n))
        for i in range(M):
            for j in range(J):
                A_ub[i, idx(i, j)] = E[config_idx, j]   # per-agent battery load
        b_ub = np.full(M, u)

        c = np.array([i for i in range(M) for _ in range(J)], float) * 1e-3  # tie-break
        bounds = [(0.0, 1.0)] * n
        return dict(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    def _binpack_standard(self, config_idx: int, usable: Optional[float] = None):
        a = self._binpack_linprog(config_idx, usable)
        return _standardize(a["c"], a["A_ub"], a["b_ub"], a["A_eq"], a["b_eq"], a["bounds"])

    def binpack_lp(self, config_idx: int, usable: Optional[float] = None) -> LPResult:
        """LP relaxation of the fixed-config bin-packing (cold solve, with basis)."""
        cs, A, b, lo, hi, n = self._binpack_standard(config_idx, usable)
        return solve_standard(cs, A, b, lo, hi)

    def binpack_warm(self, config_idx: int, basic, at_upper, usable: Optional[float] = None) -> LPResult:
        """Warm dual-simplex re-solve of the bin-packing LP from a stored basis.

        Valid when only the battery RHS changed (Type II); the stored basis stays
        dual-feasible and dual simplex restores primal feasibility in few pivots.
        """
        cs, A, b, lo, hi, n = self._binpack_standard(config_idx, usable)
        return dual_simplex(cs, A, b, lo, hi, basic, at_upper)

    def binpack_feasible(self, config_idx: int):
        """Is there an integral battery-feasible assignment for this config? (own B&B)"""
        a = self._binpack_linprog(config_idx)
        res = solve_milp(c=a["c"], A_ub=a["A_ub"], b_ub=a["b_ub"],
                         A_eq=a["A_eq"], b_eq=a["b_eq"], bounds=a["bounds"],
                         integer_mask=[1] * len(a["c"]))
        if res.status != "optimal":
            return False, {}
        return True, self.assignment_from_x(res.x)

    def assignment_from_x(self, x_struct: np.ndarray) -> Dict[int, List[int]]:
        M, J = self.scenario.num_agents, len(self.scenario.tasks)
        assign: Dict[int, List[int]] = {i: [] for i in range(M)}
        for j in range(J):
            for i in range(M):
                if x_struct[i * J + j] > 0.5:
                    assign[i].append(j)
                    break
        return assign

    def is_integral(self, x_struct: np.ndarray) -> bool:
        M, J = self.scenario.num_agents, len(self.scenario.tasks)
        xs = np.asarray(x_struct)[: M * J]
        return bool(np.all(np.abs(xs - np.round(xs)) < 1e-6))

    def round_assignment(self, x_struct: np.ndarray) -> Dict[int, List[int]]:
        """Round a (possibly fractional) bin-packing LP solution to an assignment:
        each task goes to the agent with the largest fractional share. Cheap (no
        branching) -- used by the Stage-3 warm repair; feasibility is checked by
        the caller."""
        M, J = self.scenario.num_agents, len(self.scenario.tasks)
        xs = np.asarray(x_struct)[: M * J].reshape(M, J)
        assign: Dict[int, List[int]] = {i: [] for i in range(M)}
        for j in range(J):
            assign[int(np.argmax(xs[:, j]))].append(j)
        return assign

    # ---- solving (config-cost decomposition, own B&B) ---------------------
    def solve(self, allowed_configs: Optional[List[int]] = None) -> SolveResult:
        t0 = time.perf_counter()
        ks = list(allowed_configs) if allowed_configs is not None else list(range(len(self.scenario.configs)))
        order = sorted(ks, key=self.config_cost)
        chosen, assignment = None, {}
        for k in order:
            feasible, asg = self.binpack_feasible(k)
            if feasible:
                chosen, assignment = k, asg
                break
        wall = time.perf_counter() - t0
        if chosen is None:
            return SolveResult(None, {}, INF, [], feasible=False, n_milp=1, wall_s=wall)
        loads = self.plan_per_agent_loads(chosen, assignment)
        return SolveResult(chosen, assignment, self.config_cost(chosen), loads,
                           optimality_margin=self.config_margin(chosen, allowed_configs),
                           feasible=True, n_milp=1, wall_s=wall)
