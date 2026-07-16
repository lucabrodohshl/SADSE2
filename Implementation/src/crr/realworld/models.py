"""The two model tracks, and a swappable solver backend.

Track A -- "the degenerate case we inherited"
---------------------------------------------
The paper's model: energy is agent-independent, so total cost for configuration
``k`` is ``sum_j E[k,j]`` and the optimum is ``min_k`` over configurations that
admit a battery-feasible packing. Two consequences follow that are easy to miss:

* The energy matrix is **rank 1** (``E[k,j] = c_k(odd) * L_j``): the ODD and the
  configuration scale every task identically, so no ODD or refinement can make the
  objective couple to *which agent does what*.
* "Solving" is therefore a scan over ``K`` numbers plus a feasibility check -- not a
  MILP. Any "avoid the expensive optimizer" claim measured here is measuring the
  avoidance of a scan.

Track A is retained as a **correctness control** (does CRR still conclude correctly
once the physics is realistic?), and is deliberately barred from carrying efficiency
numbers. ``assert_track_a_is_degenerate`` documents the property rather than hiding it.

Track B -- the assignment-coupled fleet
---------------------------------------
Agents are heterogeneous: each has a battery state-of-health and a home base, so
energy ``E[i,k,j]`` depends on *which* agent flies the task (transit leg + SoH).
The objective then genuinely couples to the assignment and the problem is a real
MILP. This is the only track where an efficiency claim means anything.

Formulation note (this matters by ~90x): the config-selection link uses **aggregated**
linking ``sum_{i,j} x[i,j,k] <= M*J*y_k`` (``K`` rows), not the disaggregated
``x[i,j,k] <= y_k`` (``M*J*K`` rows) used in ``src/milp_solver.py``. The disaggregated
form gives a tighter relaxation and fewer nodes, but the engine's cost is dominated by
row count, and the row blow-up loses far more than the tightening wins. Single-trip
assignment only (no task chaining): chaining would make this an energy-constrained VRP.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.domain import Configuration, DomainSpec, Parameter
from .physics import ODDPoint, task_energy, usable_budget

# Design-space box. The speed upper bound is above the M0/M1 energy-optimal cruise so
# the optimum is interior rather than pinned at a rail (checked by a validity gate).
PARAM_BOUNDS: List[Tuple[str, Tuple[float, float]]] = [
    ("speed", (3.0, 22.0)),
    ("altitude", (10.0, 120.0)),
    ("camera_res", (4.0, 20.0)),
]


@dataclass
class Agent:
    """A fleet member: battery health and a home base to fly out from."""

    soh: float                  # state of health in [0,1]
    base_xy: Tuple[float, float]


@dataclass
class FleetScenario:
    """A fixed fleet task-assignment instance."""

    domain_spec: DomainSpec
    configs: List[Configuration]
    tasks: List[Dict[str, float]]
    agents: List[Agent]
    capacity_wh: float = 300.0
    reserve: float = 0.20
    safety: float = 1.05


def make_fleet(num_agents: int, num_tasks: int, num_configs: int, seed: int = 0,
               capacity_wh: float = 300.0, reserve: float = 0.20) -> FleetScenario:
    rng = np.random.RandomState(seed * 104729 + 3)
    spec = DomainSpec([Parameter(n, b) for n, b in PARAM_BOUNDS])
    lo = np.array([b[0] for _, b in PARAM_BOUNDS])
    hi = np.array([b[1] for _, b in PARAM_BOUNDS])
    configs = [Configuration(spec, lo + rng.rand(len(PARAM_BOUNDS)) * (hi - lo))
               for _ in range(num_configs)]
    tasks = [{"length": float(rng.uniform(300.0, 1500.0)),
              "heading_deg": float(rng.uniform(0.0, 360.0)),
              "xy": (float(rng.uniform(-2000.0, 2000.0)),
                     float(rng.uniform(-2000.0, 2000.0)))}
             for _ in range(num_tasks)]
    agents = [Agent(soh=float(rng.uniform(0.75, 1.0)),
                    base_xy=(float(rng.uniform(-1500.0, 1500.0)),
                             float(rng.uniform(-1500.0, 1500.0))))
              for _ in range(num_agents)]
    return FleetScenario(domain_spec=spec, configs=configs, tasks=tasks, agents=agents,
                         capacity_wh=capacity_wh, reserve=reserve)


@dataclass
class SolveOutcome:
    status: str                       # "optimal" | "infeasible" | "node_limit"
    config_idx: Optional[int]
    assignment: Dict[int, List[int]]
    value: float
    energy_evals: int = 0             # the REAL objective work done
    solver_calls: int = 0
    wall_s: float = 0.0


# ---------------------------------------------------------------------------
# Track A -- config-determined (agent-independent) cost
# ---------------------------------------------------------------------------
class TrackAModel:
    """The paper's model under realistic physics. Cost is agent-independent."""

    def __init__(self, scn: FleetScenario, odd: ODDPoint, fidelity: str = "M0",
                 reserve: Optional[float] = None):
        self.scn = scn
        self.odd = odd
        self.fidelity = fidelity
        self.reserve = scn.reserve if reserve is None else reserve
        self._E: Optional[np.ndarray] = None
        self.energy_evals = 0

    def energy_matrix(self) -> np.ndarray:
        if self._E is None:
            K, J = len(self.scn.configs), len(self.scn.tasks)
            E = np.zeros((K, J))
            for k, cfg in enumerate(self.scn.configs):
                for j, t in enumerate(self.scn.tasks):
                    E[k, j] = task_energy(cfg, t, self.odd, fidelity=self.fidelity)
                    self.energy_evals += 1
            self._E = E
        return self._E

    def config_cost(self, k: int) -> float:
        return float(self.energy_matrix()[k].sum())

    def usable(self, agent: Agent) -> float:
        return usable_budget(self.scn.capacity_wh, self.reserve, self.scn.safety,
                             self.odd, soh=agent.soh)

    def plan_feasible(self, k: int, assignment: Dict[int, List[int]]) -> bool:
        E = self.energy_matrix()
        for i, a in enumerate(self.scn.agents):
            load = sum(E[k, j] for j in assignment.get(i, []))
            if load > self.usable(a) + 1e-9:
                return False
        return True


def track_a_coupling_report(model: TrackAModel) -> Dict[str, object]:
    """Quantify how nearly config-determined Track A's cost is.

    Under the original uniform-scale ODD model ``E[k,j] = c_k * L_j`` exactly, i.e.
    rank 1: the ODD scales every task identically, so the objective cannot couple to
    the assignment at all. The vector-wind physics here interacts wind *direction*
    with each task's *heading*, which makes ``E`` rank 2 -- wind becomes a
    config x task x ODD term rather than a pure config x ODD scale.

    That is a genuine structural improvement, but it does not rescue Track A: the
    second component carries a percent-level share of the spectrum, and Track A's
    solve is a scan over ``config_cost(k)`` **by definition of the model class**,
    whatever ``E``'s rank. Reported, not asserted, so the number goes in the paper
    rather than into an assumption.
    """
    E = model.energy_matrix()
    sv = np.linalg.svd(E, compute_uv=False)
    dominance = float(sv[1] / sv[0]) if len(sv) > 1 and sv[0] > 0 else 0.0
    return {"singular_values": sv.tolist(),
            "numerical_rank": int(np.linalg.matrix_rank(E)),
            "second_component_share": dominance,
            "effectively_config_determined": bool(dominance < 0.05)}


# ---------------------------------------------------------------------------
# Track B -- assignment-coupled (agent-dependent) cost
# ---------------------------------------------------------------------------
class TrackBModel:
    """Heterogeneous fleet: E[i,k,j] depends on the agent (transit leg + SoH)."""

    def __init__(self, scn: FleetScenario, odd: ODDPoint, fidelity: str = "M0",
                 reserve: Optional[float] = None):
        self.scn = scn
        self.odd = odd
        self.fidelity = fidelity
        self.reserve = scn.reserve if reserve is None else reserve
        self._E: Optional[np.ndarray] = None
        self.energy_evals = 0

    def energy_tensor(self) -> np.ndarray:
        """``E[i,k,j]`` -- energy for agent ``i``, config ``k``, task ``j``."""
        if self._E is None:
            M, K, J = len(self.scn.agents), len(self.scn.configs), len(self.scn.tasks)
            E = np.zeros((M, K, J))
            for i, a in enumerate(self.scn.agents):
                for j, t in enumerate(self.scn.tasks):
                    tx, ty = t["xy"]
                    dx, dy = tx - a.base_xy[0], ty - a.base_xy[1]
                    transit = float(math.hypot(dx, dy))
                    heading = float(math.degrees(math.atan2(dy, dx)))
                    for k, cfg in enumerate(self.scn.configs):
                        E[i, k, j] = task_energy(cfg, t, self.odd, fidelity=self.fidelity,
                                                 transit_m=transit,
                                                 transit_heading_deg=heading)
                        self.energy_evals += 1
            self._E = E
        return self._E

    def usable(self, i: int) -> float:
        return usable_budget(self.scn.capacity_wh, self.reserve, self.scn.safety,
                             self.odd, soh=self.scn.agents[i].soh)

    def plan_value(self, k: int, assignment: Dict[int, List[int]]) -> float:
        E = self.energy_tensor()
        return float(sum(E[i, k, j] for i, js in assignment.items() for j in js))

    def plan_feasible(self, k: int, assignment: Dict[int, List[int]]) -> bool:
        E = self.energy_tensor()
        for i in range(len(self.scn.agents)):
            load = sum(E[i, k, j] for j in assignment.get(i, []))
            if load > self.usable(i) + 1e-9:
                return False
        return True

    # -- MILP build -------------------------------------------------------
    def build_milp(self) -> dict:
        """x[i,j,k] + y[k]; aggregated linking; per-agent SoH/temperature budget."""
        E = self.energy_tensor()
        M, K, J = E.shape
        nx = M * J * K
        n = nx + K

        def xi(i, j, k):
            return (i * J + j) * K + k

        c = np.zeros(n)
        for i in range(M):
            for j in range(J):
                for k in range(K):
                    c[xi(i, j, k)] = E[i, k, j]

        rows_eq, b_eq = [], []
        for j in range(J):                       # each task assigned exactly once
            r = np.zeros(n)
            for i in range(M):
                for k in range(K):
                    r[xi(i, j, k)] = 1.0
            rows_eq.append(r); b_eq.append(1.0)
        r = np.zeros(n)                          # exactly one configuration selected
        for k in range(K):
            r[nx + k] = 1.0
        rows_eq.append(r); b_eq.append(1.0)

        rows_ub, b_ub = [], []
        for i in range(M):                       # per-agent battery budget
            r = np.zeros(n)
            for j in range(J):
                for k in range(K):
                    r[xi(i, j, k)] = E[i, k, j]
            rows_ub.append(r); b_ub.append(self.usable(i))
        for k in range(K):                       # aggregated linking (K rows, not M*J*K)
            r = np.zeros(n)
            for i in range(M):
                for j in range(J):
                    r[xi(i, j, k)] = 1.0
            r[nx + k] = -float(M * J)
            rows_ub.append(r); b_ub.append(0.0)

        return dict(c=c, A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                    A_eq=np.array(rows_eq), b_eq=np.array(b_eq),
                    bounds=[(0.0, 1.0)] * n, integer_mask=[1] * n,
                    shape=(M, J, K), nx=nx)

    def decode(self, x: np.ndarray) -> Tuple[Optional[int], Dict[int, List[int]]]:
        M, J, K = len(self.scn.agents), len(self.scn.tasks), len(self.scn.configs)
        nx = M * J * K
        y = x[nx:nx + K]
        k = int(np.argmax(y)) if np.max(y) > 0.5 else None
        assign: Dict[int, List[int]] = {i: [] for i in range(M)}
        for i in range(M):
            for j in range(J):
                for kk in range(K):
                    if x[(i * J + j) * K + kk] > 0.5:
                        assign[i].append(j)
        return k, assign


def assert_track_b_is_coupled(model: TrackBModel) -> Dict[str, object]:
    """Fail loudly if Track B silently regresses to the agent-independent scan."""
    E = model.energy_tensor()
    agent_dependent = not np.allclose(E[0], E[-1])
    ranks = [int(np.linalg.matrix_rank(E[i])) for i in range(E.shape[0])]
    return {"agent_dependent": bool(agent_dependent),
            "per_agent_ranks": ranks,
            "max_rank": int(max(ranks))}


# ---------------------------------------------------------------------------
# solver backends
# ---------------------------------------------------------------------------
def solve_own_engine(build: dict, max_nodes: int = 200000) -> Tuple[str, np.ndarray, float, int]:
    """The self-contained branch-and-bound (exact, auditable, no external solver)."""
    from src.crr.branch_and_bound import solve_milp
    r = solve_milp(c=build["c"], A_ub=build["A_ub"], b_ub=build["b_ub"],
                   A_eq=build["A_eq"], b_eq=build["b_eq"], bounds=build["bounds"],
                   integer_mask=build["integer_mask"], max_nodes=max_nodes)
    return r.status, r.x, r.obj, r.n_nodes


def solve_highs(build: dict, time_limit: float = 60.0) -> Tuple[str, np.ndarray, float, int]:
    """A production solver, used where the self-contained engine cannot reach.

    This is *not* part of the CRR artifact -- it exists so the efficiency claim can be
    measured at a fleet size where a re-solve is genuinely expensive.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    cons = []
    if len(build["A_ub"]):
        cons.append(LinearConstraint(build["A_ub"], -np.inf, build["b_ub"]))
    if len(build["A_eq"]):
        cons.append(LinearConstraint(build["A_eq"], build["b_eq"], build["b_eq"]))
    lo = np.array([b[0] for b in build["bounds"]], float)
    hi = np.array([b[1] for b in build["bounds"]], float)
    r = milp(c=build["c"], constraints=cons,
             integrality=np.asarray(build["integer_mask"], float),
             bounds=Bounds(lo, hi), options={"time_limit": time_limit})
    status = "optimal" if r.status == 0 else ("node_limit" if r.status == 1 else "infeasible")
    x = r.x if r.x is not None else np.zeros(len(build["c"]))
    obj = float(r.fun) if r.status == 0 else float("inf")
    return status, x, obj, 0


BACKENDS: Dict[str, Callable] = {"engine": solve_own_engine, "highs": solve_highs}


def solve_track_b(model: TrackBModel, backend: str = "engine", **kw) -> SolveOutcome:
    t0 = time.perf_counter()
    build = model.build_milp()
    status, x, obj, nodes = BACKENDS[backend](build, **kw)
    wall = time.perf_counter() - t0
    if status != "optimal":
        return SolveOutcome(status, None, {}, float("inf"),
                            energy_evals=model.energy_evals, solver_calls=1, wall_s=wall)
    k, assign = model.decode(x)
    return SolveOutcome("optimal", k, assign, float(obj),
                        energy_evals=model.energy_evals, solver_calls=1, wall_s=wall)


def solve_track_a(model: TrackAModel, backend: str = "engine") -> SolveOutcome:
    """Track A's 'solve': scan configurations by cost, take the cheapest packable one.

    Spelled out deliberately -- this *is* the whole optimizer for the paper's model.
    """
    from src.crr.branch_and_bound import solve_milp
    t0 = time.perf_counter()
    E = model.energy_matrix()
    K = E.shape[0]
    order = sorted(range(K), key=model.config_cost)
    calls = 0
    for k in order:
        build = _binpack_build(model, k)
        r = solve_milp(c=build["c"], A_ub=build["A_ub"], b_ub=build["b_ub"],
                       A_eq=build["A_eq"], b_eq=build["b_eq"],
                       bounds=build["bounds"], integer_mask=build["integer_mask"])
        calls += 1
        if r.status == "optimal":
            M, J = len(model.scn.agents), len(model.scn.tasks)
            assign: Dict[int, List[int]] = {i: [] for i in range(M)}
            for j in range(J):
                for i in range(M):
                    if r.x[i * J + j] > 0.5:
                        assign[i].append(j); break
            return SolveOutcome("optimal", k, assign, model.config_cost(k),
                                energy_evals=model.energy_evals, solver_calls=calls,
                                wall_s=time.perf_counter() - t0)
    return SolveOutcome("infeasible", None, {}, float("inf"),
                        energy_evals=model.energy_evals, solver_calls=calls,
                        wall_s=time.perf_counter() - t0)


def _binpack_build(model: TrackAModel, k: int) -> dict:
    E = model.energy_matrix()
    M, J = len(model.scn.agents), len(model.scn.tasks)
    n = M * J
    A_eq = np.zeros((J, n))
    for j in range(J):
        for i in range(M):
            A_eq[j, i * J + j] = 1.0
    A_ub = np.zeros((M, n))
    for i in range(M):
        for j in range(J):
            A_ub[i, i * J + j] = E[k, j]
    b_ub = np.array([model.usable(a) for a in model.scn.agents])
    c = np.array([i for i in range(M) for _ in range(J)], float) * 1e-3
    return dict(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(J),
                bounds=[(0.0, 1.0)] * n, integer_mask=[1] * n)
