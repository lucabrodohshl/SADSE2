"""A small correct MILP solver: LP-relaxation branch-and-bound with warm-started
dual-simplex children, built on :mod:`src.crr.simplex`.

Each child node tightens one variable bound and re-optimises from the parent's
optimal (hence dual-feasible) basis via the dual simplex -- the same warm-start
mechanism CRR's Stage 3 uses. Not competitive with Gurobi/CBC, but exact and
entirely self-contained (validated against scipy.optimize.milp).

Search quality
--------------
The LP relaxations are cheap; the node count is what hurts. Three orthogonal
levers, each independently switchable so their effect can be measured:

``node_select``
    ``"dfs"``    -- LIFO stack (finds incumbents fast, proves optimality slowly).
    ``"best"``   -- always expand the globally weakest bound (minimal node count
                    in theory, but no incumbent for a long time, so no pruning).
    ``"hybrid"`` -- best-first pool + *plunging*: after branching we dive into one
                    child immediately and park its sibling in the pool. Gets DFS's
                    early incumbent AND best-first's proof. Default.

``branching``
    ``"mostfrac"``    -- most-fractional variable (the classic weak default).
    ``"pseudocost"``  -- product score of historical per-unit objective gain.
    ``"reliability"`` -- pseudocost, but strong-branch (actually solve both child
                         LPs) on candidates whose pseudocosts are not yet
                         reliable. Default.

``heuristic``
    LP-guided diving at the root for an early incumbent. Any solution it returns
    satisfies Ax=b, the bounds and integrality, so it is genuinely feasible --
    but it is only ever used as an INCUMBENT (an upper bound for pruning). It is
    never reported as a proven optimum; optimality still has to be proved by
    exhausting the node pool.

``branch_priority``
    Optional per-variable priority; higher branches first among candidates whose
    scores are within a factor of the best. Fixing the high-level configuration
    variables (e.g. fleet ``y[k]``) early decomposes the subproblems.

None of these change what is proven -- only the order of exploration and the
strength of the bounds available while exploring. The proof obligation (empty
pool, every node either pruned by a bound or integral) is unchanged.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .simplex import INF, _standardize, dual_simplex, solve_standard

_INT_TOL = 1e-6
_EPS = 1e-9


@dataclass
class MILPResult:
    status: str            # "optimal" | "infeasible" | "unbounded" | "node_limit"
    x: np.ndarray          # structural solution (length n)
    obj: float
    n_nodes: int


class _Pseudocost:
    """Per-variable average objective gain per unit of fractionality moved.

    ``down[j]`` accumulates (obj_child - obj_parent) / f  over the branches that
    pushed x_j down to floor(v) (f = v - floor(v)), ``up[j]`` likewise for ceil.
    The branching score is the product of the two predicted degradations, which
    is the standard robust choice (a variable is only attractive if BOTH sides
    hurt -- that is what actually closes the gap).
    """

    def __init__(self, n):
        self.d_sum = np.zeros(n); self.d_cnt = np.zeros(n, int)
        self.u_sum = np.zeros(n); self.u_cnt = np.zeros(n, int)

    def update(self, j, side, gain, frac):
        if frac <= 1e-12 or not np.isfinite(gain) or gain < 0:
            return
        rate = gain / frac
        if side == "down":
            self.d_sum[j] += rate; self.d_cnt[j] += 1
        else:
            self.u_sum[j] += rate; self.u_cnt[j] += 1

    def _avg(self, s, cnt, j, default):
        return s[j] / cnt[j] if cnt[j] else default

    def reliable(self, j, thresh):
        return self.d_cnt[j] >= thresh and self.u_cnt[j] >= thresh

    def score(self, j, val, default):
        f = val - math.floor(val)
        down = self._avg(self.d_sum, self.d_cnt, j, default) * f
        up = self._avg(self.u_sum, self.u_cnt, j, default) * (1.0 - f)
        return max(down, 1e-6) * max(up, 1e-6)


def _fractional(lp_x, int_idx):
    v = lp_x[int_idx]
    f = np.abs(v - np.round(v))
    sel = f > _INT_TOL
    return int_idx[sel], f[sel]


def _dive_heuristic(cs, A, b, lo0, hi0, int_idx, basic, at_upper, lp_x, n,
                    max_dives=60):
    """LP-guided diving: round the LEAST fractional variable, re-solve, repeat.

    Returns (obj, x) of a genuinely feasible integral point, or (INF, None).
    Only ever used as an incumbent -- never as a claimed optimum.
    """
    lo, hi = lo0.copy(), hi0.copy()
    x, bas, au = lp_x, basic, at_upper
    for _ in range(max_dives):
        cand, frac = _fractional(x, int_idx)
        if len(cand) == 0:
            return float(cs @ x), x[:n].copy()
        # least-fractional variable -> the roundings least likely to be rejected
        k = int(np.argmin(frac))
        j = int(cand[k])
        v = x[j]
        target = math.floor(v + 0.5)
        target = min(max(target, lo[j]), hi[j])
        lo[j] = hi[j] = target
        r = dual_simplex(cs, A, b, lo, hi, bas, au, validate=False)
        if r.status != "optimal":
            return INF, None
        x, bas, au = r.x, r.basic, r.at_upper
    return INF, None


def _strong_branch(cs, A, b, lo, hi, basic, at_upper, j, val, parent_obj,
                   incumbent_obj):
    """Solve both children of a branch on j. Returns (down_obj, up_obj).

    An infeasible / bound-pruned child comes back as INF, which is exactly the
    information the caller wants: that side is closed.
    """
    objs = []
    for side in ("down", "up"):
        nl, nh = lo.copy(), hi.copy()
        if side == "down":
            nh[j] = math.floor(val + 1e-9)
        else:
            nl[j] = math.ceil(val - 1e-9)
        if np.any(nl > nh + 1e-9):
            objs.append(INF); continue
        r = dual_simplex(cs, A, b, nl, nh, basic, at_upper, validate=False)
        objs.append(r.obj if r.status == "optimal" else INF)
    return objs[0], objs[1]


def _select_branch_var(cs, A, b, lo, hi, basic, at_upper, lp_x, lp_obj, int_idx,
                       pc, branching, incumbent_obj, priority, depth,
                       reliability, max_sb_cands):
    cand, frac = _fractional(lp_x, int_idx)
    if len(cand) == 0:
        return -1

    if branching == "mostfrac":
        scores = frac
    else:
        default = max(1e-6, abs(lp_obj) * 1e-3)
        scores = np.array([pc.score(int(j), lp_x[int(j)], default) for j in cand])

        if branching == "reliability" and depth <= 12:
            unrel = [i for i, j in enumerate(cand)
                     if not pc.reliable(int(j), reliability)]
            # strong-branch the most promising unreliable candidates only
            unrel.sort(key=lambda i: -scores[i])
            for i in unrel[:max_sb_cands]:
                j = int(cand[i]); val = lp_x[j]
                d_obj, u_obj = _strong_branch(cs, A, b, lo, hi, basic, at_upper,
                                              j, val, lp_obj, incumbent_obj)
                f = val - math.floor(val)
                pc.update(j, "down", d_obj - lp_obj if np.isfinite(d_obj) else 0.0, f)
                pc.update(j, "up", u_obj - lp_obj if np.isfinite(u_obj) else 0.0, 1.0 - f)
                dg = (d_obj - lp_obj) if np.isfinite(d_obj) else 1e6
                ug = (u_obj - lp_obj) if np.isfinite(u_obj) else 1e6
                scores[i] = max(dg, 1e-6) * max(ug, 1e-6)

    if priority is not None:
        # Among candidates within 20% of the best score, prefer higher priority.
        best = float(np.max(scores))
        near = scores >= best * 0.8 - 1e-12
        prio = np.array([priority[int(j)] for j in cand], float)
        prio_masked = np.where(near, prio, -np.inf)
        top = float(np.max(prio_masked))
        near &= prio_masked >= top - 1e-12

    else:
        near = np.ones(len(cand), bool)

    # Break ties on fractionality. When the objective is (near-)degenerate --
    # e.g. bin-packing's c = i*1e-3 -- every pseudocost collapses onto the score
    # floor and the argmax degenerates into "first index", which is markedly
    # worse than most-fractional. Falling back to most-fractional among tied
    # scores recovers the classic rule exactly in that case, and costs nothing
    # when the pseudocosts are informative (then the tie set is a singleton).
    best = float(np.max(np.where(near, scores, -np.inf)))
    tied = near & (scores >= best * (1.0 - 1e-6) - 1e-12)
    return int(cand[int(np.argmax(np.where(tied, frac, -np.inf)))])


def solve_milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
               integer_mask=None, max_nodes=200000,
               node_select="hybrid", branching="reliability", heuristic=True,
               branch_priority=None, reliability=1, max_sb_cands=8):
    cs, A, b, lo0, hi0, n = _standardize(c, A_ub, b_ub, A_eq, b_eq, bounds)
    integer_mask = (np.zeros(n, bool) if integer_mask is None
                    else np.asarray(integer_mask, bool))
    int_idx = np.where(integer_mask[:n])[0]

    priority = None
    if branch_priority is not None:
        priority = np.zeros(len(cs))
        priority[:n] = np.asarray(branch_priority, float)[:n]

    root = solve_standard(cs, A, b, lo0, hi0)
    if root.status == "infeasible":
        return MILPResult("infeasible", np.zeros(n), INF, 1)
    if root.status == "unbounded":
        return MILPResult("unbounded", root.x[:n], -INF, 1)

    incumbent_obj, incumbent_x = INF, None
    if heuristic and len(int_idx):
        h_obj, h_x = _dive_heuristic(cs, A, b, lo0, hi0, int_idx, root.basic,
                                     root.at_upper, root.x, n)
        if h_x is not None:
            incumbent_obj, incumbent_x = h_obj, h_x

    pc = _Pseudocost(len(cs))
    nodes = 0
    truncated = False
    seq = 0                      # heap tie-break: FIFO among equal bounds

    # node = (lo, hi, basic, at_upper, lp_obj, lp_x, depth)
    root_node = (lo0.copy(), hi0.copy(), root.basic, root.at_upper, root.obj,
                 root.x, 0)
    pool: list = []              # best-first heap of (lp_obj, seq, node)
    stack: list = []             # DFS / plunge path

    def push(node):
        nonlocal seq
        if node_select == "dfs":
            stack.append(node)
        else:
            heapq.heappush(pool, (node[4], seq, node))
            seq += 1

    def pop():
        if node_select == "dfs":
            return stack.pop()
        if node_select == "hybrid" and stack:
            return stack.pop()          # continue the dive
        if not pool:
            return None
        return heapq.heappop(pool)[2]

    # "dfs"/"hybrid" start the dive at the root; "best" seeds the pool instead.
    if node_select == "dfs" or node_select == "hybrid":
        stack.append(root_node)
    else:
        push(root_node)

    while True:
        if node_select == "dfs":
            if not stack:
                break
        elif not stack and not pool:
            break
        node = pop()
        if node is None:
            break
        lo, hi, basic, at_upper, lp_obj, lp_x, depth = node

        nodes += 1
        if nodes > max_nodes:
            # The search was cut off, so the incumbent is NOT proven optimal.
            # Reporting "optimal" here would make a truncated search
            # indistinguishable from a proven one and silently corrupt any
            # ground truth derived from it.
            truncated = True
            break
        if lp_obj >= incumbent_obj - _EPS:           # bound prune
            continue

        frac_j = _select_branch_var(cs, A, b, lo, hi, basic, at_upper, lp_x,
                                    lp_obj, int_idx, pc, branching,
                                    incumbent_obj, priority, depth,
                                    reliability, max_sb_cands)

        if frac_j == -1:                              # integral -> candidate incumbent
            if lp_obj < incumbent_obj - 1e-12:
                incumbent_obj, incumbent_x = lp_obj, lp_x[:n].copy()
            continue

        val = lp_x[frac_j]
        f_down = val - math.floor(val)
        h_floor = hi.copy(); h_floor[frac_j] = math.floor(val + 1e-9)   # x_j <= floor
        l_ceil = lo.copy();  l_ceil[frac_j] = math.ceil(val - 1e-9)     # x_j >= ceil

        kids = []
        for side, (nl, nh) in (("down", (lo.copy(), h_floor)),
                               ("up", (l_ceil, hi.copy()))):
            if np.any(nl > nh + 1e-9):                # empty domain
                continue
            # Branching only tightens bounds; c and A are untouched, so the parent
            # basis stays dual-feasible and the check would just cost a solve.
            child = dual_simplex(cs, A, b, nl, nh, basic, at_upper, validate=False)
            if child.status != "optimal":
                continue
            pc.update(frac_j, side, child.obj - lp_obj,
                      f_down if side == "down" else 1.0 - f_down)
            if child.obj >= incumbent_obj - _EPS:     # bound prune
                continue
            kids.append((nl, nh, child.basic, child.at_upper, child.obj,
                         child.x, depth + 1))

        if node_select == "hybrid" and kids:
            # Plunge into the more promising child, park the rest in the pool.
            kids.sort(key=lambda k: k[4])
            for k in kids[1:]:
                push(k)
            stack.append(kids[0])
        else:
            for k in kids:
                push(k)

    if truncated:
        # An incumbent may exist but optimality is unproven; callers that need a
        # proven optimum (e.g. revalidation ground truth) must treat this as a
        # hard failure rather than a result.
        x = incumbent_x if incumbent_x is not None else np.zeros(n)
        return MILPResult("node_limit", x, float(incumbent_obj), nodes)
    if incumbent_x is None:
        return MILPResult("infeasible", np.zeros(n), INF, nodes)
    return MILPResult("optimal", incumbent_x, float(incumbent_obj), nodes)
