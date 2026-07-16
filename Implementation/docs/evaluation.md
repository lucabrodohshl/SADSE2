# Evaluation

> **⚠ Superseded for RQ1 and RQ3 — see [`realworld-evaluation.md`](realworld-evaluation.md).**
>
> Several results below are consequences of the experiment's construction rather than
> measurements of CRR. Each was verified by running the code:
>
> - **The Type-III result is a tautology.** `refine_type_III` is a *uniform* scale
>   `base*(1+delta)`, and `argmin_k Σ_j E[k,j]` is scale-invariant, so the ranking is
>   preserved for **any** delta (identical from 0.05 to **100**; cost-ratio spread
>   `4.4e-16`). "Type III holds at 100% because the ranking is preserved" restates the
>   operator's definition. It is also bit-identical to the wind factor the cache is
>   keyed on.
> - **"Zero MILP re-solves" understates the work done.** Stage 2's `_cheapest_config`
>   rebuilds the whole energy matrix and argmins it — **35** energy-model evaluations,
>   exactly what `solve()` performs — but is counted as `n_arith` ("no solver"). This is
>   why the 100% call reduction coexists with the ~1× wall-clock reported below.
> - **`S1 = 0` in every row is structural**, not incidental: every entry depends on
>   `obj_coeffs`/`battery_row`, so a global footprint matches all 16 entries.
> - **The `cr(e)` / ε(Z_e) machinery is not in the evaluated path.** `crr_revalidate`
>   never reads it; `residual_radius` is 11.649074 for *every* entry and ~18× larger
>   than the margin it must beat, so it could never fire.
> - **The baseline re-solves entries it knows are untouched** (`applies_to(e) == False`).
> - **N = 1**: one scenario, one seed, a 16-point linspace on a single scalar.
>
> Underneath all of them: one "expensive MILP call" here is **4.89 ms** and the entire
> cache revalidates in **78 ms**, on a 20-binary-variable problem — there is no expensive
> optimizer to avoid at this scale.
>
> **The RQ2 (correctness) result stands**, and the Stage-3 warm dual-simplex claim stands
> at **18.6×** (133 warm vs 2474 cold pivots; previously 22× against a slower cold solve).
> Type II is the only warm-startable case here — Types I/III change the constraint matrix,
> so they correctly cold-solve.

We evaluate Certified Refinement Revalidation (CRR) against three research
questions:

- **RQ1 (Efficiency).** When the optimization model is refined, how effectively
  does CRR reduce invocations of the expensive optimizer relative to full cache
  revalidation (re-solving every entry)?
- **RQ2 (Correctness).** Does CRR preserve exact optimality under the evolved
  model, and what does the prior *no-revalidation* approach cost?
- **RQ3 (Sensitivity & scalability).** How do CRR's savings vary with refinement
  type, footprint size |Δ|, and problem scale (fleet size, cache size)?

## Setup

We implement CRR on the fleet task-assignment MILP of the running example: a
fleet of *M* drones covers *J* tasks under a chosen configuration *k* (speed,
altitude, camera resolution), minimizing total fleet energy subject to a
per-agent battery-reserve budget `usable = (1-reserve)·capacity/safety`. Energy
is agent-independent, so the objective is configuration-determined
(`value = min_k Σ_j E[k,j]`) while the battery budget couples the configuration
choice to a bin-packing task assignment: tightening the budget can render the
cheapest configuration infeasible and force a costlier one.

**A self-contained optimization engine.** So that CRR's Stage-3 *warm
dual-simplex* is real rather than emulated, the entire pipeline runs on a
from-scratch solver — a bounded-variable **primal/dual revised simplex**
(`src/crr/simplex.py`) and an LP-relaxation **branch-and-bound** with
warm-started dual-simplex children (`src/crr/branch_and_bound.py`). Both are
validated against `scipy.optimize.linprog`/`milp` over randomized instances (120
oracle tests). **No external MILP solver is used.** Stage 3 warm-starts the dual
simplex from each entry's stored basis; Stage 4 is the branch-and-bound.

A cache is populated under the base model M0 with one proven-optimal entry per
operating regime (ODD), modelled as a wind factor scaling energy from *calm* to
*strong*; each entry is enriched with its certificate, dependency set, validity
ranges, and the bin-packing LP **basis** used to warm-start repairs, and inserted
into the dependency reverse index. We then apply the three refinements — Type II
(tightened battery reserve), Type III (tighter objective linearization), Type I (a
new humidity factor) — and revalidate. Baselines: *full revalidation* (re-solve
every entry; the naive N-solve upper bound and optimality ground truth) and *no
revalidation* (the stale M0 cache). All runs are deterministic.

```
python -m evaluate.experiments.crr_efficiency     # RQ1
python -m evaluate.experiments.crr_correctness    # RQ2
python -m evaluate.experiments.crr_sensitivity    # RQ3
python -m evaluate.reporting.crr_figures          # figures + tables
```

## RQ1 — Efficiency

On a cache of **N = 16** entries, a representative refinement of each type is
revalidated with **zero** MILP re-solves (Table `crr_summary`), a **100%**
reduction versus full revalidation, at exact optimality. Entries are discharged
through the cheap stages: for the moderate Type-II tightening, 1 entry certifies
directly (S2) and 15 are repaired by a warm dual-simplex re-assignment (S3);
Types III and I split across S2 and S3 with no re-solve.

As refinement **severity** grows, the population escalates through the
cost-ordered stages (Fig. `crr_stage_distribution`). For Type II the split moves
`S2/S3/S4 = 10/6/0 → 1/15/0 → 0/10/6 → 0/4/12` as the reserve tightens from 0.30
to 0.78, and the re-solve reduction degrades gracefully from **100% to 25%** — CRR
invokes the optimizer only for the entries whose *integer optimum (configuration)*
genuinely changed, and **never does worse than full revalidation**
(Fig. `crr_milp_reduction`). Type III holds at **100%** reduction across the whole
severity range: a tighter objective linearization preserves the configuration
ranking, so every optimum is corrected by an arithmetic certificate (S2) or a warm
re-assignment (S3), never a re-solve.

**Warm dual-simplex.** The Stage-3 repairs are the paper's cut-warmed repair made
concrete: the stored basis warm-starts a dual-simplex re-optimization of the
bin-packing LP after the battery RHS tightens. Across the Type-II severity sweep
the warm re-solves take **132 pivots in total versus 2 971 for cold solves — a
22× reduction** (Fig. `crr_pivots`; e.g. 22 vs 762 pivots at reserve = 0.54).
Because the warm basis is already dual-feasible, its pivot count barely grows with
problem size while the cold count climbs (RQ3), so the warm-start advantage widens
with scale.

## RQ2 — Correctness

Across **all 15** refinement points (three types × five severities) CRR's
revalidated cache is **sound**: the optimality gap against the full-revalidation
ground truth is **0.0** (exact, since the engine is exact arithmetic), confirming
the soundness guarantee (Theorem 1) at τ_req = 0.

The prior **no-revalidation** approach is, by contrast, unsafe under model
evolution: keeping the stale M0 cache leaves up to **100%** of entries
*infeasible* under a tightened Type-II battery constraint (up to 56% under Type
III) — cached plans that now violate the safety reserve. In this system the
dominant failure mode of staleness is constraint **violation** rather than mild
suboptimality — precisely the safety-relevant errors CRR eliminates while still
avoiding a full re-solve.

## RQ3 — Sensitivity & scalability

**Footprint.** A regime-scoped Type-II refinement touching *k* of the 16 regimes
is handled by the reverse index: entries outside the footprint are reused at Stage
1 without examination, so the number examined tracks |Δ|
(Fig. `crr_footprint`). Crucially, the number of *expensive* re-solves
**saturates at 3** even when the footprint grows to touch **all 16** entries —
only the few high-wind regimes whose optimum migrates ever reach a full solve.

**Scale.** Sweeping fleet size M ∈ {3,5,8} and cache size N ∈ {8,16,24} under
representative refinements, CRR sustains a **100%** re-solve reduction. The
warm-start advantage grows with scale: the Stage-3 cold LP pivot count rises with
the fleet (≈ 593 → 854 → 1 234 pivots for M = 3,5,8) while the warm dual-simplex
stays flat at ≈ 20 pivots.

## Fidelity and threats to validity

The whole pipeline runs on the self-contained engine described above — no
external solver — and every solver interaction is counted and typed (none /
arithmetic S2 / warm-LP S3 / branch-and-bound S4). Stage 3 is a genuine warm
dual-simplex re-optimization (with a cheap rounding of its LP solution; the
optimum is confirmed exactly by the engine when rounding cannot close it). The
engine is correctness-focused, not speed-optimized, so on these small instances
wall-clock is comparable to full revalidation (≈ 1×); we therefore report the
*algorithmic* metrics — expensive-solver-call reduction and simplex-pivot
reduction — which are properties of the mechanism and would translate to
wall-clock on a production solver. The study is a controlled instantiation of the
running example: absolute magnitudes depend on instance size, but the stage
distribution, the soundness result, and the footprint/scale trends are properties
of CRR itself.

## Summary

CRR revalidates a cache of proven-optimal fleet configurations under model
evolution while (RQ1) invoking the expensive optimizer only for entries whose
integer optimum genuinely changed — a 100% re-solve reduction for objective
refinements and 25–100% for constraint refinements, with the Stage-3 warm
dual-simplex using ~22× fewer pivots than a cold solve; (RQ2) preserving exact
optimality (gap = 0.0) where the prior no-revalidation approach leaves up to 100%
of entries infeasible; and (RQ3) concentrating work on the footprint (re-solves
saturating at a small constant) with savings that hold — and a warm-start
advantage that widens — as fleet and cache size grow.
