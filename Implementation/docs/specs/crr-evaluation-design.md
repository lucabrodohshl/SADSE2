# Spec — Evaluation of Certified Refinement Revalidation (CRR)

Status: approved (2026-07-15). Implements the missing evaluation for the paper
*"Efficient Runtime Self-Optimization Under Model Evolution"* (CRR).

## 1. Goal & research questions

The paper is theory-only. This work builds a **CRR engine** on the existing
task-assignment MILP + zonotope + ODD-cache substrate and evaluates it against
three research questions:

- **RQ1 (Efficiency):** When the optimization model is refined, how much does CRR
  reduce expensive MILP optimizer calls vs. full cache revalidation (re-solve all)?
- **RQ2 (Correctness/Soundness):** Does CRR preserve exact optimality under the
  evolved model (zero optimality gap at τ_req=0 vs. a ground-truth full re-solve),
  and what does the prior *no-revalidation* approach cost (stale-cache violations)?
- **RQ3 (Sensitivity/Scalability):** How do CRR's savings vary with refinement
  **type** (I/II/III), **footprint** |Δ|, and **scale** (fleet size M, cache size N)?

## 2. Background mapping (paper → code)

The model under evolution is the fleet task-assignment MILP in
`src/milp_solver.py::solve_task_assignment_milp`:
- **Decision:** `x[i,j,k]` (agent i does task j with config k), `y[k]` (config selected).
- **Objective:** minimize total energy `Σ x[i,j,k]·E[k,j]`, with `E` from the energy
  model (`drone_energy_model`), quadratic in speed (first-order/Taylor substrate).
- **Constraints:** each task assigned once; one config selected; use-only-selected;
  **per-agent battery budget** `Σ_j Σ_k x[i,j,k]·E[k,j] ≤ usable`, where
  `usable = (1-reserve)·capacity/safety`.

Refinement types map directly:
- **Type I (new factor):** add a design dimension (e.g. humidity) that adds an energy
  term / new option column. Footprint Δ = {new dimension, affected coefficients}.
- **Type II (tightened constraint):** tighten the battery reserve (raise `reserve` or
  lower `capacity/budget`) → `usable` shrinks. Footprint Δ = {battery-budget row}.
- **Type III (tightened objective):** replace the coarse energy linearization with a
  tighter one (higher-order term / `milp_solver_v2` perturbed model). Footprint Δ =
  {objective coefficients}.

## 3. CRR engine — module contracts (`src/crr/`)

All modules are pure-Python, run-from-root (`from src.crr... import ...`), and depend
only on `src/` + numpy/pulp.

### 3.1 `model.py`
- `class OptModel`: holds `energy_fn`, `battery(capacity, reserve, safety)`, `dims`
  (parameter names), `tasks`, `num_agents`. Method `solve(region_configs) ->
  SolveResult(config, assignment, value, per_agent_loads, lp_duals, reduced_costs,
  n_milp, n_lp, wall_s)`. `solve` calls `solve_task_assignment_milp`; it also solves
  the **LP relaxation** to obtain duals/reduced costs for certificates.
- `SolveResult` records solver-call counts so every path is accounted for.

### 3.2 `certificate.py`
- `class Entry`: `(zonotope Z_e, config x*_e, value v*_e, cert, dep, cr)`.
- `cert(e)`: `{assignment, per_agent_loads, optimality_margin, lp_duals,
  binding_battery_agents}`. `optimality_margin` = objective gap to the second-best
  assignment (from LP reduced costs or an explicit runner-up solve at build time).
- `dep(e)`: set of feature tags the certificate reads: `{"obj_coeffs", "battery_row",
  "task_rows", "dim:<name>"...}`.
- `cr(e)`: validity range — for **Type III** the residual radius
  `ε(Z_e) = ½·sup|f''|·diam(Z_e)²` (computed from the energy model's curvature and the
  zonotope diameter) and the reduced-cost margin; for **Type II** the battery slack
  `usable₀ − max_i load_i`; for **Type I** the new column's reduced-cost sign.
- `support(zonotope, direction)`: add `Zonotope.support(a)=aᵀc+Σ|aᵀg_i|` (new method on
  `zonotope_ops.Zonotope`) for the Type-II region peak-energy test.

### 3.3 `refinement.py`
- `refine_type_I(model, factor) -> (M1, footprint Δ)` — add dimension/energy term.
- `refine_type_II(model, new_reserve|new_budget) -> (M1, Δ)` — tighten battery.
- `refine_type_III(model, tighter_energy_fn) -> (M1, Δ)` — tighten objective.
- Each returns the evolved `OptModel` and the footprint (set of feature tags).

### 3.4 `reverse_index.py`
- `class ReverseIndex`: `insert(entry)` buckets `entry` under each tag in `dep(e)`;
  `query(Δ) -> set[Entry]` returns `∪_{f∈Δ} D[f]` (superset-safe); `adjacent(A)`
  returns zonotope-adjacent entries for Type-I space enlargement.

### 3.5 `revalidation.py` — Algorithm 1 (the 4 stages)
`crr_revalidate(cache, M1, Δ, tau_req=0.0) -> (cache', CRRMetrics)`:
- **S1** `A = index.query(Δ)`; reuse every `e∉A`; drop directional no-ops
  (tightening where entry already slack); if Δ has a new factor, `A ∪= adjacent(A)`.
- **S2** `cert_survives(e, M1)` per type (support-fn / reduced-cost / cr membership).
  Pass ⇒ status VALID, refresh `v*_e` under M1, **no solver**.
- **S3** `warm_repair(e, M1)` — LP-relaxation re-solve of the region subproblem;
  integral & feasible ⇒ REPAIRED (**LP** call), else escalate.
- **S4** batched, memoized full **MILP** re-solve of the residual, confined to
  `Z_e ∩ F₁`. Every S4 entry = one expensive solver call (deduped by `memo[hash]`).
- Soundness invariant asserted in tests: output cache is τ_req-valid under M1.

### 3.6 `baselines.py`
- `full_revalidation(cache, M1)`: re-solve MILP for every entry (N calls) = ground truth.
- `no_revalidation(cache, M1)`: keep M0 cache; report per-entry feasibility/gap under M1.
- `cold(cache, M1)`: re-solve from scratch per query (reference).

### 3.7 `metrics.py`
- `CRRMetrics`: per-stage entry counts, solver calls `{none, arith, lp, milp}`,
  wall-clock per stage, and — when ground truth supplied — per-entry optimality gap,
  max gap, and soundness pass/fail.

## 4. Experiments

Cache population: solve M0 over a grid of ODD regimes (wind × temperature, via the
existing weather scenarios) → N enriched entries with zonotope regions.

- **E1 `crr_efficiency.py` (RQ1):** fixed N, M; apply each refinement type; run CRR +
  full-revalidation. Outputs: stage histogram, MILP calls (CRR vs N), speedup.
- **E2 `crr_correctness.py` (RQ2):** per entry, CRR vs full-revalidation ground truth
  ⇒ optimality gap (expect 0). Also no-revalidation stale cache: violation rate +
  suboptimality. Outputs: gap table (CRR≈0), stale-cache cost.
- **E3 `crr_sensitivity.py` (RQ3):** sweep type ∈ {I,II,III}, footprint |Δ|
  (single-row vs multi-feature), scale M ∈ {5,10,20,50,100} and N ∈ {10,25,50,100}.
  Cap sizes so total runtime stays within a few minutes per sweep point. Outputs:
  MILP-call reduction vs |Δ|; speedup vs N and M.

Determinism: fixed seeds; the same M0 cache reused across baselines for fair comparison.

## 5. Deliverables

```
src/crr/{__init__,model,certificate,refinement,reverse_index,revalidation,baselines,metrics}.py
src/crr/{simplex,branch_and_bound}.py       # self-contained LP/MILP engine (warm dual-simplex)
evaluate/experiments/{crr_efficiency,crr_correctness,crr_sensitivity}.py
evaluate/reporting/crr_figures.py           # pgf+png figures + LaTeX tables -> results/reports
evaluate/tests/test_crr.py                  # unit + soundness (CRR == ground truth)
evaluate/tests/test_simplex.py              # LP/dual-simplex/B&B validated vs scipy
results/crr/…                               # JSON/CSV kept; figures gitignored
docs/evaluation.md + docs/evaluation.tex    # written Evaluation section (drop-in)
```

## 6. Fidelity, assumptions, honesty

- Real task-assignment MILP throughout, solved by a **self-contained engine**
  (`src/crr/simplex.py` bounded-variable primal/dual simplex + `branch_and_bound.py`),
  validated against `scipy.optimize.linprog`/`milp`. **No external solver.**
- **S3 is a true warm dual-simplex** (supersedes the earlier proxy): the stored
  basis warm-starts a dual-simplex re-optimization after the battery RHS tightens
  (~22x fewer pivots than cold); Stage 4 is warm-started branch-and-bound.
  Solver-call accounting separates none/arith/warm-LP/B&B. The engine is
  correctness-focused, so wall-clock ~1x on small instances; reported metrics are
  algorithmic (solver-call and pivot reduction).
- Certificates are **exact** where the model allows (Type II feasibility/support-fn,
  Type III reduced-cost margin vs. ε(Z_e), Type I reduced-cost sign); documented as
  such. τ_req = 0 (exact optimality) is the default; τ_req > 0 supported.

## 7. Validation

Test-first. Core soundness test: for every entry and every refinement type, the
CRR-revalidated `(config, value)` equals the full-revalidation ground truth at
τ_req = 0. Stage-logic unit tests: a slack entry hits S1/S2; a cut-off optimum reaches
S4; footprint query returns a superset of genuinely-affected entries.
