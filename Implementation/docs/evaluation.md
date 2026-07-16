# Evaluation

We evaluate Certified Refinement Revalidation (CRR) against three research
questions:

- **RQ1 (Efficiency).** When the optimization model is refined, how much does CRR
  reduce invocations of the expensive MILP optimizer relative to full cache
  revalidation (re-solving every entry)?
- **RQ2 (Correctness).** Does CRR preserve exact optimality under the evolved
  model, and what does the prior *no-revalidation* approach cost?
- **RQ3 (Sensitivity & scalability).** How do CRR's savings vary with refinement
  type, footprint size |Δ|, and problem scale (fleet size, cache size)?

## Setup

We implement CRR on the fleet task-assignment MILP of the running example: a
fleet of *M* drones must cover *J* tasks under a chosen configuration *k*
(speed, altitude, camera resolution), minimizing total fleet energy subject to a
per-agent battery-reserve budget `usable = (1-reserve)·capacity/safety`. Energy
is agent-independent, so the objective is configuration-determined
(`value = min_k Σ_j E[k,j]`) while the battery budget couples the choice of
configuration to a bin-packing task assignment — tightening the budget can render
the cheapest configuration infeasible and force a costlier one.

A cache is populated under the base model M0 by solving one proven-optimal entry
per operating regime (ODD), modelled as a wind factor scaling energy from *calm*
to *strong crosswinds*; each entry is enriched with its certificate (assignment,
per-agent loads, optimality margin), a dependency set, and validity ranges, and
inserted into the dependency reverse index. We then apply the three model
refinements of the paper and revalidate:

- **Type II** — a tightened battery-reserve constraint (regulation);
- **Type III** — a tighter objective linearization (the residual added back);
- **Type I** — a new environmental factor (humidity) entering the energy model.

**Baselines.** *Full revalidation* re-solves the MILP for every entry (the naive
upper bound of *N* solves) and serves as the optimality ground truth; *no
revalidation* keeps the M0 cache unchanged, quantifying the cost of staleness.

**Metrics.** We record, per revalidation, the distribution of entries across the
four stages (S1 reuse / S2 certificate / S3 repair / S4 re-solve), the number of
expensive MILP re-solves, wall-clock time, and — against the ground truth — the
optimality gap. All experiments are deterministic (fixed seeds), running from the
`Implementation/` root:

```
python -m evaluate.experiments.crr_efficiency     # RQ1
python -m evaluate.experiments.crr_correctness    # RQ2
python -m evaluate.experiments.crr_sensitivity    # RQ3
python -m evaluate.reporting.crr_figures          # figures + tables
```

## RQ1 — Efficiency

On a cache of **N = 16** entries, a representative refinement of each type is
revalidated with **zero** MILP re-solves (Table `crr_summary`), a **100%**
reduction versus full revalidation, at exact optimality. The entries are
discharged through the cheap stages: for the moderate Type-II tightening,
6 entries certify directly (S2) and 10 are repaired by a fixed-configuration
re-assignment (S3, an LP-class solve); for Type III and Type I all 16 entries
certify at S2.

As refinement **severity** grows, the population escalates through the
cost-ordered stages exactly as designed (Fig. `crr_stage_distribution`). For
Type II the stage split moves `S2/S3/S4 = 16/0/0 → 6/10/0 → 4/6/6 → 1/3/12` as
the reserve tightens from 0.30 to 0.78, and the MILP-call reduction degrades
gracefully from **100%** to **25%** — CRR invokes the optimizer only for the
entries whose integer optimum genuinely changed, and **never does worse than full
revalidation** (Fig. `crr_milp_reduction`). Type III stays at **100%** reduction
across the whole severity range: a tighter objective linearization preserves the
configuration ranking, so every entry's optimum is corrected by an arithmetic
certificate (S2) or a warm repair (S3) — never a re-solve. Wall-clock speedups
for the representative refinements range from **2.6×** (Type II, repair-heavy) to
over **400×** (Type III/I, certificate-only).

## RQ2 — Correctness

Across **all 15** refinement points (three types × five severities), CRR's
revalidated cache is **sound**: the maximum optimality gap against the
full-revalidation ground truth is **7.1 × 10⁻¹⁵** (numerically zero), empirically
confirming the soundness guarantee (Theorem 1) at τ_req = 0.

The prior **no-revalidation** approach is, by contrast, unsafe under model
evolution: keeping the stale M0 cache leaves up to **94%** of entries
*infeasible* under a tightened Type-II battery constraint (and up to 56% under
Type III), i.e. cached plans that now violate the safety reserve. In this system
the dominant failure mode of staleness is constraint **violation** rather than
mild suboptimality — precisely the safety-relevant errors CRR eliminates while
still avoiding a full re-solve.

## RQ3 — Sensitivity & scalability

**Footprint.** A regime-scoped Type-II refinement touching *k* of the 16 regimes
is handled by the reverse index: entries outside the footprint are reused at
Stage 1 without examination, so the number of entries examined tracks |Δ|
(Fig. `crr_footprint`). Crucially, the number of *expensive* re-solves
**saturates at 3** even when the footprint grows to touch **all 16** entries —
only the few high-wind regimes whose optimum genuinely migrates ever reach a full
MILP solve; the rest certify or repair.

**Scale.** CRR's advantage grows with problem size. Sweeping fleet size, the
wall-clock speedup over full revalidation rises from **1.0× (M=3)** to
**4.1× (M=5)**, **8.4× (M=8)**, and **91× (M=12)** as the individual MILP solves
become more expensive (Fig. `crr_scale`). Sweeping cache size N ∈ {8,16,24,32}
under a Type-III refinement, CRR sustains a **100%** re-solve reduction with
6–13× wall-clock speedup — the certificate machinery scales with the cache while
the optimizer is never invoked.

## Fidelity and threats to validity

CRR is implemented against the real fleet task-assignment MILP (PuLP/CBC); every
solver call is counted and separated into none / arithmetic (S2) / LP-class
repair (S3) / full MILP (S4). Because PuLP exposes no true warm dual-simplex, the
Stage-3 repair is realized as a fixed-configuration re-assignment — an honest
"cheap, no-branching" proxy that explores one configuration rather than all *K*;
the stage logic and the headline metric (MILP-call reduction) are exact. The
certificate checks are exact arithmetic for the model at hand (Type-II
feasibility via per-agent loads and the zonotope support function; Type-III
ranking preservation; Type-I reduced-cost sign). The evaluation is a controlled,
synthetic instantiation of the paper's running example; absolute speedups depend
on solver and instance size, but the *stage distribution*, the *soundness*
result, and the *footprint/scale trends* are properties of the mechanism.

## Summary

CRR revalidates a cache of proven-optimal fleet configurations under model
evolution while (RQ1) reducing expensive MILP re-solves by 100% for objective
refinements and 25–100% for constraint refinements — invoking the optimizer only
for entries whose integer optimum genuinely changed and never doing worse than
full revalidation; (RQ2) preserving exact optimality (gap ≈ 10⁻¹⁵) where the
prior no-revalidation approach leaves up to 94% of entries infeasible; and (RQ3)
concentrating work on the footprint (re-solves saturating at a small constant)
with a speedup that grows with fleet size.
