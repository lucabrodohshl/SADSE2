# Efficient Runtime Self-Optimization Under Model Evolution

This repository accompanies the paper **"Efficient Runtime Self-Optimization
Under Model Evolution"** ([`main.pdf`](main.pdf)), which introduces **Certified
Refinement Revalidation (CRR)**.

## The idea in one paragraph

Self-adaptive Cyber-Physical Systems (e.g. drone fleets) cache proven-optimal
configurations over regions of a large design space and reuse them instead of
re-solving an expensive MILP whenever conditions change. That cache is sound only
for a *fixed* optimization model. When engineers **refine the model** — a new
decision factor (Type I), a tightened constraint (Type II), or a tighter
objective linearization (Type III) — the cached optimality certificates may no
longer hold. CRR revalidates the cache through a **cost-ordered four-stage
pipeline** (directory lookup → certificate survival → warm dual-simplex repair →
batched re-solve), invoking the expensive optimizer **only for the entries whose
integer optimum genuinely changed**, while preserving exact optimality.

## What's in this repository

```
SADSE/
├── main.pdf              # the paper (theory + algorithm + soundness proof)
├── Implementation/       # the code: CRR engine, experiments, and evaluation
│   ├── src/crr/          # self-contained CRR engine + LP/MILP solver
│   ├── evaluate/         # experiments, reporting, tests
│   ├── results/          # experiment outputs (data kept, figures gitignored)
│   └── docs/             # documentation, incl. the written Evaluation section
└── README.md             # this file
```

`Implementation/` began as the paper's *prior approach* (a zonotope-keyed cache
with **no** revalidation) and now also contains a from-scratch implementation and
evaluation of CRR. See [`Implementation/README.md`](Implementation/README.md) for
the code layout and [`Implementation/docs/evaluation.md`](Implementation/docs/evaluation.md)
for the full evaluation.

## Evaluation at a glance (RQ1–3)

The CRR engine is built on a **self-contained optimization engine** — a
bounded-variable primal/dual simplex plus branch-and-bound, written from scratch
and validated against SciPy (no external MILP solver). Its Stage-3 repair uses a
**true warm dual-simplex**.

- **RQ1 — Efficiency.** CRR calls the expensive optimizer only when the integer
  optimum genuinely changes: on a 16-entry cache, moderate refinements of every
  type are revalidated with **0 re-solves (100 % reduction)**, degrading
  gracefully to 25 % under extreme tightening. The Stage-3 warm dual-simplex
  re-optimizes repairs in **~22× fewer pivots** than a cold solve — a gap that
  **grows with problem size**.
- **RQ2 — Correctness.** Exact optimality on every refinement point (optimality
  gap **0.0** vs. a full-re-solve ground truth), while the prior *no-revalidation*
  approach leaves up to **100 %** of entries infeasible (safety-reserve
  violations).
- **RQ3 — Sensitivity & scale.** The dependency reverse index touches only the
  refinement footprint; the number of expensive re-solves **saturates at a small
  constant** even when the footprint spans the whole cache, and the call reduction
  holds as fleet and cache size grow.

## Reproduce

```bash
cd Implementation
pip install -r requirements.txt
python -m pytest evaluate/tests/test_crr.py evaluate/tests/test_simplex.py -q   # engine + soundness
python -m evaluate.experiments.crr_efficiency      # RQ1
python -m evaluate.experiments.crr_correctness     # RQ2
python -m evaluate.experiments.crr_sensitivity     # RQ3
python -m evaluate.reporting.crr_figures           # figures + tables
```

> Note: the paper's LaTeX source (`RTOPT/`) is not present in the working tree;
> `main.pdf` is the compiled paper.
