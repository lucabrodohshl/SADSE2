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

## Two evaluations, and why there are two

The CRR engine is built on a **self-contained optimization engine** — a
bounded-variable primal/dual simplex plus branch-and-bound, written from scratch
and validated against SciPy (no external MILP solver).

**[`docs/evaluation.md`](Implementation/docs/evaluation.md)** is the original study
(16-entry cache, one seed). Several of its headline numbers turned out to be
consequences of how the experiment was built rather than measurements of CRR — most
importantly, its Type-III refinement is a *uniform* scale, which provably cannot change
`argmin_k Σ_j E[k,j]` for any severity, and its Stage-2 "certificate check" performs the
same 35 energy-model evaluations as the "expensive" solve it is credited with avoiding.
It is retained for provenance.

**[`docs/realworld-evaluation.md`](Implementation/docs/realworld-evaluation.md)** is the
replacement: a simulated operating envelope (AR(1) weather over a grid of climatologies),
vector-wind physics that genuinely re-ranks configurations (**90%** of draws, versus
**0% by construction** in the original ODD model), refinement operators that *can* move
the optimum, an honest comparator set led by **`naive_recheck`** (CRR with all its
machinery removed), and cost accounted as real work rather than only as "expensive
calls". It carries the RQ1/RQ3 claims; §1 of that document catalogues why.

### The engine now reaches a scale where the premise is real

At the originally evaluated size, one "expensive MILP call" is **4.89 ms** and the whole
16-entry cache revalidates in **78 ms** — there was no expensive optimizer to avoid. The
engine was rebuilt (LU reuse, Dantzig pricing with a Bland fallback, reliability
branching, hybrid best-first + plunging, an LP-guided dive heuristic used only as an
incumbent):

| fleet | binary vars | before | after | vs HiGHS |
|---|---|---|---|---|
| 4 × 5 × 7 | 147 | did not finish in 5 min | **383 ms** | 11.6× |
| 6 × 8 × 9 | 441 | — | **3.2 s** | 16.1× |
| 7 × 10 × 10 | 710 | — | **34.8 s** | 36.2× |

At the 6 × 8 × 9 operating point a single re-solve costs **3.2 s** on the self-contained
engine, so avoiding re-solves describes something real.

## Reproduce

```bash
cd Implementation
pip install -r requirements.txt
python -m pytest evaluate/tests/test_crr.py evaluate/tests/test_simplex.py -q   # engine + soundness
python -m evaluate.tests.bench_engine              # engine vs scipy/HiGHS oracle

# the simulation-based evaluation (RQ1/RQ3)
CRR_SEEDS=16 CRR_M=6 CRR_J=8 CRR_K=9 CRR_N=12 \
  python -m evaluate.experiments.realworld_eval

# the original study, retained for provenance
python -m evaluate.experiments.crr_efficiency      # RQ1
python -m evaluate.experiments.crr_correctness     # RQ2
python -m evaluate.experiments.crr_sensitivity     # RQ3
```

> Note: the paper's LaTeX source (`RTOPT/`) is not present in the working tree;
> `main.pdf` is the compiled paper.
