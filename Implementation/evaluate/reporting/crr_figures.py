"""Figures and LaTeX tables for the CRR evaluation (reads results/crr/*.json).

Run:  python -m evaluate.reporting.crr_figures
Writes PNG (+ best-effort PGF) to results/reports/figures/ and a LaTeX/CSV
summary table to results/reports/.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import RESULTS_DIR, REPORTS_DIR, FIGURES_DIR

CRR_DIR = RESULTS_DIR / "crr"
_STAGE_COLORS = {"S1": "#4c9f70", "S2": "#8ecae6", "S3": "#ffb703", "S4": "#e63946"}
_STAGE_LABEL = {"S1": "S1 reuse", "S2": "S2 certificate", "S3": "S3 repair", "S4": "S4 re-solve"}


def _load(name):
    with open(CRR_DIR / name) as f:
        return json.load(f)


def _save(fig, stem):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    try:
        fig.savefig(FIGURES_DIR / f"{stem}.pgf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def fig_stage_distribution(eff):
    """Stacked stage distribution across the Type-II severity sweep."""
    rows = eff["severity_sweep"]["II"]
    labels = [r["label"].replace("reserve=", "r=") for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3))
    bottom = [0] * len(rows)
    for stage in ("S1", "S2", "S3", "S4"):
        vals = [r["stages"][stage] for r in rows]
        ax.bar(labels, vals, bottom=bottom, label=_STAGE_LABEL[stage], color=_STAGE_COLORS[stage])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("cache entries")
    ax.set_xlabel("Type-II refinement severity (battery reserve)")
    ax.set_title("CRR stage distribution vs. refinement severity")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, "crr_stage_distribution")


def fig_milp_reduction(eff):
    """Expensive-solver-call reduction vs. severity, per refinement type."""
    fig, ax = plt.subplots(figsize=(6, 3))
    for kind, marker in (("II", "o"), ("III", "s"), ("I", "^")):
        rows = eff["severity_sweep"][kind]
        xs = list(range(1, len(rows) + 1))
        ys = [100 * r["milp_reduction"] for r in rows]
        ax.plot(xs, ys, marker=marker, label=f"Type {kind}")
    ax.set_xlabel("refinement severity (increasing)")
    ax.set_ylabel("MILP re-solve reduction (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("CRR vs. full revalidation: expensive-call reduction")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "crr_milp_reduction")


def fig_footprint(sens):
    """Affected-set size and CRR re-solves vs. footprint size."""
    rows = sens["footprint"]
    ks = [r["footprint_regimes"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ks, [r["affected"] for r in rows], "o-", label="entries examined")
    ax.plot(ks, [r["crr_milp"] for r in rows], "s-", color="#e63946", label="CRR MILP re-solves")
    ax.plot(ks, [r["full_milp"] for r in rows], "--", color="gray", label="full revalidation (=N)")
    ax.set_xlabel(r"footprint size $|\Delta|$ (regimes touched)")
    ax.set_ylabel("count")
    ax.set_title("Reverse-index selectivity and re-solve floor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "crr_footprint")


def fig_pivots(eff):
    """Warm dual-simplex vs. cold LP pivots for the Stage-3 repairs (Type-II sweep)."""
    rows = [r for r in eff["severity_sweep"]["II"] if r["cold_pivots"] > 0]
    labels = [r["label"].replace("reserve=", "r=") for r in rows]
    warm = [r["warm_pivots"] for r in rows]
    cold = [r["cold_pivots"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - 0.2, cold, 0.4, label="cold LP solve", color="#e63946")
    ax.bar(x + 0.2, warm, 0.4, label="warm dual-simplex", color="#4c9f70")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("simplex pivots (log scale)")
    ax.set_xlabel("Type-II refinement severity (battery reserve)")
    ax.set_title("Stage-3 repair: warm dual-simplex vs. cold solve")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    _save(fig, "crr_pivots")


def write_table(eff, corr):
    """Per-type headline table (LaTeX + CSV)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    header = ["Type", "S1", "S2", "S3", "S4", "CRR MILP", "Full MILP", "Reduction", "Max gap", "Sound"]
    lines_csv = [",".join(header)]
    rows_tex = []
    for kind in ("II", "III", "I"):
        v = eff["by_type"][kind]
        s = v["stages"]
        row = [f"Type {kind}", s["S1"], s["S2"], s["S3"], s["S4"], v["crr_milp"], v["full_milp"],
               f"{100*v['milp_reduction']:.0f}\\%", f"{v['max_gap']:.1e}", "yes" if v["sound"] else "NO"]
        lines_csv.append(",".join(str(c).replace("\\%", "%") for c in row))
        rows_tex.append(" & ".join(str(c) for c in row) + r" \\")

    (REPORTS_DIR / "crr_summary.csv").write_text("\n".join(lines_csv) + "\n")
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{CRR revalidation on a cache of %d entries under a representative refinement of each type. "
        r"Every entry is revalidated with exact optimality (gap $\approx 0$); only entries whose integer optimum "
        r"genuinely changed reach a full MILP re-solve.}" % eff["N"],
        r"\label{tab:crr-efficiency}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrrc}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
        *rows_tex,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (REPORTS_DIR / "crr_summary.tex").write_text("\n".join(tex) + "\n")


def main():
    eff = _load("efficiency.json")
    corr = _load("correctness.json")
    sens = _load("sensitivity.json")
    fig_stage_distribution(eff)
    fig_milp_reduction(eff)
    fig_footprint(sens)
    fig_pivots(eff)
    write_table(eff, corr)
    print(f"[crr_figures] figures -> {FIGURES_DIR}")
    print(f"[crr_figures] table   -> {REPORTS_DIR / 'crr_summary.tex'} (+ .csv)")


if __name__ == "__main__":
    main()
