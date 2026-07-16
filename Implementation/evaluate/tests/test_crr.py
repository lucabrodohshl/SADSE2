"""Tests for the Certified Refinement Revalidation (CRR) engine.

Run from the Implementation/ root:  python -m pytest evaluate/tests/test_crr.py -q
"""
import numpy as np

from src.zonotope_ops import Zonotope


# ---------------------------------------------------------------------------
# Zonotope.support  --  h_Z(a) = max_{x in Z} a^T x = a^T c + sum_i |a^T g_i|
# (needed for the Stage-2 Type-II "region peak energy within budget?" test)
# ---------------------------------------------------------------------------

def test_support_of_axis_aligned_box_returns_face_values():
    # box [0,10] x [0,4]  -> center (5,2), half-widths (5,2)
    z = Zonotope.from_box([(0.0, 10.0), (0.0, 4.0)])
    assert z.support(np.array([1.0, 0.0])) == 10.0   # max x
    assert z.support(np.array([0.0, 1.0])) == 4.0    # max y
    assert z.support(np.array([1.0, 1.0])) == 14.0   # max (x+y) over the box
    assert z.support(np.array([-1.0, 0.0])) == 0.0   # -min x  => 0


# ---------------------------------------------------------------------------
# OptModel  --  the fleet task-assignment MILP under a given model M
# objective is config-determined (value = min_k sum_j E[k,j]); the per-agent
# battery budget is a bin-packing feasibility constraint.
# ---------------------------------------------------------------------------
from src.crr.scenario import make_scenario
from src.crr.model import OptModel


def test_solve_picks_min_cost_config_when_battery_loose():
    scn = make_scenario(num_agents=3, num_tasks=3, num_configs=5, seed=1)
    m = OptModel.from_scenario(scn, capacity=1e6)   # effectively unconstrained battery
    r = m.solve()
    costs = [m.config_cost(k) for k in range(5)]
    assert r.feasible
    assert r.config_idx == int(np.argmin(costs))
    assert abs(r.value - min(costs)) < 1e-6
    assigned = sorted(t for ts in r.assignment.values() for t in ts)
    assert assigned == [0, 1, 2]                     # every task assigned exactly once
    assert r.n_milp == 1


def test_tighter_reserve_shrinks_usable_budget():
    scn = make_scenario(2, 2, 3, seed=2)
    loose = OptModel.from_scenario(scn, reserve=0.20)
    tight = OptModel.from_scenario(scn, reserve=0.50)
    assert tight.usable_budget() < loose.usable_budget()


def test_plan_feasible_respects_battery():
    scn = make_scenario(1, 3, 3, seed=3)             # one agent must do all 3 tasks
    r = OptModel.from_scenario(scn, capacity=1e6).solve()
    m_tight = OptModel.from_scenario(scn, capacity=1.0)
    assert not m_tight.plan_feasible(r.config_idx, r.assignment)


def test_optimality_margin_is_gap_to_runner_up():
    scn = make_scenario(3, 3, 5, seed=4)
    r = OptModel.from_scenario(scn, capacity=1e6).solve()
    costs = sorted(OptModel.from_scenario(scn, capacity=1e6).config_cost(k) for k in range(5))
    assert abs(r.optimality_margin - (costs[1] - costs[0])) < 1e-6


# ---------------------------------------------------------------------------
# Entry enrichment, model refinements, and the dependency reverse index
# ---------------------------------------------------------------------------
from src.crr.certificate import build_entry, Entry
from src.crr.refinement import refine_type_I, refine_type_II, refine_type_III
from src.crr.reverse_index import ReverseIndex


def _model(seed=1, wind=0.0, **kw):
    return OptModel.from_scenario(make_scenario(3, 3, 5, seed=seed), wind=wind, **kw)


def test_build_entry_has_optimal_plan_and_dependencies():
    m = _model(capacity=1e6)
    e = build_entry(m, name="calm", regime_name="calm")
    assert e.config_idx is not None
    assert {"obj_coeffs", "battery_row", "regime:calm"} <= e.dep
    assert abs(e.value - m.config_cost(e.config_idx)) < 1e-6


def test_type_II_global_footprint_is_battery_row_and_tightens():
    ref = refine_type_II(reserve=0.5)
    assert ref.kind == "II" and "battery_row" in ref.footprint
    m0 = _model(reserve=0.2)
    assert ref.apply(m0).usable_budget() < m0.usable_budget()


def test_type_II_scoped_footprint_targets_regimes():
    ref = refine_type_II(reserve=0.5, scope_regimes=["gust"])
    assert ref.footprint == {"regime:gust"}


def test_type_III_changes_energy_with_obj_footprint():
    ref = refine_type_III(delta=0.2)
    assert ref.kind == "III" and "obj_coeffs" in ref.footprint
    m0 = _model()
    assert not np.allclose(m0.energy_matrix(), ref.apply(m0).energy_matrix())


def test_type_I_adds_factor_with_dim_footprint():
    ref = refine_type_I(factor="humidity", strength=0.15)
    assert ref.kind == "I" and any(t.startswith("dim:") for t in ref.footprint)
    m0 = _model()
    # the new factor raises energy (humidity adds load)
    assert ref.apply(m0).energy_matrix().sum() >= m0.energy_matrix().sum()


def test_reverse_index_is_selective():
    idx = ReverseIndex()
    m = _model(capacity=1e6)
    e_calm = build_entry(m, "calm", "calm"); idx.insert(e_calm)
    e_gust = build_entry(m, "gust", "gust"); idx.insert(e_gust)
    assert idx.query({"regime:gust"}) == {e_gust}          # local: only the gust entry
    assert idx.query({"obj_coeffs"}) == {e_calm, e_gust}   # global: all entries


# ---------------------------------------------------------------------------
# The 4-stage CRR pipeline, baselines, and the SOUNDNESS property
# ---------------------------------------------------------------------------
from src.crr.cache import build_cache, default_regimes
from src.crr.revalidation import crr_revalidate
from src.crr.baselines import full_revalidation, no_revalidation


def _cache(seed=7, capacity=40.0, reserve=0.2, na=4, nt=5, nc=6):
    return build_cache(make_scenario(na, nt, nc, seed=seed), default_regimes(),
                       capacity=capacity, reserve=reserve)


def test_crr_is_sound_matches_full_revalidation_ground_truth():
    cache = _cache()
    for ref in [refine_type_II(reserve=0.45),
                refine_type_III(delta=0.25),
                refine_type_I(strength=0.2)]:
        crr_res, _ = crr_revalidate(cache, ref, tau_req=0.0)
        gt, _ = full_revalidation(cache, ref)
        for e in cache.entries:
            assert crr_res[e].feasible == gt[e].feasible, (ref.kind, e.name)
            if gt[e].feasible:
                # exact optimality: CRR's revalidated value == ground-truth optimum
                assert abs(crr_res[e].value - gt[e].value) < 1e-6, (ref.kind, e.name)


def test_crr_never_uses_more_milp_than_full_revalidation():
    cache = _cache()
    _, mc = crr_revalidate(cache, refine_type_II(reserve=0.45))
    _, mf = full_revalidation(cache, refine_type_II(reserve=0.45))
    assert mf.n_milp == len(cache.entries)     # full re-solves everything
    assert mc.n_milp <= mf.n_milp              # CRR never does worse


def test_scoped_refinement_reuses_unaffected_entries_at_stage1():
    cache = _cache()
    ref = refine_type_II(reserve=0.45, scope_regimes=["strong"])
    _, mc = crr_revalidate(cache, ref)
    # only the 'strong' entry is in the footprint; the rest are reused (S1, no solver)
    assert mc.stage_counts["S1"] == len(cache.entries) - 1


def test_no_revalidation_flags_stale_violations():
    cache = _cache()
    max_load = max((max(e.per_agent_loads) if e.per_agent_loads else 0.0) for e in cache.entries)
    tight_cap = max_load * 0.5 * 1.05 / 0.8      # usable ~ half the worst load -> some entries violate
    mn = no_revalidation(cache, refine_type_II(capacity=tight_cap))
    assert mn.violation_rate > 0.0
