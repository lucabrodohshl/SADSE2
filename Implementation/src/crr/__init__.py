"""Certified Refinement Revalidation (CRR).

An engine that revalidates a zonotope-keyed cache of proven-optimal fleet
configurations when the optimization *model* is refined (new factor / tightened
constraint / tightened objective), invoking the expensive MILP optimizer only
for entries whose integer optimum genuinely changed, while preserving exact
optimality. Implements the mechanism of the paper "Efficient Runtime
Self-Optimization Under Model Evolution".
"""
