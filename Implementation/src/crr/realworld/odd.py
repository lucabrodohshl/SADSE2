"""Operating-envelope (ODD) traces and frequency-weighted cache population.

Replaces the original evaluation's ``np.linspace(0.0, 1.6, 16)`` -- 16 points on a
single ray through one scalar wind factor, at one seed. That construction makes
cache entries near-duplicates of each other and gives the reverse index nothing to
discriminate on.

What this does instead
----------------------
* **A weather process, not a grid.** AR(1) persistence + diurnal cycles + sporadic
  fronts, following the generator already in the repo
  (``evaluate/implementation_specifics/weather.py``), which the CRR evaluation never
  used. Re-implemented here so ``src/`` does not depend on ``evaluate/``, and
  extended with a **wind direction** process -- the original tracks wind speed only,
  but direction is needed for the vector-wind physics.
* **Seeds are environments, not noise draws.** Each seed samples a *climatology*
  (latitude, day-of-year, base temperature, mean wind) from a declared grid, so
  between-seed variance reflects different operating environments rather than
  redrawn Gaussian noise around one hand-written Berlin-midsummer setting.
* **Frequency is a weight, not a selector.** Taking the top-N most-visited cells
  concentrates the cache on the climatological mode, which is exactly where the
  optimum does *not* migrate -- it would quietly cancel the physics. Entries are
  instead sampled **stratified over the wind quantiles** of the visited trace (wind
  being the only re-ranking channel), and visit frequency is carried as a reported
  weight so both a deployment-weighted and an unweighted (stress) view are available.
* **Coarse quantisation, pinned.** At the default granularity the trace shatters into
  thousands of near-unique cells and "top-N" degenerates into an arbitrary tie-break;
  the coarse setting is fixed here as a constant.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .physics import ODDPoint

# Pinned quantisation of the ODD (the "gamma" abstraction). Coarse on purpose:
# fine granularity shatters the trace into near-unique cells and makes any
# frequency-based selection an arbitrary tie-break.
GAMMA: Dict[str, float] = {
    "wind_speed": 1.0,        # [m/s]
    "wind_dir_deg": 45.0,     # [deg] -- 8 sectors
    "temperature": 2.0,       # [degC]
}


@dataclass(frozen=True)
class Climatology:
    """A declared operating environment. Seeds draw from a grid of these."""

    name: str
    lat_deg: float
    day_of_year: int
    base_temp_c: float
    base_wind: float
    prevailing_dir_deg: float


# A declared grid of environments -- deliberately diverse in temperature and
# windiness so that between-seed variance means something.
CLIMATOLOGIES: List[Climatology] = [
    Climatology("berlin_summer",   52.5, 180, 18.0, 3.5, 250.0),
    Climatology("berlin_winter",   52.5, 15,  2.0, 5.5, 240.0),
    Climatology("bergen_autumn",   60.4, 290,  8.0, 7.5, 200.0),
    Climatology("madrid_summer",   40.4, 200, 28.0, 2.5, 220.0),
    Climatology("reykjavik_spring",64.1, 100,  4.0, 9.0, 190.0),
    Climatology("milan_spring",    45.5, 120, 14.0, 2.0, 180.0),
    Climatology("aberdeen_winter", 57.1, 30,  4.0, 8.5, 260.0),
    Climatology("rome_autumn",     41.9, 300, 17.0, 3.0, 210.0),
]


def climatology_for_seed(seed: int) -> Climatology:
    return CLIMATOLOGIES[seed % len(CLIMATOLOGIES)]


def _quantize(value: float, step: float) -> float:
    return round(value / step) * step


def generate_trace(seed: int, hours: float = 240.0, dt_minutes: float = 10.0
                   ) -> Tuple[List[ODDPoint], Climatology]:
    """Simulate an ODD trace: AR(1) persistence + diurnal cycles + weather fronts."""
    clim = climatology_for_seed(seed)
    rng = np.random.RandomState(seed * 7919 + 13)
    n_steps = int(hours * 60.0 / dt_minutes)
    dt_h = dt_minutes / 60.0

    def diurnal_temp(h):
        amp = 6.0 * max(0.3, math.cos(math.radians(clim.lat_deg)) ** 2)
        return amp * math.sin(2 * math.pi * (h - 15.0) / 24.0)

    def diurnal_wind(h):
        return 1.5 + 1.0 * (math.sin(2 * math.pi * (h - 14.0) / 24.0) + 1.0)

    seasonal = 7.0 * math.sin(2 * math.pi * (clim.day_of_year - 205) / 365.0)

    temp = clim.base_temp_c + seasonal + diurnal_temp(8.0)
    wind = clim.base_wind + diurnal_wind(8.0)
    wdir = clim.prevailing_dir_deg

    phi_t, phi_w, phi_d = 0.90, 0.88, 0.94       # AR(1) persistence
    sig_t, sig_w, sig_d = 0.8, 0.9, 12.0         # process noise

    front_left, f_temp, f_wind, f_dir = 0, 0.0, 0.0, 0.0
    out: List[ODDPoint] = []
    hour = 8.0

    for _ in range(n_steps):
        if front_left <= 0 and rng.rand() < dt_h / 48.0:      # ~1 front per 48 h
            front_left = int(rng.randint(3, 12) / dt_h)
            sign = -1.0 if rng.rand() < 0.55 else 1.0
            f_temp = sign * rng.uniform(1.5, 4.0)
            f_wind = rng.uniform(1.0, 5.0)
            f_dir = rng.uniform(-90.0, 90.0)

        mu_t = clim.base_temp_c + seasonal + diurnal_temp(hour)
        mu_w = clim.base_wind + diurnal_wind(hour)
        mu_d = clim.prevailing_dir_deg
        if front_left > 0:
            mu_t += f_temp; mu_w += f_wind; mu_d += f_dir
            front_left -= 1

        temp = mu_t + phi_t * (temp - mu_t) + rng.normal(0.0, sig_t)
        wind = mu_w + phi_w * (wind - mu_w) + rng.normal(0.0, sig_w)
        wdir = mu_d + phi_d * (wdir - mu_d) + rng.normal(0.0, sig_d)

        temp = float(np.clip(temp, -20.0, 45.0))
        wind = float(np.clip(wind, 0.0, 22.0))
        wdir = float(wdir % 360.0)

        out.append(ODDPoint(
            wind_speed=_quantize(wind, GAMMA["wind_speed"]),
            wind_dir_deg=_quantize(wdir, GAMMA["wind_dir_deg"]) % 360.0,
            temperature=_quantize(temp, GAMMA["temperature"]),
        ))
        hour = (hour + dt_h) % 24.0

    return out, clim


@dataclass
class ODDCell:
    """A quantised operating region, with how often the trace visited it."""

    point: ODDPoint
    visits: int
    weight: float = 0.0


def cells_from_trace(trace: Sequence[ODDPoint]) -> List[ODDCell]:
    counts = Counter(trace)
    total = max(1, len(trace))
    cells = [ODDCell(point=p, visits=c, weight=c / total)
             for p, c in counts.items()]
    cells.sort(key=lambda c: (-c.visits, c.point.wind_speed))
    return cells


def stratified_cells(trace: Sequence[ODDPoint], n: int, seed: int = 0) -> List[ODDCell]:
    """Pick ``n`` cells stratified over the WIND quantiles of the visited trace.

    Wind is the only channel that can re-rank configurations, so stratifying on it
    keeps the cache spread across the part of the envelope where the optimum can
    actually move. Selecting the ``n`` most-visited cells instead would concentrate
    the cache on the climatological mode -- precisely where nothing migrates -- and
    would silently cancel the physics.
    """
    cells = cells_from_trace(trace)
    if len(cells) <= n:
        return cells
    rng = np.random.RandomState(seed * 31 + 7)
    winds = np.array([c.point.wind_speed for c in cells])
    edges = np.quantile(winds, np.linspace(0.0, 1.0, n + 1))
    picked: List[ODDCell] = []
    used = set()
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        in_band = [j for j, w in enumerate(winds)
                   if (lo <= w <= hi if i == n - 1 else lo <= w < hi) and j not in used]
        if not in_band:
            in_band = [j for j in range(len(cells)) if j not in used]
        if not in_band:
            break
        # within a band, prefer more-visited cells (weighted draw)
        w = np.array([cells[j].visits for j in in_band], float)
        j = int(rng.choice(in_band, p=w / w.sum()))
        used.add(j)
        picked.append(cells[j])
    picked.sort(key=lambda c: c.point.wind_speed)
    return picked


def effective_dimensionality(trace: Sequence[ODDPoint]) -> np.ndarray:
    """PCA explained-variance of the standardized trace.

    Reported as evidence that the ODD axes are actually independent, rather than one
    axis wearing several names.
    """
    X = np.array([[p.wind_speed, p.wind_dir_deg, p.temperature] for p in trace], float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cov = np.cov(X, rowvar=False)
    ev = np.linalg.eigvalsh(cov)[::-1]
    return ev / ev.sum()
