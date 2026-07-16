"""Drone energy models over a multi-dimensional ODD, and the M0 -> M1 fidelity ladder.

Why this module exists
----------------------
The original CRR evaluation scaled energy by a *uniform* factor ``E * (1 + wind)``
(``src/crr/model.py::_wind_scaled``). A uniform positive scale cannot change
``argmin_k sum_j E[k,j]``, so the cached optimum is *provably* immortal along the
ODD axis and no refinement expressed as a scale can ever move it. Worse, the
original Type-III refinement is the *same* uniform-scale operator, so it is a
tautology rather than a measurement.

What is modelled, and what is deliberately not
---------------------------------------------
Only channels that measurably reach the optimum are kept. Each was checked for
whether it can move ``argmin_k``:

* **Wind (the sole re-ranking channel).** Propulsion is driven by *airspeed*
  while task time is driven by *ground speed*, so::

      E(v, w) = (A_k + B*(v + w)^2) * L / (v * 3600)
      dE/dw   = 2B(v + w)/v * L/3600      -- depends on v

  A headwind penalises *slow* configurations disproportionately, so the ranking
  genuinely changes across the envelope and the energy-optimal speed shifts to
  ``v* = sqrt(A/B + w^2)``. Wind is applied **per leg as a vector**: an out-and-back
  route does not cancel, because the headwind leg is flown for longer than the
  tailwind leg, so energy is asymmetric even when displacement is not.
* **Temperature — a feasibility (RHS) channel only.** It derates usable battery
  capacity and (via air density) the drag coefficient. It cannot re-rank on its own
  because capacity enters only the battery constraint. Modelled with a *bounded,
  non-monotone* derate: real LiPo usable capacity peaks around 20-35 degC and falls
  off both colder and hotter. (An Arrhenius factor would be monotone increasing and
  would nonsensically predict >100% of nameplate at 50 degC; Arrhenius describes
  reaction rate and calendar ageing, not instantaneous usable capacity.)
* **Humidity — deliberately excluded from the energy model.** Over the RH span the
  cache actually covers it changes air density by ~0.1-0.2% and total power by
  <0.1%, three orders of magnitude below the config-cost differences that decide
  the argmin. It is also commonly stated with the sign backwards: moist air is
  *less* dense than dry air (H2O is 18 g/mol vs ~29 for dry air), so humidity
  slightly *reduces* required power. Including it would be physics-washing.
* **Visibility — not an ODD dimension here.** In the trace generator it is a
  deterministic function of humidity and wind (R^2 ~ 0.87) and enters no physics.

The M0 -> M1 fidelity ladder (the Type-III refinement)
-----------------------------------------------------
``M0`` is the existing published low-fidelity model: parasitic drag only,
``P ~ v_air^2`` (this is ``src/milp_solver.py::drone_energy_model``). ``M1`` is the
standard rotary-wing model (Zeng, Zhang & Lim, IEEE TWC 2019), which **adds** the
induced-power branch that dominates at low airspeed. Nothing is deleted to create
M0 -- it is a real model that a real engineer would plausibly have started from,
and M1 is a documented fidelity increase. The two differ **non-uniformly** across
configurations, which is exactly what lets a linearisation refinement re-rank them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# --- Platform constants, aligned with src/milp_solver.py::drone_energy_model ---
P0_W = 120.0              # baseline hover/cruise + avionics draw [W]
V_REF = 10.0              # reference speed for drag scaling [m/s]
K_DRAG_FRAC = 0.6         # fraction of P0 at V_REF due to parasitic drag
K_ALT_PER_M = 0.001       # +0.1% of P0 per metre of altitude
P_CAM_PER_MP = 0.25       # [W/MP]
P_SENSOR_PER_HZ = 0.10    # [W/Hz]
SENSOR_HZ = 10.0          # fixed sampling rate for this study

# --- Rotary-wing (Zeng et al. 2019) constants, Mavic/Phantom class ---
P_BLADE = 79.86           # blade profile power [W]
P_IND = 88.63             # induced power in hover [W]
U_TIP = 120.0             # rotor tip speed [m/s]
V0_ROTOR = 4.03           # mean rotor induced velocity in hover [m/s]
D0_DRAG = 0.6             # fuselage drag ratio
ROTOR_S = 0.05            # rotor solidity
ROTOR_A = 0.503           # rotor disc area [m^2]
RHO_0 = 1.225             # sea-level air density [kg/m^3]

# --- Battery derating (bounded, non-monotone; NOT Arrhenius) ---
T_COLD_REF = 25.0         # capacity plateau upper-cold reference [degC]
T_HOT_REF = 45.0          # above this, BMS derating kicks in [degC]
DERATE_COLD = 0.006       # per degC below T_COLD_REF
DERATE_HOT = 0.004        # per degC above T_HOT_REF
DERATE_FLOOR = 0.55       # never below 55% of nameplate


@dataclass(frozen=True)
class ODDPoint:
    """A point in the operational design domain.

    ``wind_speed``/``wind_dir_deg`` are a *vector*: direction matters because an
    out-and-back route does not cancel in energy.
    """

    wind_speed: float        # [m/s]
    wind_dir_deg: float      # [deg] direction the wind blows TOWARDS
    temperature: float       # [degC]

    def wind_vector(self) -> np.ndarray:
        th = math.radians(self.wind_dir_deg)
        return np.array([math.cos(th), math.sin(th)]) * self.wind_speed

    @staticmethod
    def dims() -> Tuple[str, ...]:
        return ("wind_speed", "wind_dir_deg", "temperature")


def air_density(temperature_c: float, altitude_m: float = 0.0) -> float:
    """Dry-air density [kg/m^3] from temperature and altitude (barometric)."""
    t_k = temperature_c + 273.15
    p = 101325.0 * (1.0 - 2.25577e-5 * max(0.0, altitude_m)) ** 5.25588
    return p / (287.058 * t_k)


def battery_derate(temperature_c: float) -> float:
    """Usable-capacity fraction: bounded, peaks in 25-45 degC, falls off both ways."""
    loss = (DERATE_COLD * max(0.0, T_COLD_REF - temperature_c) ** 1.1
            + DERATE_HOT * max(0.0, temperature_c - T_HOT_REF))
    return max(DERATE_FLOOR, 1.0 - loss)


def _cfg(config) -> Dict[str, float]:
    return config.as_dict() if hasattr(config, "as_dict") else dict(config)


def avionics_draw_W(config, payload_kg: float = 0.0) -> float:
    """Non-propulsion, speed-independent load [W]: altitude term, camera, sensors, payload.

    Deliberately excludes the platform's hover/cruise baseline: that is part of the
    *propulsion* model and differs between M0 and M1. Folding it in here would
    double-count it against M1's ``P_BLADE + P_IND`` hover power.
    """
    d = _cfg(config)
    return (
        P0_W * K_ALT_PER_M * max(0.0, float(d.get("altitude", 50.0)))
        + P_CAM_PER_MP * max(0.0, float(d.get("camera_res", 8.0)))
        + P_SENSOR_PER_HZ * SENSOR_HZ
        + 22.0 * max(0.0, payload_kg)
    )


# ---------------------------------------------------------------------------
# M0: the existing published low-fidelity model -- flat baseline + parasitic v^2
# ---------------------------------------------------------------------------
def propulsion_power_M0(v_air: float, rho: float = RHO_0) -> float:
    """Baseline hover/cruise draw plus a parasitic drag term quadratic in airspeed.

    This is ``src/milp_solver.py::drone_energy_model``'s propulsion model. Its
    defect as a rotorcraft model is that it is monotone increasing in airspeed: it
    has no induced-power branch, so it cannot represent the fact that a rotor works
    *harder* in slow flight than in cruise.
    """
    B = P0_W * K_DRAG_FRAC / (V_REF ** 2) * (rho / RHO_0)
    return P0_W + B * max(0.0, v_air) ** 2


# ---------------------------------------------------------------------------
# M1: rotary-wing model (Zeng, Zhang & Lim 2019) -- ADDS the induced-power branch
# ---------------------------------------------------------------------------
def propulsion_power_M1(v_air: float, rho: float = RHO_0) -> float:
    """Blade-profile + induced + parasite power for a rotary-wing platform.

    ``P_BLADE + P_IND`` is the hover power, so this term already carries the
    platform baseline that M0 models as the flat ``P0_W``.
    """
    v = max(0.0, v_air)
    blade = P_BLADE * (1.0 + 3.0 * v ** 2 / U_TIP ** 2)
    inner = math.sqrt(1.0 + v ** 4 / (4.0 * V0_ROTOR ** 4)) - v ** 2 / (2.0 * V0_ROTOR ** 2)
    induced = P_IND * math.sqrt(max(0.0, inner))
    parasite = 0.5 * D0_DRAG * rho * ROTOR_S * ROTOR_A * v ** 3
    return blade + induced + parasite


def leg_energy(config, distance_m: float, heading_deg: float, odd: ODDPoint,
               fidelity: str = "M0", payload_kg: float = 0.0) -> float:
    """Energy [Wh] to fly one straight leg of ``distance_m`` on ``heading_deg``.

    Ground speed is held at the configuration's commanded speed; the airspeed the
    rotors must produce is ``|v_ground_vec - v_wind_vec|``. Because time on the leg
    is set by ground speed, a headwind leg costs more *and* lasts longer than the
    matching tailwind leg -- the asymmetry that makes wind direction matter.
    """
    d = _cfg(config)
    v_g = max(0.1, float(d.get("speed", 10.0)))
    alt = float(d.get("altitude", 50.0))

    th = math.radians(heading_deg)
    v_ground_vec = np.array([math.cos(th), math.sin(th)]) * v_g
    v_air_vec = v_ground_vec - odd.wind_vector()
    v_air = float(np.linalg.norm(v_air_vec))

    rho = air_density(odd.temperature, alt)
    prop = propulsion_power_M0(v_air, rho) if fidelity == "M0" else propulsion_power_M1(v_air, rho)
    total_power = avionics_draw_W(config, payload_kg) + prop
    hours = max(0.0, distance_m) / (v_g * 3600.0)
    return total_power * hours


def task_energy(config, task, odd: ODDPoint, fidelity: str = "M0",
                payload_kg: float = 0.0, transit_m: float = 0.0,
                transit_heading_deg: float = 0.0) -> float:
    """Energy [Wh] for a task: optional out-and-back transit plus the survey itself.

    The transit is flown out on ``transit_heading_deg`` and back on the reverse
    heading, so the out/back wind asymmetry is retained rather than cancelled.
    """
    length = float(task.get("length", 100.0)) if isinstance(task, dict) else float(task)
    heading = float(task.get("heading_deg", 0.0)) if isinstance(task, dict) else 0.0

    e = leg_energy(config, length, heading, odd, fidelity, payload_kg)
    if transit_m > 0.0:
        e += leg_energy(config, transit_m, transit_heading_deg, odd, fidelity, payload_kg)
        e += leg_energy(config, transit_m, transit_heading_deg + 180.0, odd, fidelity, payload_kg)
    return e


def usable_budget(capacity_wh: float, reserve: float, safety: float,
                  odd: ODDPoint, soh: float = 1.0) -> float:
    """Per-agent usable energy [Wh] after reserve, safety factor, temperature and SoH."""
    nominal = (1.0 - reserve) * capacity_wh / max(1e-9, safety)
    return max(0.0, nominal * battery_derate(odd.temperature) * max(0.0, soh))
