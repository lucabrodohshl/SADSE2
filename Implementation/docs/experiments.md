# Experiments & Methodology

To address Meta-Review Point 1 ("The current approach assumes perfect knowledge... provide sensitivity & robustness analyses"), you do not need to change your core algorithm or the caching mechanism.

Instead, you need to add a "stress test" to your evaluation section. You must demonstrate how much error your system can tolerate before it breaks.

Here is a concrete plan for the experiments and analyses you can run to satisfy this request.

The Concept: "Sim-to-Real" Mismatch

Currently, your Digital Twin (DT) and Physical Twin (PT) likely use the exact same mathematical functions. To test robustness, you must decouple them:

DT (Optimization/Cache): Uses the "Ideal Model" (static 
E
E
 and 
Γ
Γ
).

PT (Evaluation): Uses a "Perturbed Model" (noisy, drifting, or biased).

Experiment A: Sensitivity to Objective Function Mismatch (
E
E
)

"What happens if the energy model is wrong (e.g., due to rotor wear or unmodeled drag)?"

1. Setup

Create a modified energy function for the Physical Twin only:

Ereal(x)=Emodel(x)⋅(1+δ)+ϵ
E
real
	​

(x)=E
model
	​

(x)⋅(1+δ)+ϵ

δ
δ
 (Bias): Represents systematic error (e.g., old battery, worn propellers).

ϵ
ϵ
 (Noise): Represents random fluctuation (Gaussian noise).

2. The Experiment Loop

Run your standard simulation (e.g., Scenario 1 from your paper) with the Cache enabled.

Step 1: DT detects ODD, queries Cache, retrieves optimal config 
xcached∗
x
cached
∗
	​

.

Step 2: Execute 
xcached∗
x
cached
∗
	​

 in the PT using 
Ereal
E
real
	​

.

Step 3: Record two metrics:

Real Energy Consumption: The value from 
Ereal
E
real
	​

.

Constraint Violation: Did 
Ereal
E
real
	​

 exceed the battery capacity?

3. What to Vary (The X-Axis)

Run the simulation multiple times, increasing the Bias (
δ
δ
) from 0% to 20%.

4. What to Plot

Create a line chart: "Impact of Model Mismatch on Feasibility"

X-Axis: Model Error Magnitude (0%, 5%, 10%, 15%, 20%).

Y-Axis (Left): Average Energy Consumption (showing it rising).

Y-Axis (Right - Bar chart): % of Missions Failed (Battery Depletion).

Expected Insight: You will likely find that for small errors (e.g., <5%), the system remains feasible because optimal solutions often lie slightly inside the boundaries, or the battery constraint wasn't the active binding constraint. For large errors (>15%), you will see failures. This quantifies the robustness.

Experiment B: Sensitivity to Abstraction Boundaries (
Γ
Γ
)

"What happens if the ODD abstraction is too rigid?"

The reviewer is worried about "Edge Cases" in your 
Γ
Γ
 function. For example, if you classify Wind Speed as:

Low: 0–5 m/s

High: 5–10 m/s

What happens when the wind is 4.9 m/s? The DT thinks it is "Low" and might pick a configuration with low engine power. If a gust hits (making it 5.1 m/s), does the drone crash?

1. Setup

Identify the boundary values in your 
Γ
Γ
 function (e.g., wind speed thresholds).

Force the simulation weather generator to hover exactly around these boundaries with high frequency.

Add Noise to the PT's environmental reading (e.g., real_wind = sensed_wind + random(-0.5, 0.5)).

2. The Experiment

Compare two baselines:

Strict Caching (Your approach): Uses the cached solution for "Low Wind" even when wind is 4.9 m/s.

Conservative/Robust Caching (Variant): If the input is within 10% of a boundary (e.g., 4.5 to 5.0 m/s), assume the worst-case bin (Treat as "High Wind").

3. What to Report

You don't necessarily need a plot here. A Table is sufficient.

Column 1: Scenario (Boundary Conditions).

Column 2: Success Rate (Standard Cache).

Column 3: Success Rate (Robust Variant - optional, only if you have time to implement).

Defense Argument: If the standard cache fails at boundaries, you can argue (in the text) that the Definition of 
Γ
Γ
 should include "Buffer Zones" or "Hysteresis" (which you mentioned in your Future Work/Discussion), thus solving the problem architecturally.

Implementation Checklist (How to code this quickly)

You can likely reuse your existing eval.py or simulation script.

1. Define the Perturbation Function:

code
Python
download
content_copy
expand_less
import numpy as np

def calculate_real_cost(predicted_cost, degradation_factor=0.0):
    # Degradation: e.g., 0.1 means 10% more energy used than modeled
    bias = 1.0 + degradation_factor
    
    # Noise: Random fluctuation +/- 2%
    noise = np.random.normal(1.0, 0.02) 
    
    return predicted_cost * bias * noise

2. Modify the Evaluation Loop:

code
Python
download
content_copy
expand_less
# ... inside your simulation loop ...

# 1. DT / Cache Logic (unchanged)
config_optimal, predicted_cost = digital_twin.get_optimization(current_odd)

# 2. PT / Reality Logic (The new part)
# Inject the mismatch here
real_cost = calculate_real_cost(predicted_cost, degradation_factor=0.10) 

# 3. Check Feasibility
battery_limit = current_odd.battery_capacity
if real_cost > battery_limit:
    log_violation("Battery Depletion", current_time)
else:
    log_success(real_cost)

3. Run the Sweep:
Script a loop to run the whole 10-hour simulation for degradation_factor = [0.0, 0.05, 0.10, 0.15, 0.20].

How to Write this in the Revision

Create a new subsection 6.4 Robustness Analysis.

"To address the concern regarding static model assumptions (Meta-Review Point 1), we performed a sensitivity analysis to quantify the impact of model-reality mismatch. We introduced a synthetic Sim-to-Real gap where the Physical Twin's energy consumption deviates from the Digital Twin's model by a parameterized bias 
δ
δ
 and Gaussian noise 
ϵ∼N(0,0.02)
ϵ∼N(0,0.02)
."

Then, describe your results:

"Fig. X shows that our approach remains robust up to a systematic model error of X%. Specifically, the Zonotope constraints provide a natural safety margin; even when energy consumption was 5% higher than predicted, ODD violations remained near zero (0.1%). However, as expected, beyond 10% error, the cached configurations—which are optimized to be tight against constraints—began to violate battery limits, highlighting the need for the feedback mechanisms discussed in Section 7."

Why this works:

It directly answers the "What if?" question with data.

It admits the limitation (it breaks at >10% error) which makes you look scientifically honest.

It perfectly sets up the "Feedback Loop" discussion (Meta-Review Point 2) as the solution to that limitation.