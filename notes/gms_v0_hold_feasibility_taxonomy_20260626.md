# TRACER GMS V0 Hold Feasibility Taxonomy

Date: 2026-06-26  
Branch: telecommunicating-structure

## 1. Purpose

This note documents why the V0 Gait Mode Selector classifies some slippery / soft terrain cases as:

- no_valid_forward_recovery_needed

rather than as:

- active_hold_or_slow_probe
- slow_forward
- cautious_forward

The key finding is that several terrains failed even under a zero-vx hold command. Therefore, the failure is not simply caused by excessive commanded forward velocity.

The current conclusion is not that TRACER must immediately implement a recovery primitive. Instead, the conclusion is:

- these terrains are outside the currently valid high-level action / low-level controller regime
- GMS should output a critical invalid-primitive flag
- the system should not attempt forward traversal on these terrains in V0
- these cases should be used as negative / critical examples for RAM, GMS, and Objective Selector learning

## 2. Tested hold command

The common zero-vx hold command used for the feasibility check was:

- layout: [yaw_rate, vx, vy, body_height, swing_clearance, enable]
- command: [0.0, 0.0, 0.0, 0.325, 0.04, 1.0]

Expected behavior:

- vx near zero
- body height remains near the commanded height
- x/y drift remains small
- roll/pitch remain bounded
- no body collapse

Observed behavior on critical slippery / soft terrains:

- unintended drift
- body height collapse
- severe roll/pitch in some cases
- fall / invalid hold

This means that these terrains should not be assigned a forward traversal command merely by reducing speed.

## 3. V0 hold feasibility evidence

The distilled evidence is stored in:

- data/rollout_metrics/gms_hold_feasibility_matrix_v0.csv
- data/rollout_metrics/gms_hold_feasibility_matrix_v0.json

Current critical cases:

| terrain_name | world_name | result |
|---|---|---|
| icy_slippery_even_forward | tracer_slippery_flat | hold failed |
| sponge_like_even_forward | tracer_sponge_like_flat | hold failed |
| icy_slippery_downslope5_forward | tracer_slippery_downslope_5deg | hold failed |
| sponge_like_downslope5_forward | tracer_sponge_firm_downslope_5deg | hold failed |

All four are assigned:

- hold_feasible: false
- decision: no_valid_forward_recovery_needed

## 4. Final V0 GMS taxonomy

The current V0 GMS taxonomy is:

- normal_forward
- high_clearance_slow_probe
- cautious_probe
- no_valid_forward_recovery_needed

### normal_forward

Used when the terrain is feasible for normal forward locomotion.

Examples:

- solid_even_forward
- solid_downslope5_forward
- rough_bumps_start020_forward

### high_clearance_slow_probe

Used when the terrain is traversable, but requires more cautious foot clearance or slower progression.

Example:

- solid_upslope5_forward

### cautious_probe

Used for terrains where forward motion may be possible, but should be slow and monitored.

Example:

- rough_bumps_start018_forward

### no_valid_forward_recovery_needed

Used when the current forward / hold primitive is not valid under the current controller.

Current examples:

- icy_slippery_even_forward
- sponge_like_even_forward
- icy_slippery_downslope5_forward
- sponge_like_downslope5_forward
- mud_slippery_even_forward
- icy_slippery_upslope5_forward
- mud_slippery_upslope5_forward
- sponge_like_upslope5_forward

For V0, this mode should be interpreted as a flag, not as a new recovery primitive.

The flag means:

- do not issue a forward traversal command
- record the state as critical / invalid primitive
- use the trajectory as negative evidence for RAM, GMS, and Objective Selector training
- defer concrete recovery primitive design to a later low-level controller stage

## 5. Why active_hold_or_slow_probe is deprecated in V0

The previous interpretation was:

- if terrain is risky, reduce vx and try active hold or slow probe

The zero-vx hold experiments show this is too optimistic.

If the robot fails even at:

- vx = 0.0
- enable = 1.0
- body-height hold command active

then the problem is not just forward velocity.

The terrain/controller pair has entered a regime where the current primitive cannot guarantee posture stability.

Therefore, active_hold_or_slow_probe is deprecated for V0 critical terrain handling.

It may be revisited later only after the low-level controller is improved with stronger stance control, friction-aware MPC/WBC, adaptive impedance residuals, or an explicit recovery/safety layer.

## 6. Connection to TRACER modules

### RAM

RAM should detect the mismatch between commanded behavior and actual physical response.

Example:

Commanded:

- vx = 0
- body height hold
- low drift expected

Observed:

- drift
- z collapse
- roll/pitch instability
- fall

This is exactly the type of online reality-gap / mismatch signal that RAM should encode into:

- rho
- sigma
- risk heads
- recovery_needed / invalid_primitive likelihood

### Gait Mode Selector

GMS is directly supported by these results.

The experiments show that GMS should not be a simple speed-scaling module.

Instead, GMS must answer:

- is this locomotion primitive valid in the current terrain/controller regime?

If not, it should output:

- no_valid_forward_recovery_needed

In V0, this is a flag that prevents invalid forward traversal, not a request to execute a new recovery primitive.

### Objective Selector

Objective Selector is not directly validated by these hold tests yet, because beta is still mostly fixed or externally configured.

However, these results strongly motivate Objective Selector.

For feasible terrain, the desired objective can prioritize progress, stability, and energy depending on context.

For critical terrain, the desired objective should shift away from progress and toward safety / non-deployment / invalid-action avoidance.

In beta notation:

Solid terrain:

- beta_v high
- beta_s moderate
- beta_e moderate

Rough but feasible terrain:

- beta_v and beta_s balanced

Critical slippery / soft terrain:

- beta_s dominant
- beta_v very low
- beta_e secondary

The newly exposed beta environment hook in the Isaac reward config allows controlled beta-conditioned experiments.

## 7. Immediate next step

The next step is not to build a recovery primitive.

The immediate next step is to verify the closed-loop TRACER structure on feasible terrains where locomotion can be maintained.

Target validation route:

- terrain / context
- GMS mode
- Objective beta setting
- high-level planner or policy table
- Decoder / Mapper
- low-level reference
- A1-QP-MPC Gazebo controller
- trajectory-level rollout metrics

The first validation target is:

- Does the high-level planner with the low-level controller work on feasible terrains?

Critical terrains should remain in the dataset as invalid / negative examples, but they should not dominate the next closed-loop validation step.

## 8. V0 conclusion

The V0 evidence supports the following interpretation:

Some slippery / soft terrains are not merely slow locomotion cases. They are invalid primitive cases under the current controller.

Therefore, the V0 GMS should classify them as:

- no_valid_forward_recovery_needed

For now, this mode is a flag, not a new primitive.

This closes the V0 semantic decision and motivates the next stage:

1. validate closed-loop high-level-to-low-level locomotion on feasible terrains
2. collect trajectory-level rollout data
3. train RAM / GMS / Objective Selector from simulation rollouts
4. later transfer the learned adaptation stack to the real robot
