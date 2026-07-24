# Phase-L2A Feature Source and Supervision Audit v0

## 1. Purpose

Phase-L2A audits the existing Phase-K runtime logs to determine whether
state/context/action features can be reconstructed without new rollouts.

It also clarifies the distinction between rollout-level promotion decisions and
causal action-quality supervision.

---

## 2. Feature Sources

### D5 policy_inputs_v0.csv

Primary source for:

- context
- x
- y
- beta_motion
- beta_stability
- beta_energy
- RAM slip proxy
- RAM roughness proxy
- RAM sigma
- empirical/default reference

This stream provides the canonical state/objective/adaptation backbone.

### D7 objective selector log

Supplementary source for:

- predicted beta
- raw beta
- prior beta
- actual beta
- objective scores
- rollout/context lateral metrics
- RAM proxy means
- reference means

D7 is useful for selector analysis but is not required as the primary L2 state
stream when policy_inputs is available.

### J4 projected reference log

Primary source for numerical candidate action effects:

- empirical reference
- projected candidate reference
- delta_vx
- delta_yaw_rate
- delta_body_h
- delta_clearance
- action residual parameters

Candidate action values must be read from J4 rather than inferred from candidate
names.

### J7 guarded routing log

Authoritative source for actual candidate exposure:

- empirical reference
- projected reference
- actual output reference
- j7_accept
- j7_reason
- j7_source

A candidate is considered actually exposed to the controller only when:

```text
j7_accept == 1
and
j7_source == projected
Candidate proposals rejected by J7 are not treated as realized action exposure.

K4 override log

Audit source for manual Phase-K4C candidate overrides.

It confirms whether the intended profile replaced the upstream J3 candidate.

K4 is useful for validating intervention identity but J7 remains authoritative
for whether the projected command actually reached the active controller route.

K6 outcome dataset

Source for rollout-level outcome and promotion decision:

goal_reached
final_x
final_y
max_abs_y
mean_abs_y
risk_score
decision

K6 decision labels describe rollout-level promotion evidence.

They must not automatically be interpreted as causal per-action labels.

3. Candidate Exposure Definition

Phase-L2 defines an actual candidate exposure row/window using:

J7:
    context != unknown
    j7_accept == 1
    j7_source == projected

Contiguous accepted rows with the same context and action ID form one candidate
exposure window.

Startup rows with unknown context are excluded.

Rejected projected candidates are not associated with realized action effects.

4. Flat Slow03 Finding

The actual Flat Slow03 override is:

vx_scale = 0.97

For the nominal flat empirical reference:

empirical vx = 0.2100
projected vx = 0.2037
delta vx     = -0.0063

Therefore Flat Slow03 represents a 3 percent velocity scaling, not a direct
-0.03 m/s velocity offset.

The candidate was actively accepted by J7 during the flat exposure region.

5. Flat Noop Finding

Flat Noop uses:

vx_scale = 1.0
body_h_delta = 0
clearance_delta = 0

The projected reference is numerically equivalent to the empirical reference.

Despite this, its K6/K7 rollout decision is reject.

This demonstrates that a rollout-level reject label cannot be treated as direct
evidence that the candidate action itself caused degradation.

Rollout stochasticity, reset variability, trajectory history, and other
uncontrolled factors may contribute to the final rollout metrics.

Flat Noop should therefore be retained as a control-equivalent/no-op evidence
sample.

6. Supervision Separation

Phase-L2 separates:

rollout_decision_label

from:

action_effect_label

Initial rollout labels:

empirical
reject
shadow_only

Initial evidence roles:

reference_anchor
control_equivalent_noop
near_miss_exposure
rollout_associated_negative_unpaired

No candidate receives a causal positive or causal negative action-effect label
during L2A.

7. Dataset Granularity

Raw 10 Hz rows must not be treated as independent supervised samples.

The target Phase-L2 granularity is:

one rollout
x one context segment
x one realized candidate exposure window

For each exposure window, aggregate:

context
start/end x and y
mean/max abs(y)
beta statistics
RAM proxy statistics
empirical reference statistics
candidate projected reference statistics
action residual statistics
exposure duration
J7 routing evidence
rollout-level outcome

This prevents pseudo-replication of one rollout label across hundreds of
correlated time samples.

8. Phase-L2 Decision

Phase-L2 will first construct a window-level evidence dataset.

It is not yet treated as the final selector training dataset.

Direct classifier or RL training is deferred until candidate effects can be
compared against suitable empirical references using segment-level or paired
outcome evidence.
